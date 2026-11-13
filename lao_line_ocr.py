#!/usr/bin/env python3
"""
CPU OCR for Lao pages: a text-line detector and a Lao CTC line recognizer on ONNX Runtime.

Same two-stage shape as iapp/OpenThai2.0-OCR-Text for Thai:
    page -> PP-OCRv6-tiny detector (used as shipped) -> line boxes in reading order
         -> crops, 48 px high -> Lao recognizer (PP-OCRv6-tiny fine-tuned by train_ppocr_rec.py)
         -> greedy CTC decode -> normalize_lao_text -> text

Model dir (written by train_ppocr_rec.py):
    det.onnx       detector
    rec.onnx       recognizer, output = per-frame class probabilities
    rec_dict.txt   one character per line; CTC classes are blank + dictionary + space

Usage:
    python lao_line_ocr.py models/lao_tiny page.png
    python lao_line_ocr.py models/lao_tiny page.png --json
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

REC_HEIGHT = 48
DET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
DET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ----------------------------------------------------------------------------
#  Shared with training (train_ppocr_rec.py): keep both sides identical
# ----------------------------------------------------------------------------

def load_bgr(path):
    """Read an image as 3-channel BGR uint8 (also works with non-ASCII paths)."""
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'cannot read image: {path}')
    return img


def prep_line(img, height=REC_HEIGHT, max_w=3200):
    """BGR line image -> float32 (3, height, W) in [-1, 1]; W follows the aspect ratio, at most max_w."""
    h, w = img.shape[:2]
    rw = min(max_w, max(8, math.ceil(height * w / max(h, 1))))
    x = cv2.resize(img, (rw, height)).astype(np.float32) / 255.0
    return ((x - 0.5) / 0.5).transpose(2, 0, 1)


def pad_batch(lines):
    """Stack (3, H, W_i) arrays into (N, 3, H, max W); padding is 0 (mid-grey after normalizing)."""
    batch = np.zeros((len(lines), 3, lines[0].shape[1], max(x.shape[2] for x in lines)), np.float32)
    for i, x in enumerate(lines):
        batch[i, :, :, :x.shape[2]] = x
    return batch


def load_chars(path):
    """CTC class list: index 0 = blank, then the dictionary, then space (PaddleOCR's layout)."""
    with open(path, encoding='utf-8') as f:
        return [''] + [c for c in f.read().split('\n') if c] + [' ']


def ctc_greedy(probs, chars):
    """(T, C) per-frame probabilities -> (text, mean confidence of the kept frames)."""
    ids, conf = probs.argmax(1), probs.max(1)
    keep = np.ones_like(ids, bool)
    keep[1:] = ids[1:] != ids[:-1]
    keep &= ids != 0
    text = ''.join(chars[i] if i < len(chars) else '' for i in ids[keep])
    return text, float(conf[keep].mean()) if keep.any() else 0.0


# ----------------------------------------------------------------------------
#  Detection helpers (DB post-processing, as in PaddleOCR)
# ----------------------------------------------------------------------------

def _order_points(pts):
    """4 points -> top-left, top-right, bottom-right, bottom-left."""
    pts = pts[np.argsort(pts[:, 0])]
    left, right = pts[:2][np.argsort(pts[:2, 1])], pts[2:][np.argsort(pts[2:, 1])]
    return np.array([left[0], right[0], right[1], left[1]], np.float32)


def _min_box(contour):
    rect = cv2.minAreaRect(contour)
    return _order_points(cv2.boxPoints(rect)), min(rect[1])


def _box_score(prob, box):
    h, w = prob.shape
    xs, ys = box[:, 0], box[:, 1]
    x0, x1 = int(np.clip(np.floor(xs.min()), 0, w - 1)), int(np.clip(np.ceil(xs.max()), 0, w - 1))
    y0, y1 = int(np.clip(np.floor(ys.min()), 0, h - 1)), int(np.clip(np.ceil(ys.max()), 0, h - 1))
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    cv2.fillPoly(mask, [(box - [x0, y0]).astype(np.int32)], 1)
    return cv2.mean(prob[y0:y1 + 1, x0:x1 + 1], mask)[0]


def _unclip(box, ratio):
    import pyclipper
    area, length = cv2.contourArea(box), cv2.arcLength(box, True)
    if length == 0:
        return None
    off = pyclipper.PyclipperOffset()
    off.AddPath([tuple(p) for p in box.tolist()], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    out = off.Execute(area * ratio / length)
    return np.array(out[0], np.float32) if out else None


def crop_line(img, box):
    """Cut one quad out of the page and straighten it into a line image."""
    w = max(1, int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3]))))
    h = max(1, int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2]))))
    m = cv2.getPerspectiveTransform(box.astype(np.float32), np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32))
    return cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)


def reading_order(boxes):
    """Group boxes into rows (top to bottom), each row left to right. Returns lists of box indices."""
    if not boxes:
        return []
    cy = [float(b[:, 1].mean()) for b in boxes]
    cx = [float(b[:, 0].mean()) for b in boxes]
    tol = 0.5 * float(np.median([np.linalg.norm(b[0] - b[3]) for b in boxes]))
    rows = []
    for i in sorted(range(len(boxes)), key=lambda k: cy[k]):
        if rows and abs(cy[i] - np.mean([cy[j] for j in rows[-1]])) <= tol:
            rows[-1].append(i)
        else:
            rows.append([i])
    return [sorted(r, key=lambda k: cx[k]) for r in rows]


# ----------------------------------------------------------------------------
#  Engine
# ----------------------------------------------------------------------------

class LaoLineOCR:
    def __init__(self, model_dir, threads=1, det_limit=1280, det_thresh=0.3, box_thresh=0.6,
                 unclip=1.5, v_pad=0.0, batch=16):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.det = ort.InferenceSession(os.path.join(model_dir, 'det.onnx'), opts, providers=['CPUExecutionProvider'])
        self.rec = ort.InferenceSession(os.path.join(model_dir, 'rec.onnx'), opts, providers=['CPUExecutionProvider'])
        self.chars = load_chars(os.path.join(model_dir, 'rec_dict.txt'))
        self.det_limit, self.det_thresh, self.box_thresh = det_limit, det_thresh, box_thresh
        self.unclip, self.v_pad, self.batch = unclip, v_pad, batch
        cv2.setNumThreads(threads)

    def detect(self, img):
        """Page (BGR) -> list of 4x2 float32 boxes (TL, TR, BR, BL) in page pixels."""
        H0, W0 = img.shape[:2]
        pad = max(8, int(0.05 * min(H0, W0)))  # white border: text touching the edge still closes
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        h, w = img.shape[:2]
        s = min(self.det_limit / max(h, w), 4.0)
        rh, rw = max(32, round(h * s / 32) * 32), max(32, round(w * s / 32) * 32)
        x = cv2.resize(img, (rw, rh)).astype(np.float32) / 255.0
        x = ((x - DET_MEAN) / DET_STD).transpose(2, 0, 1)[None]
        prob = self.det.run(None, {self.det.get_inputs()[0].name: x})[0][0, 0]
        contours, _ = cv2.findContours((prob > self.det_thresh).astype(np.uint8), cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        sx, sy = w / prob.shape[1], h / prob.shape[0]
        boxes = []
        for cnt in contours[:3000]:
            box, side = _min_box(cnt)
            if side < 3 or _box_score(prob, box) < self.box_thresh:
                continue
            big = _unclip(box, self.unclip)
            if big is None:
                continue
            box, side = _min_box(big.reshape(-1, 1, 2))
            if side < 5:
                continue
            box[:, 0] = np.clip(box[:, 0] * sx - pad, 0, W0 - 1)
            box[:, 1] = np.clip(box[:, 1] * sy - pad, 0, H0 - 1)
            if self.v_pad:  # room for Lao marks above/below the core band: grow height only
                hh = max(np.linalg.norm(box[0] - box[3]), 1.0) * self.v_pad / 2
                box[[0, 1], 1] = np.clip(box[[0, 1], 1] - hh, 0, H0 - 1)
                box[[2, 3], 1] = np.clip(box[[2, 3], 1] + hh, 0, H0 - 1)
            if np.ptp(box[:, 0]) >= 2 and np.ptp(box[:, 1]) >= 2:
                boxes.append(box)
        return boxes

    def recognize(self, crops):
        """Line images (BGR) -> [(text, confidence)], batched by similar width."""
        lines = [prep_line(c) for c in crops]
        out = [('', 0.0)] * len(lines)
        order = sorted(range(len(lines)), key=lambda i: lines[i].shape[2])
        for b in range(0, len(order), self.batch):
            idx = order[b:b + self.batch]
            probs = self.rec.run(None, {self.rec.get_inputs()[0].name: pad_batch([lines[i] for i in idx])})[0]
            for n, i in enumerate(idx):
                out[i] = ctc_greedy(probs[n][:max(1, lines[i].shape[2] // 8)], self.chars)
        return out

    def read(self, image):
        """Image path or BGR array -> {'text', 'lines': [{'text', 'confidence', 'box', 'row'}]}."""
        from gen_data_from_text import normalize_lao_text
        img = load_bgr(image) if isinstance(image, str) else image
        boxes = self.detect(img)
        rows = reading_order(boxes)
        flat = [i for r in rows for i in r]
        results = self.recognize([crop_line(img, boxes[i]) for i in flat])
        lines, by_box = [], dict(zip(flat, results))
        for row_no, r in enumerate(rows):
            for i in r:
                text, conf = by_box[i]
                lines.append({'text': normalize_lao_text(text), 'confidence': round(conf, 4),
                              'box': boxes[i].round(1).tolist(), 'row': row_no})
        text = '\n'.join(' '.join(ln['text'] for ln in lines if ln['row'] == n and ln['text'])
                         for n in range(len(rows)))
        return {'text': text, 'lines': lines}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model_dir', help='Dir with det.onnx, rec.onnx, rec_dict.txt')
    p.add_argument('images', nargs='+')
    p.add_argument('--json', action='store_true', help='Print lines with boxes and confidences as JSON')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--v-pad', type=float, default=0.0,
                   help='Grow each box height by this share, if tone marks or vowels get cut off (e.g. 0.2)')
    args = p.parse_args(argv)

    ocr = LaoLineOCR(args.model_dir, threads=args.threads, v_pad=args.v_pad)
    for path in args.images:
        result = ocr.read(path)
        if args.json:
            print(json.dumps({'image': path, **result}, ensure_ascii=False, indent=1))
        else:
            if len(args.images) > 1:
                print(f'==> {path} <==')
            print(result['text'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
