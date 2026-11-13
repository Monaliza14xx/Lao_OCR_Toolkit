#!/usr/bin/env python3
"""
Fine-tune the PP-OCRv6-tiny line recognizer for Lao in PyTorch, then export it for lao_line_ocr.py.

The recipe of iapp/OpenThai2.0-OCR-Text, for Lao: keep the pretrained backbone, give the
last layer one output per Lao dictionary character (+ CTC blank + space), train with CTC
on 48 px high line images. The detector is used as shipped.

Input: a dataset dir made by to_paddleocr.py (train_list.txt, val_list.txt, lao_dict.txt).

    python train_ppocr_rec.py train --data paddle_data --out models/lao_tiny --epochs 30
    python train_ppocr_rec.py export --out models/lao_tiny       # re-export only
    python lao_line_ocr.py models/lao_tiny page.png

--init picks the starting weights: the official PP-OCRv6-tiny recognizer (default), any
transformers checkpoint of the same model, a .pt state dict (e.g. OpenThai's
pytorch/rec_weights.pt), or "none" for random weights. Tensors whose shape does not
match (the output layer) start fresh.

Output dir: best/ (transformers checkpoint, lowest val CER), rec.onnx, rec_dict.txt, det.onnx.
"""

import argparse
import math
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from gen_data_from_text import normalize_lao_text
from lao_line_ocr import REC_HEIGHT, load_bgr, load_chars, pad_batch, prep_line, ctc_greedy
from tesstrain_core.metrics import strict_cer_counts

DEFAULT_INIT = 'PaddlePaddle/PP-OCRv6_tiny_rec_safetensors'
DEFAULT_DET = 'PaddlePaddle/PP-OCRv6_tiny_det_onnx'
STRIDE = 8  # input pixels per output frame

# Architecture of PaddlePaddle/PP-OCRv6_tiny_rec_safetensors (config.json), for --init none / .pt
TINY_CONFIG = {
    'backbone_config': {
        'model_type': 'pp_lcnet_v4', 'stem_channels': [3, 24, 48], 'stem_type': 'small',
        'out_features': ['stage1', 'stage2', 'stage3', 'stage4'], 'out_indices': [1, 2, 3, 4],
        'block_configs': [
            [[3, 48, 48, 1, True]],
            [[3, 48, 48, 1, False]],
            [[3, 48, 96, [2, 1], False], [3, 96, 96, 1, True], [3, 96, 96, 1, False]],
            [[3, 96, 160, [2, 1], False], [3, 160, 160, 1, True], [3, 160, 160, 1, False],
             [3, 160, 160, 1, False]],
        ],
    },
    'hidden_size': 80,
}


# ----------------------------------------------------------------------------
#  Model
# ----------------------------------------------------------------------------

def build_model(num_classes, init=DEFAULT_INIT):
    from transformers import PPOCRV6TinyRecConfig, PPOCRV6TinyRecForTextRecognition
    if init == 'none' or init.endswith('.pt'):
        model = PPOCRV6TinyRecForTextRecognition(PPOCRV6TinyRecConfig(**TINY_CONFIG, head_out_channels=num_classes))
        if init.endswith('.pt'):
            load_matching(model, torch.load(init, map_location='cpu', weights_only=True))
    else:
        model = PPOCRV6TinyRecForTextRecognition.from_pretrained(init)
    fc2 = model.head.fc2
    if fc2.out_features != num_classes:
        model.head.fc2 = nn.Linear(fc2.in_features, num_classes)
        model.config.head_out_channels = num_classes
    return model


def load_matching(model, state):
    """Load every tensor of `state` whose name and shape match the model; report what was skipped."""
    for key in ('state_dict', 'model', 'net'):  # 'net': OpenThai's pytorch/rec_weights.pt
        if isinstance(state, dict) and isinstance(state.get(key), dict):
            state = state[key]
    state = {k.removeprefix('module.').removeprefix('m.'): v for k, v in state.items()}
    own = model.state_dict()
    ok = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    if not ok:
        raise ValueError('no tensor in the .pt file matches PP-OCRv6-tiny rec (wrong file?)')
    model.load_state_dict(ok, strict=False)
    print(f'  init: loaded {len(ok)}/{len(own)} tensors, {len(own) - len(ok)} start fresh')


def ctc_logits(model, x):
    """(N, 3, 48, W) -> (N, W/8, C) raw scores. Same layers as the model's forward, minus its softmax."""
    h = model.model(x).last_hidden_state.squeeze(2)
    hd = model.head
    h = hd.act_fn(hd.norm1(hd.conv1(h)))
    h = hd.act_fn(hd.norm2(hd.conv2(h)))
    return hd.fc2(hd.fc1(h.transpose(1, 2)))


class ProbsModel(nn.Module):
    """What rec.onnx computes: per-frame class probabilities."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return F.softmax(ctc_logits(self.model, x), dim=-1)


# ----------------------------------------------------------------------------
#  Data
# ----------------------------------------------------------------------------

class LineList(Dataset):
    def __init__(self, data_dir, list_file, char_to_id, max_w):
        self.data_dir, self.char_to_id, self.max_w = data_dir, char_to_id, max_w
        self.rows, unknown = [], 0
        with open(list_file, encoding='utf-8') as f:
            for line in f:
                path, _, label = line.rstrip('\n').partition('\t')
                if not label:
                    continue
                if any(c not in char_to_id for c in label):
                    unknown += 1
                    continue
                self.rows.append((path, label))
        if unknown:
            print(f'  {list_file}: skipped {unknown} lines with characters missing from the dictionary')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, label = self.rows[i]
        x = prep_line(load_bgr(os.path.join(self.data_dir, path)), REC_HEIGHT, self.max_w)
        return x, [self.char_to_id[c] for c in label], label


def collate(items):
    xs, ys, labels = zip(*items)
    return (torch.from_numpy(pad_batch(xs)),
            torch.tensor([t for y in ys for t in y], dtype=torch.long),
            torch.tensor([len(y) for y in ys], dtype=torch.long),
            torch.tensor([max(1, x.shape[2] // STRIDE) for x in xs], dtype=torch.long),
            list(labels))


def frames_needed(ids):
    """Fewest CTC frames that can emit `ids`: one per label, plus a blank between equal neighbours."""
    return len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))


# ----------------------------------------------------------------------------
#  Train / evaluate / export
# ----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, chars, device, show=0):
    model.eval()
    errors = total = 0
    shown = []
    for x, _, _, in_lens, labels in loader:
        probs = F.softmax(ctc_logits(model, x.to(device)), dim=-1).float().cpu().numpy()
        for p, n, gt in zip(probs, in_lens.tolist(), labels):
            pred = normalize_lao_text(ctc_greedy(p[:n], chars)[0])
            e, t = strict_cer_counts(pred, gt)
            errors, total = errors + e, total + t
            if len(shown) < show:
                shown.append((gt, pred))
    model.train()
    return errors / max(total, 1), shown


def cmd_train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dict_path = os.path.join(args.data, 'lao_dict.txt')
    chars = load_chars(dict_path)
    char_to_id = {c: i for i, c in enumerate(chars) if c}

    train_set = LineList(args.data, os.path.join(args.data, 'train_list.txt'), char_to_id, args.max_w)
    val_set = LineList(args.data, os.path.join(args.data, 'val_list.txt'), char_to_id, args.max_w)
    if not len(train_set):
        sys.exit('  ERROR: train_list.txt has no usable lines')
    if not len(val_set):
        sys.exit('  ERROR: val_list.txt is empty; run to_paddleocr.py with --val-ratio > 0')

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = build_model(len(chars), args.init).to(device)
    print(f'  model: {sum(p.numel() for p in model.parameters()):,} parameters, {len(chars)} classes, device {device}')
    print(f'  data: {len(train_set)} train, {len(val_set)} val lines')

    kw = dict(collate_fn=collate, num_workers=args.workers, pin_memory=device.type == 'cuda')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False, **kw)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kw)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=min(0.3, 500 / steps))
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    os.makedirs(args.out, exist_ok=True)
    shutil.copy(dict_path, os.path.join(args.out, 'rec_dict.txt'))
    best = math.inf
    model.train()
    for epoch in range(1, args.epochs + 1):
        t0, loss_sum, n_batches, too_long = time.time(), 0.0, 0, 0
        for x, targets, tgt_lens, in_lens, _ in train_loader:
            # A line too long for its image width cannot be emitted by CTC; zero_infinity drops it.
            pos = 0
            for L, n in zip(tgt_lens.tolist(), in_lens.tolist()):
                too_long += frames_needed(targets[pos:pos + L].tolist()) > n
                pos += L
            log_probs = F.log_softmax(ctc_logits(model, x.to(device)).float(), dim=-1)
            loss = ctc(log_probs.transpose(0, 1), targets, in_lens, tgt_lens)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            loss_sum, n_batches = loss_sum + loss.item(), n_batches + 1

        cer, samples = evaluate(model, val_loader, chars, device, show=3)
        mark = ''
        if cer < best:
            best, mark = cer, '  * best, saved'
            model.save_pretrained(os.path.join(args.out, 'best'))
        print(f'  epoch {epoch}/{args.epochs}  loss {loss_sum / max(n_batches, 1):.4f}  val CER {cer:.4f} '
              f'(accuracy {100 * (1 - cer):.1f})  {time.time() - t0:.0f}s{mark}')
        if too_long:
            print(f'    {too_long} lines were too long for their image width and gave no gradient')
        for gt, pred in samples:
            print(f'    GT:   {gt}\n    PRED: {pred}')

    print(f'\n  best val CER {best:.4f}')
    cmd_export(args)


def cmd_export(args):
    from transformers import PPOCRV6TinyRecForTextRecognition
    best = os.path.join(args.out, 'best')
    if not os.path.isdir(best):
        sys.exit(f'  ERROR: no checkpoint in {best}; train first')
    model = ProbsModel(PPOCRV6TinyRecForTextRecognition.from_pretrained(best)).eval()
    onnx_path = os.path.join(args.out, 'rec.onnx')
    torch.onnx.export(model, torch.zeros(1, 3, REC_HEIGHT, 320), onnx_path, input_names=['x'],
                      output_names=['probs'], dynamic_axes={'x': {0: 'n', 3: 'w'}, 'probs': {0: 'n', 1: 't'}},
                      opset_version=17, dynamo=False)

    import onnxruntime as ort
    x = np.random.default_rng(0).uniform(-1, 1, (2, 3, REC_HEIGHT, 200)).astype(np.float32)
    ref = model(torch.from_numpy(x)).detach().numpy()
    got = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']).run(None, {'x': x})[0]
    diff = float(np.abs(ref - got).max())
    print(f'  exported {onnx_path} (ONNX vs PyTorch max difference {diff:.2e})')
    if diff > 1e-3:
        sys.exit('  ERROR: ONNX output does not match PyTorch')

    det_path = os.path.join(args.out, 'det.onnx')
    if args.det == 'none':
        pass
    elif os.path.isfile(args.det):
        shutil.copy(args.det, det_path)
    elif not os.path.exists(det_path):
        from huggingface_hub import hf_hub_download
        shutil.copy(hf_hub_download(args.det, 'inference.onnx'), det_path)
    if os.path.exists(det_path):
        print(f'  detector: {det_path}')
    print(f'\n  Read a page:  python lao_line_ocr.py {args.out} page.png')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train', help='Fine-tune, keep the best val checkpoint, then export')
    t.add_argument('--data', required=True, help='Dataset dir from to_paddleocr.py')
    t.add_argument('--init', default=DEFAULT_INIT, help=f'Starting weights (default {DEFAULT_INIT})')
    t.add_argument('--epochs', type=int, default=30)
    t.add_argument('--batch_size', type=int, default=64)
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--weight_decay', type=float, default=1e-4)
    t.add_argument('--max_w', type=int, default=1600, help='Widest line image in pixels after scaling to 48 px high')
    t.add_argument('--workers', type=int, default=4)
    t.add_argument('--device', default='', help='cuda / cpu (default: cuda if available)')
    t.add_argument('--seed', type=int, default=42)
    e = sub.add_parser('export', help='Export best/ to rec.onnx and fetch the detector')
    for q in (t, e):
        q.add_argument('--out', required=True, help='Model dir')
        q.add_argument('--det', default=DEFAULT_DET,
                       help=f'Detector: a .onnx file, a Hugging Face repo (default {DEFAULT_DET}) or "none"')
    args = p.parse_args(argv)
    (cmd_train if args.cmd == 'train' else cmd_export)(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
