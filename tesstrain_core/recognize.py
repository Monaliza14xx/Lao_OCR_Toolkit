import os
import sys
import re
import numpy as np
import torch
from PIL import Image
from .utils import resolve_paths
from .model import build_model
from .dawg import DAWGDecoder, NgramLM
from .metrics import ctc_decode

def page_segment_lines(pil_img, min_line_height=10, merge_gap=5):
    arr = np.array(pil_img, dtype=np.uint8)
    h, w = arr.shape

    threshold = np.mean(arr) * 0.8
    binary = (arr < threshold).astype(np.uint8) 

    h_proj = binary.sum(axis=1)

    ink_threshold = max(w * 0.005, 2)
    is_text = h_proj > ink_threshold

    lines = []
    in_line = False
    y_start = 0
    for y in range(h):
        if is_text[y] and not in_line:
            y_start = y
            in_line = True
        elif not is_text[y] and in_line:
            if y - y_start >= min_line_height:
                lines.append((y_start, y))
            in_line = False
    if in_line and h - y_start >= min_line_height:
        lines.append((y_start, h))

    merged = []
    for start, end in lines:
        if merged and start - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    lines = merged

    if not lines:
        return [(pil_img, 0)]

    padding = 4
    result = []
    for y_start, y_end in lines:
        y1 = max(0, y_start - padding)
        y2 = min(h, y_end + padding)

        line_binary = binary[y1:y2, :]
        v_proj = line_binary.sum(axis=0)
        cols = np.where(v_proj > 0)[0]
        if len(cols) == 0:
            continue
        x1 = max(0, cols[0] - padding)
        x2 = min(w, cols[-1] + 1 + padding)

        line_img = pil_img.crop((x1, y1, x2, y2))
        result.append((line_img, y1))

    return result if result else [(pil_img, 0)]

def page_segment_blocks(pil_img, min_block_height=20):
    arr = np.array(pil_img, dtype=np.uint8)
    h, w = arr.shape
    threshold = np.mean(arr) * 0.8
    binary = (arr < threshold).astype(np.uint8)

    v_proj = binary.sum(axis=0)
    col_threshold = max(h * 0.01, 2)
    is_content = v_proj > col_threshold

    columns = []
    in_col = False
    x_start = 0
    min_col_width = max(w * 0.05, 20)
    for x in range(w):
        if is_content[x] and not in_col:
            x_start = x
            in_col = True
        elif not is_content[x] and in_col:
            if x - x_start >= min_col_width:
                columns.append((x_start, x))
            in_col = False
    if in_col and w - x_start >= min_col_width:
        columns.append((x_start, w))

    if not columns:
        columns = [(0, w)]

    all_lines = []
    for block_idx, (x1, x2) in enumerate(columns):
        col_img = pil_img.crop((x1, 0, x2, h))
        lines = page_segment_lines(col_img)
        for line_idx, (line_img, _y_off) in enumerate(lines):
            all_lines.append((line_img, block_idx, line_idx))

    return all_lines

def recognize_line(model, pil_line, target_h, device, inv_charset,
                   beam_width=0, dawg_decoder=None):
    w, h = pil_line.size
    if h != target_h and h > 0:
        scale = target_h / h
        pil_line = pil_line.resize((max(1, int(w * scale)), target_h), Image.BILINEAR)

    arr = np.array(pil_line, dtype=np.float32)
    tensor = torch.from_numpy((arr - 127.5) / 127.5).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        if out.dim() == 4:
            B, C, H, W = out.shape
            out = out.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)

    decoded = ctc_decode(out[0], beam_width=beam_width, dawg_decoder=dawg_decoder)
    return ''.join(inv_charset.get(l, '?') for l in decoded)

def cmd_recognize(args):
    """Run OCR inference on an image using the trained PyTorch model."""
    p = resolve_paths(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    psm = getattr(args, 'psm', 7)
    beam_width = getattr(args, 'beam_width', 0)

    best_pt = p.model_output + '_best.pt'
    final_pt = p.model_output + '_final.pt'
    pt_path = best_pt if os.path.exists(best_pt) else final_pt
    if not os.path.exists(pt_path):
        print(f"Error: No checkpoint at {best_pt} or {final_pt}")
        sys.exit(1)

    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    spec = ckpt['net_spec'].replace('###', str(ckpt['num_classes']))
    model = build_model(spec, ckpt['num_classes']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    inv_charset = {v: k for k, v in ckpt['charset'].items()}

    target_h = 48
    m = re.match(r'\[?\d+,(\d+),', ckpt.get('net_spec', ''))
    if m: target_h = int(m.group(1))

    img_path = args.image
    if not os.path.exists(img_path):
        print(f"Error: Image not found: {img_path}")
        sys.exit(1)

    pil = Image.open(img_path).convert('L')
    decode_method = 'beam' if beam_width > 1 else 'greedy'

    dawg_decoder = None
    wordlist_path = getattr(args, 'wordlist', '')
    lm_corpus_path = getattr(args, 'lm_corpus', '')
    lm_weight = getattr(args, 'lm_weight', 0.5)
    if beam_width > 1:
        lm = None
        if lm_corpus_path:
            lm = NgramLM(char_order=3, word_order=2)
            if os.path.isdir(lm_corpus_path):
                lm.train_from_gt_dir(lm_corpus_path)
            else:
                lm.train_from_file(lm_corpus_path)
            if lm.total_chars == 0:
                lm = None

        dawg_decoder = DAWGDecoder.from_files(
            inv_charset=inv_charset,
            wordlist=wordlist_path if wordlist_path else None,
            lm=lm,
            lm_weight=lm_weight,
        )
        if dawg_decoder.word_dawg.word_count > 0 and lm:
            decode_method = f'beam+DAWG+LM ({dawg_decoder.word_dawg.word_count} words)'
        elif lm:
            decode_method = 'beam+LM'
        elif dawg_decoder.word_dawg.word_count > 0:
            decode_method = f'beam+DAWG ({dawg_decoder.word_dawg.word_count} words)'

    print(f"\n  Image:  {img_path} ({pil.size[0]}x{pil.size[1]})")
    print(f"  PSM:    {psm}  |  Decode: {decode_method}" +
          (f" (width={beam_width})" if beam_width > 1 else ""))
    print()

    if psm == 7 or psm == 13:
        text = recognize_line(model, pil, target_h, device, inv_charset,
                              beam_width, dawg_decoder)
        print(f"  {text}")

    elif psm == 6:
        lines = page_segment_lines(pil)
        print(f"  Lines found: {len(lines)}")
        print(f"  {'─' * 50}")
        for _i, (line_img, _y_off) in enumerate(lines):
            text = recognize_line(model, line_img, target_h, device, inv_charset,
                                  beam_width, dawg_decoder)
            print(f"  {text}")
        print(f"  {'─' * 50}")

    elif psm == 3:
        all_lines = page_segment_blocks(pil)
        print(f"  Blocks: {max(b for _, b, _ in all_lines) + 1 if all_lines else 0}  "
              f"Lines: {len(all_lines)}")
        print(f"  {'─' * 50}")
        current_block = -1
        for line_img, block_idx, _line_idx in all_lines:
            if block_idx != current_block:
                if current_block >= 0:
                    print()
                current_block = block_idx
            text = recognize_line(model, line_img, target_h, device, inv_charset,
                                  beam_width, dawg_decoder)
            print(f"  {text}")
        print(f"  {'─' * 50}")

    else:
        print(f"  Unsupported PSM mode: {psm}")
        print("  Supported: 3 (full auto), 6 (single block), 7 (single line), 13 (raw line)")

    print()
