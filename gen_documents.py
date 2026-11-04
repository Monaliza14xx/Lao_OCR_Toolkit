#!/usr/bin/env python3
"""
gen_documents.py — Synthetic Lao document pages → realistically cropped line images.

Each page is laid out like a Lao official letter (see tesstrain_core/doc_synth.py),
"scanned" (skew, blur, paper tint, uneven light, resolution loss, noise, specks,
JPEG, photocopy look), then straightened and segmented by exactly the same code
that segments real scans (tesstrain_core/doc_layout.py, used by segment_document.py).

So the synthetic line crops inherit the real pipeline's behaviour: residual skew,
crop margins, neighbouring marks, stamp/signature overlap, bilingual lines.

Labels: every word drawn on the page is recorded with its box. A segmented line
is labelled with the words that lie completely inside its crop, left to right.
A crop that cuts through a word, or that holds words from two text lines, is
discarded — labels are never guessed.

Output (OUTPUT_DIR):
    doc<NNNNNN>_l<NNN>.png + .gt.txt (+ .lstmf with --to_lstmf)
    pages/                      first --save_pages pages: scan, layout preview, words.json
    summary.json                lines found / kept / dropped, word coverage

Usage:
    python3 gen_documents.py --corpus corpus.txt --output_dir data/lao-docs \\
        --font fonts/Phetsarath-Regular.ttf fonts/Phetsarath-Bold.ttf fonts/saysettha_ot.ttf \\
        --sec_font fonts/TimesNewerRoman-*.otf --count 500 --workers 8
"""

import argparse
import json
import multiprocessing
import os
import random
import sys
import time

import numpy as np

from gen_data_from_text import image_to_lstmf, load_corpus
from tesstrain_core.doc_layout import crop_line, deskew, draw_layout, rotate_points, segment_lines
from tesstrain_core.doc_synth import Content, FontSet, make_letter, scan

_CFG = {}


def _init(cfg):
    global _CFG
    _CFG = cfg
    _CFG['fonts'] = FontSet(cfg['font_paths'], cfg['sec_font_paths'])


def _overlap(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def label_line(crop_box, line_box, words):
    """Text of the words fully inside crop_box, or (None, reason).

    A word only partly inside the crop is a problem when it belongs to this
    line (its vertical centre lies within the line's box): then the crop cuts it
    and the line is dropped. Slivers of the neighbouring lines' marks poking in
    from above or below are normal in real crops and don't change the label.
    """
    inside = []
    for w in words:
        bx = w['box']
        a = max(1e-6, (bx[2] - bx[0]) * (bx[3] - bx[1]))
        frac = _overlap(bx, crop_box) / a
        if frac >= 0.85:
            inside.append(w)
        elif frac > 0.15 and line_box[1] <= (bx[1] + bx[3]) / 2 <= line_box[3]:
            return None, 'cuts a word'
    if not inside:
        return None, 'no words (graphic)'
    centres = [(w['box'][1] + w['box'][3]) / 2 for w in inside]
    heights = [w['box'][3] - w['box'][1] for w in inside]
    if max(centres) - min(centres) > 0.6 * float(np.median(heights)):
        return None, 'two text lines'
    inside.sort(key=lambda w: w['box'][0])
    return ' '.join(w['text'] for w in inside), None


def make_page(idx):
    cfg = _CFG
    rng = random.Random(cfg['seed'] * 1_000_003 + idx)
    np.random.seed((cfg['seed'] * 1_000_003 + idx) % 2 ** 32)
    content = Content(cfg['lines'], rng)
    page, words = make_letter(content, cfg['fonts'], rng, dpi=cfg['dpi'])
    size = page.size

    scanned, skew = scan(page, rng, max_skew=cfg['max_skew'])
    straight, fix = deskew(scanned)
    for w in words:   # move the word's corners through both rotations, then box them
        l, t, r, b = w['box']
        pts = rotate_points(rotate_points([(l, t), (r, t), (r, b), (l, b)], skew, size), fix, size)
        w['box'] = (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))

    lines, info = segment_lines(straight)
    margin = max(2, int(round(cfg['margin'] * info['core_height'])))
    stats = {'lines': len(lines), 'kept': 0, 'dropped': {}, 'words': len(words), 'words_kept': 0}
    for n, ln in enumerate(lines, start=1):
        L, T, R, B = ln['box']
        crop_box = (max(0, L - margin), max(0, ln['limits'][0], T - margin),
                    min(size[0], R + margin), min(size[1], ln['limits'][1], B + margin))
        text, reason = label_line(crop_box, ln['box'], words)
        if text is None:
            stats['dropped'][reason] = stats['dropped'].get(reason, 0) + 1
            continue
        img = crop_line(straight, ln['box'], margin, ln['limits']).convert('L')
        name = f"doc{idx:06d}_l{n:03d}"
        out = cfg['output_dir']
        if cfg['ext'] == 'jpg':
            img.save(os.path.join(out, name + '.jpg'), quality=95)
        else:
            img.save(os.path.join(out, name + '.png'))
        with open(os.path.join(out, name + '.gt.txt'), 'w', encoding='utf-8') as f:
            f.write(text)
        if cfg['to_lstmf']:
            with open(os.path.join(out, name + '.lstmf'), 'wb') as f:
                f.write(image_to_lstmf(img, text, name))
        stats['kept'] += 1
        stats['words_kept'] += len(text.split())

    if idx < cfg['save_pages']:
        pdir = os.path.join(cfg['output_dir'], 'pages')
        os.makedirs(pdir, exist_ok=True)
        straight.save(os.path.join(pdir, f'doc{idx:06d}.jpg'), quality=90)
        draw_layout(straight, lines, info).save(os.path.join(pdir, f'doc{idx:06d}_layout.png'))
        with open(os.path.join(pdir, f'doc{idx:06d}_words.json'), 'w', encoding='utf-8') as f:
            json.dump({'skew': skew, 'deskew': fix, 'words': words}, f, ensure_ascii=False)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--corpus', required=True, help='Lao text, one line/sentence per line')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--font', nargs='+', required=True, help='Lao fonts (.ttf/.otf); bold variants are paired by name')
    ap.add_argument('--sec_font', nargs='*', default=[], help='Fonts for digits/Latin (Times-like serif recommended)')
    ap.add_argument('--count', type=int, default=100, help='Number of pages (default: 100)')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--max_skew', type=float, default=1.5, help='Max scan skew in degrees (default: 1.5)')
    ap.add_argument('--margin', type=float, default=0.25, help='Crop margin × letter-body height (default: 0.25)')
    ap.add_argument('--ext', default='png', choices=['png', 'jpg'])
    ap.add_argument('--to_lstmf', action='store_true')
    ap.add_argument('--save_pages', type=int, default=3, help='Save the first N pages + layout previews')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    lines = load_corpus(args.corpus)
    if not lines:
        sys.exit("Corpus is empty.")
    sec = args.sec_font or [os.path.join('fonts', f) for f in sorted(os.listdir('fonts'))
                            if 'times' in f.lower()] if os.path.isdir('fonts') else []
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = dict(lines=lines, font_paths=args.font, sec_font_paths=sec, dpi=args.dpi,
               max_skew=args.max_skew, margin=args.margin, ext=args.ext, to_lstmf=args.to_lstmf,
               save_pages=args.save_pages, output_dir=args.output_dir, seed=args.seed)

    fonts = FontSet(args.font, sec)
    print(f"  Lao fonts: {[os.path.basename(r) + (' + ' + os.path.basename(b) if b else '') for r, b in fonts.families]}")
    if fonts.skipped:
        print(f"  Skipped (HarfBuzz output broken for these): {[os.path.basename(p) for p in fonts.skipped]}")
    print(f"  Digit/Latin fonts: {[os.path.basename(p) for p in fonts.sec]}")
    print(f"  Generating {args.count} pages with {args.workers} worker(s)...")

    t0 = time.time()
    total = {'lines': 0, 'kept': 0, 'dropped': {}, 'words': 0, 'words_kept': 0}
    pages = range(args.count)
    if args.workers > 1:
        with multiprocessing.Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
            results = pool.imap_unordered(make_page, pages)
            for k, s in enumerate(results, start=1):
                _accumulate(total, s)
                if k % 20 == 0:
                    print(f"  [{k}/{args.count}] {total['kept']} lines")
    else:
        _init(cfg)
        for k in pages:
            _accumulate(total, make_page(k))
    elapsed = time.time() - t0

    total['word_coverage'] = round(total['words_kept'] / max(1, total['words']), 3)
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(total, f, indent=2)
    print(f"\n  ✓ {args.count} pages in {elapsed:.0f}s → {total['kept']} labelled line images")
    print(f"  Lines found {total['lines']}, kept {total['kept']}, dropped {total['dropped']}")
    print(f"  Words drawn that ended up in a labelled line: {total['word_coverage']:.0%}")


def _accumulate(total, s):
    for k in ('lines', 'kept', 'words', 'words_kept'):
        total[k] += s[k]
    for r, n in s['dropped'].items():
        total['dropped'][r] = total['dropped'].get(r, 0) + n


if __name__ == '__main__':
    main()
