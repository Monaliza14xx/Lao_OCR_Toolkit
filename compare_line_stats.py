#!/usr/bin/env python3
"""
compare_line_stats.py — Is synthetic line data statistically close to real scans?

Compares two folders of line images (e.g. real crops from segment_document.py
and synthetic crops from gen_documents.py) on the properties that matter once
the trainer rescales every line to a fixed height:

    core_frac    share of the crop height taken by the letter bodies (text scale)
    aspect       width / height (line length)
    paper        median background level      paper_noise  background std
    ink          median ink level             ink_share    share of ink pixels

Prints p10 / median / p90 for each folder. Large differences point to a domain
gap worth fixing in the generator before training.

Usage:
    python3 compare_line_stats.py --real data/real-lines --synthetic data/lao-docs
"""

import argparse
import glob
import os
import random

import numpy as np
from PIL import Image

from tesstrain_core.doc_layout import otsu_threshold

IMG = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


def _runs(flags):
    padded = np.concatenate([[False], flags, [False]])
    d = np.diff(padded.astype(np.int8))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def line_stats(path):
    g = np.asarray(Image.open(path).convert('L')).astype(np.float32)
    ink = g < min(otsu_threshold(g.astype(np.uint8)), 200)
    prof = ink.sum(axis=1)
    core_h = max((e - s for s, e in _runs(prof >= 0.25 * prof.max())), default=0) if prof.max() else 0
    bg = g[~ink]
    return {'core_frac': core_h / g.shape[0], 'aspect': g.shape[1] / g.shape[0],
            'paper': float(np.median(bg)) if bg.size else np.nan,
            'paper_noise': float(np.std(bg[bg > np.percentile(bg, 20)])) if bg.size > 10 else np.nan,
            'ink': float(np.median(g[ink])) if ink.any() else np.nan, 'ink_share': float(ink.mean())}


def images(folder, limit, seed=0):
    files = sorted(f for f in glob.glob(os.path.join(folder, '**', '*'), recursive=True)
                   if f.lower().endswith(IMG) and '_layout' not in f and os.sep + 'pages' + os.sep not in f
                   and not os.path.basename(f).startswith('page'))
    random.Random(seed).shuffle(files)
    return files[:limit]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--real', required=True, help='Folder with real line crops')
    ap.add_argument('--synthetic', required=True, nargs='+', help='One or more folders with synthetic line crops')
    ap.add_argument('--limit', type=int, default=500, help='Max images per folder (default: 500)')
    args = ap.parse_args()

    print('p10 / median / p90')
    for name, folder in [('real', args.real)] + [(os.path.basename(os.path.normpath(f)), f) for f in args.synthetic]:
        files = images(folder, args.limit)
        if not files:
            print(f'  {name}: no images')
            continue
        rows = [line_stats(f) for f in files]
        cells = []
        for k in rows[0]:
            v = np.array([r[k] for r in rows], dtype=np.float64)
            cells.append(f"{k} {np.nanpercentile(v, 10):.2f}/{np.nanmedian(v):.2f}/{np.nanpercentile(v, 90):.2f}")
        print(f"  {name[:14]:14s} n={len(rows):4d}  " + '  '.join(cells))


if __name__ == '__main__':
    main()
