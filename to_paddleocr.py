#!/usr/bin/env python3
"""
Convert image + .gt.txt line pairs into a PaddleOCR text-recognition dataset.

Writes to --out:
    images/<source>/<...>.png   copied line images (skipped with --no-copy)
    train_list.txt              "<image path>\\t<label>" per line
    val_list.txt                random --val-ratio share of the input dirs
    test_list.txt               every pair from --test dirs (only if given)
    lao_dict.txt                one character per line, for Global.character_dict_path

Labels are put in standard Lao storage order (NFC, no zero-width chars, ໍ+າ → ຳ,
tone marks before ຳ) with the same normalize_lao_text used for synthetic data,
so one glyph never has two spellings. Space is left out of the dictionary;
PaddleOCR adds it when use_space_char is true.

Example:
    python to_paddleocr.py data/synth data/real-train --test data/real-test --out paddle_data
"""

import argparse
import os
import random
import shutil
import sys

from gen_data_from_text import normalize_lao_text

IMAGE_EXTS = ('.png', '.tif', '.tiff', '.jpg', '.jpeg')


def find_pairs(root):
    """Yield (image path, normalized label) for every image under root with a non-empty .gt.txt."""
    for dirpath, _, files in os.walk(root):
        for name in sorted(files):
            base, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_EXTS:
                continue
            gt_path = os.path.join(dirpath, base + '.gt.txt')
            if not os.path.exists(gt_path):
                continue
            with open(gt_path, encoding='utf-8-sig') as f:
                label = normalize_lao_text(f.read())
            if label:
                yield os.path.join(dirpath, name), label


def build_dict(labels):
    """Sorted unique characters of all labels, without space."""
    return sorted({c for label in labels for c in label} - {' '})


def write_list(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for img, label in rows:
            f.write(f'{img}\t{label}\n')


def place_images(pairs, src_root, out, copy):
    """Return rows with image paths as PaddleOCR will read them (relative to --out, or absolute)."""
    rows = []
    tag = os.path.basename(os.path.normpath(src_root)) or 'data'
    for img, label in pairs:
        if not copy:
            rows.append((os.path.abspath(img), label))
            continue
        rel = os.path.join('images', tag, os.path.relpath(img, src_root))
        os.makedirs(os.path.dirname(os.path.join(out, rel)), exist_ok=True)
        shutil.copy2(img, os.path.join(out, rel))
        rows.append((rel.replace(os.sep, '/'), label))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('inputs', nargs='+', help='Dirs with image + .gt.txt pairs (searched recursively), split into train/val')
    p.add_argument('--test', nargs='*', default=[], help='Dirs used only for test_list.txt (e.g. held-out real lines)')
    p.add_argument('--out', required=True, help='Output dataset dir')
    p.add_argument('--val-ratio', type=float, default=0.05, help='Share of input pairs for val_list.txt (default 0.05)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-copy', action='store_true',
                   help='Do not copy images; write absolute paths (then set data_dir to "/")')
    args = p.parse_args(argv)

    if not 0 <= args.val_ratio < 1:
        p.error('--val-ratio must be in [0, 1)')
    for d in args.inputs + args.test:
        if not os.path.isdir(d):
            p.error(f'not a directory: {d}')
    tags = [os.path.basename(os.path.normpath(d)) for d in args.inputs + args.test]
    if not args.no_copy and len(set(tags)) != len(tags):
        p.error('input dirs must have different folder names (images are copied to images/<folder name>/)')

    os.makedirs(args.out, exist_ok=True)
    copy = not args.no_copy

    train_val = []
    for d in args.inputs:
        rows = place_images(list(find_pairs(d)), d, args.out, copy)
        print(f'  {d}: {len(rows)} pairs')
        train_val += rows
    test = []
    for d in args.test:
        rows = place_images(list(find_pairs(d)), d, args.out, copy)
        print(f'  {d}: {len(rows)} pairs (test)')
        test += rows

    if not train_val:
        print('  ERROR: no image + .gt.txt pairs found in the input dirs.', file=sys.stderr)
        return 1

    random.Random(args.seed).shuffle(train_val)
    n_val = round(len(train_val) * args.val_ratio)
    val, train = train_val[:n_val], train_val[n_val:]

    write_list(os.path.join(args.out, 'train_list.txt'), train)
    write_list(os.path.join(args.out, 'val_list.txt'), val)
    if args.test:
        write_list(os.path.join(args.out, 'test_list.txt'), test)

    all_labels = [label for _, label in train_val + test]
    chars = build_dict(all_labels)
    with open(os.path.join(args.out, 'lao_dict.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(chars) + '\n')

    test_only = sorted(set(build_dict(label for _, label in test)) - set(build_dict(label for _, label in train)))
    max_len = max(len(label) for label in all_labels)
    out = os.path.abspath(args.out)
    data_dir = '/' if args.no_copy else out

    print(f'\n  train {len(train)}, val {len(val)}, test {len(test)}')
    print(f'  dictionary: {len(chars)} characters (+ space via use_space_char)')
    if test_only:
        print(f'  WARNING: {len(test_only)} characters appear in test but never in train: {"".join(test_only)}')
    print(f'  longest label: {max_len} codepoints')
    print('\n  PaddleOCR overrides (add to tools/train.py -o ...):')
    print(f'    Global.character_dict_path={out}/lao_dict.txt')
    print('    Global.use_space_char=True')
    print(f'    Global.max_text_length={max_len}')
    print(f'    Train.dataset.data_dir={data_dir}')
    print(f"    Train.dataset.label_file_list=['{out}/train_list.txt']")
    print(f'    Eval.dataset.data_dir={data_dir}')
    print(f"    Eval.dataset.label_file_list=['{out}/val_list.txt']")
    return 0


if __name__ == '__main__':
    sys.exit(main())
