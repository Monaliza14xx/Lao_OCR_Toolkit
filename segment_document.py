#!/usr/bin/env python3
"""
segment_document.py — Layout analysis + line segmentation of scanned documents.

Turns scanned PDFs or page images into one image per text line, ready to be
transcribed into ground truth (or compared against synthetic data).

For every input document it writes, under OUTPUT_DIR/<document name>/:
    page<N>.png                 rendered page (at --dpi), straightened (deskewed)
    page<N>_layout.png          preview: green = text line, red = graphic, blue = rule
    p<N>_l<NNN>.png             line crops in reading order
    p<N>_l<NNN>.ocr.txt         Tesseract's DRAFT text for that line (if tesseract is
                                installed). Correct it and save as .gt.txt to use the
                                line as ground truth.
    lines.json                  boxes and draft text for every line

Usage:
    python3 segment_document.py --input scan.pdf --output_dir data/real-lines
    python3 segment_document.py --input scans/ --output_dir data/real-lines --dpi 300

Needs scipy, and pypdfium2 for PDFs. Scanned documents can contain personal
data: keep the output out of public repos (data/ is git-ignored).
"""

import argparse
import json
import os
import re
import sys

from tesstrain_core.doc_layout import (IMAGE_EXTS, PDF_EXTS, crop_line, deskew, draft_text,
                                       draw_layout, load_pages, segment_lines,
                                       tesseract_available)


def safe_name(path, index):
    stem = os.path.splitext(os.path.basename(path))[0]
    # Replace only characters that are unsafe in file names; keep Lao marks intact.
    stem = re.sub(r'[\\/:*?"<>|\s]+', '_', stem).strip('_.')
    return stem[:60] or f'doc{index:03d}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, nargs='+', help='PDF / image files or folders')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--dpi', type=int, default=300, help='PDF render resolution (default: 300)')
    ap.add_argument('--margin', type=float, default=0.25,
                    help='Crop margin as a fraction of the letter-body height (default: 0.25)')
    ap.add_argument('--no_deskew', action='store_true', help='Do not straighten pages first')
    ap.add_argument('--draft_lang', default='lao',
                    help="Tesseract language for draft text; 'none' to skip (default: lao)")
    args = ap.parse_args()

    files = []
    for item in args.input:
        if os.path.isdir(item):
            files += sorted(os.path.join(item, f) for f in os.listdir(item)
                            if f.lower().endswith(PDF_EXTS + IMAGE_EXTS))
        else:
            files.append(item)
    if not files:
        sys.exit("No input documents found.")
    drafts = args.draft_lang != 'none' and tesseract_available()
    if args.draft_lang != 'none' and not drafts:
        print("  Note: tesseract not found — no draft text will be written.")

    total = 0
    for di, path in enumerate(files):
        out = os.path.join(args.output_dir, safe_name(path, di))
        os.makedirs(out, exist_ok=True)
        meta = {'source': os.path.basename(path), 'dpi': args.dpi, 'pages': []}
        for pi, page in enumerate(load_pages(path, dpi=args.dpi), start=1):
            angle = 0.0
            if not args.no_deskew:
                page, angle = deskew(page)
            page.save(os.path.join(out, f'page{pi}.png'))
            lines, info = segment_lines(page)
            draw_layout(page, lines, info).save(os.path.join(out, f'page{pi}_layout.png'))
            margin = max(2, int(round(args.margin * info['core_height'])))
            page_meta = {'page': pi, 'deskew_angle': round(angle, 2),
                         'core_height': info['core_height'], 'lines': []}
            for n, ln in enumerate(lines, start=1):
                name = f'p{pi}_l{n:03d}'
                crop = crop_line(page, ln['box'], margin, ln['limits'])
                crop.save(os.path.join(out, name + '.png'))
                text = draft_text(crop, args.draft_lang) if drafts else ''
                if drafts:
                    with open(os.path.join(out, name + '.ocr.txt'), 'w', encoding='utf-8') as f:
                        f.write(text)
                page_meta['lines'].append({'name': name, 'box': list(ln['box']), 'draft_text': text})
            print(f"  {os.path.basename(path)} page {pi}: {len(lines)} text lines, "
                  f"{len(info['graphics'])} graphics, {len(info['rules'])} rules, "
                  f"letter-body height {info['core_height']:.0f}px, deskewed {angle:+.2f}°")
            total += len(lines)
            meta['pages'].append(page_meta)
        with open(os.path.join(out, 'lines.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n  ✓ {total} line images from {len(files)} document(s) → {args.output_dir}")
    if drafts:
        print("  Draft text is in *.ocr.txt — correct it and save as *.gt.txt to use as ground truth.")


if __name__ == '__main__':
    main()
