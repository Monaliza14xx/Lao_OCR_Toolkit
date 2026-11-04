#!/usr/bin/env python3
"""
compare_pt_vs_tess.py
────────────────────────────────────────────────────────────────────────────
Compare Tesseract (.traineddata) models against PyTorch (.pt) models.
Useful for validating the export process and benchmarking GPU vs CPU inference.

Usage
─────
  python compare_pt_vs_tess.py
  python compare_pt_vs_tess.py --image_folder data/lao-ground-truth-test
"""

import os
import csv
import glob
import time
import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image as _PILImage

# ── Project imports ──────────────────────────────────────────────────────────
from tesseract_tools import (
    find_test_pairs,
    edit_distance,
    align_strings,
    get_component_type,
    LAO_COMPONENT_MAP,
    COER_COLS,
)
from tesstrain_core.model import build_model
from tesstrain_core.recognize import recognize_line

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    print("[Warning] pytesseract not found — Tesseract inference will fail.")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_FOLDER = "data/lao-ground-truth-test"
TESSERACT_PATH = "tesseract"
TESS_CONFIG = "--psm 7 --oem 1"
OUT_DIR = "data/cross_comparison"

# Specific pairs as requested by the user
MODEL_PAIRS = [
    {
        "name": "lao48cnn16",
        "tess": "data/lao48cnn16_final.traineddata",
        "pt": "data/lao48cnn16/lao48cnn16_final.pt"
    },
    {
        "name": "lao48cnn32",
        "tess": "data/lao48cnn32_final.traineddata",
        "pt": "data/lao48cnn32/lao48cnn32_final.pt"
    },
    {
        "name": "lao48cnn16t16",
        "tess": "data/lao48cnn16t16_final.traineddata",
        "pt": "data/lao48cnn16t16/lao48cnn16t16_final.pt"
    },
    {
        "name": "lao48cnn16t32",
        "tess": "data/lao48cnn16t32_final.traineddata",
        "pt": "data/lao48cnn16t32/lao48cnn16t32_final.pt"
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def run_tesseract(image_path, traineddata_path):
    """Run Tesseract using a specific .traineddata file."""
    tessdata_dir = os.path.dirname(traineddata_path)
    model_name = os.path.basename(traineddata_path).replace(".traineddata", "")
    config = f'--tessdata-dir "{tessdata_dir}" -l {model_name} {TESS_CONFIG}'
    try:
        img = _PILImage.open(image_path).convert('L')
        text = pytesseract.image_to_string(img, config=config)
        return text.strip()
    except Exception:
        return ""

def load_pt_model(pt_path, device):
    """Load PyTorch model and metadata from checkpoint."""
    if not os.path.exists(pt_path):
        return None, None, None
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    spec = ckpt['net_spec'].replace('###', str(ckpt['num_classes']))
    model = build_model(spec, ckpt['num_classes']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    inv_charset = {v: k for k, v in ckpt['charset'].items()}
    
    target_h = 48
    m = re.match(r'\[?\d+,(\d+),', ckpt.get('net_spec', ''))
    if m: target_h = int(m.group(1))
    
    return model, inv_charset, target_h

def run_pytorch(image_path, model, inv_charset, target_h, device):
    """Run PyTorch inference on a single image line."""
    try:
        pil = _PILImage.open(image_path).convert('L')
        return recognize_line(model, pil, target_h, device, inv_charset, beam_width=0)
    except Exception:
        return ""

# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(image_folder, out_dir, num_samples, active_pairs, device):
    print("=" * 70)
    print(f"  Evaluating Dataset: {image_folder}")
    print("=" * 70)

    os.makedirs(out_dir, exist_ok=True)

    # Find test pairs
    test_pairs = find_test_pairs(image_folder)
    if not test_pairs:
        print(f"  ERROR: No *.gt.txt pairs found in {image_folder}")
        return
    
    if num_samples > 0:
        test_pairs = test_pairs[:num_samples]
        print(f"  Limiting to {num_samples} samples.")
        
    print(f"  Found {len(test_pairs)} test pairs.\n")

    predictions = {} # {model_key: {img_path: text}}
    model_types = {} # {model_key: type_string}

    for pair in active_pairs:
        name = pair["name"]
        tess_path = pair["tess"]
        pt_path = pair["pt"]

        # ── Tesseract ────────────────────────────────────────────────────────
        tess_key = f"{name}_Tess"
        if os.path.exists(tess_path):
            print(f"  [Tess] Running {tess_key}...")
            pred_map = {}
            for img_path, _ in test_pairs:
                pred_map[img_path] = run_tesseract(img_path, tess_path)
            predictions[tess_key] = pred_map
            model_types[tess_key] = "Tesseract (.traineddata)"
        else:
            print(f"  [Skip] Tesseract model not found: {tess_path}")

        # ── PyTorch ─────────────────────────────────────────────────────────
        pt_key = f"{name}_PT"
        model, charset, th = load_pt_model(pt_path, device)
        if model:
            print(f"  [PT  ] Running {pt_key}...")
            pred_map = {}
            for img_path, _ in test_pairs:
                pred_map[img_path] = run_pytorch(img_path, model, charset, th, device)
            predictions[pt_key] = pred_map
            model_types[pt_key] = "PyTorch (.pt)"
        else:
            print(f"  [Skip] PyTorch model not found: {pt_path}")

    if not predictions:
        print("  ERROR: No models found to compare.")
        return

    # Compute metrics
    print(f"\n  Computing metrics for {len(predictions)} models...")
    from compare_cross import compute_metrics as compute_cross_metrics
    results = compute_cross_metrics(predictions, test_pairs)

    # Export CSVs
    from compare_cross import export_cross_csv, export_predictions_csv
    model_list = sorted(predictions.keys())
    export_cross_csv(results, model_types, model_list, out_dir)
    export_predictions_csv(predictions, test_pairs, model_types, out_dir)

    # Console Summary
    print()
    print("=" * 95)
    print(f"  {'Model':<30} | {'Type':<25} | {'CER (%)':>8} | {'Accuracy (%)':>12}")
    print("-" * 95)
    for m in model_list:
        r = results[m]
        print(f"  {m:<30} | {model_types[m]:<25} | {r['cer']*100:>8.2f} | {r['acc']*100:>12.2f}")
    print("=" * 95)
    print(f"\n  Done. Results saved to {out_dir}/\n")

def main():
    parser = argparse.ArgumentParser(description="Tesseract vs PyTorch Accuracy Comparison")
    parser.add_argument("--image_folder", default=IMAGE_FOLDER)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--models", nargs="+", help="List of models to compare", default=[])
    parser.add_argument("--tess_dir", default="/content/tesstrain-gpu/tesseract_data", help="Tesseract traineddata directory")
    parser.add_argument("--pt_dir", default="/content/tesstrain-gpu/data", help="PyTorch model directory")
    parser.add_argument("--num-samples", type=int, default=0, help="Limit number of samples to evaluate")
    parser.add_argument("--custom_img_dir", default="data/lao-ground-truth-test", help="First dataset (test)")
    parser.add_argument("--custom_realworld_img_dir", default="data/lao-real-documents", help="Second dataset (realworld)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("  Model Comparison: Tesseract (.traineddata) vs PyTorch (.pt)")
    print("=" * 70)
    print(f"  Device       : {device}")
    print()

    if args.models:
        active_pairs = []
        for name in args.models:
            tess_matches = glob.glob(f"{args.tess_dir}/{name}*.traineddata")
            tess_path = tess_matches[0] if tess_matches else f"{args.tess_dir}/{name}.traineddata"
            
            pt_matches = glob.glob(f"{args.pt_dir}/{name}/*.pt")
            pt_path = pt_matches[0] if pt_matches else f"{args.pt_dir}/{name}/{name}.pt"
            
            active_pairs.append({"name": name, "tess": tess_path, "pt": pt_path})
    else:
        active_pairs = MODEL_PAIRS

    # Run for custom_img_dir
    if args.custom_img_dir and os.path.exists(args.custom_img_dir):
        out_test = os.path.join(args.out_dir, "test")
        run_evaluation(args.custom_img_dir, out_test, args.num_samples, active_pairs, device)
        
    # Run for custom_realworld_img_dir
    if args.custom_realworld_img_dir and os.path.exists(args.custom_realworld_img_dir):
        out_realworld = os.path.join(args.out_dir, "realworld")
        run_evaluation(args.custom_realworld_img_dir, out_realworld, args.num_samples, active_pairs, device)

if __name__ == "__main__":
    main()
