"""
compare_cross.py
────────────────────────────────────────────────────────────────────────────
Cross-compare original Tesseract models against our fine-tuned models on the
same test image set.  Outputs CSV reports only (no plots).

Folder layout expected
──────────────────────
  data/
    lao48cnn16_final.traineddata      ← our fine-tuned models
    lao48cnn32_final.traineddata
    lao48cnn16t16_final.traineddata
    lao48cnn16t32_final.traineddata
    lao-ground-truth-test/            ← *.png / *.jpg + matching *.gt.txt

Original Tesseract models (e.g. "lao", "eng") are looked up in the system
tessdata directory (or the path set in ORIGINAL_TESSDATA_DIR below).

Output (in data/cross_comparison/)
────────────────────────────────────
  cross_report.csv             – CER / WER / CoER / Accuracy per model with
                                 model_type column (Original / Fine-tuned)
  cross_report_coer.csv        – Per-category CoER for every model
  cross_report_characters.csv  – Per-character error breakdown
  cross_report_mistakes.csv    – Top substitution / deletion / insertion errors

Usage
─────
  python experiments/compare_cross.py
  python experiments/compare_cross.py --image_folder data/lao-ground-truth-test
  python experiments/compare_cross.py --out_dir data/cross_comparison

"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: tesstrain_core, tesseract_tools

import os
import sys
import csv
import time
import argparse
from collections import Counter, defaultdict

# ── re-use all shared utilities from tesseract_tools.py ──────────────────────
from tesseract_tools import (
    find_test_pairs,
    edit_distance,
    align_strings,
    get_component_type,
    LAO_COMPONENT_MAP,
    fold_sara_am,
    COER_COLS,
)

try:
    import pytesseract
    from PIL import Image as _PILImage
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    print("[Warning] pytesseract / Pillow not found — inference will fail.")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  (edit these to match your environment)
# ══════════════════════════════════════════════════════════════════════════════

# ── Test images ───────────────────────────────────────────────────────────────
IMAGE_FOLDER = "data/lao-ground-truth-test"

# ── Tesseract binary ──────────────────────────────────────────────────────────
TESSERACT_PATH = "tesseract"

# ── Original Tesseract models ─────────────────────────────────────────────────
# Set to None to use the system default (Tesseract finds it automatically).
# On most Linux systems this is /usr/share/tessdata/
# On macOS with Homebrew: /usr/local/share/tessdata/
ORIGINAL_TESSDATA_DIR = None    # e.g. "/usr/share/tessdata"

ORIGINAL_MODELS = [
    "lao",      # Standard Lao model bundled with Tesseract
    # "lao_best",   # uncomment if you also have tessdata-best lao
    # "eng",        # add any others you want to compare against
]

# ── Fine-tuned models (in data/) ─────────────────────────────────────────────
CUSTOM_TESSDATA_DIR = "data"

CUSTOM_MODELS = [
    "lao48cnn16_final",
    "lao48cnn32_final",
    "lao48cnn16t16_final",
    "lao48cnn16t32_final",
]

# ── PSM / OEM config ──────────────────────────────────────────────────────────
TESS_CONFIG = "--psm 7 --oem 1"

# ── Output ────────────────────────────────────────────────────────────────────
OUT_DIR = "data/cross_comparison"

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def _run(image_path: str, model_name: str, tessdata_dir: str | None) -> str:
    """Run Tesseract on one image and return stripped predicted text."""
    if tessdata_dir:
        config = f'--tessdata-dir "{tessdata_dir}" -l {model_name} {TESS_CONFIG}'
    else:
        config = f'-l {model_name} {TESS_CONFIG}'
    try:
        img  = _PILImage.open(image_path)
        text = pytesseract.image_to_string(img, config=config)
        return text.strip()
    except Exception as exc:
        print(f"  ⚠ Tesseract failed on {image_path}: {exc}", file=sys.stderr)
        return ""


def run_all_models(test_pairs: list) -> tuple[dict, dict]:
    """
    Run inference for every original + custom model.

    Returns
    -------
    predictions : {model_name: {img_path: pred_str}}
    model_types : {model_name: "Original Tesseract" | "Fine-tuned"}
    """
    predictions: dict = {}
    model_types: dict = {}

    # ── Original Tesseract ────────────────────────────────────────────────────
    for m in ORIGINAL_MODELS:
        td = ORIGINAL_TESSDATA_DIR  # may be None
        if td:
            td_file = os.path.join(td, f"{m}.traineddata")
            if not os.path.exists(td_file):
                print(f"  [Skip-original] {m} — not found in {td}")
                continue
        print(f"  Running original [{m}] on {len(test_pairs)} images...")
        pred_map: dict = {}
        t0 = time.time()
        for idx, (img_path, _) in enumerate(test_pairs):
            pred_map[img_path] = _run(img_path, m, td)
            if (idx + 1) % 100 == 0:
                print(f"    {idx+1}/{len(test_pairs)}  ({time.time()-t0:.1f}s)")
        predictions[m] = pred_map
        model_types[m] = "Original Tesseract"
        print(f"    Done in {time.time()-t0:.1f}s")

    # ── Fine-tuned models ─────────────────────────────────────────────────────
    for m in CUSTOM_MODELS:
        td_file = os.path.join(CUSTOM_TESSDATA_DIR, f"{m}.traineddata")
        if not os.path.exists(td_file):
            print(f"  [Skip-custom] {m} — {td_file} not found")
            continue
        print(f"  Running fine-tuned [{m}] on {len(test_pairs)} images...")
        pred_map = {}
        t0 = time.time()
        for idx, (img_path, _) in enumerate(test_pairs):
            pred_map[img_path] = _run(img_path, m, CUSTOM_TESSDATA_DIR)
            if (idx + 1) % 100 == 0:
                print(f"    {idx+1}/{len(test_pairs)}  ({time.time()-t0:.1f}s)")
        predictions[m] = pred_map
        model_types[m] = "Fine-tuned"
        print(f"    Done in {time.time()-t0:.1f}s")

    return predictions, model_types


# ══════════════════════════════════════════════════════════════════════════════
# METRICS  (same logic as tesseract_tools.compute_metrics)
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(predictions: dict, test_pairs: list) -> dict:
    """
    Returns {model_name: {cer, wer, coer, acc, eval_time,
                           char_stats, mistake_ctr}}
    """
    results: dict = {}
    for model_name, pred_map in predictions.items():
        total_cer_num  = 0
        total_cer_den  = 0
        total_wer_dist = 0
        total_wer_wds  = 0
        coer_stats = {cat: {"total": 0, "errors": 0} for cat in LAO_COMPONENT_MAP}
        coer_stats["other"] = {"total": 0, "errors": 0}
        char_stats  = defaultdict(lambda: {"total": 0, "correct": 0, "errors": 0})
        mistake_ctr = Counter()
        t0 = time.time()

        for img_path, truth in test_pairs:
            pred = fold_sara_am(pred_map.get(img_path, ""))
            truth = fold_sara_am(truth)

            # CER
            d = edit_distance(pred, truth)
            total_cer_num += d
            total_cer_den += max(len(truth), 1)

            # WER
            tw = truth.split()
            pw = pred.split()
            total_wer_dist += edit_distance(pw, tw)
            total_wer_wds  += max(len(tw), 1)

            # Per-character alignment
            edits = align_strings(pred, truth)
            for op, p_c, t_c in edits:
                ch  = t_c if t_c else p_c
                cat = get_component_type(ch)
                if t_c:
                    char_stats[t_c]["total"]   += 1
                    coer_stats[cat]["total"]   += 1
                if op != "mat":
                    if t_c:
                        char_stats[t_c]["errors"] += 1
                        coer_stats[cat]["errors"] += 1
                    mistake_ctr[(op, p_c, t_c)] += 1
                else:
                    if t_c:
                        char_stats[t_c]["correct"] += 1

        eval_time = time.time() - t0
        cer      = total_cer_num  / max(total_cer_den, 1)
        wer      = total_wer_dist / max(total_wer_wds, 1)
        coer     = {cat: s["errors"] / max(s["total"], 1)
                    for cat, s in coer_stats.items() if s["total"] > 0}
        coer_avg = sum(coer.values()) / max(len(coer), 1)
        acc      = max(0.0, 1.0 - (cer + wer + coer_avg) / 3.0)

        results[model_name] = {
            "cer": cer, "wer": wer, "coer": coer, "acc": acc,
            "eval_time": eval_time,
            "char_stats": char_stats,
            "mistake_ctr": mistake_ctr,
        }
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CSV EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_predictions_csv(predictions: dict, test_pairs: list, model_types: dict, out_dir: str):
    """
    Export line-by-line predictions compared to ground truth.
    Format: gt_name, gt, prediction, model, engine
    """
    out_path = os.path.join(out_dir, "predictions_report.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gt_name", "gt", "prediction", "model", "engine"])
        for model, pred_map in predictions.items():
            engine = model_types.get(model, "Unknown")
            for img_path, gt_str in test_pairs:
                gt_name = os.path.basename(img_path)
                pred_str = pred_map.get(img_path, "")
                w.writerow([gt_name, gt_str, pred_str, model, engine])
    print(f"  ✓ Predictions CSV saved to: {out_path}")


def export_cross_csv(results: dict, model_types: dict,
                     model_list: list, out_dir: str):
    """Write all four CSV files to out_dir."""

    # ── 1. Global summary ─────────────────────────────────────────────────────
    global_path = os.path.join(out_dir, "cross_report.csv")
    with open(global_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["Model", "Model Type",
                  "CER (%)", "WER (%)", "Accuracy (%)", "Avg CoER (%)",
                  "Eval Time (s)"]
        for _, col in COER_COLS:
            header.append(f"CoER {col} (%)")
        w.writerow(header)

        for m in model_list:
            r = results.get(m)
            if not r:
                continue
            coer_avg = (sum(r["coer"].values()) / max(len(r["coer"]), 1)) * 100
            row = [
                m,
                model_types.get(m, "Unknown"),
                f"{r['cer']*100:.4f}",
                f"{r['wer']*100:.4f}",
                f"{r['acc']*100:.4f}",
                f"{coer_avg:.4f}",
                f"{r['eval_time']:.1f}",
            ]
            for cat, _ in COER_COLS:
                row.append(f"{r['coer'].get(cat, 0)*100:.4f}")
            w.writerow(row)
    print(f"  ✓ Global report        → {global_path}")

    # ── 2. CoER breakdown ─────────────────────────────────────────────────────
    coer_path = os.path.join(out_dir, "cross_report_coer.csv")
    with open(coer_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Category", "Category Label"] +
                   [f"{m} CoER (%)" for m in model_list] +
                   [f"{m} Type" for m in model_list])
        for cat, label in COER_COLS:
            row = [cat, label]
            for m in model_list:
                r = results.get(m)
                row.append(f"{r['coer'].get(cat, 0)*100:.4f}" if r else "")
            for m in model_list:
                row.append(model_types.get(m, ""))
            w.writerow(row)
    print(f"  ✓ CoER breakdown       → {coer_path}")

    # ── 3. Character-level ────────────────────────────────────────────────────
    all_chars = sorted(set(
        ch for m in model_list if results.get(m)
        for ch in results[m]["char_stats"].keys()
    ))
    char_path = os.path.join(out_dir, "cross_report_characters.csv")
    with open(char_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["Character", "Category"]
        for m in model_list:
            t = model_types.get(m, "")
            header += [f"{m} [{t}] GT",
                       f"{m} [{t}] Correct",
                       f"{m} [{t}] Errors",
                       f"{m} [{t}] Error Rate (%)"]
        w.writerow(header)
        for ch in all_chars:
            cat = get_component_type(ch)
            row = [ch, cat]
            for m in model_list:
                r = results.get(m)
                cs = r["char_stats"].get(ch, {"total": 0, "correct": 0, "errors": 0}) if r else {}
                t_  = cs.get("total", 0)
                c_  = cs.get("correct", 0)
                e_  = cs.get("errors", 0)
                row += [t_, c_, e_, f"{(e_/t_*100) if t_ else 0.0:.2f}"]
            w.writerow(row)
    print(f"  ✓ Character report     → {char_path}")

    # ── 4. Mistakes ───────────────────────────────────────────────────────────
    mistakes_path = os.path.join(out_dir, "cross_report_mistakes.csv")
    with open(mistakes_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Model Type", "Operation",
                    "GT Char", "Pred Char", "Count"])
        for m in model_list:
            r = results.get(m)
            if not r:
                continue
            errors = [(op, p, t, cnt)
                      for (op, p, t), cnt in r["mistake_ctr"].items()
                      if op != "mat"]
            errors.sort(key=lambda x: x[3], reverse=True)
            for op, p, t, cnt in errors:
                w.writerow([m, model_types.get(m, ""), op,
                             t or "", p or "", cnt])
    print(f"  ✓ Mistakes report      → {mistakes_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: dict, model_types: dict, model_list: list):
    print()
    print("=" * 85)
    print("  Cross-Comparison Results: Original Tesseract  vs  Fine-tuned Models")
    print("=" * 85)
    hdr = (f"  {'Model':<26} {'Type':<22} "
           f"{'CER (%)':>9} {'WER (%)':>9} {'Acc (%)':>9} {'CoER (%)':>10}")
    print(hdr)
    print("  " + "─" * 83)

    # group: originals first, then fine-tuned
    for group in ("Original Tesseract", "Fine-tuned"):
        group_models = [m for m in model_list
                        if model_types.get(m) == group and results.get(m)]
        if not group_models:
            continue
        for m in group_models:
            r = results[m]
            coer_avg = (sum(r["coer"].values()) / max(len(r["coer"]), 1)) * 100
            print(f"  {m:<26} {group:<22} "
                  f"{r['cer']*100:>9.2f} {r['wer']*100:>9.2f} "
                  f"{r['acc']*100:>9.2f} {coer_avg:>10.2f}")
        print("  " + "─" * 83)

    print()
    # Delta table: best fine-tuned vs best original
    orig_models    = [m for m in model_list if model_types.get(m) == "Original Tesseract" and results.get(m)]
    custom_models  = [m for m in model_list if model_types.get(m) == "Fine-tuned" and results.get(m)]
    if orig_models and custom_models:
        best_orig   = min(orig_models,   key=lambda m: results[m]["cer"])
        best_custom = min(custom_models, key=lambda m: results[m]["cer"])
        ro, rc = results[best_orig], results[best_custom]
        delta_cer  = ro["cer"]  - rc["cer"]
        delta_wer  = ro["wer"]  - rc["wer"]
        delta_acc  = rc["acc"]  - ro["acc"]
        print(f"  Best original  : {best_orig}  (CER {ro['cer']*100:.2f}%)")
        print(f"  Best fine-tuned: {best_custom}  (CER {rc['cer']*100:.2f}%)")
        print(f"  Δ CER  = {delta_cer*100:+.2f}%   (positive = fine-tuned is better)")
        print(f"  Δ WER  = {delta_wer*100:+.2f}%")
        print(f"  Δ Acc  = {delta_acc*100:+.2f}%")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global ORIGINAL_TESSDATA_DIR
    parser = argparse.ArgumentParser(
        description="Cross-compare original Tesseract vs fine-tuned models (CSV only).")
    parser.add_argument("--image_folder",   default=IMAGE_FOLDER,
                        help=f"Folder with test images + .gt.txt (default: {IMAGE_FOLDER})")
    parser.add_argument("--out_dir",        default=OUT_DIR,
                        help=f"Output directory for CSVs (default: {OUT_DIR})")
    parser.add_argument("--orig_tessdata",  default=ORIGINAL_TESSDATA_DIR,
                        help="Path to original tessdata dir (default: system default)")
    args = parser.parse_args()

    ORIGINAL_TESSDATA_DIR = args.orig_tessdata

    print("=" * 70)
    print("  Cross-Comparison: Original Tesseract  vs  Fine-tuned OCR")
    print("=" * 70)
    print(f"  Original models : {ORIGINAL_MODELS}")
    print(f"  Fine-tuned      : {CUSTOM_MODELS}")
    print(f"  Image folder    : {args.image_folder}")
    print(f"  Output dir      : {args.out_dir}")
    print()

    if not HAS_TESSERACT:
        print("  ERROR: pytesseract / Pillow not installed.")
        print("  Run:   pip install pytesseract pillow")
        return

    try:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    except Exception:
        pass

    os.makedirs(args.out_dir, exist_ok=True)

    # Find test pairs
    test_pairs = find_test_pairs(args.image_folder)
    if not test_pairs:
        print(f"  ERROR: No *.gt.txt pairs found in {args.image_folder}")
        return
    print(f"  Found {len(test_pairs)} test pairs.\n")

    # Run inference
    predictions, model_types = run_all_models(test_pairs)
    if not predictions:
        print("  ERROR: No models produced predictions.")
        return

    model_list = list(predictions.keys())

    # Compute metrics
    print(f"\n  Computing metrics for {len(model_list)} models...")
    results = compute_metrics(predictions, test_pairs)

    # Print summary
    print_summary(results, model_types, model_list)

    # Export CSVs
    print("  Exporting CSVs...")
    export_cross_csv(results, model_types, model_list, args.out_dir)
    export_predictions_csv(predictions, test_pairs, model_types, args.out_dir)

    print()
    print("=" * 70)
    print(f"  Done.  All CSVs saved to: {args.out_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
