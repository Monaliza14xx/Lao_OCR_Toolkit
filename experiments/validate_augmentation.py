#!/usr/bin/env python3
"""
validate_augmentation.py — Phase 3: Visual & Statistical Validation

Validates that the newly augmented synthetic data's visual distribution
matches real-world document characteristics measured in Phase 1.

Generates:
  - validation_report.md   : side-by-side comparison metrics
  - comparison_gallery/    : visual gallery of real vs. synthetic pairs
  - kl_divergence.json     : KL divergence scores for key features

Usage:
    python3 experiments/validate_augmentation.py \
        --real_dir /path/to/lao-real-documents \
        --synth_dir data/lao-augmented-v2 \
        --analysis_params /path/to/analysis_output/augmentation_params.json \
        --n_synth_sample 100
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# ============================================================================
#  Feature extraction (mirrors analyze_real_data.py)
# ============================================================================

def extract_features(img_path):
    """Extract key visual features from a single image."""
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.float32)

    # Background stats
    bg_mask = arr > 200
    bg_mean = float(arr[bg_mask].mean()) if bg_mask.any() else 255.0
    bg_std = float(arr[bg_mask].std()) if bg_mask.any() else 0.0

    # Contrast
    p5 = float(np.percentile(arr, 5))
    p95 = float(np.percentile(arr, 95))
    michelson = (p95 - p5) / (p95 + p5 + 1e-6)

    # Noise (Laplacian residual)
    from PIL import ImageFilter
    smoothed = np.array(img.filter(ImageFilter.MedianFilter(3)), dtype=np.float32)
    residual = arr - smoothed
    noise_sigma = float(residual.std())

    # Sharpness
    lap = np.array(img.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    sharpness = float(lap.var())

    h, w = arr.shape
    return {
        "height": h, "width": w,
        "bg_mean": bg_mean, "bg_std": bg_std,
        "michelson": michelson,
        "noise_sigma": noise_sigma,
        "sharpness": sharpness,
    }


def collect_features(directory, n_sample=None, ext=(".jpg", ".png", ".tif")):
    """Collect features from all (or sampled) images in a directory."""
    paths = [p for p in Path(directory).glob("*") if p.suffix in ext]
    if not paths:
        print(f"  WARNING: No images found in {directory}")
        return []
    if n_sample and len(paths) > n_sample:
        paths = random.sample(paths, n_sample)
    features = []
    for p in paths:
        try:
            features.append(extract_features(p))
        except Exception as e:
            print(f"  ⚠ Skipping {p}: {e}")
    return features


# ============================================================================
#  KL Divergence (histogram-based)
# ============================================================================

def kl_divergence(p_vals, q_vals, n_bins=30):
    """
    Compute KL divergence D(P||Q) between two distributions.
    P = real, Q = synthetic (so we measure how well synthetic approximates real).
    Lower is better. < 0.1 = excellent, 0.1–0.5 = acceptable, > 0.5 = poor.
    """
    if not p_vals or not q_vals:
        return float("nan")

    all_vals = p_vals + q_vals
    lo, hi = min(all_vals), max(all_vals)
    if lo == hi:
        return 0.0

    bins = np.linspace(lo, hi, n_bins + 1)
    p_hist, _ = np.histogram(p_vals, bins=bins, density=True)
    q_hist, _ = np.histogram(q_vals, bins=bins, density=True)

    eps = 1e-9
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()

    kl = float(np.sum(p_hist * np.log(p_hist / q_hist)))
    return round(kl, 4)


def grade_kl(kl):
    if kl <= 0.05:
        return "🟢 Excellent"
    if kl <= 0.15:
        return "🟢 Good"
    if kl <= 0.50:
        return "🟡 Acceptable"
    return "🔴 Poor"


# ============================================================================
#  Gallery Generator
# ============================================================================

def save_comparison_gallery(real_dir, synth_dir, output_dir, n_pairs=10):
    """
    Create a gallery of side-by-side comparisons: real image (left) vs. nearest
    synthetic image (right). Saved as PNG strips.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        return []

    real_paths = sorted(Path(real_dir).glob("*.jpg"))[:n_pairs]
    synth_paths = sorted(Path(synth_dir).glob("*.jpg"))
    if not synth_paths:
        synth_paths = sorted(Path(synth_dir).glob("*.png"))

    gallery_files = []
    for i, real_path in enumerate(real_paths):
        try:
            real_img = Image.open(real_path).convert("RGB")
            # Pick a random synthetic image
            synth_path = random.choice(synth_paths) if synth_paths else None
            if synth_path:
                synth_img = Image.open(synth_path).convert("RGB")
            else:
                synth_img = Image.new("RGB", real_img.size, (200, 200, 200))

            # Resize both to same height
            target_h = 120
            scale_r = target_h / real_img.height
            scale_s = target_h / synth_img.height
            rw = int(real_img.width * scale_r)
            sw = int(synth_img.width * scale_s)
            real_resized = real_img.resize((rw, target_h), Image.LANCZOS)
            synth_resized = synth_img.resize((sw, target_h), Image.LANCZOS)

            # Cap width at 800px each
            rw = min(rw, 800)
            sw = min(sw, 800)
            real_resized = real_resized.crop((0, 0, rw, target_h))
            synth_resized = synth_resized.crop((0, 0, sw, target_h))

            # Combine side-by-side
            label_h = 20
            gap = 10
            total_w = rw + sw + gap
            strip = Image.new("RGB", (total_w, target_h + label_h), (240, 240, 240))
            strip.paste(real_resized, (0, label_h))
            strip.paste(synth_resized, (rw + gap, label_h))

            out_path = Path(output_dir) / f"compare_{i+1:02d}_{real_path.stem}.png"
            strip.save(out_path)
            gallery_files.append(out_path)
        except Exception as e:
            print(f"  ⚠ Could not build comparison strip for {real_path}: {e}")

    return gallery_files


# ============================================================================
#  Report Generator
# ============================================================================

def build_validation_report(real_feats, synth_feats, kl_scores, gallery_files,
                             real_dir, synth_dir, real_params=None):
    """Generate a Markdown validation report."""

    def fmts(feats, key):
        vals = [f[key] for f in feats if key in f]
        if not vals:
            return "N/A"
        return f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"

    kl_rows = "\n".join(
        f"| {feat} | {kl:.4f} | {grade_kl(kl)} |"
        for feat, kl in kl_scores.items()
    )

    gallery_section = ""
    if gallery_files:
        gallery_section = "\n## 4. Visual Comparison Gallery\n\n"
        gallery_section += "Side-by-side: **real image (left)** vs. **synthetic augmented (right)**.\n\n"
        for gf in gallery_files[:5]:
            gallery_section += f"![{gf.stem}]({gf})\n\n"

    # Verdict
    failing = [k for k, v in kl_scores.items() if v > 0.5]
    passing = [k for k, v in kl_scores.items() if v <= 0.5]
    if not failing:
        verdict = "✅ **PASS** — All features within acceptable KL divergence (< 0.5). The augmented data distribution adequately approximates real-world characteristics."
    else:
        verdict = (f"⚠️ **PARTIAL PASS** — {len(passing)}/{len(kl_scores)} features pass. "
                   f"Features needing improvement: {', '.join(failing)}.")

    report = f"""# Augmentation Validation Report

**Real-World Data:** `{real_dir}`
**Synthetic Data:** `{synth_dir}`

---

## 1. Distribution Comparison

| Feature | Real (mean ± std) | Synthetic (mean ± std) |
|---|---|---|
| Background mean | {fmts(real_feats, 'bg_mean')} | {fmts(synth_feats, 'bg_mean')} |
| Background std (texture) | {fmts(real_feats, 'bg_std')} | {fmts(synth_feats, 'bg_std')} |
| Michelson Contrast | {fmts(real_feats, 'michelson')} | {fmts(synth_feats, 'michelson')} |
| Noise σ | {fmts(real_feats, 'noise_sigma')} | {fmts(synth_feats, 'noise_sigma')} |
| Sharpness (Laplacian var) | {fmts(real_feats, 'sharpness')} | {fmts(synth_feats, 'sharpness')} |

---

## 2. KL Divergence Scores D(Real ∥ Synthetic)

> Lower is better: 🟢 < 0.15 (Good), 🟡 0.15–0.50 (Acceptable), 🔴 > 0.50 (Poor)

| Feature | KL Divergence | Grade |
|---|---|---|
{kl_rows}

---

## 3. Overall Verdict

{verdict}

{gallery_section}
---

*Generated by `validate_augmentation.py` — Phase 3 of the Real-World Lao Document Augmentation Pipeline.*
"""
    return report


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate augmented data against real-world distribution")
    parser.add_argument("--real_dir", required=True,
                        help="Directory with real-world images")
    parser.add_argument("--synth_dir", required=True,
                        help="Directory with newly generated augmented images")
    parser.add_argument("--analysis_params", default=None,
                        help="Path to augmentation_params.json from Phase 1 (optional)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: synth_dir/validation)")
    parser.add_argument("--n_synth_sample", type=int, default=200,
                        help="How many synthetic images to sample for stats (default: 200)")
    parser.add_argument("--n_gallery", type=int, default=10,
                        help="Number of gallery comparison pairs (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.synth_dir) / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Phase 3: Augmentation Validation")
    print(f"{'='*60}\n")

    # Load optional Phase 1 params for reference
    real_params = None
    if args.analysis_params and Path(args.analysis_params).exists():
        real_params = json.loads(Path(args.analysis_params).read_text())
        print(f"  Loaded Phase 1 calibration: {args.analysis_params}")

    print(f"  Extracting features from real images ({args.real_dir})...")
    real_feats = collect_features(args.real_dir, n_sample=None)
    print(f"  → {len(real_feats)} real images")

    print(f"  Extracting features from synthetic images ({args.synth_dir})...")
    synth_feats = collect_features(args.synth_dir, n_sample=args.n_synth_sample)
    print(f"  → {len(synth_feats)} synthetic images sampled")

    if not real_feats or not synth_feats:
        print("  ERROR: Not enough images to compare. Check --real_dir and --synth_dir.")
        sys.exit(1)

    # Compute KL divergences
    feature_keys = ["bg_mean", "bg_std", "michelson", "noise_sigma", "sharpness"]
    feature_labels = {
        "bg_mean": "Background Mean",
        "bg_std": "Background Std (texture)",
        "michelson": "Michelson Contrast",
        "noise_sigma": "Noise σ",
        "sharpness": "Sharpness (Laplacian var)",
    }

    kl_scores = {}
    print("\n  KL Divergence D(Real ∥ Synthetic):")
    for key in feature_keys:
        real_vals = [f[key] for f in real_feats if key in f]
        synth_vals = [f[key] for f in synth_feats if key in f]
        kl = kl_divergence(real_vals, synth_vals)
        kl_scores[feature_labels[key]] = kl
        grade = grade_kl(kl)
        print(f"  {grade} {feature_labels[key]}: KL={kl:.4f}")

    # Save KL divergence JSON
    kl_path = output_dir / "kl_divergence.json"
    kl_path.write_text(json.dumps(kl_scores, indent=2), encoding="utf-8")
    print(f"\n  ✓ KL scores: {kl_path}")

    # Generate gallery
    gallery_dir = output_dir / "comparison_gallery"
    print("\n  Generating visual comparison gallery...")
    gallery_files = save_comparison_gallery(
        args.real_dir, args.synth_dir, gallery_dir, n_pairs=args.n_gallery)
    print(f"  ✓ Gallery: {len(gallery_files)} comparison images in {gallery_dir}")

    # Generate report
    report = build_validation_report(
        real_feats, synth_feats, kl_scores, gallery_files,
        args.real_dir, args.synth_dir, real_params)

    report_path = output_dir / "validation_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  ✓ Validation report: {report_path}")

    # Print overall verdict
    failing = [k for k, v in kl_scores.items() if v > 0.5]
    print(f"\n  {'PASS ✅' if not failing else 'PARTIAL PASS ⚠️'}")
    if failing:
        print(f"  Features needing improvement: {', '.join(failing)}")
    print()


if __name__ == "__main__":
    main()
