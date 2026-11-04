#!/usr/bin/env python3
"""
benchmark.py
────────────────────────────────────────────────────────────────────────────
Full Tesseract vs PyTorch Benchmark Suite.

Reads pre-computed results from:
  - tesseract_data/{model}/      → Tesseract (.traineddata) training logs
  - data/{model}/training.log    → PyTorch (.pt) training logs (CSV format)
  - data/comparison/report.csv   → PyTorch inference metrics (from torch_train.py compare)
  - tesseract_data/tesseract_comparison/  → Tesseract inference metrics

Then generates:
  1. Training curves (8 plots) from both logs
  2. Unified side-by-side Tess vs PT bar charts
  3. Combined benchmark_report.csv + benchmark_report.md
  4. TensorBoard export for all scalars

Usage:
  python3 experiments/benchmark.py
  bash run.sh benchmark
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: tesstrain_core, tesseract_tools

import os
import sys

# Silence TensorFlow and CUDA noise (Must be set before any imports that trigger TF/logging)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
logging.getLogger('absl').setLevel(logging.ERROR)

import glob
import re
import csv
import argparse
import shutil
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter
        HAS_TB = True
    except ImportError:
        HAS_TB = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
MODELS_I48    = ["LaoCRNN-I48K3C16", "LaoCRNN-I48K3C32", "LaoCRNN-I48K5C16", "LaoCRNN-I48K5C32"]
MODELS_I64    = ["LaoCRNN-I64K3C16", "LaoCRNN-I64K3C32", "LaoCRNN-I64K5C16", "LaoCRNN-I64K5C32"]
MODELS        = MODELS_I48 + MODELS_I64

TESS_ROOT     = "tesseract_data"
PT_ROOT       = "data"
PT_COMPARE    = "data/comparison"
TESS_COMPARE  = "tesseract_data/tesseract_comparison"
OUT_DIR       = "benchmark_output"
REPORT_DIR    = os.path.join(OUT_DIR, "reports")
PLOT_DIR      = os.path.join(OUT_DIR, "plots")
ENGINE_DIR    = os.path.join(PLOT_DIR, "engine_details")
TB_DIR        = os.path.join(OUT_DIR, "tensorboard")

# ── Group output directories ─────────────────────────────────────────────────
I48_TESS_DIR  = os.path.join(PLOT_DIR, "I48_tesseract")
I48_PT_DIR    = os.path.join(PLOT_DIR, "I48_PT")
I64_PT_DIR    = os.path.join(PLOT_DIR, "I64_PT")
PT_GLOBAL_DIR = os.path.join(PLOT_DIR, "PT_Global")

# Group 1: I48 Tesseract vs PyTorch
G1_DIR        = os.path.join(PLOT_DIR, "I48_tess_vs_pt")
G1_CURVE      = os.path.join(G1_DIR, "training_curves")
G1_3D         = os.path.join(G1_DIR, "cross_engine_3d")
G1_2D         = os.path.join(G1_DIR, "cross_engine_2d")
G1_COMPARE    = os.path.join(G1_DIR, "comparisons")

# Group 2: I64 PyTorch models (compared among themselves)

G2_DIR        = os.path.join(PLOT_DIR, "I64_pytorch")
G2_CURVE      = os.path.join(G2_DIR, "training_curves")
G2_3D         = os.path.join(G2_DIR, "cross_model_3d")
G2_2D         = os.path.join(G2_DIR, "cross_model_2d")
G2_COMPARE    = os.path.join(G2_DIR, "comparisons")

# Group 3: PyTorch I48 vs I64 (resolution comparison)
G3_DIR        = os.path.join(PLOT_DIR, "I48_vs_I64_pytorch")
G3_3D         = os.path.join(G3_DIR, "cross_resolution_3d")
G3_2D         = os.path.join(G3_DIR, "cross_resolution_2d")
G3_COMPARE    = os.path.join(G3_DIR, "comparisons")

# Lao component columns for CoER
COER_COLS = [
    ("tone",         "Tone Mark"),
    ("upper_vowel",  "Upper Vowel"),
    ("consonant",    "Consonant"),
    ("before_vowel", "Before Vowel"),
    ("rear_vowel",   "After Vowel"),
    ("lower_vowel",  "Lower Vowel"),
    ("special",      "Special"),
    ("latin",        "Latin"),
    ("space",        "Space"),
    ("symbol",       "Symbol"),
    ("digit",        "Digit"),
    ("lao_number",   "Lao Number"),
]

# TensorBoard-inspired palette — high contrast, paper-friendly
PALETTE = [
    '#FF6F00',   # amber
    '#1565C0',   # blue
    '#2E7D32',   # green
    '#C62828',   # red
    '#6A1B9A',   # purple
    '#00838F',   # cyan
    '#EF6C00',   # orange
    '#AD1457',   # pink
    '#283593',   # indigo
    '#558B2F',   # light-green
]

# ══════════════════════════════════════════════════════════════════════════════
# 1. TESSERACT TRAINING LOG PARSER
#    format: "At iteration N/N/N, mean rms=X%, BCER train=Y%, BWER train=Z%"
# ══════════════════════════════════════════════════════════════════════════════
TESS_TRAIN_RE = re.compile(
    r"At iteration (\d+)/\d+/\d+,\s+"
    r"mean rms=([\d.]+)%.*?"
    r"BCER train=([\d.]+)%.*?"
    r"BWER train=([\d.]+)%"
)
TESS_EVAL_FILE_OLD = re.compile(r"[\w]+_[\d.]+_(\d+)_\d+\.eval\.log$")
TESS_EVAL_FILE_NEW = re.compile(r"[\w]+_iter(\d+)\.eval\.log$")
TESS_EVAL_RE       = re.compile(r"BCER eval=([\d.]+),\s+BWER eval=([\d.]+)")
TESS_CHECK_RE      = re.compile(r"Iteration (\d+):\s+BCER train=([\d.]+)%")



def parse_validating_log(path):
    """
    Parse 'time' command output in validating.log.
    Returns number of seconds (float) or None.
    Example line: 'real	1230m16.607s'
    """
    if not os.path.exists(path):
        return None
    # Pattern to match 'real' followed by minutes and seconds
    # Handles both 'XmYs' and 'Ys' formats
    time_re = re.compile(r"real\s+(?:(\d+)m)?([\d.]+)s")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                m = time_re.search(line)
                if m:
                    mins = int(m.group(1)) if m.group(1) else 0
                    secs = float(m.group(2))
                    return mins * 60 + secs
    except Exception as e:
        print(f"  ⚠ Could not read {path}: {e}", file=sys.stderr)
    return None


def parse_tess_training(model_name):
    """Parse Tesseract lstmtraining logs → train + eval series."""
    log_path = os.path.join(TESS_ROOT, model_name, "training.log")
    eval_dir = os.path.join(TESS_ROOT, model_name, "eval")

    train_series = []
    start_ts = None
    end_ts = None
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("[") and len(line) > 20 and line[20] == "]":
                    try:
                        dt = datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S")
                        if start_ts is None: start_ts = dt
                        end_ts = dt
                    except ValueError: pass
                m = TESS_TRAIN_RE.search(line)
                if m:
                    train_series.append({
                        "iteration": int(m.group(1)),
                        "mean_rms":  float(m.group(2)),
                        "bcer":      float(m.group(3)),
                        "bwer":      float(m.group(4)),
                    })
    train_series.sort(key=lambda x: x["iteration"])
    max_iter = train_series[-1]["iteration"] if train_series else 0

    eval_series = []
    if os.path.exists(eval_dir):
        for fpath in glob.glob(os.path.join(eval_dir, "*.eval.log")):
            fname = os.path.basename(fpath)
            iteration = -1
            m = TESS_EVAL_FILE_NEW.search(fname) or TESS_EVAL_FILE_OLD.search(fname)
            if m:
                iteration = int(m.group(1))
            if iteration >= 0:
                with open(fpath, encoding="utf-8") as fh:
                    for line in fh:
                        em = TESS_EVAL_RE.search(line)
                        if em:
                            eval_series.append({
                                "iteration": iteration,
                                "bcer": float(em.group(1)),
                                "bwer": float(em.group(2)),
                            })
    eval_series.sort(key=lambda x: x["iteration"])

    # Double check training.log for checkpoint BCERs (Tesseract 5 standard)
    check_series = []
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                mc = TESS_CHECK_RE.search(line)
                if mc:
                    check_series.append({
                        "iteration": int(mc.group(1)),
                        "bcer":      float(mc.group(2)),
                    })
    check_series.sort(key=lambda x: x["iteration"])

    train_time_h = 0.0
    if start_ts and end_ts:
        train_time_h = (end_ts - start_ts).total_seconds() / 3600.0

    val_time_s = 0.0
    found_vlog = False
    for vlog in [os.path.join(TESS_ROOT, model_name, "validating.log"),
                 os.path.join(eval_dir, "validating.log")]:
        v_sec = parse_validating_log(vlog)
        if v_sec is not None:
            val_time_s = v_sec
            found_vlog = True
            break

    best_eval = min(eval_series, key=lambda x: x["bcer"], default=None)
    return {
        "train_series": train_series,
        "check_series": check_series,
        "eval_series":  eval_series,
        "best_bcer":    best_eval["bcer"] if best_eval else None,
        "best_iter":    best_eval["iteration"] if best_eval else None,
        "max_iter":     max_iter,
        "train_time_h": train_time_h,
        "val_time_s":   val_time_s if found_vlog else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. PYTORCH TRAINING LOG PARSER
#    format: CSV with header from TrainingLogger
# ══════════════════════════════════════════════════════════════════════════════
def parse_pt_training(model_name):
    """Parse PyTorch training.log CSV → train + eval series."""
    log_path = os.path.join(PT_ROOT, model_name, "training.log")
    if not os.path.exists(log_path):
        return {"train_series": [], "eval_series": [], "best_bcer": None, "best_iter": None}

    train_series = []
    eval_series  = []
    start_ts = None
    end_ts = None

    with open(log_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row.get("timestamp", "")
            if ts_str:
                try:
                    dt = datetime.fromisoformat(ts_str)
                    if start_ts is None: start_ts = dt
                    end_ts = dt
                except ValueError: pass

            try:
                it = int(row.get("iteration", "") or 0)
            except ValueError:
                continue
            if it == 0:
                continue

            def _f(k):
                v = row.get(k, "")
                return float(v) if v else None

            # Train row
            tr_cer = _f("train_cer")
            tr_wer = _f("train_wer")
            # PyTorch CSV uses 'train_loss', Tesseract uses 'mean_rms'
            rms    = _f("train_loss") or _f("mean_rms")
            if tr_cer is not None:
                train_series.append({
                    "iteration": it,
                    "bcer": tr_cer * 100,   # convert 0-1 → %
                    "bwer": (tr_wer or 0) * 100,
                    "mean_rms": (rms or 0) * 100 if rms and rms <= 1 else (rms or 0),
                })

            # Eval row
            ev_cer  = _f("eval_cer")
            ev_wer  = _f("eval_wer")
            ev_loss = _f("eval_loss")
            if ev_cer is not None:
                eval_series.append({
                    "iteration": it,
                    "bcer": ev_cer * 100,
                    "bwer": (ev_wer or 0) * 100,
                    "eval_loss": (ev_loss or 0) * 100 if ev_loss and ev_loss <= 1 else (ev_loss or 0),
                })

    train_series.sort(key=lambda x: x["iteration"])
    eval_series.sort(key=lambda x: x["iteration"])
    max_iter = train_series[-1]["iteration"] if train_series else 0

    train_time_h = 0.0
    if start_ts and end_ts:
        train_time_h = (end_ts - start_ts).total_seconds() / 3600.0

    best_eval = min(eval_series, key=lambda x: x["bcer"], default=None)
    return {
        "train_series": train_series,
        "eval_series":  eval_series,
        "best_bcer":    best_eval["bcer"] if best_eval else None,
        "best_iter":    best_eval["iteration"] if best_eval else None,
        "max_iter":     max_iter,
        "train_time_h": train_time_h,
        "val_time_s":   train_time_h * 3600.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. INFERENCE RESULTS: read comparison CSVs
# ══════════════════════════════════════════════════════════════════════════════
def load_pt_inference_results():
    """Read data/comparison/report.csv → dict keyed by model_name.
    torch_train.py compare writes:
      Model | ... | Test CER | Test WER | Test Accuracy | Test Eval Time (s) | Train Time (h) | Test CoER: Tone Mark | ...
    Values in CER/WER/Accuracy are fractions (0-1). CoER columns use the label name directly.
    """
    csv_path = os.path.join(PT_COMPARE, "report.csv")
    results = {}
    if not os.path.exists(csv_path):
        print(f"  [PT]   No inference CSV at {csv_path}. Run: python3 torch_train.py compare ...")
        return results
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("Model", "").strip()
            if not name:
                continue
            try:
                coer = {}
                for cat, label in COER_COLS:
                    # compare.py writes "Test CoER: Tone Mark" etc.
                    v = row.get(f"Test CoER: {label}",
                        row.get(f"Real-world CoER: {label}",
                        row.get(label, row.get(cat, ""))))
                    if v not in (None, ""):
                        try:
                            raw = float(v)
                            # If already fraction-like (<= 1.5), keep; else divide by 100
                            coer[cat] = raw if raw <= 1.5 else raw / 100.0
                        except ValueError:
                            pass

                def _f(keys):
                    for k in keys:
                        v = row.get(k, "")
                        if v not in (None, ""):
                            try:
                                return float(v)
                            except ValueError:
                                pass
                    return 0.0

                results[name] = {
                    # Test split is primary; fall back to old column names
                    "cer":        _f(["Test CER", "CER"]),
                    "wer":        _f(["Test WER", "WER"]),
                    "acc":        _f(["Test Accuracy", "Accuracy"]),
                    "eval_time":  _f(["Test Eval Time (s)", "Eval Time (s)"]),
                    "rw_cer":     _f(["Real-world Test CER", "Real-world CER", "Real-world CER (%)"]),
                    "rw_wer":     _f(["Real-world Test WER", "Real-world WER", "Real-world WER (%)"]),
                    "rw_acc":     _f(["Real-world Test Accuracy", "Real-world Accuracy", "Real-world Accuracy (%)"]),
                    "rw_eval_time": _f(["Real-world Test Eval Time (s)", "Real-world Test Time (s)", "Real-world Eval Time (s)"]),
                    "train_time": _f(["Train Time (h)"]),
                    "val_time_s": _f(["Eval CER"]) * 0,  # not in this CSV
                    "coer":       coer,
                }
            except (ValueError, TypeError):
                pass
    print(f"  [PT]   Loaded inference results for: {list(results.keys())}")
    return results



def load_tess_inference_results():
    """
    Read tesseract_data/tesseract_comparison/report.csv.
    tesseract_tools.py writes:
      Model | CER (%) | WER (%) | Accuracy (%) | Eval Time (s) | Test Time (s) | Train Time (s) | Train Time (HH:MM:SS) | CoER Tone Mark (%) | ...
    All metric values are already in percent (e.g. 3.79 means 3.79%).
    We divide by 100 to store as fractions (0-1), consistent with PT side.
    """
    csv_path = os.path.join(TESS_COMPARE, "report.csv")
    if not os.path.exists(csv_path):
        print(f"  [Tess] No inference CSV at {csv_path}. Run: ./run.sh compare")
        return {}

    results = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("Model", "").strip()
            if name.endswith("_final"):
                name = name[:-6]  # Strip '_final' suffix
            if not name:
                continue
            try:
                def _pct(keys, default=None):
                    """Read value (already in %) → return as 0-1 fraction."""
                    for k in keys:
                        v = row.get(k, "")
                        if v not in (None, ""):
                            return float(v) / 100.0
                    return 0.0

                coer = {}
                for cat, label in COER_COLS:
                    # tesseract_tools writes "CoER Tone Mark (%)" etc.
                    v = row.get(f"CoER {label} (%)", row.get(f"CoER {label}", ""))
                    if v not in (None, ""):
                        try:
                            coer[cat] = float(v) / 100.0
                        except ValueError:
                            pass

                results[name] = {
                    "cer":        _pct(["CER (%)", "CER"]),
                    "wer":        _pct(["WER (%)", "WER"]),
                    "acc":        _pct(["Accuracy (%)", "Acc (%)", "Accuracy"]),
                    "eval_time":  float(row.get("Test Time (s)", row.get("Eval Time (s)", 0)) or 0),
                    "rw_cer":     _pct(["Real-world CER (%)", "Real-world CER", "Real-world Test CER"]),
                    "rw_wer":     _pct(["Real-world WER (%)", "Real-world WER", "Real-world Test WER"]),
                    "rw_acc":     _pct(["Real-world Accuracy (%)", "Real-world Accuracy", "Real-world Test Accuracy"]),
                    "rw_eval_time": float(row.get("Real-world Test Time (s)", row.get("Real-world Test Eval Time (s)", 0)) or 0),
                    "val_time_s": float(row.get("Eval Time (s)", 0) or 0) if "Test Time (s)" in row else None,
                    "train_time": float(row.get("Train Time (s)", 0) or 0) / 3600.0,
                    "coer":       coer,
                }
            except (ValueError, TypeError) as e:
                print(f"  [Tess] Warning parsing {name}: {e}")

    if results:
        print(f"  [Tess] Loaded inference results for: {list(results.keys())}")
    else:
        print(f"  [Tess] CSV exists but no rows parsed from {csv_path}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4. PLOTTING  (TensorBoard-inspired style)
# ══════════════════════════════════════════════════════════════════════════════
DPI = 300  # Paper-quality output

def _setup_ax(ax, title, xlabel, ylabel):
    """Quick axis setup — theme is applied globally by apply_premium_theme."""
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=10)


def render_training_plots(tess_logs, pt_logs):
    """Render the training-curve plots with TensorBoard-inspired style."""
    from tesstrain_core.plotting import (
        apply_premium_theme, smooth_ema, add_best_annotation,
        tb_plot_series, tb_save, fill_between_smooth,
    )
    os.makedirs(PLOT_DIR, exist_ok=True)

    def _get(logs, model, series, key):
        s = logs.get(model, {}).get(series, [])
        return [(d["iteration"], d[key]) for d in s if key in d]

    def _plot_combined(title, ylabel, filename, series_data, log_y=False, ylim=None, save_dir=G1_CURVE):
        """Combined multi-model overlay — TensorBoard style."""
        fig, ax = plt.subplots(figsize=(10, 5.5))
        apply_premium_theme(fig, ax)
        ax.set_title(title, fontweight='bold', fontsize=14, pad=12)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        if log_y: ax.set_yscale('log')
        if ylim: ax.set_ylim(ylim)

        for _i, (label, data, color) in enumerate(series_data):
            if not data: continue
            tb_plot_series(ax, data, color, label,
                           smoothing=0.6, raw_alpha=0.18, raw_lw=0.6,
                           smooth_lw=2.2, fill=True)

        ax.legend(loc='best', fontsize=9)
        fig.tight_layout()
        tb_save(fig, os.path.join(save_dir, filename), dpi=DPI)
        print(f"    ✓ plots/{os.path.relpath(os.path.join(save_dir, filename), PLOT_DIR)}")

    def _plot_3d_waterfall_combined(title, zlabel, filename, series_data, save_dir):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3-D projection)
        from matplotlib.collections import PolyCollection
        import matplotlib.colors as mcolors
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        # White background for 3D plots
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.xaxis.pane.set_facecolor('white')
        ax.yaxis.pane.set_facecolor('white')
        ax.zaxis.pane.set_facecolor('white')

        # Remove edge lines from panes
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')

        # Crisp, light grey gridlines
        ax.grid(True, color='#eeeeee', linestyle='-', linewidth=1.0)
        ax.xaxis._axinfo["grid"].update({"color": "#dddddd", "linewidth": 1.0})
        ax.yaxis._axinfo["grid"].update({"color": "#dddddd", "linewidth": 1.0})
        ax.zaxis._axinfo["grid"].update({"color": "#dddddd", "linewidth": 1.0})
        
        yticks = []
        yticklabels = []
        verts = []
        poly_colors = []
        
        # Decide the floor correctly based on metric type
        z_floor = 0.0
        
        for i, (label, data, color) in enumerate(series_data):
            if not data: continue
            sm = smooth_ema(data, weight=0.6)
            xs = [d[0] for d in sm]
            zs = [d[1] for d in sm]
            
            y_depth = len(series_data) - i - 1  # Reverse order for better visibility depending on view
            
            # Draw the solid leading edge
            ax.plot(xs, zs, zs=y_depth, zdir='y', color=color, lw=2.0)
            
            # Polygons to floor
            v = [(xs[0], z_floor)] + list(zip(xs, zs)) + [(xs[-1], z_floor)]
            verts.append(v)
            poly_colors.append(mcolors.to_rgba(color, alpha=0.35))
                
            yticks.append(y_depth)
            yticklabels.append(label)
            
        # Draw the shaded sections
        poly = PolyCollection(verts, facecolors=poly_colors, edgecolors='none')
        ax.add_collection3d(poly, zs=yticks, zdir='y')
            
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=10)
        ax.set_xlabel("Iterations", labelpad=15, fontsize=12)
        ax.set_zlabel(zlabel, labelpad=15, fontsize=12)
        ax.set_title(title, fontweight="bold", fontsize=16, pad=20)
        
        # Adjust zoom and margins to account for polygon rendering
        ax.set_zlim(bottom=z_floor)
        
        # Optimal viewing angle for standing lines
        ax.view_init(elev=25, azim=-55)
        
        os.makedirs(save_dir, exist_ok=True)
        fig.tight_layout()
        tb_save(fig, os.path.join(save_dir, filename), dpi=DPI)
        print(f"    ✓ plots/{os.path.relpath(os.path.join(save_dir, filename), PLOT_DIR)}")

    # ══════════════════════════════════════════════════════════════════════════
    # Helper: render a group of subplots (per-model panels)
    # ══════════════════════════════════════════════════════════════════════════
    def _render_subplots(group_models, tess, pt, curve_dir, tag, overlay_engines=True, fname_suffix=""):
        """Render 4 per-model subplot sheets — TensorBoard style."""
        os.makedirs(curve_dir, exist_ok=True)
        titles = [
            # (title,                                   filename,                   key,    series,          ylabel,        is_eval)
            (f"Train CER — {tag}",  "plot_train_loss_CER.png", "bcer", "train_series", "BCER (%)",  False),
            (f"Train WER — {tag}",  "plot_train_loss_WER.png", "bwer", "train_series", "BWER (%)",  False),
            (f"Eval CER — {tag}",   "plot_eval_loss_CER.png",  "bcer", "eval_series",  "BCER (%)",  True),
            (f"Eval WER — {tag}",   "plot_eval_loss_WER.png",  "bwer", "eval_series",  "BWER (%)",  True),
            (f"Train Accuracy — {tag}", "plot_train_acc.png",  "bcer", "train_series", "Accuracy (%)", False),
            (f"Eval Accuracy — {tag}",  "plot_eval_acc.png",   "bcer", "eval_series",  "Accuracy (%)", True),
        ]
        for title, fname, key, series, ylabel, is_eval in titles:
            if fname_suffix:
                fname = fname.replace(".png", f"_{fname_suffix}.png")
            n = len(group_models)
            ncols = min(4, n)
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
            axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            apply_premium_theme(fig, axes_flat)
            fig.suptitle(title, fontweight='bold', fontsize=15, y=0.99)

            for idx, m in enumerate(group_models):
                ax = axes_flat[idx]
                color = PALETTE[idx % len(PALETTE)]
                t_data, p_data = [], []

                # ── Triple BCER panel (CER train only) ──
                if fname == "plot_train_loss_CER.png":
                    if overlay_engines:
                        t_iter  = _get(tess, m, "train_series", "bcer")
                        t_check = _get(tess, m, "check_series", "bcer")
                        t_eval  = _get(tess, m, "eval_series",  "bcer")
                        if t_iter:
                            tb_plot_series(ax, t_iter, color, 'Tess Train (Iter)',
                                           smoothing=0.6, raw_alpha=0.12, smooth_lw=1.2, fill=False)
                        if t_check:
                            tb_plot_series(ax, t_check, color, 'Tess Train (Chk)',
                                           smoothing=0.5, raw_alpha=0.15, smooth_lw=1.8, fill=True)
                            add_best_annotation(ax, t_check, color, mode='min')
                        elif t_iter:
                            add_best_annotation(ax, t_iter, color, mode='min')
                        if t_eval:
                            ax.plot(*zip(*t_eval), color='#333333', marker='*',
                                    markersize=5, alpha=0.6, linestyle='None',
                                    label='Tess Eval', zorder=8)

                    p_iter = _get(pt, m, "train_series", "bcer")
                    p_eval = _get(pt, m, "eval_series",  "bcer")
                    if p_iter:
                        tb_plot_series(ax, p_iter, color, 'PT Train',
                                       smoothing=0.6, raw_alpha=0.15, smooth_lw=2.2, fill=True)
                        add_best_annotation(ax, p_iter, color, mode='min')
                    if p_eval:
                        ax.plot(*zip(*p_eval), color='#C62828', marker='o',
                                markersize=4, alpha=0.55, linestyle='None',
                                label='PT Eval', zorder=8)

                # ── Standard panel ──
                else:
                    t_data = _get(tess, m, series, key) if overlay_engines else []
                    p_data = _get(pt, m, series, key)
                    if key == "bcer" and "Acc" in ylabel:
                        t_data = [(it, 100 - v) for it, v in t_data]
                        p_data = [(it, 100 - v) for it, v in p_data]

                    if t_data:
                        sm_t = smooth_ema(t_data, weight=0.5 if is_eval else 0.6)
                        ax.plot(*zip(*t_data), color=color, alpha=0.12, lw=0.6)
                        ax.plot(*zip(*sm_t), color=color, lw=1.8, linestyle='--', label='Tess Train' if not is_eval else 'Tess (CPU)')
                        fill_between_smooth(ax, t_data, sm_t, color, alpha=0.05)
                        add_best_annotation(ax, t_data, color, mode='max' if 'Acc' in ylabel else 'min')
                    if p_data:
                        sm_p = smooth_ema(p_data, weight=0.5 if is_eval else 0.6)
                        ax.plot(*zip(*p_data), color=color, alpha=0.15, lw=0.6)
                        ax.plot(*zip(*sm_p), color=color, lw=2.2, label='PT Train' if not is_eval else 'PT (GPU)')
                        fill_between_smooth(ax, p_data, sm_p, color, alpha=0.06)
                        add_best_annotation(ax, p_data, color, mode='max' if 'Acc' in ylabel else 'min')

                    # Overlay Eval points if this is a Train plot
                    if not is_eval:
                        t_eval = _get(tess, m, "eval_series", key) if overlay_engines else []
                        p_eval = _get(pt, m, "eval_series", key)
                        if key == "bcer" and "Acc" in ylabel:
                            t_eval = [(it, 100 - v) for it, v in t_eval]
                            p_eval = [(it, 100 - v) for it, v in p_eval]

                        if t_eval:
                            ax.plot(*zip(*t_eval), color='#333333', marker='*',
                                    markersize=5, alpha=0.6, linestyle='None',
                                    label='Tess Eval', zorder=8)
                        if p_eval:
                            ax.plot(*zip(*p_eval), color='#C62828', marker='o',
                                    markersize=4, alpha=0.55, linestyle='None',
                                    label='PT Eval', zorder=8)

                title_m = m if m.startswith('LaoCRNN-') else f'LaoCRNN-{m}'
                ax.set_title(f'Model: {title_m}', fontsize=11, fontweight='bold')
                ax.set_xlabel('Iteration'); ax.set_ylabel(ylabel)
                ax.legend(fontsize=7, loc='best')
                if ("BCER" in ylabel or "BWER" in ylabel) and (len(t_data) > 0 or len(p_data) > 0 or fname in ("plot_train_loss_CER.png", "plot_train_loss_WER.png")):
                    ax.set_yscale('log')

            for idx in range(n, len(axes_flat)):
                axes_flat[idx].set_visible(False)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            tb_save(fig, os.path.join(curve_dir, fname), dpi=DPI)
            print(f"    ✓ {os.path.relpath(os.path.join(curve_dir, fname), PLOT_DIR)}")

    # ══════════════════════════════════════════════════════════════════════════
    # Helper: render paired metric charts for paper (1×2 layout)
    # ══════════════════════════════════════════════════════════════════════════
    def _render_paper_pairs(group_models, tess, pt, curve_dir, tag, overlay_engines=True):
        """
        Generate 2 paper-ready figures, each containing 2 side-by-side panels:
          • plot_accuracy.png  →  [Train Accuracy | Eval Accuracy]
          • plot_error.png     →  [Train BCER     | Eval BCER    ]
        All models are overlaid in each panel using TensorBoard style.
        """
        os.makedirs(curve_dir, exist_ok=True)

        pairs = [
            # (filename,       left_title,          right_title,          left_series, right_series, transform,    ylabel_left,      ylabel_right,    log_y)
            ("plot_accuracy.png", "Training Accuracy", "Evaluation Accuracy", "train_series", "eval_series", lambda v: 100-v, "Accuracy (%)", "Accuracy (%)", False),
            ("plot_error.png",    "Training BCER",     "Evaluation BCER",     "train_series", "eval_series", None,            "BCER (%)",     "BCER (%)",     True),
        ]

        for fname, left_title, right_title, left_series, right_series, transform, yl, yr, log_y in pairs:
            fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))
            apply_premium_theme(fig, [ax_l, ax_r])
            fig.suptitle(f"{tag}", fontweight='bold', fontsize=14, y=1.0)

            for panel_idx, (ax, panel_title, series_key) in enumerate([
                (ax_l, left_title, left_series),
                (ax_r, right_title, right_series),
            ]):
                ax.set_title(panel_title, fontweight='bold', fontsize=12)
                ax.set_xlabel('Iteration')
                ax.set_ylabel(yl if panel_idx == 0 else yr)
                if log_y:
                    ax.set_yscale('log')

                for i, m in enumerate(group_models):
                    color = PALETTE[i % len(PALETTE)]
                    short = m.replace('LaoCRNN-', '')
                    label_name = f"LaoCRNN-{short}"

                    # Tesseract data
                    if overlay_engines:
                        t_data = _get(tess, m, series_key, "bcer")
                        if transform and t_data:
                            t_data = [(it, transform(v)) for it, v in t_data]
                        if t_data:
                            tb_plot_series(ax, t_data, color, f'{label_name} Tess',
                                           smoothing=0.6, raw_alpha=0.10, smooth_lw=1.4,
                                           fill=False)

                    # PyTorch data
                    p_data = _get(pt, m, series_key, "bcer")
                    if transform and p_data:
                        p_data = [(it, transform(v)) for it, v in p_data]
                    if p_data:
                        tb_plot_series(ax, p_data, color, f'{label_name} PT',
                                       smoothing=0.6, raw_alpha=0.15, smooth_lw=2.0,
                                       fill=True)

                ax.legend(fontsize=7, loc='best', ncol=2)

            fig.tight_layout(rect=[0, 0, 1, 0.96])
            tb_save(fig, os.path.join(curve_dir, fname), dpi=DPI)
            print(f"    ✓ {os.path.relpath(os.path.join(curve_dir, fname), PLOT_DIR)}")


    # ─────────────────────────────────────────────────────────────────────────
    # Helper: render training_curves subplots broken into individual kernel groups
    # ─────────────────────────────────────────────────────────────────────────
    def _render_subplots_by_kernel(all_models, tess, pt, base_curve_dir,
                                   tag, overlay_engines=False):
        """
        Extract all unique K values from the models, and for each K, generate
        a dedicated figure with subplots (e.g. K=3 -> C16 and C32 subplots).
        Saves them in base_curve_dir/by_kernel/
        """
        import re
        from collections import defaultdict
        
        by_k = defaultdict(list)
        for m in all_models:
            m_k = re.search(r'K(\d+)', m)
            if m_k:
                by_k[m_k.group(1)].append(m)
                
        if not by_k:
            return
            
        subdir = os.path.join(base_curve_dir, "by_kernel")
        os.makedirs(subdir, exist_ok=True)
        
        for k_val in sorted(by_k.keys(), key=int):
            filtered = by_k[k_val]
            print(f"      ── Kernel group K={k_val}: {filtered}")
            _render_subplots(filtered, tess, pt, subdir,
                             f"{tag} [K={k_val}]",
                             overlay_engines=overlay_engines,
                             fname_suffix=f"K{k_val}")

    def _render_kernel_group_overlay_charts(group_models, tess, pt, out_dir, tag):
        """
        Generate 2D line plot comparison charts (overlaying models of different K)
        separated into folders for each kernel types and split by channels (C16, C32, etc.)
        using standard matplotlib style and 1x3 grid layout.
        """
        os.makedirs(out_dir, exist_ok=True)
        
        # Split group_models by channel C value
        from collections import defaultdict
        by_c = defaultdict(list)
        for m in group_models:
            meta = parse_model_meta(m)
            if meta["C"] is not None:
                by_c[meta["C"]].append(m)
                
        metrics_train = [
            ("Train Accuracy", "bcer", "train_series", "Accuracy (%)", lambda v: 100 - v, False),
            ("Train Loss CER", "bcer", "train_series", "BCER (%)", None, True),
            ("Train Loss WER", "bwer", "train_series", "BWER (%)", None, False),
        ]
        
        metrics_eval = [
            ("Eval Accuracy",  "bcer", "eval_series",  "Accuracy (%)", lambda v: 100 - v, False),
            ("Eval Loss CER",  "bcer", "eval_series",  "BCER (%)", None, False),
            ("Eval Loss WER",  "bwer", "eval_series",  "BWER (%)", None, False),
        ]
        
        for c_val in sorted(by_c.keys()):
            models = by_c[c_val]
            # Sort models by K value
            models.sort(key=lambda m: parse_model_meta(m)["K"] or 0)
            
            def render_3col_fig(metrics_list, prefix):
                # Standard matplotlib style without dark background theme
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                fig.suptitle(f"{tag} ({prefix}) - C{c_val}", fontweight='bold', fontsize=14)
                
                for ax, (title, key, series_key, ylabel, transform, log_y) in zip(axes, metrics_list):
                    for i, m in enumerate(models):
                        color = PALETTE[i % len(PALETTE)]
                        d = _get(pt, m, series_key, key)
                        if transform and d:
                            d = [(it, transform(v)) for it, v in d]
                            
                        if d:
                            title_name = m if m.startswith("LaoCRNN-") else f"LaoCRNN-{m}"
                            x_vals = [pt[0] for pt in d]
                            y_vals = [pt[1] for pt in d]
                            ax.plot(x_vals, y_vals, label=title_name, color=color, linewidth=2)
                    
                    ax.set_title(title, fontweight='bold', fontsize=12)
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel(ylabel)
                    if log_y:
                        ax.set_yscale('log')
                    ax.grid(True, linestyle='--', alpha=0.6)
                    ax.legend(fontsize=9, loc='best')
                
                fig.tight_layout()
                fname = f"plot_{prefix.lower()}_metrics_C{c_val}.png"
                fig.savefig(os.path.join(out_dir, fname), dpi=DPI, bbox_inches='tight')
                plt.close(fig)
                print(f"      ✓ Overlay Grid: {os.path.relpath(os.path.join(out_dir, fname), PLOT_DIR)}")

            render_3col_fig(metrics_train, "Train")
            render_3col_fig(metrics_eval, "Eval")

    def _render_summary_grid(group_models, save_dir, filename, tag, engines="both"):
        """Render a 2×2 metric grid — TensorBoard style."""
        os.makedirs(save_dir, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        apply_premium_theme(fig, axes.flatten())
        fig.suptitle(f'Metric Evolution: {tag}', fontweight='bold', fontsize=16, y=0.98)

        panels = [
            ('Error Evolution (BCER)', 'bcer',      'train_series', 'BCER (%)', False),
            ('Validation Error',       'eval_loss', 'eval_series',  'Error (%)', True),
            ('Training Accuracy',      'bcer',      'train_series', 'Accuracy (%)', False),
            ('Validation Accuracy',    'bcer',      'eval_series',  'Accuracy (%)', True),
        ]

        for ax_idx, (title, key, series, ylabel, is_eval) in enumerate(panels):
            ax = axes.flatten()[ax_idx]
            ax.set_title(title, fontweight='bold', fontsize=13)
            ax.set_xlabel('Iterations')
            ax.set_ylabel(ylabel)

            for i, m in enumerate(group_models):
                color = PALETTE[i % len(PALETTE)]
                transform = (lambda x: 100 - x) if 'Accuracy' in title else (lambda x: x)
                title_m = m if m.startswith("LaoCRNN-") else f"LaoCRNN-{m}"
                lbl = f' ({title_m})' if len(group_models) > 1 else ''

                # ── Error Evolution panel ──
                if title == 'Error Evolution (BCER)':
                    if engines in ('both', 'tess'):
                        t_iter  = _get(tess_logs, m, 'train_series', 'bcer')
                        t_check = _get(tess_logs, m, 'check_series', 'bcer')
                        t_eval  = _get(tess_logs, m, 'eval_series',  'bcer')
                        if t_iter:
                            d = [(it, transform(v)) for it, v in t_iter]
                            tb_plot_series(ax, d, color, f'Tess Iter{lbl}',
                                           smoothing=0.6, smooth_lw=1.0, raw_alpha=0.08, fill=False)
                        if t_check:
                            d = [(it, transform(v)) for it, v in t_check]
                            tb_plot_series(ax, d, color, f'Tess Chk{lbl}',
                                           smoothing=0.5, smooth_lw=1.6, raw_alpha=0.12, fill=True)
                        if t_eval:
                            d = [(it, transform(v)) for it, v in t_eval]
                            ax.plot(*zip(*d), color='#333333', marker='*',
                                    markersize=4, alpha=0.5, linestyle='None')

                    if engines in ('both', 'pt'):
                        p_iter = _get(pt_logs, m, 'train_series', 'bcer')
                        p_eval = _get(pt_logs, m, 'eval_series',  'bcer')
                        if p_iter:
                            d = [(it, transform(v)) for it, v in p_iter]
                            tb_plot_series(ax, d, color, f'PT Train{lbl}',
                                           smoothing=0.6, smooth_lw=2.0, fill=True)
                        if p_eval:
                            d = [(it, transform(v)) for it, v in p_eval]
                            ax.plot(*zip(*d), color='#C62828', marker='o',
                                    markersize=3, alpha=0.4, linestyle='None')

                # ── Standard panel ──
                else:
                    t_data, p_data = [], []
                    if engines in ('both', 'tess'):
                        tk = 'bcer' if (is_eval and key == 'eval_loss') else key
                        d = _get(tess_logs, m, series, tk)
                        if d: t_data = [(it, transform(v)) for it, v in d]
                    if engines in ('both', 'pt'):
                        d = _get(pt_logs, m, series, key)
                        if d: p_data = [(it, transform(v)) for it, v in d]
                        if is_eval and key == 'eval_loss' and not p_data:
                            d = _get(pt_logs, m, series, 'bcer')
                            if d: p_data = [(it, transform(v)) for it, v in d]

                    if t_data:
                        tb_plot_series(ax, t_data, color, f'Tess{lbl}',
                                       smoothing=0.6, smooth_lw=1.6, fill=True)
                    if p_data:
                        tb_plot_series(ax, p_data, color, f'PT{lbl}',
                                       smoothing=0.6, smooth_lw=2.0, fill=True)

            ax.legend(fontsize=7, loc='best')
            if 'BCER' in title or 'Loss' in title or 'Error' in title:
                ax.set_yscale('log')

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        tb_save(fig, os.path.join(save_dir, filename), dpi=DPI)
        print(f"    ✓ {os.path.relpath(os.path.join(save_dir, filename), PLOT_DIR)}")



    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 1 — I48: Tesseract vs PyTorch
    # ══════════════════════════════════════════════════════════════════════════
    print("    ── Group 1: I48 Tesseract vs PyTorch ──")
    os.makedirs(G1_CURVE, exist_ok=True)
    os.makedirs(G1_3D, exist_ok=True)

    # Combined line plots
    for metric_key, ylabel, fname, log_y, ylim, transform in [
        ("bcer", "Training error (BCER %)", "plot_loss_combined.png", True, None, None),
        ("bcer", "Training accuracy ", "plot_acc_combined.png", False, (0, 105), lambda v: 100 - v),
        ("bwer", "Training WER", "plot_wer_combined.png", False, None, None),
    ]:
        series_all  = []
        series_tess = []
        series_pt   = []
        for i, m in enumerate(MODELS_I48):
            t_data = _get(tess_logs, m, "train_series", metric_key)
            p_data = _get(pt_logs, m, "train_series", metric_key)
            if transform:
                t_data = [(it, transform(v)) for it, v in t_data] if t_data else []
                p_data = [(it, transform(v)) for it, v in p_data] if p_data else []
            title_m = m if m.startswith("LaoCRNN-") else f"LaoCRNN-{m}"
            if t_data:
                name = f"{title_m} (Tess)"
                color = PALETTE[(i*2) % len(PALETTE)]
                series_all.append((name, t_data, color))
                series_tess.append((title_m, t_data, color))
            if p_data:
                name = f"{title_m} (PT)"
                color = PALETTE[(i*2+1) % len(PALETTE)]
                series_all.append((name, p_data, color))
                series_pt.append((title_m, p_data, color))
        
        _plot_combined(f"I48 Engine Comparison: {ylabel}", ylabel, fname, series_all, log_y=log_y, ylim=ylim, save_dir=G1_CURVE)

    def parse_model_meta(label):
        """Extract I, K, C integer values from a model label string."""
        import re
        I_val = re.search(r'I(\d+)', label)
        K_val = re.search(r'K(\d+)', label)
        C_val = re.search(r'C(\d+)', label)
        return {
            "I": int(I_val.group(1)) if I_val else None,
            "K": int(K_val.group(1)) if K_val else None,
            "C": int(C_val.group(1)) if C_val else None,
        }

    # ── Predefined research comparison groups ─────────────────────────────────
    # Each entry: (group_dir_name, group_title, filter_fn)
    # filter_fn(meta) -> True if the model belongs in this comparison group.
    # Each group is then automatically split by C (filter depth) for the final plots.
    COMPARISON_GROUPS = [
        (
            "1_resolution_I48_vs_I64",
            "Input Resolution: I48 vs I64  (K=3,5 shared arch)",
            lambda m: m["K"] in {3, 5},           # include both I48 and I64
        ),
        (
            "2_odd_vs_even_kernel",
            "Odd vs Even Kernel  (I64, K=2,3,4,5)",
            lambda m: m["I"] == 64 and m["K"] in {2, 3, 4, 5},
        ),
        (
            "3_small_to_large_kernel",
            "Small→Large Kernel  (I64, K=3,5,7,9)",
            lambda m: m["I"] == 64 and m["K"] in {3, 5, 7, 9},
        ),
    ]

    def plot_3d_comparison_groups(all_series, save_dir, zlabel, title_sfx, base_fname):
        """
        For each predefined COMPARISON_GROUPS entry:
          - filter all_series to models matching the group
          - split the filtered list by C value → one plot per C
          - save to  save_dir/<group_dir_name>/plot_<fname>_C<n>.png
        """
        for dir_name, group_title, filter_fn in COMPARISON_GROUPS:
            group_sub = os.path.join(save_dir, dir_name)
            os.makedirs(group_sub, exist_ok=True)

            # Split by C value
            from collections import defaultdict
            by_c = defaultdict(list)
            for item in all_series:
                meta = parse_model_meta(item[0])
                if filter_fn(meta) and meta["C"] is not None:
                    by_c[meta["C"]].append(item)

            if not by_c:
                continue

            for c_val in sorted(by_c.keys()):
                slist = by_c[c_val]
                _plot_3d_waterfall_combined(
                    f"{group_title} | C={c_val}: {title_sfx}",
                    zlabel,
                    f"{base_fname}_C{c_val}.png",
                    slist,
                    save_dir=group_sub,
                )

    def plot_2d_comparison_groups(all_series, save_dir, ylabel, title_sfx, base_fname, log_y=False, ylim=None):
        """
        For each predefined COMPARISON_GROUPS entry:
          - filter all_series to models matching the group
          - split the filtered list by C value → one plot per C
          - save to  save_dir/<group_dir_name>/plot_<fname>_C<n>.png
        """
        for dir_name, group_title, filter_fn in COMPARISON_GROUPS:
            group_sub = os.path.join(save_dir, dir_name)
            os.makedirs(group_sub, exist_ok=True)

            # Split by C value
            from collections import defaultdict
            by_c = defaultdict(list)
            for item in all_series:
                meta = parse_model_meta(item[0])
                if filter_fn(meta) and meta["C"] is not None:
                    by_c[meta["C"]].append(item)

            if not by_c:
                continue

            for c_val in sorted(by_c.keys()):
                slist = by_c[c_val]
                _plot_combined(
                    f"{group_title} | C={c_val}: {title_sfx}",
                    ylabel,
                    f"{base_fname}_C{c_val}.png",
                    slist,
                    log_y=log_y,
                    ylim=ylim,
                    save_dir=group_sub,
                )


    # ── 3D Waterfall Plots: 4 variants (Train Loss, Train Acc, Eval Loss, Eval Acc) ──
    _3d_specs = [
        # (title_suffix, zlabel,          fname_suffix,     series_key,     metric_key, transform)
        ("Training Loss",     "BCER (%)",      "train_loss",     "train_series", "bcer", None),
        ("Training Accuracy", "Accuracy (%)",  "train_acc",      "train_series", "bcer", lambda v: 100 - v),
        ("Eval Loss",         "BCER (%)",      "eval_loss",      "eval_series",  "bcer", None),
        ("Eval Accuracy",     "Accuracy (%)",  "eval_acc",       "eval_series",  "bcer", lambda v: 100 - v),
    ]
    for title_sfx, zlabel, fname_sfx, series_key, metric_key, transform in _3d_specs:
        series_all  = []
        series_tess = []
        series_pt   = []
        for i, m in enumerate(MODELS_I48):
            t_data = _get(tess_logs, m, series_key, metric_key)
            p_data = _get(pt_logs,  m, series_key, metric_key)
            if transform:
                t_data = [(it, transform(v)) for it, v in t_data] if t_data else []
                p_data = [(it, transform(v)) for it, v in p_data] if p_data else []
            title_m = m if m.startswith("LaoCRNN-") else f"LaoCRNN-{m}"
            if t_data:
                series_all.append((f"{title_m} (Tess)", t_data, PALETTE[(i*2) % len(PALETTE)]))
                series_tess.append((title_m, t_data, PALETTE[(i*2) % len(PALETTE)]))
            if p_data:
                series_all.append((f"{title_m} (PT)", p_data, PALETTE[(i*2+1) % len(PALETTE)]))
                series_pt.append((title_m, p_data, PALETTE[(i*2+1) % len(PALETTE)]))

        # Build combined series for comparison groups (Tess + PT labels preserved)
        plot_3d_comparison_groups(
            all_series=series_all,
            save_dir=G1_3D,
            zlabel=zlabel,
            title_sfx=title_sfx,
            base_fname=f"plot_3d_{fname_sfx}",
        )
        plot_2d_comparison_groups(
            all_series=series_all,
            save_dir=G1_2D,
            ylabel=zlabel,
            title_sfx=title_sfx,
            base_fname=f"plot_2d_{fname_sfx}",
            log_y=(fname_sfx in ("train_loss", "eval_loss")),
            ylim=(0, 105) if "accuracy" in title_sfx.lower() else None,
        )

    # Individual engine summary grids
    _render_summary_grid(MODELS_I48, I48_TESS_DIR, "summary_metrics.png", "I48 Tesseract Only", engines="tess")
    _render_summary_grid(MODELS_I48, I48_PT_DIR,   "summary_metrics.png", "I48 PyTorch Only",   engines="pt")
    _render_summary_grid(MODELS_I48, G1_DIR,       "summary_metrics_combined.png", "I48 Tesseract vs PyTorch", engines="both")

    # Per-model subplots (Tess overlay) — all models flat
    _render_subplots(MODELS_I48, tess_logs, pt_logs, G1_CURVE, "I48 Tess vs PT", overlay_engines=True)
    # Per-model subplots split by kernel group
    _render_subplots_by_kernel(MODELS_I48, tess_logs, pt_logs, G1_CURVE, "I48 Tess vs PT", overlay_engines=True)

    # Paper-ready paired figures (2 charts per figure)
    _render_paper_pairs(MODELS_I48, tess_logs, pt_logs, G1_CURVE, "I48 Tesseract vs PyTorch", overlay_engines=True)

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 2 — I64: PyTorch models compared among themselves
    # ══════════════════════════════════════════════════════════════════════════
    print("    ── Group 2: I64 PyTorch (internal comparison) ──")
    os.makedirs(G2_CURVE, exist_ok=True)
    os.makedirs(G2_3D, exist_ok=True)

    for metric_key, ylabel, fname, log_y, ylim, transform in [
        ("bcer", "Training error (BCER %)", "plot_loss_combined.png", True, None, None),
        ("bcer", "Training accuracy ", "plot_acc_combined.png", False, (0, 105), lambda v: 100 - v),
        ("bwer", "Training WER", "plot_wer_combined.png", False, None, None),
    ]:
        series = []
        for i, m in enumerate(MODELS_I64):
            p_data = _get(pt_logs, m, "train_series", metric_key)
            if transform:
                p_data = [(it, transform(v)) for it, v in p_data] if p_data else []
            if p_data:
                title_m = m if m.startswith("LaoCRNN-") else f"LaoCRNN-{m}"
                series.append((title_m, p_data, PALETTE[i % len(PALETTE)]))
        
        _plot_combined(f"I64 PyTorch: {ylabel}", ylabel, fname, series, log_y=log_y, ylim=ylim, save_dir=G2_CURVE)

    for title_sfx, zlabel, fname_sfx, series_key, metric_key, transform in _3d_specs:
        series = []
        for i, m in enumerate(MODELS_I64):
            p_data = _get(pt_logs, m, series_key, metric_key)
            if transform:
                p_data = [(it, transform(v)) for it, v in p_data] if p_data else []
            if p_data:
                title_m = m if m.startswith("LaoCRNN-") else f"LaoCRNN-{m}"
                series.append((title_m, p_data, PALETTE[i % len(PALETTE)]))

        plot_3d_comparison_groups(
            all_series=series,
            save_dir=G2_3D,
            zlabel=zlabel,
            title_sfx=title_sfx,
            base_fname=f"plot_3d_{fname_sfx}",
        )
        plot_2d_comparison_groups(
            all_series=series,
            save_dir=G2_2D,
            ylabel=zlabel,
            title_sfx=title_sfx,
            base_fname=f"plot_2d_{fname_sfx}",
            log_y=(fname_sfx in ("train_loss", "eval_loss")),
            ylim=(0, 105) if "accuracy" in title_sfx.lower() else None,
        )
        plot_3d_comparison_groups(
            all_series=series,
            save_dir=I64_PT_DIR,
            zlabel=zlabel,
            title_sfx=title_sfx,
            base_fname=f"plot_3d_{fname_sfx}",
        )

    # Individual engine summary grid
    _render_summary_grid(MODELS_I64, I64_PT_DIR, "summary_metrics.png", "I64 PyTorch Only", engines="pt")

    # Per-model subplots (PT only) — all models flat
    _render_subplots(MODELS_I64, tess_logs, pt_logs, G2_CURVE, "I64 PyTorch", overlay_engines=False)
    # Per-model subplots split by kernel group
    _render_subplots_by_kernel(MODELS_I64, tess_logs, pt_logs, G2_CURVE, "I64 PyTorch", overlay_engines=False)

    # ── Render training_curves/kernel_group subplots ─────────────────────────
    kernel_group_dir = os.path.join(G2_CURVE, "kernel_group")
    os.makedirs(kernel_group_dir, exist_ok=True)
    
    # Sort helper to keep subplots logically ordered
    def sort_key(model_name):
        meta = parse_model_meta(model_name)
        return (meta["K"] or 0, meta["C"] or 0)

    # Group 1: k2,3,4,5 (Odd vs Even Kernel)
    k2345_models = []
    for m in MODELS_I64:
        m_k = re.search(r'K(\d+)', m)
        if m_k and int(m_k.group(1)) in {2, 3, 4, 5}:
            k2345_models.append(m)
            
    if k2345_models:
        print(f"      ── Kernel group k2,3,4,5: {k2345_models}")
        k2345_models.sort(key=sort_key)
        _render_subplots(k2345_models, tess_logs, pt_logs, kernel_group_dir,
                         "I64 PyTorch [K=2,3,4,5]",
                         overlay_engines=False,
                         fname_suffix="odd_vs_even_kernel")
        # Overlay charts folder
        k2345_subdir = os.path.join(kernel_group_dir, "odd_vs_even_kernel_depth(C16 and C32)")
        _render_kernel_group_overlay_charts(k2345_models, tess_logs, pt_logs, k2345_subdir,
                                            "I64 PyTorch [K=2,3,4,5]")
                         
    # Group 2: K3,5,7,9 (Small to Large Kernel)
    k3579_models = []
    for m in MODELS_I64:
        m_k = re.search(r'K(\d+)', m)
        if m_k and int(m_k.group(1)) in {3, 5, 7, 9}:
            k3579_models.append(m)
            
    if k3579_models:
        print(f"      ── Kernel group K3,5,7,9: {k3579_models}")
        k3579_models.sort(key=sort_key)
        _render_subplots(k3579_models, tess_logs, pt_logs, kernel_group_dir,
                         "I64 PyTorch [K=3,5,7,9]",
                         overlay_engines=False,
                         fname_suffix="small_to_large_kernel")
        # Overlay charts folder
        k3579_subdir = os.path.join(kernel_group_dir, "small_to_large_kernel_depth(C16 and C32)")
        _render_kernel_group_overlay_charts(k3579_models, tess_logs, pt_logs, k3579_subdir,
                                            "I64 PyTorch [K=3,5,7,9]")

    # Group 3: K5 Depth (c16, 32, 64)
    k5_depth_models = []
    for m in MODELS_I64:
        m_k = re.search(r'K(\d+)', m)
        if m_k and int(m_k.group(1)) == 5:
            k5_depth_models.append(m)
            
    if k5_depth_models:
        k5_depth_subdir = os.path.join(kernel_group_dir, "depth_k5(C16, C32 and C64)")
        os.makedirs(k5_depth_subdir, exist_ok=True)
        k5_depth_models.sort(key=sort_key)
        
        metrics_train = [
            ("Train Accuracy", "bcer", "train_series", "Accuracy (%)", lambda v: 100 - v, False),
            ("Train Loss CER", "bcer", "train_series", "BCER (%)", None, True),
            ("Train Loss WER", "bwer", "train_series", "BWER (%)", None, False),
        ]
        
        metrics_eval = [
            ("Eval Accuracy",  "bcer", "eval_series",  "Accuracy (%)", lambda v: 100 - v, False),
            ("Eval Loss CER",  "bcer", "eval_series",  "BCER (%)", None, False),
            ("Eval Loss WER",  "bwer", "eval_series",  "BWER (%)", None, False),
        ]
        
        def render_k5_3col_fig(metrics_list, prefix):
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f"I64 PyTorch [K=5] Depth Comparison ({prefix})", fontweight='bold', fontsize=14)
            
            for ax, (title, key, series_key, ylabel, transform, log_y) in zip(axes, metrics_list):
                for i, m in enumerate(k5_depth_models):
                    color = PALETTE[i % len(PALETTE)]
                    d = _get(pt_logs, m, series_key, key)
                    if transform and d:
                        d = [(it, transform(v)) for it, v in d]
                        
                    if d:
                        title_name = m if m.startswith("LaoCRNN-") else f"LaoCRNN-{m}"
                        x_vals = [pt[0] for pt in d]
                        y_vals = [pt[1] for pt in d]
                        ax.plot(x_vals, y_vals, label=title_name, color=color, linewidth=2)
                        # Add best point annotation if possible
                        try:
                            from tesstrain_core.plotting import add_best_annotation
                            add_best_annotation(ax, d, color, mode='max' if 'Accuracy' in title else 'min')
                        except ImportError:
                            pass
                
                ax.set_title(title, fontweight='bold', fontsize=12)
                ax.set_xlabel('Iteration')
                ax.set_ylabel(ylabel)
                if log_y:
                    ax.set_yscale('log')
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend(fontsize=9, loc='best')
            
            fig.tight_layout()
            fname = f"plot_{prefix.lower()}_metrics.png"
            fig.savefig(os.path.join(k5_depth_subdir, fname), dpi=DPI, bbox_inches='tight')
            plt.close(fig)
            print(f"      ✓ K5 Depth Overlay Grid: {os.path.relpath(os.path.join(k5_depth_subdir, fname), PLOT_DIR)}")

        render_k5_3col_fig(metrics_train, "Train")
        render_k5_3col_fig(metrics_eval, "Eval")

    # Paper-ready paired figures (2 charts per figure)
    _render_paper_pairs(MODELS_I64, tess_logs, pt_logs, G2_CURVE, "I64 PyTorch", overlay_engines=False)

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 3 — PyTorch I48 vs I64 (resolution comparison)
    # ══════════════════════════════════════════════════════════════════════════
    print("    ── Group 3: PyTorch I48 vs I64 ──")
    G3_CURVE = os.path.join(G3_DIR, "training_curves")
    os.makedirs(G3_CURVE, exist_ok=True)
    os.makedirs(G3_3D, exist_ok=True)

    # Pair models by architecture: K3C16@48 vs K3C16@64, etc.
    ARCH_PAIRS = [("K3C16", 0), ("K3C32", 1), ("K5C16", 2), ("K5C32", 3)]
    for metric_key, ylabel, fname, log_y, ylim, transform in [
        ("bcer", "Training error (BCER %)", "plot_loss_combined.png", True, None, None),
        ("bcer", "Training accuracy ", "plot_acc_combined.png", False, (0, 105), lambda v: 100 - v),
        ("bwer", "Training WER", "plot_wer_combined.png", False, None, None),
    ]:
        series = []
        for arch, ci in ARCH_PAIRS:
            m48 = f"LaoCRNN-I48{arch}"
            m64 = f"LaoCRNN-I64{arch}"
            d48 = _get(pt_logs, m48, "train_series", metric_key)
            d64 = _get(pt_logs, m64, "train_series", metric_key)
            if transform:
                d48 = [(it, transform(v)) for it, v in d48]
                d64 = [(it, transform(v)) for it, v in d64]
            if d48: series.append((f"{arch} @48px", d48, PALETTE[(ci*2) % len(PALETTE)]))
            if d64: series.append((f"{arch} @64px", d64, PALETTE[(ci*2+1) % len(PALETTE)]))
        _plot_combined(f"PyTorch Resolution: {ylabel}", ylabel, fname, series, log_y=log_y, ylim=ylim, save_dir=G3_CURVE)

        # Build a combined I48+I64 series list for the resolution comparison groups
        series_all_res = []
        for arch, ci in ARCH_PAIRS:
            m48 = f"LaoCRNN-I48{arch}"
            m64 = f"LaoCRNN-I64{arch}"
            d48 = _get(pt_logs, m48, series_key, metric_key)
            d64 = _get(pt_logs, m64, series_key, metric_key)
            if transform:
                d48 = [(it, transform(v)) for it, v in d48] if d48 else []
                d64 = [(it, transform(v)) for it, v in d64] if d64 else []
            if d48: series_all_res.append((m48, d48, PALETTE[(ci*2) % len(PALETTE)]))
            if d64: series_all_res.append((m64, d64, PALETTE[(ci*2+1) % len(PALETTE)]))
        plot_3d_comparison_groups(
            all_series=series_all_res,
            save_dir=G3_3D,
            zlabel=zlabel,
            title_sfx=title_sfx,
            base_fname=f"plot_3d_{fname_sfx}",
        )
        plot_2d_comparison_groups(
            all_series=series_all_res,
            save_dir=G3_2D,
            ylabel=zlabel,
            title_sfx=title_sfx,
            base_fname=f"plot_2d_{fname_sfx}",
            log_y=(fname_sfx in ("train_loss", "eval_loss")),
            ylim=(0, 105) if "accuracy" in title_sfx.lower() else None,
        )

    # Summary grid for Group 3 (All 8 models overlaid)
    _render_summary_grid(MODELS_I48 + MODELS_I64, G3_DIR, "summary_metrics_combined.png", "PT I48 vs I64 Resolution", engines="pt")

    # We also render per-model subplots but here group them by Arch instead.
    # To reuse _render_subplots easily, we can simulate an overlay by passing I48 as "tess" and I64 as "pt".
    # But wait, _render_subplots expects Model Names.
    # Let's create a custom subplots generator for Group 3:
    titles = [
        ("Train CER (I48 vs I64)",      "plot_train_loss_CER.png", "bcer", "train_series", "BCER (%)",     False),
        ("Train WER (I48 vs I64)",      "plot_train_loss_WER.png", "bwer", "train_series", "BWER (%)",     False),
        ("Eval CER (I48 vs I64)",       "plot_eval_loss_CER.png",  "bcer", "eval_series",  "BCER (%)",     True),
        ("Eval WER (I48 vs I64)",       "plot_eval_loss_WER.png",  "bwer", "eval_series",  "BWER (%)",     True),
        ("Training Accuracy (I48 vs I64)", "plot_train_acc.png",   "bcer", "train_series", "Accuracy (%)", False),
        ("Eval Accuracy (I48 vs I64)",     "plot_eval_acc.png",    "bcer", "eval_series",  "Accuracy (%)", True),
    ]
    for title, fname, key, series, ylabel, is_eval in titles:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes_flat = axes.flatten()
        apply_premium_theme(fig, axes_flat)
        fig.suptitle(title, fontweight='bold', fontsize=15, y=0.99)
        
        for idx, (arch, ci) in enumerate(ARCH_PAIRS):
            ax = axes_flat[idx]
            m48 = f"LaoCRNN-I48{arch}"
            m64 = f"LaoCRNN-I64{arch}"
            
            d48 = _get(pt_logs, m48, series, key)
            d64 = _get(pt_logs, m64, series, key)
            if key == "bcer" and "accuracy" in ylabel.lower():
                d48 = [(it, 100 - v) for it, v in d48]
                d64 = [(it, 100 - v) for it, v in d64]
                
            color48 = PALETTE[(ci*2) % len(PALETTE)]
            color64 = PALETTE[(ci*2+1) % len(PALETTE)]
            
            if d48:
                sm48 = smooth_ema(d48, weight=0.5 if is_eval else 0.6)
                ax.plot(*zip(*d48), color=color48, alpha=0.12, lw=0.6)
                ax.plot(*zip(*sm48), color=color48, lw=1.8, linestyle='--', label='48px Input')
                fill_between_smooth(ax, d48, sm48, color48, alpha=0.05)
                add_best_annotation(ax, d48, color48, mode='max' if 'accuracy' in ylabel.lower() else 'min')

            if d64:
                sm64 = smooth_ema(d64, weight=0.5 if is_eval else 0.6)
                ax.plot(*zip(*d64), color=color64, alpha=0.15, lw=0.6)
                ax.plot(*zip(*sm64), color=color64, lw=2.2, label='64px Input')
                fill_between_smooth(ax, d64, sm64, color64, alpha=0.06)
                add_best_annotation(ax, d64, color64, mode='max' if 'accuracy' in ylabel.lower() else 'min')

            if not is_eval:
                e48 = _get(pt_logs, m48, "eval_series", key)
                e64 = _get(pt_logs, m64, "eval_series", key)
                if key == "bcer" and "accuracy" in ylabel.lower():
                    e48 = [(it, 100 - v) for it, v in e48] if e48 else []
                    e64 = [(it, 100 - v) for it, v in e64] if e64 else []
                
                if e48:
                    ax.plot(*zip(*e48), color='#333333', marker='*', markersize=5,
                            alpha=0.6, linestyle='None', label='48px Eval', zorder=8)
                if e64:
                    ax.plot(*zip(*e64), color='#C62828', marker='o', markersize=4,
                            alpha=0.55, linestyle='None', label='64px Eval', zorder=8)

            ax.set_title(f'Architecture: {arch}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Iteration'); ax.set_ylabel(ylabel)
            ax.legend(fontsize=8, loc='best')
            if ('bcer' in ylabel.lower() or 'bwer' in ylabel.lower()) and (len(d48) > 0 or len(d64) > 0):
                ax.set_yscale('log')

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        tb_save(fig, os.path.join(G3_CURVE, fname), dpi=DPI)
        print(f"    ✓ {os.path.relpath(os.path.join(G3_CURVE, fname), PLOT_DIR)}")

    # Paper-ready paired figures for Group 3 (all I48+I64 models overlaid)
    ALL_PT_MODELS = MODELS_I48 + MODELS_I64
    _render_paper_pairs(ALL_PT_MODELS, tess_logs, pt_logs, G3_CURVE, "PyTorch I48 vs I64", overlay_engines=False)

def render_comparison_plots(tess_infer, pt_infer):
    """Render comparison bar charts split into 3 groups — TensorBoard style."""
    from tesstrain_core.plotting import apply_premium_theme, tb_save

    def _bar_chart_tess_vs_pt(models, out_dir, tag):
        """Group 1 style: Tesseract vs PyTorch side-by-side bars."""
        os.makedirs(out_dir, exist_ok=True)
        valid = [m for m in models if m in tess_infer or m in pt_infer]
        if not valid: return
        x = np.arange(len(valid)); width = 0.35
        for metric, label, title in [
            ("cer", "CER (%)", f"CER — {tag}"),
            ("wer", "WER (%)", f"WER — {tag}"),
            ("acc", "Accuracy (%)", f"Accuracy — {tag}"),
        ]:
            fig, ax = plt.subplots(figsize=(11, 5.5))
            apply_premium_theme(fig, ax)
            tv = [tess_infer.get(m, {}).get(metric, 0) * 100 if tess_infer.get(m) else 0 for m in valid]
            pv = [pt_infer.get(m, {}).get(metric, 0) * 100   if pt_infer.get(m)   else 0 for m in valid]
            b1 = ax.bar(x - width/2, tv, width, label="Tesseract", color=PALETTE[1], edgecolor='white', linewidth=0.5)
            b2 = ax.bar(x + width/2, pv, width, label="PyTorch",   color=PALETTE[0], edgecolor='white', linewidth=0.5)
            for bar in list(b1) + list(b2):
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, f"{h:.1f}",
                            ha="center", va="bottom", fontsize=8, fontweight="bold", color='#333333')
            _setup_ax(ax, title, "Model", label)
            labels = [m if m.startswith("LaoCRNN-") else f"LaoCRNN-{m}" for m in valid]
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
            ax.legend(fontsize=9); fig.tight_layout()
            fname = f"plot_compare_{metric}.png"
            tb_save(fig, os.path.join(out_dir, fname), dpi=DPI)
            print(f"    ✓ {os.path.relpath(os.path.join(out_dir, fname), PLOT_DIR)}")

        # Speed
        if any(tess_infer.get(m, {}).get("eval_time") for m in valid) or \
           any(pt_infer.get(m, {}).get("eval_time") for m in valid):
            fig, ax = plt.subplots(figsize=(11, 5.5))
            apply_premium_theme(fig, ax)
            tt = [tess_infer.get(m, {}).get("eval_time", 0) for m in valid]
            pt = [pt_infer.get(m, {}).get("eval_time", 0)   for m in valid]
            ax.bar(x - width/2, tt, width, label="Tesseract", color=PALETTE[1], edgecolor='white', linewidth=0.5)
            ax.bar(x + width/2, pt, width, label="PyTorch",   color=PALETTE[0], edgecolor='white', linewidth=0.5)
            _setup_ax(ax, f"Inference Speed — {tag}", "Model", "Time (s)")
            labels = [m if m.startswith("LaoCRNN-") else f"LaoCRNN-{m}" for m in valid]
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
            ax.legend(fontsize=9); fig.tight_layout()
            tb_save(fig, os.path.join(out_dir, "plot_compare_speed.png"), dpi=DPI)
            print(f"    ✓ {os.path.relpath(os.path.join(out_dir, 'plot_compare_speed.png'), PLOT_DIR)}")

    def _bar_chart_pt_only(models, out_dir, tag):
        """Group 2/3 style: PyTorch models compared among themselves."""
        os.makedirs(out_dir, exist_ok=True)
        valid = [m for m in models if m in pt_infer]
        if not valid: return
        x = np.arange(len(valid)); colors = [PALETTE[i % len(PALETTE)] for i in range(len(valid))]
        for metric, label, title in [
            ("cer", "CER (%)", f"CER — {tag}"),
            ("wer", "WER (%)", f"WER — {tag}"),
            ("acc", "Accuracy (%)", f"Accuracy — {tag}"),
        ]:
            fig, ax = plt.subplots(figsize=(11, 5.5))
            apply_premium_theme(fig, ax)
            vals = [pt_infer.get(m, {}).get(metric, 0) * 100 if pt_infer.get(m) else 0 for m in valid]
            bars = ax.bar(x, vals, 0.6, color=colors, edgecolor='white', linewidth=0.5)
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, f"{h:.1f}",
                            ha="center", va="bottom", fontsize=8, fontweight="bold", color='#333333')
            _setup_ax(ax, title, "Model", label)
            labels = [m if m.startswith("LaoCRNN-") else f"LaoCRNN-{m}" for m in valid]
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
            fig.tight_layout()
            fname = f"plot_compare_{metric}.png"
            tb_save(fig, os.path.join(out_dir, fname), dpi=DPI)
            print(f"    ✓ {os.path.relpath(os.path.join(out_dir, fname), PLOT_DIR)}")

    def _bar_chart_resolution(out_dir):
        """Group 3: I48 vs I64 side-by-side for same architecture."""
        os.makedirs(out_dir, exist_ok=True)
        archs = ["K3C16", "K3C32", "K5C16", "K5C32"]
        valid_archs = [a for a in archs
                       if pt_infer.get(f"LaoCRNN-I48{a}") or pt_infer.get(f"LaoCRNN-I64{a}")]
        if not valid_archs: return
        x = np.arange(len(valid_archs)); width = 0.35
        for metric, label, title in [
            ("cer", "CER (%)", "CER — PT I48 vs I64"),
            ("wer", "WER (%)", "WER — PT I48 vs I64"),
            ("acc", "Accuracy (%)", "Accuracy — PT I48 vs I64"),
        ]:
            fig, ax = plt.subplots(figsize=(11, 5.5))
            apply_premium_theme(fig, ax)
            v48 = [pt_infer.get(f"LaoCRNN-I48{a}", {}).get(metric, 0) * 100 for a in valid_archs]
            v64 = [pt_infer.get(f"LaoCRNN-I64{a}", {}).get(metric, 0) * 100 for a in valid_archs]
            b1 = ax.bar(x - width/2, v48, width, label="I48 (48px)", color=PALETTE[2], edgecolor='white', linewidth=0.5)
            b2 = ax.bar(x + width/2, v64, width, label="I64 (64px)", color=PALETTE[4], edgecolor='white', linewidth=0.5)
            for bar in list(b1) + list(b2):
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, f"{h:.1f}",
                            ha="center", va="bottom", fontsize=8, fontweight="bold", color='#333333')
            _setup_ax(ax, title, "Architecture", label)
            ax.set_xticks(x); ax.set_xticklabels(valid_archs, fontsize=10)
            ax.legend(fontsize=9); fig.tight_layout()
            fname = f"plot_compare_{metric}.png"
            tb_save(fig, os.path.join(out_dir, fname), dpi=DPI)

            print(f"    ✓ {os.path.relpath(os.path.join(out_dir, fname), PLOT_DIR)}")

    # ── Execute all 3 groups ──────────────────────────────────────────────────
    print("    ── Group 1: I48 Tess vs PT ──")
    _bar_chart_tess_vs_pt(MODELS_I48, G1_COMPARE, "I48 Tess vs PT")

    print("    ── Group 2: I64 PyTorch ──")
    _bar_chart_pt_only(MODELS_I64, G2_COMPARE, "I64 PyTorch")

    print("    ── Group 3: PT I48 vs I64 ──")
    _bar_chart_resolution(G3_COMPARE)

    # Copy engine-specific plots into detailed subdirectories
    os.makedirs(I48_TESS_DIR, exist_ok=True)
    os.makedirs(I48_PT_DIR, exist_ok=True)
    os.makedirs(I64_PT_DIR, exist_ok=True)
    os.makedirs(PT_GLOBAL_DIR, exist_ok=True)

    if os.path.exists(TESS_COMPARE):
        for f in glob.glob(os.path.join(TESS_COMPARE, "*.png")):
            try: shutil.copy2(f, I48_TESS_DIR)
            except OSError as e: print(f"  ⚠ Could not copy {f}: {e}", file=sys.stderr)
            
    if os.path.exists(PT_COMPARE):
        for f in glob.glob(os.path.join(PT_COMPARE, "*.png")):
            fname = os.path.basename(f)
            dest = PT_GLOBAL_DIR
            if "-I48" in fname: dest = I48_PT_DIR
            elif "-I64" in fname: dest = I64_PT_DIR
            try: shutil.copy2(f, dest)
            except OSError as e: print(f"  ⚠ Could not copy {f}: {e}", file=sys.stderr)

    # Move any dashboards dumped into PLOT_DIR directly by external scripts
    for f in glob.glob(os.path.join(PLOT_DIR, "plot_dashboard_*.png")):
        fname = os.path.basename(f)
        try:
            if "I48" in fname:
                # Distinguish Tesseract vs PT based on how it was generated...
                # Actually, tesseract_tools and torch_train.py might overwrite each other if they use the same name.
                # Is there a suffix? Both Tesseract and PT might name their dashboards plot_dashboard_LaoCRNN-I48K3C16.png.
                # If they do, they overwrite each other in PLOT_DIR!
                # Wait, tesseract_tools saves to 'plots'. torch_train might save elsewhere.
                # Since we don't have torch_train output logged right now, let's just move them to I48_tesseract for now if generated by run.sh all, 
                # but let's copy to both if we can't tell, or assume tesseract tools dumped it if PT_COMPARE is missing...
                # Let's cleanly copy them to the tess directory, since PyTorch usually has PT_COMPARE. 
                shutil.move(f, I48_TESS_DIR)  # Since tesseract_tools outputs here currently
            elif "I64" in fname:
                shutil.move(f, I64_PT_DIR)
        except (OSError, shutil.Error) as e:
            print(f"  ⚠ Could not move {f}: {e}", file=sys.stderr)
            
    # Move training time plots too
    for f in glob.glob(os.path.join(PLOT_DIR, "plot_training_time_*.png")):
        fname = os.path.basename(f)
        try:
            if "I48" in fname: shutil.move(f, I48_TESS_DIR)
            elif "I64" in fname: shutil.move(f, I64_PT_DIR)
        except (OSError, shutil.Error) as e: print(f"  ⚠ Could not move {f}: {e}", file=sys.stderr)

    print("    ✓ Copied engine analytics to I48_PT, I64_PT, and I48_tesseract")

# ══════════════════════════════════════════════════════════════════════════════
# 5. TENSORBOARD EXPORT
# ══════════════════════════════════════════════════════════════════════════════
def export_tensorboard(tess_logs, pt_logs, tess_infer, pt_infer):
    if not HAS_TB:
        print("    [Skip] TensorBoard not installed. Run: pip install tensorboard")
        return
    os.makedirs(TB_DIR, exist_ok=True)

    for model in MODELS:
        # Tesseract training scalars
        tl = tess_logs.get(model, {})
        if tl.get("train_series"):
            writer = SummaryWriter(os.path.join(TB_DIR, f"{model}_Tess_Train"))
            for d in tl["train_series"]:
                writer.add_scalar("Tess/Train_BCER",    d["bcer"],     d["iteration"])
                writer.add_scalar("Tess/Train_BWER",    d["bwer"],     d["iteration"])
                writer.add_scalar("Tess/Train_Loss",    d["mean_rms"], d["iteration"])
                writer.add_scalar("Tess/Train_Acc",     100 - d["bcer"], d["iteration"])
            for d in tl.get("eval_series", []):
                writer.add_scalar("Tess/Eval_BCER", d["bcer"], d["iteration"])
                writer.add_scalar("Tess/Eval_BWER", d["bwer"], d["iteration"])
                writer.add_scalar("Tess/Eval_Acc",  100 - d["bcer"], d["iteration"])
            writer.close()

        # PyTorch training scalars
        pl = pt_logs.get(model, {})
        if pl.get("train_series"):
            writer = SummaryWriter(os.path.join(TB_DIR, f"{model}_PT_Train"))
            for d in pl["train_series"]:
                writer.add_scalar("PT/Train_CER",   d["bcer"],     d["iteration"])
                writer.add_scalar("PT/Train_WER",   d["bwer"],     d["iteration"])
                writer.add_scalar("PT/Train_Loss",  d["mean_rms"], d["iteration"])
                writer.add_scalar("PT/Train_Acc",   100 - d["bcer"], d["iteration"])
            for d in pl.get("eval_series", []):
                writer.add_scalar("PT/Eval_CER",  d["bcer"], d["iteration"])
                writer.add_scalar("PT/Eval_WER",  d["bwer"], d["iteration"])
                writer.add_scalar("PT/Eval_Acc",  100 - d["bcer"], d["iteration"])
            writer.close()

    # Benchmark inference metrics (one scalar per model, step=1)
    for model in MODELS:
        writer = SummaryWriter(os.path.join(TB_DIR, f"{model}_Benchmark"))
        tr = tess_infer.get(model)
        pr = pt_infer.get(model)
        if tr:
            writer.add_scalar("Benchmark/Tess_CER", tr["cer"] * 100, 1)
            writer.add_scalar("Benchmark/Tess_WER", tr["wer"] * 100, 1)
            writer.add_scalar("Benchmark/Tess_Acc", tr["acc"] * 100, 1)
        if pr:
            writer.add_scalar("Benchmark/PT_CER",  pr["cer"] * 100, 1)
            writer.add_scalar("Benchmark/PT_WER",  pr["wer"] * 100, 1)
            writer.add_scalar("Benchmark/PT_Acc",  pr["acc"] * 100, 1)
        writer.close()

    print(f"    Exported. Launch: tensorboard --logdir={TB_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.5. CSV EXPORT PER MODEL
# ══════════════════════════════════════════════════════════════════════════════
def export_model_csvs(tess_logs, pt_logs):
    csv_dir = os.path.join(OUT_DIR, "csv_exports")
    os.makedirs(csv_dir, exist_ok=True)
    
    for model in MODELS:
        # PyTorch
        pl = pt_logs.get(model, {})
        if pl.get("train_series"):
            pt_train_path = os.path.join(csv_dir, f"{model}_PT_train.csv")
            with open(pt_train_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Iteration", "Loss", "CER(%)", "WER(%)", "Accuracy(%)"])
                for d in pl["train_series"]:
                    w.writerow([d.get("iteration", ""), d.get("mean_rms", ""), d.get("bcer", ""), d.get("bwer", ""), 100 - d.get("bcer", 100) if d.get("bcer") is not None else ""])
                    
        if pl.get("eval_series"):
            pt_eval_path = os.path.join(csv_dir, f"{model}_PT_eval.csv")
            with open(pt_eval_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Iteration", "Loss", "CER(%)", "WER(%)", "Accuracy(%)"])
                for d in pl["eval_series"]:
                    w.writerow([d.get("iteration", ""), d.get("eval_loss", ""), d.get("bcer", ""), d.get("bwer", ""), 100 - d.get("bcer", 100) if d.get("bcer") is not None else ""])
                    
        # Tesseract
        tl = tess_logs.get(model, {})
        if tl.get("train_series"):
            tess_train_path = os.path.join(csv_dir, f"{model}_Tess_train.csv")
            with open(tess_train_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Iteration", "MeanRMS", "BCER(%)", "BWER(%)", "Accuracy(%)"])
                for d in tl["train_series"]:
                    w.writerow([d.get("iteration", ""), d.get("mean_rms", ""), d.get("bcer", ""), d.get("bwer", ""), 100 - d.get("bcer", 100) if d.get("bcer") is not None else ""])
                    
        if tl.get("eval_series"):
            tess_eval_path = os.path.join(csv_dir, f"{model}_Tess_eval.csv")
            with open(tess_eval_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Iteration", "BCER(%)", "BWER(%)", "Accuracy(%)"])
                for d in tl["eval_series"]:
                    w.writerow([d.get("iteration", ""), d.get("bcer", ""), d.get("bwer", ""), 100 - d.get("bcer", 100) if d.get("bcer") is not None else ""])
                    
        if tl.get("check_series"):
            tess_check_path = os.path.join(csv_dir, f"{model}_Tess_check.csv")
            with open(tess_check_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Iteration", "BCER(%)", "Accuracy(%)"])
                for d in tl["check_series"]:
                    w.writerow([d.get("iteration", ""), d.get("bcer", ""), 100 - d.get("bcer", 100) if d.get("bcer") is not None else ""])

    print(f"    ✓ Exported per-model CSVs to {os.path.relpath(csv_dir, os.getcwd())}/")


# ══════════════════════════════════════════════════════════════════════════════
# 6. REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════
def write_reports(tess_logs, pt_logs, tess_infer, pt_infer):
    os.makedirs(OUT_DIR, exist_ok=True)

    # CSV
    csv_path = os.path.join(REPORT_DIR, "benchmark_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        coer_headers = [f"CoER {label}" for _, label in COER_COLS]
        w.writerow(["Model", "Engine", "CER(%)", "WER(%)", "Acc(%)",
                    "RW_CER(%)", "RW_WER(%)", "RW_Acc(%)",
                    "Total Iterations", "TrainTime(h)", "EvalTime(m)", "TestTime(m)", "RW_TestTime(m)",
                    "BestEvalBCER(%)", "BestEvalIter"] + coer_headers)
        for model in MODELS:
            for engine, infer, logs in [
                ("Tesseract", tess_infer, tess_logs),
                ("PyTorch",   pt_infer,   pt_logs),
            ]:
                r = infer.get(model)
                lg = logs.get(model, {})
                if r is None:
                    w.writerow([model, engine] + ["--"] * (13 + len(COER_COLS)))
                    continue
                coer_vals = [f"{r['coer'].get(cat, 0)*100:.2f}" for cat, _ in COER_COLS]
                r_tt = r.get("train_time", 0)
                final_tt = r_tt if r_tt > 0 else lg.get("train_time_h", 0)
                
                # Validation time: favor tesseract_tools computed value if available
                val_s = r.get("val_time_s") if r and r.get("val_time_s") is not None else lg.get("val_time_s")
                val_m_str = f"{(val_s / 60.0):.2f}" if val_s is not None else "--"
                
                # Test time: the inference loop speed
                test_s = r.get("eval_time", 0) if r else 0
                rw_test_s = r.get("rw_eval_time", 0) if r else 0
                
                w.writerow([
                    model, engine,
                    f"{r['cer']*100:.2f}" if r else "--",
                    f"{r['wer']*100:.2f}" if r else "--",
                    f"{r['acc']*100:.2f}" if r else "--",
                    f"{r['rw_cer']*100:.2f}" if r and 'rw_cer' in r else "--",
                    f"{r['rw_wer']*100:.2f}" if r and 'rw_wer' in r else "--",
                    f"{r['rw_acc']*100:.2f}" if r and 'rw_acc' in r else "--",
                    f"{lg.get('max_iter', '--')}",
                    f"{final_tt:.4f}",
                    val_m_str,
                    f"{(test_s / 60.0):.2f}",
                    f"{(rw_test_s / 60.0):.2f}",
                    f"{lg.get('best_bcer', '--')}",
                    f"{lg.get('best_iter', '--')}",
                ] + coer_vals)
    print("    ✓ reports/benchmark_report.csv")

    # Markdown
    md_path = os.path.join(REPORT_DIR, "benchmark_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Benchmark Report: Tesseract vs PyTorch\n\n")
        f.write("## Global Performance\n\n")
        f.write("| Model | Engine | CER (%) | WER (%) | Acc (%) | RW CER (%) | RW WER (%) | RW Acc (%) | Total Iterations | Train Time (h) | Val Time (m) | Test Time (m) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for model in MODELS:
            for engine, infer, logs in [
                ("Tesseract", tess_infer, tess_logs),
                ("PyTorch",   pt_infer,   pt_logs),
            ]:
                r = infer.get(model)
                lg = logs.get(model, {})
                # Validation time
                val_s = r.get("val_time_s") if r and r.get("val_time_s") is not None else lg.get("val_time_s")
                val_m_str = f"{(val_s / 60.0):.2f}" if val_s is not None else "--"
                
                # Test time
                test_m = (r.get('eval_time', 0) / 60.0) if r else 0
                
                r_tt = r.get("train_time", 0) if r else 0
                final_tt = r_tt if r_tt > 0 else lg.get("train_time_h", 0)
                
                max_it = lg.get("max_iter", "--")
                
                if r:
                    rw_c = f"{r.get('rw_cer', 0)*100:.2f}" if 'rw_cer' in r else "--"
                    rw_w = f"{r.get('rw_wer', 0)*100:.2f}" if 'rw_wer' in r else "--"
                    rw_a = f"{r.get('rw_acc', 0)*100:.2f}" if 'rw_acc' in r else "--"
                    f.write(f"| {model} | {engine} | {r['cer']*100:.2f} | "
                            f"{r['wer']*100:.2f} | {r['acc']*100:.2f} | {rw_c} | {rw_w} | {rw_a} | {max_it} | "
                            f"{final_tt:.4f} | {val_m_str} | {test_m:.2f} |\n")
                else:
                    f.write(f"| {model} | {engine} | -- | -- | -- | -- | -- | -- | {max_it} | {final_tt:.4f} | {val_m_str} | -- |\n")

        f.write("\n## Training Log Summary\n\n")
        f.write("| Model | Best Eval BCER (Tess %) | Best Eval Iter (Tess) | Best Eval CER (PT %) | Best Eval Iter (PT) |\n")
        f.write("|---|---|---|---|---|\n")
        for model in MODELS:
            tl = tess_logs.get(model, {})
            pl = pt_logs.get(model, {})
            f.write(f"| {model} | {tl.get('best_bcer','--')} | {tl.get('best_iter','--')} "
                    f"| {pl.get('best_bcer','--')} | {pl.get('best_iter','--')} |\n")

        f.write("\n## Component Error Rates (CoER)\n\n")
        coer_header = " | ".join(label for _, label in COER_COLS)
        f.write(f"| Model | Engine | {coer_header} |\n")
        f.write("|---|---|" + "|".join(["---"] * len(COER_COLS)) + "|\n")
        for model in MODELS:
            for engine, infer in [("Tesseract", tess_infer), ("PyTorch", pt_infer)]:
                r = infer.get(model)
                if r:
                    vals = " | ".join(f"{r['coer'].get(cat, 0)*100:.2f}%" for cat, _ in COER_COLS)
                    f.write(f"| {model} | {engine} | {vals} |\n")
                else:
                    f.write(f"| {model} | {engine} | " + " | ".join(["--"] * len(COER_COLS)) + " |\n")

        # ─── Visual Analysis Section ──────────────────────────────────────────
        f.write("\n## 📈 Visual Performance Analysis\n\n")

        # ── Group 1: I48 Tess vs PT ──────────────────────────────────────
        f.write("### 🔷 Group 1: I48 — Tesseract vs PyTorch\n\n")
        f.write("Cross-engine comparison for the 48px input height models.\n\n")
        f.write("| CER | Accuracy | Speed |\n")
        f.write("|---|---|---|\n")
        f.write("| ![CER](../../plots/I48_tess_vs_pt/comparisons/plot_compare_cer.png) ")
        f.write("| ![Acc](../../plots/I48_tess_vs_pt/comparisons/plot_compare_acc.png) ")
        f.write("| ![Speed](../../plots/I48_tess_vs_pt/comparisons/plot_compare_speed.png) |\n\n")
        f.write("#### 3D Convergence (I48 Tess vs PT)\n\n")
        f.write("| Training Loss | Training Accuracy | Eval Loss | Eval Accuracy |\n")
        f.write("|---|---|---|---|\n")
        f.write("| ![Loss](../../plots/I48_tess_vs_pt/cross_engine_3d/plot_3d_train_loss.png) ")
        f.write("| ![Acc](../../plots/I48_tess_vs_pt/cross_engine_3d/plot_3d_train_acc.png) ")
        f.write("| ![ELoss](../../plots/I48_tess_vs_pt/cross_engine_3d/plot_3d_eval_loss.png) ")
        f.write("| ![EAcc](../../plots/I48_tess_vs_pt/cross_engine_3d/plot_3d_eval_acc.png) |\n\n")

        f.write("#### Per-Model Training Curves (I48)\n\n")
        f.write("| Train Loss | Eval Error | Train Acc | Eval Acc |\n")
        f.write("|---|---|---|---|\n")
        g1 = "../../plots/I48_tess_vs_pt/training_curves"
        f.write(f"| ![Loss]({g1}/plot_train_loss.png) | ![Eval]({g1}/plot_eval_loss.png) ")
        f.write(f"| ![TAcc]({g1}/plot_train_acc.png) | ![EAcc]({g1}/plot_eval_acc.png) |\n\n")

        # ── Group 2: I64 PyTorch ─────────────────────────────────────────
        f.write("---\n\n")
        f.write("### 🟣 Group 2: I64 — PyTorch Models (Internal Comparison)\n\n")
        f.write("Comparison among the 64px input height PyTorch models (no Tesseract).\n\n")
        f.write("| CER | WER | Accuracy |\n")
        f.write("|---|---|---|\n")
        f.write("| ![CER](../../plots/I64_pytorch/comparisons/plot_compare_cer.png) ")
        f.write("| ![WER](../../plots/I64_pytorch/comparisons/plot_compare_wer.png) ")
        f.write("| ![Acc](../../plots/I64_pytorch/comparisons/plot_compare_acc.png) |\n\n")
        f.write("#### 3D Convergence (I64 PyTorch)\n\n")
        f.write("| Training Loss | Training Accuracy | Eval Loss | Eval Accuracy |\n")
        f.write("|---|---|---|---|\n")
        f.write("| ![Loss](../../plots/I64_pytorch/cross_model_3d/plot_3d_train_loss.png) ")
        f.write("| ![Acc](../../plots/I64_pytorch/cross_model_3d/plot_3d_train_acc.png) ")
        f.write("| ![ELoss](../../plots/I64_pytorch/cross_model_3d/plot_3d_eval_loss.png) ")
        f.write("| ![EAcc](../../plots/I64_pytorch/cross_model_3d/plot_3d_eval_acc.png) |\n\n")

        f.write("#### Per-Model Training Curves (I64)\n\n")
        f.write("| Train Loss | Eval Error | Train Acc | Eval Acc |\n")
        f.write("|---|---|---|---|\n")
        g2 = "../../plots/I64_pytorch/training_curves"
        f.write(f"| ![Loss]({g2}/plot_train_loss.png) | ![Eval]({g2}/plot_eval_loss.png) ")
        f.write(f"| ![TAcc]({g2}/plot_train_acc.png) | ![EAcc]({g2}/plot_eval_acc.png) |\n\n")

        # ── Group 3: PT I48 vs I64 ───────────────────────────────────────
        f.write("---\n\n")
        f.write("### 🟢 Group 3: PyTorch I48 vs I64 (Resolution Comparison)\n\n")
        f.write("Same architecture, different input resolution (48px vs 64px).\n\n")
        f.write("| CER | WER | Accuracy |\n")
        f.write("|---|---|---|\n")
        f.write("| ![CER](../../plots/I48_vs_I64_pytorch/comparisons/plot_compare_cer.png) ")
        f.write("| ![WER](../../plots/I48_vs_I64_pytorch/comparisons/plot_compare_wer.png) ")
        f.write("| ![Acc](../../plots/I48_vs_I64_pytorch/comparisons/plot_compare_acc.png) |\n\n")
        f.write("#### 3D Convergence (I48 vs I64)\n\n")
        f.write("| Training Loss | Training Accuracy | Eval Loss | Eval Accuracy |\n")
        f.write("|---|---|---|---|\n")
        f.write("| ![Loss](../../plots/I48_vs_I64_pytorch/cross_resolution_3d/plot_3d_train_loss.png) ")
        f.write("| ![Acc](../../plots/I48_vs_I64_pytorch/cross_resolution_3d/plot_3d_train_acc.png) ")
        f.write("| ![ELoss](../../plots/I48_vs_I64_pytorch/cross_resolution_3d/plot_3d_eval_loss.png) ")
        f.write("| ![EAcc](../../plots/I48_vs_I64_pytorch/cross_resolution_3d/plot_3d_eval_acc.png) |\n\n")

        f.write("#### Per-Model Training Curves (I48 vs I64)\n\n")
        f.write("| Train Loss | Eval Error | Train Acc | Eval Acc |\n")
        f.write("|---|---|---|---|\n")
        g3 = "../../plots/I48_vs_I64_pytorch/training_curves"
        f.write(f"| ![Loss]({g3}/plot_train_loss.png) | ![Eval]({g3}/plot_eval_loss.png) ")
        f.write(f"| ![TAcc]({g3}/plot_train_acc.png) | ![EAcc]({g3}/plot_eval_acc.png) |\n\n")

        # ── Engine-Specific 3D Analysis ──────────────────────────────────
        f.write("### 🧊 Engine-Specific Convergence (3D Waterfall)\n\n")
        f.write("Visualizing internal convergence per engine/resolution group.\n\n")

        f.write("#### I48 Tesseract 3D plots\n")
        f.write("| Training Loss | Training Accuracy | Eval Loss | Eval Accuracy |\n")
        f.write("|---|---|---|---|\n")
        f.write("| ![Loss](../../plots/I48_tesseract/plot_3d_train_loss.png) ")
        f.write("| ![Acc](../../plots/I48_tesseract/plot_3d_train_acc.png) ")
        f.write("| ![ELoss](../../plots/I48_tesseract/plot_3d_eval_loss.png) ")
        f.write("| ![EAcc](../../plots/I48_tesseract/plot_3d_eval_acc.png) |\n\n")

        f.write("#### I48 PyTorch 3D plots\n")
        f.write("| Training Loss | Training Accuracy | Eval Loss | Eval Accuracy |\n")
        f.write("|---|---|---|---|\n")
        f.write("| ![Loss](../../plots/I48_PT/plot_3d_train_loss.png) ")
        f.write("| ![Acc](../../plots/I48_PT/plot_3d_train_acc.png) ")
        f.write("| ![ELoss](../../plots/I48_PT/plot_3d_eval_loss.png) ")
        f.write("| ![EAcc](../../plots/I48_PT/plot_3d_eval_acc.png) |\n\n")

        f.write("#### I64 PyTorch 3D plots\n")
        f.write("| Training Loss | Training Accuracy | Eval Loss | Eval Accuracy |\n")
        f.write("|---|---|---|---|\n")
        f.write("| ![Loss](../../plots/I64_PT/plot_3d_train_loss.png) ")
        f.write("| ![Acc](../../plots/I64_PT/plot_3d_train_acc.png) ")
        f.write("| ![ELoss](../../plots/I64_PT/plot_3d_eval_loss.png) ")
        f.write("| ![EAcc](../../plots/I64_PT/plot_3d_eval_acc.png) |\n\n")

        # ── Metric Evolution Grids ──────────────────────────────────────
        f.write("---\n\n")
        f.write("### 🔳 Metric Evolution Grids (2x2 Summary)\n\n")
        f.write("A consolidated view of Train/Val Loss and Train/Val Accuracy for each major research group.\n\n")

        f.write("#### I48 Engines: Single vs. Overlaid\n")
        f.write("| Tesseract (48px) | PyTorch (48px) | Cross-Engine Comparison |\n")
        f.write("|---|---|---|\n")
        f.write("| ![Tess](../../plots/I48_tesseract/summary_metrics.png) ")
        f.write("| ![PT](../../plots/I48_PT/summary_metrics.png) ")
        f.write("| ![Both](../../plots/I48_tess_vs_pt/summary_metrics_combined.png) |\n\n")

        f.write("#### Resolution Comparison & I64 PyTorch\n")
        f.write("| I48 vs I64 Resolution Comparison | I64 PyTorch Only |\n")
        f.write("|---|---|\n")
        f.write("| ![Res](../../plots/I48_vs_I64_pytorch/summary_metrics_combined.png) ")
        f.write("| ![I64](../../plots/I64_PT/summary_metrics.png) |\n\n")

        # ── Model Dashboards & Error Analytics ───────────────────────────
        f.write("---\n\n")
        f.write("### 🔍 Model Dashboards & Deep Analytics\n\n")
        f.write("Detailed training convergence and recognition profiles for every model.\n\n")
        f.write("| Model | Tesseract (CPU/LSTM) | PyTorch (GPU/CRNN) |\n")
        f.write("|---|---|---|\n")
        for model in MODELS:
            t_dash = f"../../plots/I48_tesseract/plot_dashboard_{model}.png"
            if "I48" in model:
                p_dash = f"../../plots/I48_PT/plot_dashboard_{model}.png"
            else:
                p_dash = f"../../plots/I64_PT/plot_dashboard_{model}.png"
                
            if "I64" in model:
                f.write(f"| **{model}** | N/A | [🚀 Dashboard]({p_dash}) |\n")
            else:
                f.write(f"| **{model}** | [📊 Dashboard]({t_dash}) | [🚀 Dashboard]({p_dash}) |\n")

        f.write("\n### ⚖️ Error Distribution Analytics\n\n")
        f.write("| Engine | Global Metrics | Component Sensitivity (CoER) | Worst Sample Analysis |\n")
        f.write("|---|---|---|---|\n")

        t_global = "../../plots/I48_tesseract/plot_global_bar.png"
        t_coer   = "../../plots/I48_tesseract/plot_coer_bar.png"
        t_worst  = "../../plots/I48_tesseract/plot_worst_coer_grid.png"
        f.write(f"| **Tesseract (I48)** | [🖼️ View]({t_global}) | [🧩 View]({t_coer}) | [⚠️ View]({t_worst}) |\n")

        p_global = "../../plots/PT_Global/plot_global_bar.png"
        p_coer   = "../../plots/PT_Global/plot_coer_bar.png"
        p_worst  = "../../plots/PT_Global/plot_worst_coer_grid.png"
        f.write(f"| **PyTorch (Global)** | [🖼️ View]({p_global}) | [🧩 View]({p_coer}) | [⚠️ View]({p_worst}) |\n")

    print(f"    ✓ {md_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global MODELS, MODELS_I48, MODELS_I64

    parser = argparse.ArgumentParser(description="Benchmark: Tesseract vs PyTorch")
    parser.add_argument("--skip_plots", action="store_true", help="Skip plot rendering")
    parser.add_argument("--no-plots",   action="store_true", dest="no_plots",
                        help="Alias for --skip_plots. Skip all plot rendering.")
    parser.add_argument(
        "--models", nargs="+", default=None, metavar="MODEL",
        help="Restrict benchmark to these model names only. "
             "Example: --models LaoCRNN-I48K3C16 LaoCRNN-I48K3C32",
    )
    args = parser.parse_args()

    # --no-plots is an alias for --skip_plots
    if args.no_plots:
        args.skip_plots = True

    # Override global model lists when --models is supplied
    if args.models:
        MODELS    = args.models
        MODELS_I48 = [m for m in MODELS if "I48" in m]
        MODELS_I64 = [m for m in MODELS if "I64" in m]
        print(f"  Models override: {MODELS}")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(ENGINE_DIR, exist_ok=True)
    os.makedirs(TB_DIR, exist_ok=True)
    
    # Engine-specific detail folders
    os.makedirs(I48_TESS_DIR, exist_ok=True)
    os.makedirs(I48_PT_DIR, exist_ok=True)
    os.makedirs(I64_PT_DIR, exist_ok=True)
    os.makedirs(PT_GLOBAL_DIR, exist_ok=True)
    
    # Group dirs are created on-demand in render functions
    
    sep = "═════════════════════════════════════════════════════════════════"
    print(sep)
    print("  📊  Full Benchmark: Tesseract (.traineddata) vs PyTorch (.pt)")
    print(sep)

    # ── Step 1: Parse training logs ──────────────────────────────────────────
    print("\n[1] 🔎 Parsing training logs...")
    tess_logs = {}
    pt_logs   = {}
    for m in MODELS:
        tess_logs[m] = parse_tess_training(m)
        pt_logs[m]   = parse_pt_training(m)
        tb = tess_logs[m]["best_bcer"]
        pb = pt_logs[m]["best_bcer"]
        print(f"    {m:<16}  Tess best BCER={tb:.2f}%  |  PT best CER={pb:.2f}%"
              if tb is not None and pb is not None
              else f"    {m:<16}  Tess={tb}  |  PT={pb}")

    # ── Step 2: Load inference results ───────────────────────────────────────
    print("\n[2] 📦 Loading inference results...")
    tess_infer = load_tess_inference_results()
    pt_infer   = load_pt_inference_results()

    # ── Step 3: Training-curve plots ─────────────────────────────────────────
    if not args.skip_plots:
        print("\n[3] 📈 Rendering training-curve plots...")
        try:
            render_training_plots(tess_logs, pt_logs)
        except Exception as e:
            print(f"    [Warning] Training plots failed: {e}")

        # ── Step 4: Tess vs PT comparison bar charts ──────────────────────────
        print("\n[4] ⚖️  Rendering Tess vs PT comparison plots...")
        try:
            render_comparison_plots(tess_infer, pt_infer)
        except Exception as e:
            print(f"    [Warning] Comparison plots failed: {e}")
    else:
        print("\n[3-4] Skipping plots (--skip_plots).")

    # ── Step 5: TensorBoard ───────────────────────────────────────────────────
    print("\n[5] 🌐 Exporting to TensorBoard...")
    try:
        export_tensorboard(tess_logs, pt_logs, tess_infer, pt_infer)
    except Exception as e:
        print(f"    [Warning] TensorBoard export failed: {e}")

    # ── Step 5.5: CSV Exports ─────────────────────────────────────────────────
    print("\n[5.5] 📁 Exporting per-model CSV data...")
    try:
        export_model_csvs(tess_logs, pt_logs)
    except Exception as e:
        print(f"    [Warning] CSV export failed: {e}")

    # ── Step 6: Reports ───────────────────────────────────────────────────────
    print("\n[6] 📝 Writing reports (CSV + Markdown)...")
    try:
        write_reports(tess_logs, pt_logs, tess_infer, pt_infer)
    except Exception as e:
        print(f"    [Warning] Report generation failed: {e}")

    print("\n" + sep)
    print("  ✅  Benchmark complete!")
    print(f"  Reports → {os.path.relpath(REPORT_DIR, os.getcwd())}/")
    print(f"  Plots   → {os.path.relpath(PLOT_DIR, os.getcwd())}/")
    if HAS_TB:
        print(f"  Launch TensorBoard: tensorboard --logdir={os.path.relpath(TB_DIR, os.getcwd())}")
    print(sep)


if __name__ == "__main__":
    main()
