"""
tesseract_tools.py
──────────────────────────────────────────────────────────────────────────────
Combined tool:

  PART 1 – Training Log Plots
  ────────────────────────────
  Parse Tesseract LSTM training.log (and eval/*.eval.log) files and
  produce publication-ready PNG plots saved to plots/.

  Output plots
  ────────────
    plot_loss_combined.png   – Mean-RMS (≈loss) for all models on one axes
    plot_acc_combined.png    – Char Accuracy (100-BCER) for all models
    plot_train_loss.png      – Mean-RMS per model (subplots)
    plot_eval_loss.png       – BCER eval per model (sparse checkpoints)
    plot_train_acc.png       – Char Accuracy (train) per model (subplots)
    plot_eval_acc.png        – Char Accuracy (eval) per model (sparse)
    plot_wer_train.png       – Word Error Rate (train) all models combined
    plot_wer_eval.png        – Word Error Rate (eval) sparse combined

  PART 2 – Model Comparison (requires compiled .traineddata + test images)
  ─────────────────────────────────────────────────────────────────────────
  Run all models on a test image set, compute CER / WER / CoER / Accuracy,
  export CSV reports, and generate bar-chart + demo-grid plots.

  PART 3 – Training Time Report
  ──────────────────────────────
  Parse timestamps embedded in training.log (written by the segmented
  training script) to produce:
    • A printed table: segment boundaries, wall-clock duration,
      iterations/second, and BCER at the end of each segment.
    • report_training_time.csv  – machine-readable version of the table.
    • plot_training_time.png    – dual-axis chart: time-per-segment bars
      with BCER-over-iteration overlay.

Usage
─────
  # Generate training plots only (default)
  python tesseract_tools.py plots

  # Run model comparison only
  python tesseract_tools.py compare

  # Training time report (one model)
  python tesseract_tools.py timereport --model_name lao48cnn32

  # Training time report for ALL models found under data/
  python tesseract_tools.py timereport

  # Run both
  python tesseract_tools.py all

Requirements
────────────
  pip install matplotlib numpy pytesseract pillow

──────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import csv
import glob
import random
import time
import argparse
from collections import Counter, defaultdict
from pathlib import Path


import matplotlib
matplotlib.use("Agg")   # headless / no display required
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# §1  SHARED CONFIG & THEME
# ══════════════════════════════════════════════════════════════════════════════

# ─── Plot output ──────────────────────────────────────────────────────────────
PLOTS_DIR    = "plots"           # default output folder for training plots
COMPARE_DIR  = "data/tesseract_comparison"   # output for comparison results

# ─── Tesseract comparison config ─────────────────────────────────────────────
IMAGE_FOLDER        = "data/lao-ground-truth-test"   # jpg+gt.txt pairs
CUSTOM_TESSDATA_DIR = "tesseract_data"                  # where *.traineddata live
TESSERACT_PATH      = "tesseract"
LAO_FONT_PATH       = "font/phetsarath_ot.ttf"
TESS_CONFIG         = "--psm 7 --oem 1"

# The compiled model names (without .traineddata extension).
# These files must exist inside CUSTOM_TESSDATA_DIR:
#   data/lao48cnn16_final.traineddata
#   data/lao48cnn32_final.traineddata  etc.
CUSTOM_MODEL_NAMES = [
    "LaoCRNN-I48K3C16", "LaoCRNN-I48K3C32",
    "LaoCRNN-I48K5C16", "LaoCRNN-I48K5C32",
    "I64K2C16",         "I64K2C32",
    "LaoCRNN-I64K3C16", "LaoCRNN-I64K3C32",
    "I64K4C16",         "I64K4C32",
    "LaoCRNN-I64K5C16", "LaoCRNN-I64K5C32", "LaoCRNN-I64K5C64",
    "LaoCRNN-I64K7C16", "LaoCRNN-I64K7C32",
    "LaoCRNN-I64K9C16", "LaoCRNN-I64K9C32",
]

# ─── Light theme ──────────────────────────────────────────────────────────────
STYLE_BG   = "#ffffff"     # figure background
STYLE_AX   = "#f7f8fc"     # axes face
STYLE_GRID = "#dde1ea"     # grid lines
STYLE_TEXT = "#1a1d27"     # all text / ticks / labels

PALETTE = [                # line / bar colours (one per model)
    "#1a73e8",             # blue
    "#e8710a",             # orange
    "#188038",             # green
    "#d93025",             # red
    "#7b2fb5",             # purple
    "#0097a7",             # teal
    "#555555",             # gray
    "#c62828",             # red variant
    "#1565c0",             # blue variant
    "#2e7d32",             # green variant
]

# Premium Theme Alignment (matching tesstrain_core/plotting.py)
C_TRAIN      = '#90CAF9'    # Light blue
C_EVAL       = '#FF8A65'    # Orange
C_SMOOTH     = '#1565C0'    # Blue (smoothed train)
C_ESMOOTH    = '#C62828'    # Red (smoothed eval)
C_ACC_T      = '#2E7D32'    # Green
C_ACC_E      = '#E65100'    # Orange
STYLE_BG     = '#ffffff'
STYLE_AX     = '#f8f8f8'
STYLE_GRID   = '#dddddd'
STYLE_TEXT   = '#333333'
STYLE_BORDER = '#cccccc'

TITLE_FS  = 15
LABEL_FS  = 12
TICK_FS   = 10
LEGEND_FS = 11
LINE_W    = 2.0
MARKER_SZ = 7
FIG_DPI   = 150


# ══════════════════════════════════════════════════════════════════════════════
# §2  TRAINING LOG PARSING
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_RE = re.compile(
    r"At iteration (\d+)/\d+/\d+, "
    r"mean rms=([\d.]+)%, "
    r"delta=([\d.]+)%, "
    r"BCER train=([\d.]+)%, "
    r"BWER train=([\d.]+)%"
)
# Old segmented-training format:  modelname_<bcer>_<iter1>_<iter2>.eval.log
EVAL_FILE_RE_OLD = re.compile(r"[\w]+_([\d.]+)_(\d+)_(\d+)\.eval\.log$")
# New segmented-training format:  modelname_iter<NNNNNN>.eval.log
EVAL_FILE_RE_NEW = re.compile(r"[\w]+_iter(\d+)\.eval\.log$")
EVAL_LOG_RE      = re.compile(r"BCER eval=([\d.]+),\s+BWER eval=([\d.]+)")


def load_training_log(path):
    """Return sorted list of (iteration, rms, delta, bcer, bwer) tuples."""
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            m = TRAIN_RE.search(line)
            if m:
                rows.append((
                    int(m.group(1)),
                    float(m.group(2)),   # mean rms  (≈ loss)
                    float(m.group(3)),   # delta
                    float(m.group(4)),   # BCER train (char error rate)
                    float(m.group(5)),   # BWER train (word error rate)
                ))
    rows.sort(key=lambda r: r[0])
    return rows


def load_eval_logs(eval_dir):
    """
    Return sorted list of (iteration, bcer_eval, bwer_eval) from all
    *.eval.log files in eval_dir.

    Handles two filename conventions:
      OLD (classic)  : <model>_<bcer>_<iter_start>_<iter_end>.eval.log
                       iteration = iter_start (group 2 of OLD regex)
      NEW (segmented): <model>_iter<NNNNNN>.eval.log
                       iteration = the numeric suffix
    """
    rows = []
    for fpath in glob.glob(os.path.join(eval_dir, "*.eval.log")):
        fname = os.path.basename(fpath)

        # Try new format first (more specific pattern)
        m_new = EVAL_FILE_RE_NEW.search(fname)
        if m_new:
            iteration = int(m_new.group(1))
        else:
            # Fall back to old format
            m_old = EVAL_FILE_RE_OLD.search(fname)
            if not m_old:
                continue   # unrecognised filename
            iteration = int(m_old.group(2))  # iter_start

        with open(fpath, "r", encoding="utf-8") as fh:
            for line in fh:
                m_log = EVAL_LOG_RE.search(line)
                if m_log:
                    rows.append((
                        iteration,
                        float(m_log.group(1)),   # BCER eval
                        float(m_log.group(2)),   # BWER eval
                    ))
    rows.sort(key=lambda r: r[0])
    return rows


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
    except Exception:
        pass
    return None


def parse_plain_training_time(path):
    """
    Standard Tesseract log fallback.
    Scans the file for timestamps at start of line: [YYYY-MM-DD HH:MM:SS]
    Returns total duration in seconds.
    """
    if not os.path.exists(path):
        return 0.0
    start_ts = None
    end_ts = None
    # Matches [2026-04-16 10:48:25]
    ts_re = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                m = ts_re.search(line)
                if m:
                    try:
                        dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                        if start_ts is None:
                            start_ts = dt
                        end_ts = dt
                    except ValueError:
                        continue
    except Exception:
        pass

    if start_ts and end_ts:
        return (end_ts - start_ts).total_seconds()
    return 0.0


def discover_models(data_dir):
    """Return list of (model_name, training_log_path, eval_dir_or_None)."""
    models = []
    for model_dir in sorted(Path(data_dir).iterdir()):
        if not model_dir.is_dir():
            continue
        log_path = model_dir / "training.log"
        if not log_path.exists():
            continue
        eval_dir = model_dir / "eval"
        models.append((
            model_dir.name,
            str(log_path),
            str(eval_dir) if eval_dir.is_dir() else None,
        ))
    return models


# ══════════════════════════════════════════════════════════════════════════════
# §3  THEME HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def apply_theme(fig, axes_list):
    """Reset to the original style of matplotlib."""
    import matplotlib.pyplot as plt
    plt.rcdefaults()


def save_fig(fig, path):
    """Tight-layout, save, and close a figure."""
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# §4  TRAINING-LOG PLOT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_loss_combined(model_data, out_dir):
    """Mean-RMS (≈ training loss) for all models on one axes."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, train_rows, _) in enumerate(model_data):
        if not train_rows:
            continue
        iters = [r[0] for r in train_rows]
        rms   = [r[1] for r in train_rows]
        label_name = name if name.startswith("LaoCRNN-") else f"LaoCRNN-{name}"
        ax.plot(iters, rms, label=label_name, color=PALETTE[i % len(PALETTE)],
                linewidth=LINE_W)
    ax.set_xlabel("Iteration", fontsize=LABEL_FS)
    ax.set_ylabel("Mean RMS (%)", fontsize=LABEL_FS)
    ax.set_title("Training Loss (Mean RMS) — All Models", fontsize=TITLE_FS)
    ax.legend(fontsize=LEGEND_FS)
    apply_theme(fig, [ax])
    save_fig(fig, os.path.join(out_dir, "plot_loss_combined.png"))


def plot_acc_combined(model_data, out_dir):
    """Character Accuracy (100-BCER) for all models on one axes."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, train_rows, _) in enumerate(model_data):
        if not train_rows:
            continue
        iters = [r[0] for r in train_rows]
        acc   = [100.0 - r[3] for r in train_rows]
        label_name = name if name.startswith("LaoCRNN-") else f"LaoCRNN-{name}"
        ax.plot(iters, acc, label=label_name, color=PALETTE[i % len(PALETTE)],
                linewidth=LINE_W)
    ax.set_xlabel("Iteration", fontsize=LABEL_FS)
    ax.set_ylabel("Character Accuracy (%)", fontsize=LABEL_FS)
    ax.set_title("Training Character Accuracy (100 – BCER) — All Models",
                 fontsize=TITLE_FS)
    ax.legend(fontsize=LEGEND_FS)
    apply_theme(fig, [ax])
    save_fig(fig, os.path.join(out_dir, "plot_acc_combined.png"))


def _smooth(values, window):
    """Rolling mean with reflect-padding so edges are not clipped."""
    if len(values) < window:
        window = max(1, len(values))
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="reflect")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def _rolling_std(values, window):
    """Rolling std with the same reflect-padding trick."""
    if len(values) < window:
        window = max(1, len(values))
    padded = np.pad(values, (window // 2, window // 2), mode="reflect")
    result = np.array([
        padded[i: i + window].std()
        for i in range(len(values))
    ])
    return result


def plot_accuracy_premium(model_data, out_dir):
    """
    Premium combined accuracy plot reflecting reference style:
      • Dashed line (color) – raw training accuracy (100 - BCER_train)
      • Solid line  (color) – validation accuracy checkpoints (100 - BCER_eval)
      • Shaded band (color) – ± rolling std around training curve
      • ★ star marker (color) – best checkpoint annotated with its value
    """
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor(STYLE_BG)
    ax.set_facecolor(STYLE_AX)

    for i, (name, train_rows, eval_rows) in enumerate(model_data):
        color = PALETTE[i % len(PALETTE)]

        # ── Training data (Dashed) ──────────────────────────────────
        if train_rows:
            t_iters = np.array([r[0] for r in train_rows])
            t_acc   = np.array([100.0 - r[3] for r in train_rows])
            
            # Smoothing for the band
            win = max(5, len(t_iters) // 10)
            smoothed = _smooth(t_acc, win)
            std_band = _rolling_std(t_acc, win)

            ax.plot(t_iters, t_acc, color=color, linewidth=0.8,
                    linestyle="--", alpha=0.4, zorder=2)
            ax.fill_between(t_iters, 
                            np.clip(smoothed - std_band, 0, 100),
                            np.clip(smoothed + std_band, 0, 100),
                            color=color, alpha=0.1, zorder=1)

        # ── Validation data (Solid) ─────────────────────────────────
        if eval_rows:
            e_iters = np.array([r[0] for r in eval_rows])
            e_acc   = np.array([100.0 - r[1] for r in eval_rows])
            
            # Plot validation line (Solid)
            ax.plot(e_iters, e_acc, color=color, linewidth=2.5,
                    linestyle="-", zorder=4, label=name)

            # Best point marker
            best_idx = int(np.argmax(e_acc))
            bx, by   = e_iters[best_idx], e_acc[best_idx]
            
            ax.scatter([bx], [by], marker="*", s=250, color=color,
                       edgecolors="white", linewidths=0.5, zorder=6)
            ax.text(bx, by + 1.2, f"{by/100:.3f}", 
                    color=color, fontweight="bold", ha="center", fontsize=10)
        elif train_rows:
            # If no eval, show smoothed training as solid
            ax.plot(t_iters, smoothed, color=color, linewidth=2.5,
                    linestyle="-", zorder=3, label=name)

    # Styling
    from matplotlib.lines import Line2D
    legend_extras = [
        Line2D([0], [0], color="#555555", lw=2.5, linestyle="-",  label="(Validation)"),
        Line2D([0], [0], color="#555555", lw=1.2, linestyle="--", label="(Training)"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + legend_extras, labels + ["(Validation)", "(Training)"],
              loc="lower right", fontsize=10, framealpha=0.9,
              facecolor=STYLE_AX, edgecolor=STYLE_GRID, labelcolor=STYLE_TEXT)

    ax.set_xlabel("Iteration", fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=LABEL_FS, fontweight="bold")
    ax.set_title("Character Accuracy (Combined)", fontsize=TITLE_FS+1, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    apply_theme(fig, [ax])
    save_fig(fig, os.path.join(out_dir, "plot_acc_premium.png"))


def plot_loss_premium(model_data, out_dir):
    """
    Premium combined CTC Loss plot (Mean RMS fallback):
      • Logarithmic Y-axis (very important for reference style)
      • Solid = Best estimate / Validation
      • Dashed = Raw Training noise
      • Bands = Stability area
    """
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor(STYLE_BG)
    ax.set_facecolor(STYLE_AX)

    for i, (name, train_rows, eval_rows) in enumerate(model_data):
        color = PALETTE[i % len(PALETTE)]

        if train_rows:
            t_iters = np.array([r[0] for r in train_rows])
            # Use BCER train (r[3]) instead of Mean RMS (r[1]) to ensure
            # training and validation are on the same scale (Error Rate)
            t_loss  = np.array([r[3] / 100.0 for r in train_rows]) 
            
            win = max(5, len(t_iters) // 10)
            smoothed = _smooth(t_loss, win)
            std_band = _rolling_std(t_loss, win)

            # Raw training (Dashed)
            ax.plot(t_iters, t_loss, color=color, linewidth=0.9,
                    linestyle="--", alpha=0.35, zorder=2)
            
            # Bands
            ax.fill_between(t_iters, 
                            np.maximum(1e-5, smoothed - std_band),
                            smoothed + std_band,
                            color=color, alpha=0.08, zorder=1)
            
            # Validation metrics (BCER eval)
            if eval_rows:
                e_iters = np.array([r[0] for r in eval_rows])
                e_loss  = np.array([r[1] / 100.0 for r in eval_rows]) # BCER eval
                ax.plot(e_iters, e_loss, color=color, linewidth=2.5,
                        linestyle="-", zorder=4, label=name)
                
                # Markers for best eval point
                best_idx = int(np.argmin(e_loss))
                bx, by   = e_iters[best_idx], e_loss[best_idx]
                ax.scatter([bx], [by], marker="*", s=250, color=color,
                           edgecolors="white", linewidths=0.5, zorder=6)
                ax.text(bx, by - (by*0.2), f"{by:.3f}", 
                        color=color, fontweight="bold", ha="center", va="top", fontsize=10)
            else:
                # Fallback to smoothed training as solid line
                ax.plot(t_iters, smoothed, color=color, linewidth=2.5,
                        linestyle="-", zorder=3, label=name)

    # Style
    ax.set_yscale('log')
    from matplotlib.lines import Line2D
    legend_extras = [
        Line2D([0], [0], color="#555555", lw=2.5, linestyle="-",  label="(Validation / BCER)"),
        Line2D([0], [0], color="#555555", lw=1.2, linestyle="--", label="(Training / BCER)"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + legend_extras, labels + ["(Validation)", "(Training)"],
              loc="upper right", fontsize=10, framealpha=0.9,
              facecolor=STYLE_AX, edgecolor=STYLE_GRID, labelcolor=STYLE_TEXT)

    ax.set_xlabel("Iteration", fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel("Error Rate (Loss)", fontsize=LABEL_FS, fontweight="bold")
    ax.set_title("CTC Performance (Combined Error Rate)", fontsize=TITLE_FS+1, fontweight="bold")
    ax.set_xlim(left=0)
    apply_theme(fig, [ax])
    save_fig(fig, os.path.join(out_dir, "plot_loss_premium.png"))


def _train_subplots(model_data, metric_idx, ylabel, title, fname, out_dir,
                    transform=None):
    """Generic per-model subplot grid from training data."""
    n = len(model_data)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(7 * cols, 4 * rows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]
    last_i = 0
    for i, (name, train_rows, _) in enumerate(model_data):
        ax = axes_flat[i]
        last_i = i
        title_name = name if name.startswith("LaoCRNN-") else f"LaoCRNN-{name}"
        if train_rows:
            iters  = [r[0] for r in train_rows]
            values = [r[metric_idx] for r in train_rows]
            if transform:
                values = [transform(v) for v in values]
            ax.plot(iters, values, color=PALETTE[i % len(PALETTE)],
                    linewidth=LINE_W, label=title_name)
            ax.legend(fontsize=LEGEND_FS)
        ax.set_xlabel("Iteration", fontsize=LABEL_FS)
        ax.set_ylabel(ylabel, fontsize=LABEL_FS)
        ax.set_title(title_name, fontsize=TITLE_FS)
    for j in range(last_i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(title, fontsize=TITLE_FS + 2, color=STYLE_TEXT, y=1.01)
    apply_theme(fig, axes_flat)
    save_fig(fig, os.path.join(out_dir, fname))


def _eval_subplots(model_data, eval_metric_idx, ylabel, title, fname,
                   out_dir, transform=None):
    """Generic per-model subplot grid from eval data (sparse checkpoints)."""
    with_eval = [(n, td, ed) for n, td, ed in model_data if ed]
    if not with_eval:
        print(f"  Skipping {fname}: no eval data found for any model.")
        return
    n = len(with_eval)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(7 * cols, 4 * rows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]
    last_i = 0
    for i, (name, _, eval_rows) in enumerate(with_eval):
        ax = axes_flat[i]
        last_i = i
        title_name = name if name.startswith("LaoCRNN-") else f"LaoCRNN-{name}"
        if eval_rows:
            iters  = [r[0] for r in eval_rows]
            values = [r[eval_metric_idx] for r in eval_rows]
            if transform:
                values = [transform(v) for v in values]
            color = PALETTE[i % len(PALETTE)]
            ax.plot(iters, values, color=color, linewidth=LINE_W,
                    marker="o", markersize=MARKER_SZ, label=title_name)
            ax.legend(fontsize=LEGEND_FS)
        ax.set_xlabel("Iteration", fontsize=LABEL_FS)
        ax.set_ylabel(ylabel, fontsize=LABEL_FS)
        ax.set_title(title_name, fontsize=TITLE_FS)
    for j in range(last_i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(title, fontsize=TITLE_FS + 2, color=STYLE_TEXT, y=1.01)
    apply_theme(fig, axes_flat)
    save_fig(fig, os.path.join(out_dir, fname))


def plot_wer_combined(model_data, which, out_dir):
    """Word Error Rate – combined overlay for train or eval."""
    assert which in ("train", "eval")
    fig, ax = plt.subplots(figsize=(10, 5))
    if which == "train":
        for i, (name, train_rows, _) in enumerate(model_data):
            if not train_rows:
                continue
            iters = [r[0] for r in train_rows]
            wer   = [r[4] for r in train_rows]   # BWER train
            title_name = name if name.startswith("LaoCRNN-") else f"LaoCRNN-{name}"
            ax.plot(iters, wer, label=title_name,
                    color=PALETTE[i % len(PALETTE)], linewidth=LINE_W)
        ax.set_title("Word Error Rate (Train) — All Models", fontsize=TITLE_FS)
        fname = "plot_wer_train.png"
    else:
        for i, (name, _, eval_rows) in enumerate(model_data):
            if not eval_rows:
                continue
            iters = [r[0] for r in eval_rows]
            wer   = [r[2] for r in eval_rows]    # BWER eval
            title_name = name if name.startswith("LaoCRNN-") else f"LaoCRNN-{name}"
            ax.plot(iters, wer, label=title_name,
                    color=PALETTE[i % len(PALETTE)], linewidth=LINE_W,
                    marker="o", markersize=MARKER_SZ)
        ax.set_title("Word Error Rate (Eval) — All Models", fontsize=TITLE_FS)
        fname = "plot_wer_eval.png"
    ax.set_xlabel("Iteration", fontsize=LABEL_FS)
    ax.set_ylabel("Word Error Rate (%)", fontsize=LABEL_FS)
    ax.legend(fontsize=LEGEND_FS)
    apply_theme(fig, [ax])
    save_fig(fig, os.path.join(out_dir, fname))


def plot_training_dashboard(model_data, out_dir):
    """Generate a unified 6-panel (3x2) dashboard:
    Column 1: train_acc, train_loss_cer (train CER), train_loss_wer (train WER)
    Column 2: eval_acc, eval_loss_cer (eval CER), eval_loss_wer (eval WER)
    """
    from matplotlib.ticker import MaxNLocator
    
    for i, (name, train_rows, eval_rows) in enumerate(model_data):
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        title_name = name if name.startswith("LaoCRNN-") else f"LaoCRNN-{name}"
        fig.suptitle(f"{title_name} — Tesseract Training Dashboard", 
                     fontsize=20, fontweight="bold", y=0.98)
        
        apply_theme(fig, axes.flatten())
        
        # --- Column 1: Train ---
        # 1. train_acc (row 0, col 0)
        ax = axes[0, 0]
        if train_rows:
            iters = [r[0] for r in train_rows]
            t_acc = [100.0 - r[3] for r in train_rows]
            ax.plot(iters, t_acc, color=C_ACC_T, alpha=0.2, lw=0.8)
            ax.plot(iters, _smooth(t_acc, 10), color=C_ACC_T, lw=1.5, label="Train Acc")
        ax.set_title("Training Character Accuracy", fontweight="bold")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
        ax.legend(loc="lower right")
        
        # 2. train_loss_cer (row 1, col 0)
        ax = axes[1, 0]
        if train_rows:
            iters = [r[0] for r in train_rows]
            t_cer = [r[3] for r in train_rows]
            ax.plot(iters, t_cer, color=C_TRAIN, alpha=0.3, lw=0.8)
            ax.plot(iters, _smooth(t_cer, 10), color=C_SMOOTH, lw=2, label="Train CER")
        ax.set_title("Training Character Error Rate (CER)", fontweight="bold")
        ax.set_ylabel("CER (%)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right")
        
        # 3. train_loss_wer (row 2, col 0)
        ax = axes[2, 0]
        if train_rows:
            iters = [r[0] for r in train_rows]
            t_wer = [r[4] for r in train_rows]
            ax.plot(iters, t_wer, color=C_TRAIN, alpha=0.3, lw=0.8)
            ax.plot(iters, _smooth(t_wer, 10), color=C_SMOOTH, lw=2, label="Train WER")
        ax.set_title("Training Word Error Rate (WER)", fontweight="bold")
        ax.set_ylabel("WER (%)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right")
        ax.set_xlabel("Iteration")
        
        # --- Column 2: Eval ---
        # 4. eval_acc (row 0, col 1)
        ax = axes[0, 1]
        if eval_rows:
            e_iters = [r[0] for r in eval_rows]
            e_acc   = [100.0 - r[1] for r in eval_rows]
            ax.plot(e_iters, e_acc, color=C_ACC_E, marker="o", markersize=4, lw=2, label="Eval Acc")
        ax.set_title("Evaluation Character Accuracy", fontweight="bold")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
        ax.legend(loc="lower right")
        
        # 5. eval_loss_cer (row 1, col 1)
        ax = axes[1, 1]
        if eval_rows:
            e_iters = [r[0] for r in eval_rows]
            e_cer   = [r[1] for r in eval_rows]
            ax.plot(e_iters, e_cer, color=C_ESMOOTH, marker="o", markersize=4, lw=2, label="Eval CER")
        ax.set_title("Evaluation Character Error Rate (CER)", fontweight="bold")
        ax.set_ylabel("CER (%)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right")
        
        # 6. eval_loss_wer (row 2, col 1)
        ax = axes[2, 1]
        if eval_rows:
            e_iters = [r[0] for r in eval_rows]
            e_wer   = [r[2] for r in eval_rows]
            ax.plot(e_iters, e_wer, color=C_ESMOOTH, marker="o", markersize=4, lw=2, label="Eval WER")
        ax.set_title("Evaluation Word Error Rate (WER)", fontweight="bold")
        ax.set_ylabel("WER (%)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right")
        ax.set_xlabel("Iteration")
        
        for a in axes.flatten():
            a.xaxis.set_major_locator(MaxNLocator(integer=True))
            
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fname = f"plot_dashboard_{name}.png"
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✓ Dashboard [{name}] → {path}")


def run_plots(data_dir="data", out_dir=PLOTS_DIR):
    """Main entry for the training-log plot pipeline."""
    script_dir = Path(__file__).parent.resolve()
    data_path  = script_dir / data_dir
    out_path   = script_dir / out_dir
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Training Log Plots")
    print("=" * 70)
    print(f"  Models dir : {data_path}")
    print(f"  Output dir : {out_path}\n")

    models = discover_models(data_path)
    if not models:
        print("  No models with training.log found – exiting.")
        return

    model_data = []
    for name, log_path, eval_dir in models:
        print(f"  Loading [{name}]")
        train_rows = load_training_log(log_path)
        eval_rows  = load_eval_logs(eval_dir) if eval_dir else []
        print(f"    train rows: {len(train_rows)},  eval checkpoints: {len(eval_rows)}")
        model_data.append((name, train_rows, eval_rows))

    print()
    out = str(out_path)

    # Combined overlays
    plot_loss_combined(model_data, out)
    plot_acc_combined(model_data, out)
    plot_loss_premium(model_data, out)      # New premium loss curve
    plot_accuracy_premium(model_data, out)   # Updated premium accuracy curve
    plot_wer_combined(model_data, "train", out)
    plot_wer_combined(model_data, "eval",  out)

    # Subplots and Dashboards
    plot_training_dashboard(model_data, out)
    
    _train_subplots(model_data, metric_idx=1,
                    ylabel="Mean RMS (%)",
                    title="Training Loss (Mean RMS) — Per Model",
                    fname="plot_train_loss.png", out_dir=out)
    _train_subplots(model_data, metric_idx=3,
                    ylabel="Character Accuracy (%)",
                    title="Training Character Accuracy — Per Model",
                    fname="plot_train_acc.png", out_dir=out,
                    transform=lambda v: 100.0 - v)

    # Per-model subplots – eval (index 1 = BCER eval, index 2 = BWER eval)
    _eval_subplots(model_data, eval_metric_idx=1,
                   ylabel="BCER Eval (%)",
                   title="Eval Character Error Rate — Per Model",
                   fname="plot_eval_loss.png", out_dir=out)
    _eval_subplots(model_data, eval_metric_idx=1,
                   ylabel="Character Accuracy (%)",
                   title="Eval Character Accuracy — Per Model",
                   fname="plot_eval_acc.png", out_dir=out,
                   transform=lambda v: 100.0 - v)

    print(f"\n  All plots saved to: {out_path}/")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# §5  LAO COMPONENT MAP
# ══════════════════════════════════════════════════════════════════════════════

LAO_COMPONENT_MAP = {
    "tone":         ["່", "້", "໊", "໋"],
    "consonant":    ["ກ","ຂ","ຄ","ງ","ຈ","ຊ","ຍ","ດ","ຕ","ຖ","ທ","ນ","ບ","ປ",
                     "ຜ","ຝ","ພ","ຟ","ມ","ຢ","ຣ","ລ","ວ","ສ","ຫ","ອ","ຮ","ໜ","ໝ","ຽ",
                     "ໆ", "ຯ"],
    "lao_number":   ["໐","໑","໒","໓","໔","໕","໖","໗","໘","໙"],
    "digit":        ["0","1","2","3","4","5","6","7","8","9"],
    "symbol":       list('.,\"?/!:;- +*=()[]{}\'"@#$%^&_|\\<>«»—…""\'\'•·°±×÷₭¥€£$'),
    "before_vowel": ["ແ","ໂ","ເ","ໄ","ໃ"],
    "rear_vowel":   ["ະ","າ","ຳ"],
    "upper_vowel":  ["ັ","ິ","ີ","ົ","ຶ","ື","໌","ໍ"],
    "lower_vowel":  ["ຸ","ູ"],
    "special":      ["ຼ"],
}

_CHAR_TO_CAT = {}
for _cat, _chars in LAO_COMPONENT_MAP.items():
    for _ch in _chars:
        _CHAR_TO_CAT[_ch] = _cat


def get_component_type(ch: str) -> str:
    return _CHAR_TO_CAT.get(ch, "other")


# ══════════════════════════════════════════════════════════════════════════════
# §6  EDIT DISTANCE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def edit_distance(a, b):
    """Generic Levenshtein distance over any sequence."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost)
        prev = curr
    return prev[lb]


def align_strings(pred: str, truth: str):
    """Return list of (op, pred_char, truth_char) via Levenshtein backtrace."""
    m, n = len(pred), len(truth)
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if pred[i-1] == truth[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if pred[i-1] == truth[j-1] else 1):
            op = "mat" if pred[i-1] == truth[j-1] else "sub"
            ops.append((op, pred[i-1], truth[j-1]))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(("ins", pred[i-1], None))
            i -= 1
        else:
            ops.append(("del", None, truth[j-1]))
            j -= 1
    return list(reversed(ops))


# ══════════════════════════════════════════════════════════════════════════════
# §7  TESSERACT INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

try:
    import pytesseract
    from PIL import Image as _PILImage
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

# ── PyTorch Imports ──────────────────────────────────────────────────────────
try:
    import torch
    from tesstrain_core.model import build_model
    from tesstrain_core.recognize import recognize_line
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

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
    m_match = re.match(r'\[?\d+,(\d+),', ckpt.get('net_spec', ''))
    if m_match: target_h = int(m_match.group(1))
    
    return model, inv_charset, target_h

def run_pytorch(image_path, model, inv_charset, target_h, device):
    """Run PyTorch inference on a single image line."""
    try:
        from PIL import Image as _PILImage
        pil = _PILImage.open(image_path).convert('L')
        return recognize_line(model, pil, target_h, device, inv_charset, beam_width=0)
    except Exception:
        return ""



def find_test_pairs(image_folder: str):
    """Find (image_path, ground_truth) pairs (image + matching .gt.txt)."""
    pairs = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        for img_path in sorted(glob.glob(os.path.join(image_folder, ext))):
            base    = os.path.splitext(img_path)[0]
            gt_path = base + ".gt.txt"
            if os.path.exists(gt_path):
                with open(gt_path, encoding="utf-8") as f:
                    gt = f.read().strip()
                pairs.append((img_path, gt))
    return pairs


def run_tesseract(image_path: str, model_name: str, tessdata_dir: str = CUSTOM_TESSDATA_DIR) -> str:
    """Run Tesseract with the custom model and return predicted text."""
    config = f'--tessdata-dir "{tessdata_dir}" -l {model_name} {TESS_CONFIG}'
    try:
        img  = _PILImage.open(image_path)
        text = pytesseract.image_to_string(img, config=config)
        return text.strip()
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# §8  METRIC COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

COER_COLS = [
    ("tone",         "Tone Mark"),
    ("upper_vowel",  "Upper Vowel"),
    ("consonant",    "Consonant"),
    ("before_vowel", "Before Vowel"),
    ("rear_vowel",   "After Vowel"),
    ("lower_vowel",  "Lower Vowel"),
    ("symbol",       "Symbol"),
    ("digit",        "Digit"),
    ("lao_number",   "Lao Number"),
    ("special",      "Special")
]


def compute_metrics(predictions: dict, test_pairs: list, inference_times: dict = None):
    """
    predictions : {model_name: {img_path: pred_str}}
    inference_times: {model_name: total_inference_seconds}
    Returns     : {model_name: {cer, wer, coer, acc, eval_time, char_stats, mistake_counter}}
    """
    results = {}
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
            pred = pred_map.get(img_path, "")

            # CER
            d = edit_distance(pred, truth)
            total_cer_num += d
            total_cer_den += max(len(truth), 1)

            # WER
            tw = truth.split()
            pw = pred.split()
            total_wer_dist += edit_distance(pw, tw)
            total_wer_wds  += max(len(tw), 1)

            # Alignment for CoER + character stats
            edits = align_strings(pred, truth)
            for op, p_c, t_c in edits:
                ch  = t_c if t_c else p_c
                cat = get_component_type(ch)
                if t_c:
                    char_stats[t_c]["total"]  += 1
                    coer_stats[cat]["total"]  += 1
                if op != "mat":
                    if t_c:
                        char_stats[t_c]["errors"]  += 1
                        coer_stats[cat]["errors"]  += 1
                    mistake_ctr[(op, p_c, t_c)] += 1
                else:
                    if t_c:
                        char_stats[t_c]["correct"] += 1

        if inference_times and model_name in inference_times:
            eval_time = inference_times[model_name]
        else:
            eval_time = time.time() - t0
            
        cer  = total_cer_num  / max(total_cer_den, 1)
        wer  = total_wer_dist / max(total_wer_wds, 1)
        coer = {cat: s["errors"] / max(s["total"], 1)
                for cat, s in coer_stats.items() if s["total"] > 0}
        coer_avg = sum(coer.values()) / max(len(coer), 1)
        acc  = max(0.0, 1.0 - (cer + wer + coer_avg) / 3.0)

        results[model_name] = {
            "cer": cer, "wer": wer, "coer": coer, "acc": acc,
            "eval_time": eval_time,
            "char_stats": char_stats,
            "mistake_ctr": mistake_ctr,
        }
    return results


# ══════════════════════════════════════════════════════════════════════════════
# §9  REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: dict, models: list, train_times: dict = None, eval_times: dict = None, title="Tesseract Model Comparison Results"):
    train_times = train_times or {}
    eval_times  = eval_times or {}
    print(f"\n  ==== {title} ====")
    header = (f"  {'Model':<22} {'CER (%)':<10} {'WER (%)':<10} "
              f"{'Acc (%)':<9} {'Eval Time':<12} {'Test Time':<12} {'Train Time':<12}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in models:
        r = results.get(m)
        
        # Train Time
        train_s = train_times.get(m)
        if train_s is not None:
            tt_mm, tt_ss = divmod(int(train_s), 60)
            tt_hh, tt_mm = divmod(tt_mm, 60)
            train_str = f"{tt_hh:02d}:{tt_mm:02d}:{tt_ss:02d}"
        else:
            train_str = "    —"
            
        # Eval Time (from validating.log)
        eval_s = eval_times.get(m)
        if eval_s is not None:
            et_mm, et_ss = divmod(int(eval_s), 60)
            et_hh, et_mm = divmod(et_mm, 60)
            eval_str = f"{et_hh:02d}:{et_mm:02d}:{et_ss:02d}"
        else:
            eval_str = "    —"

        if r:
            # Test Time (inference)
            test_s = r['eval_time'] # This is currently 'inf_time'
            test_str = f"{test_s:>11.1f}s"
            
            print(f"  {m:<22} {r['cer']*100:>10.2f} {r['wer']*100:>10.2f} "
                  f"{r['acc']*100:>9.2f} {eval_str:<12} {test_str:<12} {train_str:<12}")
        else:
            print(f"  {m:<22} {'--':>10} {'--':>10} {'--':>9} {'--':>12} {'--':>12} {train_str:<12}")
    print()


def export_csv_report(results: dict, models: list, out_dir: str,
                      train_times: dict = None, eval_times: dict = None,
                      realworld_results: dict = None):
    """
    Write report.csv, report_characters.csv, and report_mistakes.csv.
    train_times : {model_name: total_train_seconds}  (optional)
    """
    train_times = train_times or {}
    # Global metrics
    csv_path = os.path.join(out_dir, "report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Model", "CER (%)", "WER (%)", "Accuracy (%)",
                  "Eval Time (s)", "Test Time (s)", "Train Time (s)", "Train Time (HH:MM:SS)"]
        if realworld_results is not None:
            header += ["Real-world CER (%)", "Real-world WER (%)", "Real-world Accuracy (%)", "Real-world Test Time (s)"]
            
        for _, col in COER_COLS:
            header.append(f"CoER {col} (%)")
        if realworld_results is not None:
            for _, col in COER_COLS:
                header.append(f"Real-world CoER {col} (%)")
        writer.writerow(header)
        for m in models:
            r = results.get(m)
            if not r: continue
            train_s = train_times.get(m)
            if train_s is not None:
                tt_mm, tt_ss = divmod(int(train_s), 60)
                tt_hh, tt_mm = divmod(tt_mm, 60)
                train_hms = f"{tt_hh:02d}:{tt_mm:02d}:{tt_ss:02d}"
                train_s_str = f"{train_s:.1f}"
            else:
                train_s_str = ""
                train_hms   = ""
            eval_s = eval_times.get(m)
            eval_s_str = f"{eval_s:.1f}" if eval_s is not None else ""
            
            row = [m, f"{r['cer']*100:.2f}", f"{r['wer']*100:.2f}",
                   f"{r['acc']*100:.2f}", eval_s_str, f"{r['eval_time']:.1f}",
                   train_s_str, train_hms]
                   
            if realworld_results is not None:
                rw_r = realworld_results.get(m)
                if rw_r:
                    row += [f"{rw_r['cer']*100:.2f}", f"{rw_r['wer']*100:.2f}",
                            f"{rw_r['acc']*100:.2f}", f"{rw_r['eval_time']:.1f}"]
                else:
                    row += ["", "", "", ""]
                    
            for cat, _ in COER_COLS:
                row.append(f"{r['coer'].get(cat, 0)*100:.2f}")
                
            if realworld_results is not None:
                rw_r = realworld_results.get(m)
                if rw_r:
                    for cat, _ in COER_COLS:
                        row.append(f"{rw_r['coer'].get(cat, 0)*100:.2f}")
                else:
                    row += [""] * len(COER_COLS)
                    
            writer.writerow(row)
    print(f"  ✓ Global CSV  → {csv_path}")

    # Character-level report
    all_chars = sorted(set(
        ch for m in models if results.get(m)
        for ch in results[m]["char_stats"].keys()
    ))
    char_csv = os.path.join(out_dir, "report_characters.csv")
    with open(char_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Character", "Category"]
        for m in models:
            header += [f"{m} GT", f"{m} Correct", f"{m} Errors",
                       f"{m} Error Rate (%)"]
        writer.writerow(header)
        for ch in all_chars:
            cat = get_component_type(ch)
            row = [ch, cat]
            for m in models:
                cs = (results[m]["char_stats"].get(ch,
                      {"total": 0, "correct": 0, "errors": 0})
                      if results.get(m) else {})
                t = cs.get("total", 0)
                c = cs.get("correct", 0)
                e = cs.get("errors", 0)
                row += [t, c, e, f"{(e/t*100) if t else 0.0:.1f}"]
            writer.writerow(row)
    print(f"  ✓ Char CSV    → {char_csv}")

    # Mistakes report
    mistakes_csv = os.path.join(out_dir, "report_mistakes.csv")
    with open(mistakes_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Operation", "GT Char", "Pred Char", "Count"])
        for m in models:
            if not results.get(m): continue
            errors = [(op, p, t, cnt)
                      for (op, p, t), cnt in results[m]["mistake_ctr"].items()
                      if op != "mat"]
            errors.sort(key=lambda x: x[3], reverse=True)
            for op, p, t, cnt in errors:
                writer.writerow([m, op, t or "", p or "", cnt])
    print(f"  ✓ Mistakes CSV→ {mistakes_csv}\n")


# ══════════════════════════════════════════════════════════════════════════════
# §10 COMPARISON PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_global_bar(results: dict, models: list, out_dir: str):
    """Grouped bar chart comparing CER, WER, Avg CoER and Accuracy per model."""
    valid = [m for m in models if results.get(m)]
    if not valid:
        return

    colors = plt.get_cmap("tab10").colors
    x      = np.arange(len(valid))
    width  = 0.2

    cer_v  = [results[m]["cer"] * 100 for m in valid]
    wer_v  = [results[m]["wer"] * 100 for m in valid]
    coer_v = [(sum(results[m]["coer"].values()) / max(len(results[m]["coer"]), 1)) * 100
              if results[m]["coer"] else 0.0 for m in valid]
    acc_v  = [results[m]["acc"] * 100 for m in valid]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(STYLE_BG)
    ax.set_facecolor(STYLE_AX)

    rects1 = ax.bar(x - width*1.5, cer_v,  width, label="CER (%)",      color=colors[0])
    rects2 = ax.bar(x - width*0.5, wer_v,  width, label="WER (%)",      color=colors[1])
    rects3 = ax.bar(x + width*0.5, coer_v, width, label="Avg CoER (%)", color=colors[2])
    rects4 = ax.bar(x + width*1.5, acc_v,  width, label="Accuracy (%)", color=colors[3])

    for rects in [rects1, rects2, rects3, rects4]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9,
                        fontweight="bold", rotation=90 if h > 0 else 0,
                        color=STYLE_TEXT)

    ax.set_ylabel("Percentage (%)", fontweight="bold", fontsize=12, color=STYLE_TEXT)
    ax.set_title("Tesseract Model — Global Performance",
                 fontweight="bold", fontsize=14, pad=16, color=STYLE_TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(valid, fontweight="bold", fontsize=11, color=STYLE_TEXT)
    ax.tick_params(colors=STYLE_TEXT)
    ax.spines[:].set_color(STYLE_GRID)
    ax.grid(axis="y", linestyle="--", alpha=0.7, color=STYLE_GRID)
    ax.legend(fontsize=11, bbox_to_anchor=(1.01, 1), loc="upper left",
              facecolor=STYLE_AX, edgecolor=STYLE_GRID, labelcolor=STYLE_TEXT)
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_global_bar.png")
    fig.savefig(path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Global bar chart → {path}")


def plot_coer_bar(results: dict, models: list, out_dir: str):
    valid = [m for m in models if results.get(m)]
    if not valid: return

    cats   = [c for c, _ in COER_COLS
              if any(results[m]["coer"].get(c) is not None for m in valid)]
    
    # Sort categories from 'less to more' based on average error rate
    def cat_avg_error(c):
        vals = [results[m]["coer"].get(c, 0) for m in valid]
        return sum(vals) / len(vals) if vals else 0
        
    cats.sort(key=cat_avg_error)
    
    labels = [dict(COER_COLS)[c] for c in cats]
    colors = plt.get_cmap("tab10").colors
    x      = np.arange(len(cats))
    width  = 0.8 / max(len(valid), 1)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(STYLE_BG)
    ax.set_facecolor(STYLE_AX)
    for i, m in enumerate(valid):
        vals   = [results[m]["coer"].get(c, 0) * 100 for c in cats]
        offset = (i - len(valid) / 2 + 0.5) * width
        rects  = ax.bar(x + offset, vals, width, label=m,
                        color=colors[i % len(colors)])
        for rect in rects:
            h = rect.get_height()
            if h > 0.5:
                ax.annotate(f"{h:.1f}",
                            xy=(rect.get_x() + rect.get_width()/2, h),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7,
                            fontweight="bold", rotation=90, color=STYLE_TEXT)

    ax.set_ylabel("Error Rate (%)", fontweight="bold", color=STYLE_TEXT)
    ax.set_title("Component-Level Error Rate (CoER)",
                 fontweight="bold", fontsize=14, pad=16, color=STYLE_TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right",
                       fontweight="bold", color=STYLE_TEXT)
    ax.tick_params(colors=STYLE_TEXT)
    ax.spines[:].set_color(STYLE_GRID)
    ax.grid(axis="y", linestyle="--", alpha=0.6, color=STYLE_GRID)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left",
              facecolor=STYLE_AX, edgecolor=STYLE_GRID,
              labelcolor=STYLE_TEXT)
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_coer_bar.png")
    fig.savefig(path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ CoER bar chart  → {path}")


def plot_demo_grid(predictions: dict, test_pairs: list,
                   models: list, out_dir: str, out_filename: str = "plot_demo_grid.png"):
    """Pick a random sample, show image + GT/Pred alignment per model."""
    if not test_pairs: return
    valid = [m for m in models if m in predictions]
    if not valid: return

    candidates = [(img, gt) for img, gt in test_pairs
                  if any(predictions[m].get(img, gt) != gt for m in valid)]
    demo_pair = random.choice(candidates) if candidates else random.choice(test_pairs)
    img_path, gt = demo_pair

    try:
        pil_img = _PILImage.open(img_path)
    except Exception:
        return

    fp_sm    = (fm.FontProperties(fname=LAO_FONT_PATH, size=11)
                if os.path.exists(LAO_FONT_PATH)
                else fm.FontProperties(family="monospace", size=11))
    op_colors = {"mat": "#2ca02c", "sub": "#d62728",
                 "del": "#e08000", "ins": "#9467bd"}
    op_labels = {"mat": "MATCH",  "sub": "SUB", "del": "DEL", "ins": "INS"}
    MAX_COLS = 20
    CELL_W   = 14.0 / MAX_COLS
    CHAR_H   = 0.52
    ARROW_H  = 0.38
    LABEL_H  = 0.30

    img_w_px, img_h_px = pil_img.size
    IMG_H   = max(14.0 * (img_h_px / max(img_w_px, 1)), 0.8)
    panel_h = 0.40 + IMG_H + 0.10 + CHAR_H + ARROW_H + CHAR_H + LABEL_H + 0.15

    fig, axs = plt.subplots(len(valid), 1,
                            figsize=(14, panel_h * len(valid) + 0.5))
    fig.patch.set_facecolor(STYLE_BG)
    if len(valid) == 1:
        axs = [axs]
    fig.suptitle("Tesseract Random Sample Comparison",
                 fontsize=14, fontweight="bold", y=1.0, color=STYLE_TEXT)

    for i, model_name in enumerate(valid):
        pred_str = predictions[model_name].get(img_path, "")
        edits    = align_strings(pred_str, gt)[:MAX_COLS]
        total_h  = panel_h
        ax = axs[i]
        ax.set_facecolor(STYLE_BG)
        ax.set_xlim(0, 14); ax.set_ylim(0, total_h); ax.axis("off")
        cy = total_h - 0.05
        ax.text(0.15, cy,
                f"Model: {model_name}   |   GT: {gt}   |   Pred: {pred_str}",
                fontsize=10, fontweight="bold", va="top",
                color=STYLE_TEXT, clip_on=False)
        cy -= 0.38
        inset = ax.inset_axes([0.0, (cy - IMG_H) / total_h, 1.0,
                               IMG_H / total_h])
        inset.imshow(pil_img, cmap="gray", aspect="auto",
                     extent=[0, img_w_px, 0, img_h_px])
        inset.set_xlim(0, img_w_px); inset.set_ylim(0, img_h_px)
        inset.axis("off")
        cy -= IMG_H + 0.10
        TRUTH_Y = cy - CHAR_H / 2
        ARROW_Y = cy - CHAR_H - ARROW_H / 2
        PRED_Y  = cy - CHAR_H - ARROW_H - CHAR_H / 2
        LBL_Y   = cy - CHAR_H - ARROW_H - CHAR_H - LABEL_H / 2
        ax.text(-0.1, TRUTH_Y, "GT:",   fontsize=8, fontweight="bold",
                ha="right", va="center", clip_on=False, color=STYLE_TEXT)
        ax.text(-0.1, PRED_Y,  "Pred:", fontsize=8, fontweight="bold",
                ha="right", va="center", clip_on=False, color=STYLE_TEXT)
        for col, (op, p_c, t_c) in enumerate(edits):
            color = op_colors.get(op, "#888888")
            cx = col * CELL_W + CELL_W / 2
            if t_c:
                ax.add_patch(FancyBboxPatch(
                    (cx - CELL_W*0.44, TRUTH_Y - CHAR_H*0.5),
                    CELL_W*0.88, CHAR_H, boxstyle="round,pad=0.03",
                    fc="white", ec=color, lw=1.8, clip_on=False))
                ax.text(cx, TRUTH_Y, t_c, fontproperties=fp_sm,
                        ha="center", va="center", clip_on=False)
            ax.text(cx, ARROW_Y, "↓" if op != "ins" else "↑",
                    fontsize=10, ha="center", va="center",
                    color=color, fontweight="bold", clip_on=False)
            if p_c:
                ax.add_patch(FancyBboxPatch(
                    (cx - CELL_W*0.44, PRED_Y - CHAR_H*0.5),
                    CELL_W*0.88, CHAR_H, boxstyle="round,pad=0.03",
                    fc="white", ec=color, lw=1.8, clip_on=False))
                ax.text(cx, PRED_Y, p_c, fontproperties=fp_sm,
                        ha="center", va="center", clip_on=False)
            ax.text(cx, LBL_Y, op_labels.get(op, op),
                    fontsize=6, ha="center", va="center",
                    color=color, fontweight="bold", clip_on=False)

    plt.tight_layout(pad=0.6, h_pad=0.8)
    path = os.path.join(out_dir, out_filename)
    fig.savefig(path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Demo grid       → {path}")


def plot_worst_coer_grid(predictions: dict, test_pairs: list, results: dict,
                         models: list, out_dir: str):
    """Find the category with highest average error and show its hardest sample."""
    import matplotlib.font_manager as fm
    from matplotlib.patches import FancyBboxPatch

    valid = [m for m in models if m in predictions and results.get(m)]
    if not valid: return

    # 1. Identify worst category
    avg_errors = {}
    for cat, _ in COER_COLS:
        vals = [results[m]["coer"].get(cat, 0.0) for m in valid]
        if vals:
            avg_errors[cat] = sum(vals) / len(vals)

    if not avg_errors: return
    worst_cat = max(avg_errors, key=avg_errors.get)
    worst_cat_name = dict(COER_COLS).get(worst_cat, worst_cat)

    # 2. Find samples containing that category
    target_chars = LAO_COMPONENT_MAP.get(worst_cat, [])
    candidates = [(img, gt) for img, gt in test_pairs 
                  if any(c in gt for c in target_chars)]
    
    if not candidates: return
    
    # Pick the "hardest" candidate (highest error across models)
    sample_scores = []
    for img, gt in candidates:
        err_sum = 0
        for m in valid:
            pred = predictions[m].get(img, "")
            err_sum += edit_distance(pred, gt)
        sample_scores.append((img, gt, err_sum))
    
    sample_scores.sort(key=lambda x: x[2], reverse=True)
    img_path, gt, _ = sample_scores[0]

    try:
        pil_img = _PILImage.open(img_path)
    except Exception:
        return

    fp_sm = (fm.FontProperties(fname=LAO_FONT_PATH, size=11)
             if os.path.exists(LAO_FONT_PATH) else fm.FontProperties(family="monospace", size=11))
    
    op_colors = {"mat": "#2ca02c", "sub": "#d62728", "del": "#e08000", "ins": "#9467bd"}
    op_labels = {"mat": "MATCH",  "sub": "SUB", "del": "DEL", "ins": "INS"}
    MAX_COLS = 20
    CELL_W   = 14.0 / MAX_COLS
    CHAR_H   = 0.52
    ARROW_H  = 0.38
    LABEL_H  = 0.30

    img_w_px, img_h_px = pil_img.size
    IMG_H = max(14.0 * (img_h_px / max(img_w_px, 1)), 0.8)
    panel_h = 0.40 + IMG_H + 0.10 + CHAR_H + ARROW_H + CHAR_H + LABEL_H + 0.15

    fig, axs = plt.subplots(len(valid), 1, figsize=(14, panel_h * len(valid) + 0.7))
    fig.patch.set_facecolor(STYLE_BG)
    if len(valid) == 1: axs = [axs]
    fig.suptitle(f"Worst Category Analysis: {worst_cat_name}", 
                 fontsize=16, fontweight="bold", y=0.99, color=STYLE_TEXT)

    for i, model_name in enumerate(valid):
        pred_str = predictions[model_name].get(img_path, "")
        edits    = align_strings(pred_str, gt)[:MAX_COLS]
        ax = axs[i]; ax.set_facecolor(STYLE_BG); ax.axis("off")
        ax.set_xlim(0, 14); ax.set_ylim(0, panel_h)
        cy = panel_h - 0.05
        ax.text(0.15, cy, f"Model: {model_name}   |   GT: {gt}   |   Pred: {pred_str}",
                fontsize=10, fontweight="bold", va="top", color=STYLE_TEXT)
        cy -= 0.38
        inset = ax.inset_axes([0.0, (cy - IMG_H)/panel_h, 1.0, IMG_H/panel_h])
        inset.imshow(pil_img, cmap="gray", aspect="auto")
        inset.axis("off")
        cy -= IMG_H + 0.10
        
        TRUTH_Y, ARROW_Y, PRED_Y, LBL_Y = cy-0.26, cy-0.64, cy-1.02, cy-1.32
        for col, (op, p_c, t_c) in enumerate(edits):
            color = op_colors.get(op, "#888888")
            cx = col * CELL_W + CELL_W / 2
            if t_c:
                ax.add_patch(FancyBboxPatch((cx-CELL_W*0.44, TRUTH_Y-CHAR_H*0.5), CELL_W*0.88, CHAR_H, 
                                            boxstyle="round,pad=0.03", fc="white", ec=color, lw=1.8))
                ax.text(cx, TRUTH_Y, t_c, fontproperties=fp_sm, ha="center", va="center")
            ax.text(cx, ARROW_Y, "↓" if op != "ins" else "↑", fontsize=10, ha="center", va="center", color=color, fontweight="bold")
            if p_c:
                ax.add_patch(FancyBboxPatch((cx-CELL_W*0.44, PRED_Y-CHAR_H*0.5), CELL_W*0.88, CHAR_H, 
                                            boxstyle="round,pad=0.03", fc="white", ec=color, lw=1.8))
                ax.text(cx, PRED_Y, p_c, fontproperties=fp_sm, ha="center", va="center")
            ax.text(cx, LBL_Y, op_labels.get(op, op), fontsize=6, ha="center", va="center", color=color, fontweight="bold")

    plt.tight_layout(pad=0.8)
    path = os.path.join(out_dir, "plot_worst_coer_grid.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Worst category grid → {path}")


def plot_metric_explanation(predictions: dict, test_pairs: list,
                            models: list, out_dir: str):
    """One figure per (model, sample), saved to  out_dir/<model>/sample_NN_<stem>.png

    Output layout:
        out_dir/
          <ModelA>/
            sample_01_<stem>.png
            sample_02_<stem>.png
            ...
          <ModelB>/
            sample_01_<stem>.png
            ...

    All models see the SAME GT samples (test_pairs is shared).
    """
    valid = [m for m in models if m in predictions]
    if not valid or not test_pairs:
        return

    # ── Register & activate the Lao font ─────────────────────────────────────
    _lao_font_path = (
        LAO_FONT_PATH
        if os.path.exists(LAO_FONT_PATH)
        else str(Path(__file__).parent / LAO_FONT_PATH)
    )
    _lao_family = None
    if os.path.exists(_lao_font_path):
        fm.fontManager.addfont(_lao_font_path)
        fm._fmcache = None
        _prop = fm.FontProperties(fname=_lao_font_path)
        _lao_family = _prop.get_name()   # e.g. "Phetsarath OT"

    _prev_family = plt.rcParams.get("font.family", ["sans-serif"])
    _prev_sans   = list(plt.rcParams.get("font.sans-serif", []))
    if _lao_family:
        plt.rcParams["font.family"]     = "sans-serif"
        plt.rcParams["font.sans-serif"] = [_lao_family] + _prev_sans

    lao_font_lg = fm.FontProperties(fname=_lao_font_path, size=16) \
                  if os.path.exists(_lao_font_path) \
                  else fm.FontProperties(size=16)
    lao_font_sm = fm.FontProperties(fname=_lao_font_path, size=13) \
                  if os.path.exists(_lao_font_path) \
                  else fm.FontProperties(size=13)

    cat_colors = {
        "consonant":    "#1f77b4", "tone":        "#d62728",
        "before_vowel": "#2ca02c", "rear_vowel":  "#9467bd",
        "upper_vowel":  "#ff7f0e", "lower_vowel": "#8c564b",
        "symbol":       "#7f7f7f", "lao_number":  "#17becf",
        "digit":        "#bcbd22", "special":     "#e377c2",
        "other":        "#aaaaaa",
    }

    MAX_COLS   = 14
    CELL_W     = 14.0 / MAX_COLS
    CHAR_H     = 0.55
    LABEL_H    = 0.45
    ROW_GAP    = 0.20
    EDIT_ROW_H = 2.0
    HDR_H      = 0.45
    SEP_H      = 0.30
    FORMULA_H  = 0.65

    n_leg_rows     = (len(cat_colors) - 1) // 5 + 1
    legend_block_h = 0.44 + n_leg_rows * 0.30

    # ── Shared GT samples (same order for every model) ────────────────────────
    _shared = list(test_pairs)
    if not _shared:
        print("  [Metric Explanation] No test pairs available.")
        return

    op_c = {"mat": "#2ca02c", "sub": "#d62728",
             "del": "#e08000", "ins": "#9467bd"}
    op_l = {"mat": "MATCH",   "sub": "SUB",
             "del": "DEL",    "ins": "INS"}

    # ── Outer loop: model → inner loop: sample ────────────────────────────────
    for demo_m in valid:
        # Create a subfolder for this model
        model_dir = os.path.join(out_dir, demo_m)
        os.makedirs(model_dir, exist_ok=True)

        for s_idx, (img_f, demo_truth) in enumerate(_shared):
            demo_pred = predictions[demo_m].get(img_f, demo_truth)

            # ── Figure height for this single sample ──────────────────────────
            coer_chars = list(demo_truth)          # all chars, no truncation
            coer_rows  = [coer_chars[i:i + MAX_COLS]
                          for i in range(0, len(coer_chars), MAX_COLS)]
            edits      = align_strings(demo_pred, demo_truth)
            edit_rows  = [edits[i:i + MAX_COLS]    # all edits, no truncation
                          for i in range(0, len(edits), MAX_COLS)]

            # If the GT image exists, read its aspect ratio to allocate space
            img_h_alloc = 0.0
            pil_img = None
            if os.path.exists(img_f) and not img_f.endswith(".gt.txt"):
                try:
                    pil_img = plt.imread(img_f)
                    h, w = pil_img.shape[:2]
                    img_w_alloc = 8.0
                    img_h_alloc = (h / w) * img_w_alloc
                    if img_h_alloc > 3.0:
                        img_h_alloc = 3.0
                    img_h_alloc += 0.8  # Add space for GT Image header
                except Exception:
                    pass

            n_coer_rows = len(coer_rows)
            n_edit_rows = len(edit_rows)
            fig_h = (HDR_H * 2
                     + img_h_alloc
                     + n_coer_rows * (CHAR_H + LABEL_H + ROW_GAP)
                     + 0.10 + legend_block_h + SEP_H
                     + HDR_H + n_edit_rows * EDIT_ROW_H
                     + FORMULA_H + 0.5)

            try:
                fig, ax = plt.subplots(1, 1, figsize=(14, max(fig_h, 4)))
                fig.patch.set_facecolor(STYLE_BG)
                ax.set_facecolor(STYLE_BG)
                ax.axis("off")
                ax.set_xlim(0, 14)
                ax.set_ylim(0, fig_h)

                # Title: model name + sample index
                stem = os.path.splitext(os.path.basename(img_f))[0]
                ax.set_title(
                    f"Model: {demo_m}   |   Sample {s_idx + 1}: {stem}",
                    fontsize=11, fontweight="bold",
                    loc="center", pad=6, color="#444444",
                )
                cy = fig_h - 0.15

                # ── Ground Truth Image ───────────────────────────────────────
                if pil_img is not None:
                    ax.text(0.15, cy, "Ground Truth Image", fontsize=13, fontweight="bold",
                            va="top", color="#222222", clip_on=False)
                    cy -= 0.5
                    
                    # Draw the image centered
                    img_w_in_plot = 8.0
                    h, w = pil_img.shape[:2]
                    img_h_in_plot = (h / w) * img_w_in_plot
                    if img_h_in_plot > 3.0:
                        img_h_in_plot = 3.0
                        img_w_in_plot = (w / h) * img_h_in_plot
                        
                    left = (14.0 - img_w_in_plot) / 2
                    right = left + img_w_in_plot
                    bottom = cy - img_h_in_plot
                    top = cy
                    
                    cmap = "gray" if len(pil_img.shape) == 2 else None
                    ax.imshow(pil_img, extent=[left, right, bottom, top], cmap=cmap, aspect="auto")
                    
                    cy = bottom - 0.3

                # ── CoER section ──────────────────────────────────────────────
                lbl = f'CoER  —  GT = "{demo_truth}"'
                ax.text(0.15, cy, lbl, fontproperties=lao_font_lg,
                        fontsize=13, fontweight="bold",
                        va="top", color="#222222", clip_on=False)
                cy -= HDR_H
                
                lbl_pred = f'Pred = "{demo_pred}"'
                ax.text(0.15, cy, lbl_pred, fontproperties=lao_font_lg,
                        fontsize=13, fontweight="bold",
                        va="top", color="#222222", clip_on=False)
                cy -= HDR_H

                for row in coer_rows:
                    char_cy  = cy - CHAR_H / 2
                    label_cy = cy - CHAR_H - LABEL_H / 2
                    for col, ch in enumerate(row):
                        cat   = get_component_type(ch)
                        color = cat_colors.get(cat, "#aaaaaa")
                        cx    = col * CELL_W + CELL_W / 2
                        ax.add_patch(FancyBboxPatch(
                            (cx - CELL_W * 0.42, char_cy - CHAR_H * 0.5),
                            CELL_W * 0.84, CHAR_H,
                            boxstyle="round,pad=0.04",
                            fc=color + "22", ec=color, lw=1.8, clip_on=False))
                        ax.text(cx, char_cy, ch, fontproperties=lao_font_lg,
                                ha="center", va="center",
                                color="#111111", clip_on=False)
                        ax.text(cx, label_cy, cat.replace("_", "\n"),
                                fontsize=6.5, ha="center", va="center",
                                color=color, fontweight="bold",
                                linespacing=1.15, clip_on=False)
                    cy -= CHAR_H + LABEL_H + ROW_GAP

                # ── Colour legend (always shown, once per figure) ─────────────
                cy -= 0.05
                ax.plot([0.1, 13.9], [cy, cy], color="#cccccc", lw=0.8)
                cy -= 0.22
                for li, (cat, color) in enumerate(cat_colors.items()):
                    lx = (li % 5) * (14.0 / 5) + 0.25
                    ly = cy - (li // 5) * 0.30
                    ax.add_patch(FancyBboxPatch(
                        (lx - 0.18, ly - 0.09), 0.18, 0.18,
                        boxstyle="round,pad=0.02",
                        fc=color + "44", ec=color, lw=1, clip_on=False))
                    ax.text(lx + 0.10, ly, cat.replace("_", " "),
                            fontsize=7.5, va="center",
                            color=color, fontweight="bold")
                cy -= n_leg_rows * 0.30 + 0.22

                # ── CER / WER alignment section ───────────────────────────────
                ax.plot([0.1, 13.9], [cy, cy], color="#cccccc", lw=0.8)
                cy -= SEP_H

                lbl2 = "CER & WER Alignment"
                ax.text(0.15, cy, lbl2, fontsize=13, fontweight="bold",
                        va="top", color="#222222", clip_on=False)
                cy -= HDR_H

                for row_edits in edit_rows:
                    TRUTH_Y = cy - 0.40
                    ARROW_Y = cy - 0.90
                    PRED_Y  = cy - 1.40
                    OPLBL_Y = cy - 1.78
                    ax.text(-0.05, TRUTH_Y, "GT:",   fontsize=9,
                            fontweight="bold", ha="right", va="center",
                            clip_on=False, color=STYLE_TEXT)
                    ax.text(-0.05, PRED_Y,  "Pred:", fontsize=9,
                            fontweight="bold", ha="right", va="center",
                            clip_on=False, color=STYLE_TEXT)
                    for col, (op, p_c, t_c) in enumerate(row_edits):
                        color = op_c.get(op, "#888888")
                        cx = col * CELL_W + CELL_W / 2
                        if t_c:
                            ax.add_patch(FancyBboxPatch(
                                (cx - CELL_W * 0.42, TRUTH_Y - 0.28),
                                CELL_W * 0.84, 0.56,
                                boxstyle="round,pad=0.04",
                                fc="white", ec=color, lw=2, clip_on=False))
                            ax.text(cx, TRUTH_Y, t_c,
                                    fontproperties=lao_font_sm,
                                    ha="center", va="center", clip_on=False)
                        ax.text(cx, ARROW_Y,
                                "↓" if op != "ins" else "↑",
                                fontsize=13, ha="center", va="center",
                                color=color, fontweight="bold", clip_on=False)
                        if p_c:
                            ax.add_patch(FancyBboxPatch(
                                (cx - CELL_W * 0.42, PRED_Y - 0.28),
                                CELL_W * 0.84, 0.56,
                                boxstyle="round,pad=0.04",
                                fc="white", ec=color, lw=2, clip_on=False))
                            ax.text(cx, PRED_Y, p_c,
                                    fontproperties=lao_font_sm,
                                    ha="center", va="center", clip_on=False)
                        ax.text(cx, OPLBL_Y, op_l.get(op, op),
                                fontsize=7, ha="center", va="top",
                                color=color, fontweight="bold", clip_on=False)
                    cy -= EDIT_ROW_H

                # ── Metric formula box ────────────────────────────────────────
                cer_val = edit_distance(demo_pred, demo_truth) \
                          / max(len(demo_truth), 1)
                wer_val = (edit_distance(demo_pred.split(),
                                         demo_truth.split())
                           / max(len(demo_truth.split()), 1))
                formula = (f"Results:  CER = {cer_val*100:.1f}%"
                           f"  |  WER = {wer_val*100:.1f}%\n"
                           "Formula:  CER = (Sub + Del + Ins) / Total GT Chars")
                ax.text(0.2, cy - 0.05, formula,
                        fontsize=9, style="italic", va="top",
                        bbox=dict(fc="#fffde7", ec="#f0c000",
                                  boxstyle="round,pad=0.5", lw=1.2),
                        clip_on=False)

                # ── Save ──────────────────────────────────────────────────────
                plt.subplots_adjust(left=0.06, right=0.98,
                                    top=0.97, bottom=0.01)
                out_fname = f"sample_{s_idx + 1:02d}_{stem}.png"
                expl_path = os.path.join(model_dir, out_fname)
                fig.savefig(expl_path, dpi=200, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
                plt.close(fig)
                print(f"  ✓ [{demo_m}] sample {s_idx + 1:02d} → {expl_path}")

            except Exception as _me:
                plt.close("all")
                print(f"  [Error — {demo_m} sample {s_idx + 1}] {_me}")

        print(f"  → {demo_m}/  ({len(_shared)} figures)")
    print()

    # ── Restore original rcParams ─────────────────────────────────────────────
    plt.rcParams["font.family"]     = _prev_family
    plt.rcParams["font.sans-serif"] = _prev_sans


# ══════════════════════════════════════════════════════════════════════════════
# §11  STANDALONE METRIC EXPLANATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_metric_explanation(
    out_dir: str = COMPARE_DIR,
    tessdata_dir: str = CUSTOM_TESSDATA_DIR,
    image_folder: str = IMAGE_FOLDER,
    custom_gt_file: str = None,
    num_samples: int = 5,
    min_length: int = 3,
    max_length: int = None,
    mode: str = "metric_explanation",
    model_names: list = None,
):
    """Generate metric-explanation diagrams (CoER + CER/WER alignment).

    Two modes:
      1. Default  – pick ``num_samples`` (default 5) from IMAGE_FOLDER,
                   run all available Tesseract models on them.
      2. Custom   – pass a .gt.txt file via --custom-gt=<path>;
                   the text is used as the GT and the model's OCR output
                   is compared against it (if a matching image exists), or
                   the text itself is used as both GT and prediction to
                   visualise the pure CoER component breakdown.
    """
    # Resolve the effective model list (CLI override or module default)
    _active_models = model_names if model_names else CUSTOM_MODEL_NAMES

    script_dir = Path(__file__).parent.resolve()
    out_path   = script_dir / out_dir
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Metric Explanation Diagrams")
    print("=" * 70)
    print(f"  Active models ({len(_active_models)}): {', '.join(_active_models)}")

    # ── Build test_pairs and predictions ─────────────────────────────────────
    if custom_gt_file:
        gt_path = Path(custom_gt_file).expanduser().resolve()
        if not gt_path.exists():
            print(f"  ERROR: --custom-gt file not found: {gt_path}")
            return

        gt_text = gt_path.read_text(encoding="utf-8").strip()
        if not gt_text:
            print(f"  ERROR: --custom-gt file is empty: {gt_path}")
            return

        print(f"  Custom GT file : {gt_path}")
        print(f"  GT text        : {repr(gt_text[:80])}{'…' if len(gt_text)>80 else ''}")

        # Look for a paired image (same stem, any image ext)
        paired_image = None
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            candidate = gt_path.with_suffix(ext)
            if candidate.exists():
                paired_image = str(candidate)
                break

        # Fake image key (used as dict key; None if no image found)
        img_key = paired_image if paired_image else str(gt_path)
        test_pairs = [(img_key, gt_text)]

        # Build predictions: if we have a real image + tessdata → run OCR;
        # otherwise use GT itself so the diagram still shows the CoER breakdown.
        predictions: dict = {}
        
        available_tess = []
        if paired_image and HAS_TESSERACT:
            available_tess = [
                m for m in _active_models
                if (script_dir / tessdata_dir / f"{m}.traineddata").exists()
            ]
        
        available_pt = []
        if paired_image and HAS_PYTORCH:
            for m in _active_models:
                pt_path = script_dir / "data" / m / f"{m}_final.pt"
                if not pt_path.exists():
                    pt_path = script_dir / "data" / m / f"{m}_best.pt"
                if pt_path.exists():
                    available_pt.append((m, pt_path))

        total_models = len(available_tess) + len(available_pt)
        if total_models > 0:
            print(f"  Running OCR with {total_models} model(s) on paired image…")
            for mname in available_tess:
                pred_text = run_tesseract(paired_image, mname, tessdata_dir=str(script_dir / tessdata_dir))
                predictions[f"{mname}_Tess"] = {img_key: pred_text}
                print(f"    [{mname}_Tess] pred: {repr(pred_text[:60])}")
            
            if HAS_PYTORCH and available_pt:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                for mname, pt_path in available_pt:
                    model, charset, th = load_pt_model(str(pt_path), device)
                    if model:
                        pred_text = run_pytorch(paired_image, model, charset, th, device)
                        predictions[f"{mname}_PT"] = {img_key: pred_text}
                        print(f"    [{mname}_PT] pred: {repr(pred_text[:60])}")
        else:
            if not paired_image:
                print("  No paired image found – using GT as prediction "
                      "(CoER breakdown only).")
            else:
                print("  No compiled models found – using GT as prediction "
                      "(CoER breakdown only).")
            predictions["GT-only"] = {img_key: gt_text}

    else:
        # ── Default mode: randomly pick num_samples from IMAGE_FOLDER ─────────
        img_folder = str(script_dir / image_folder)
        print(f"  Image folder : {img_folder}")
        all_pairs = find_test_pairs(img_folder)
        if not all_pairs:
            print(f"  ERROR: No image+.gt.txt pairs found in {img_folder}")
            print("  Tip: pass --custom-gt=<file.gt.txt> to use a specific text.")
            return
        print(f"  Found {len(all_pairs)} total test pairs.")

        # Filter by length: min_length lower-bound, optional max_length upper-bound
        if max_length is not None:
            candidates = [(img, gt) for img, gt in all_pairs
                          if min_length <= len(gt) <= max_length]
        else:
            candidates = [(img, gt) for img, gt in all_pairs
                          if min_length <= len(gt)]
        if not candidates:
            candidates = list(all_pairs)
        random.shuffle(candidates)
        test_pairs = candidates[:num_samples]

        print(f"  Selected {len(test_pairs)} sample(s) (of {len(all_pairs)}) for diagrams:")
        for img_p, gt in test_pairs:
            print(f"    {os.path.basename(img_p)!r:40s}  GT: {repr(gt)}")

        # ── Run ALL available Tesseract and PyTorch models on selected samples ──
        predictions = {}
        
        available_tess = [
            m for m in _active_models
            if (script_dir / tessdata_dir / f"{m}.traineddata").exists()
        ]
        
        available_pt = []
        if HAS_PYTORCH:
            for m in _active_models:
                pt_path = script_dir / "data" / m / f"{m}_final.pt"
                if not pt_path.exists():
                    pt_path = script_dir / "data" / m / f"{m}_best.pt"
                if pt_path.exists():
                    available_pt.append((m, pt_path))

        total_models = len(available_tess) + len(available_pt)
        if total_models == 0:
            print("  No compiled models found – using GT as prediction "
                  "(CoER breakdown only).")
            predictions["GT-only"] = {img: gt for img, gt in test_pairs}
        else:
            print(f"  Running {total_models} model(s) on {len(test_pairs)} sample(s)…")
            
            for mname in available_tess:
                print(f"    [{mname}_Tess]", end="", flush=True)
                pred_map = {}
                for img_path, _ in test_pairs:
                    pred_map[img_path] = run_tesseract(
                        img_path, mname,
                        tessdata_dir=str(script_dir / tessdata_dir))
                predictions[f"{mname}_Tess"] = pred_map
                print(" ✓")
                
            if HAS_PYTORCH and available_pt:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                for mname, pt_path in available_pt:
                    print(f"    [{mname}_PT]", end="", flush=True)
                    model, charset, th = load_pt_model(str(pt_path), device)
                    pred_map = {}
                    if model:
                        for img_path, _ in test_pairs:
                            pred_map[img_path] = run_pytorch(img_path, model, charset, th, device)
                    predictions[f"{mname}_PT"] = pred_map
                    print(" ✓")

    # ── Generate diagrams ─────────────────────────────────────────────────────
    models = list(predictions.keys())
    print(f"\n  Generating diagrams for: {models}")
    
    if mode == "inspection":
        out_path = out_path / "inspection"
        out_path.mkdir(parents=True, exist_ok=True)
        print(f"  Output dir : {out_path}\n")
        for idx, (img, gt) in enumerate(test_pairs):
            stem = Path(img).stem
            out_name = f"sample_{idx+1:02d}_{stem}.png"
            plot_demo_grid(predictions, [(img, gt)], models, str(out_path), out_filename=out_name)
    else:
        print(f"  Output dir : {out_path}\n")
        plot_metric_explanation(predictions, test_pairs, models, str(out_path))
        
    print(f"✓ Done.  Diagrams saved to: {out_path}/")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# §12 COMPARISON PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_compare(out_dir=COMPARE_DIR, tessdata_dir=CUSTOM_TESSDATA_DIR, logs_dir="data", skip_plots=False, model_names=None, realworld_dir=None):
    """Main entry for the model comparison pipeline."""
    script_dir = Path(__file__).parent.resolve()
    active_models = model_names if model_names else CUSTOM_MODEL_NAMES
    print("=" * 70)
    print("  Tesseract Model Comparison")
    print("=" * 70)
    print(f"  Models       : {', '.join(active_models)}")
    print(f"  Tessdata dir : {tessdata_dir}")
    print(f"  Image folder : {IMAGE_FOLDER}")
    if realworld_dir:
        print(f"  Real-world   : {realworld_dir}")
    print(f"  Output dir   : {out_dir}")
    print()

    if not HAS_TESSERACT:
        print("  ERROR: pytesseract / Pillow not installed.")
        print("  Run:  pip install pytesseract pillow")
        return

    os.makedirs(out_dir, exist_ok=True)

    # Validate tessdata
    missing = [m for m in active_models
               if not os.path.exists(
                   os.path.join(tessdata_dir, f"{m}.traineddata"))]
    if missing:
        print(f"  [Warning] Missing .traineddata: {missing}")

    # Find test pairs
    test_pairs = find_test_pairs(IMAGE_FOLDER)
    if not test_pairs:
        print(f"  ERROR: No image+.gt.txt pairs found in {IMAGE_FOLDER}")
        return
    print(f"  Found {len(test_pairs)} test pairs.\n")

    realworld_pairs = []
    if realworld_dir:
        if os.path.exists(realworld_dir):
            realworld_pairs = find_test_pairs(realworld_dir)
            if not realworld_pairs:
                print(f"  [Warning] No image+.gt.txt pairs found in real-world dir: {realworld_dir}")
            else:
                print(f"  Found {len(realworld_pairs)} real-world test pairs.\n")
        else:
            print(f"  [Warning] Real-world directory does not exist: {realworld_dir}")

    predictions = {}
    inference_times = {}
    realworld_predictions = {}
    realworld_inference_times = {}

    for model_name in active_models:
        td_path = os.path.join(tessdata_dir, f"{model_name}.traineddata")
        if not os.path.exists(td_path):
            print(f"  [Skip] {model_name} — .traineddata not found.")
            continue
        print(f"  Running {model_name} on {len(test_pairs)} images (Test)...")
        pred_map = {}
        t0 = time.time()
        for idx, (img_path, _) in enumerate(test_pairs):
            pred_map[img_path] = run_tesseract(img_path, model_name, tessdata_dir=tessdata_dir)
            if (idx + 1) % 100 == 0:
                print(f"    {idx+1}/{len(test_pairs)}  ({time.time()-t0:.1f}s)")
        inf_time = time.time() - t0
        predictions[model_name] = pred_map
        inference_times[model_name] = inf_time
        print(f"    Done {model_name} in {inf_time:.1f}s")

        if realworld_pairs:
            print(f"  Running {model_name} on {len(realworld_pairs)} images (Real-world)...")
            rw_pred_map = {}
            t0 = time.time()
            for idx, (img_path, _) in enumerate(realworld_pairs):
                rw_pred_map[img_path] = run_tesseract(img_path, model_name, tessdata_dir=tessdata_dir)
                if (idx + 1) % 100 == 0:
                    print(f"    {idx+1}/{len(realworld_pairs)}  ({time.time()-t0:.1f}s)")
            rw_inf_time = time.time() - t0
            realworld_predictions[model_name] = rw_pred_map
            realworld_inference_times[model_name] = rw_inf_time
            print(f"    Done {model_name} (Real-world) in {rw_inf_time:.1f}s")

    if not predictions:
        print("  ERROR: No models produced predictions.")
        return

    # ── Collect Validation Time from validating.log ──
    eval_times: dict = {}
    print("  Collecting validation times from validating.log files...")
    for mname in list(predictions.keys()):
        # validating.log is in tessdata_dir / model_name / validating.log
        vlog_file = script_dir / tessdata_dir / mname / "validating.log"
        v_sec = parse_validating_log(str(vlog_file))
        if v_sec is not None:
            eval_times[mname] = v_sec
            mm, ss = divmod(int(v_sec), 60)
            hh, mm = divmod(mm, 60)
            print(f"    [{mname}] Eval Time: {hh:02d}:{mm:02d}:{ss:02d}")
        else:
            print(f"    [{mname}] no validating.log found at {vlog_file.relative_to(script_dir)}")

    # ── Collect total training time for each model from its training.log ──
    # Model names like 'lao48cnn16_final' → log at data/lao48cnn16/training.log
    _strip_suffixes = ("_final", "_fast", "_best")
    train_times: dict = {}
    print("  Collecting training times from training.log files...")
    for mname in list(predictions.keys()):
        base = mname
        for sfx in _strip_suffixes:
            if base.endswith(sfx):
                base = base[: -len(sfx)]
                break
        # Training logs are in logs_dir / model_name / training.log
        log_file = script_dir / logs_dir / base / "training.log"
        if log_file.exists():
            segs = parse_training_time_report(str(log_file))
            total_s = sum(s["wall_sec"] or 0.0 for s in segs)
            
            # Fallback to plain timestamp subtraction if no segments found
            if total_s <= 0:
                total_s = parse_plain_training_time(str(log_file))
                source_str = "timestamps"
            else:
                source_str = f"{len(segs)} segments"

            if total_s > 0:
                train_times[mname] = total_s
                mm, ss = divmod(int(total_s), 60)
                hh, mm = divmod(mm, 60)
                print(f"    [{mname}] {hh:02d}:{mm:02d}:{ss:02d}  ({source_str}, "
                      f"log: {log_file.relative_to(script_dir)})")
            else:
                print(f"    [{mname}] training.log found but no timing data recovered")
        else:
            print(f"    [{mname}] no training.log found at {log_file.relative_to(script_dir)}")
    print()

    # Metrics
    print("\n  Computing metrics (Test Split)...")
    results = compute_metrics(predictions, test_pairs, inference_times=inference_times)
    model_list = list(predictions.keys())

    realworld_results = None
    if realworld_pairs:
        print("\n  Computing metrics (Real-world Split)...")
        realworld_results = compute_metrics(realworld_predictions, realworld_pairs, inference_times=realworld_inference_times)

    print_summary(results, model_list, train_times=train_times, eval_times=eval_times, title="Tesseract Model Comparison Results (Test Split)")
    if realworld_results:
        print_summary(realworld_results, model_list, train_times=train_times, eval_times=eval_times, title=f"🌎 Real-world Testing Split (Custom Evaluation on: {realworld_dir})")

    # CoER table
    print("  ==== Component-Level Error Rate (CoER) - Test Split ====")
    col_names = [name for _, name in COER_COLS]
    header = f"  {'Model':<22} " + " ".join(f"{n[:10]:>11}" for n in col_names)
    print(header)
    for m in model_list:
        r = results.get(m)
        if not r: continue
        row = f"  {m:<22} "
        for cat, _ in COER_COLS:
            row += f"{r['coer'].get(cat, 0)*100:>11.2f}"
        print(row)
    print()

    if realworld_results:
        print(f"  ==== Component-Level Error Rate (CoER) - Real-world Test Set ({realworld_dir}) ====")
        print(header)
        for m in model_list:
            r = realworld_results.get(m)
            if not r: continue
            row = f"  {m:<22} "
            for cat, _ in COER_COLS:
                row += f"{r['coer'].get(cat, 0)*100:>11.2f}"
            print(row)
        print()

    # Top mistakes
    print("  ==== Top 3 Mistakes per Model (Test Split) ====")
    for m in model_list:
        r = results.get(m)
        if not r: continue
        errors = [(op, p, t, cnt)
                  for (op, p, t), cnt in r["mistake_ctr"].items()
                  if op != "mat"]
        errors.sort(key=lambda x: x[3], reverse=True)
        parts = []
        for op, p, t, cnt in errors[:3]:
            if op == "sub":  parts.append(f"'{t}'→'{p}' ({cnt})")
            elif op == "del": parts.append(f"Missed '{t}' ({cnt})")
            elif op == "ins": parts.append(f"Extra '{p}' ({cnt})")
        print(f"  [{m}] {', '.join(parts) or 'None'}")
    print()

    if realworld_results:
        print("  ==== Top 3 Mistakes per Model (Real-world Split) ====")
        for m in model_list:
            r = realworld_results.get(m)
            if not r: continue
            errors = [(op, p, t, cnt)
                      for (op, p, t), cnt in r["mistake_ctr"].items()
                      if op != "mat"]
            errors.sort(key=lambda x: x[3], reverse=True)
            parts = []
            for op, p, t, cnt in errors[:3]:
                if op == "sub":  parts.append(f"'{t}'→'{p}' ({cnt})")
                elif op == "del": parts.append(f"Missed '{t}' ({cnt})")
                elif op == "ins": parts.append(f"Extra '{p}' ({cnt})")
            print(f"  [{m}] {', '.join(parts) or 'None'}")
        print()

    # CSVs
    export_csv_report(results, model_list, out_dir, train_times=train_times, eval_times=eval_times, realworld_results=realworld_results)

    # Plots
    if not skip_plots:
        print("  Generating comparison plots...")
        plot_global_bar(results, model_list, out_dir)
        plot_coer_bar(results, model_list, out_dir)
        plot_demo_grid(predictions, test_pairs, model_list, out_dir)
        plot_worst_coer_grid(predictions, test_pairs, results, model_list, out_dir)
        plot_metric_explanation(predictions, test_pairs, model_list, out_dir)

        # Balance check: copy training plots into comparison dir
        print("  Consolidating plots for benchmark balance...")
        import shutil
        plots_src = Path(__file__).parent / PLOTS_DIR
        if plots_src.exists():
            for f in plots_src.glob("*.png"):
                dest = Path(out_dir) / f.name
                if not dest.exists():
                    shutil.copy2(f, out_dir)
                    print(f"    + {f.name}")
    else:
        print("  Skipping comparison plots (--no-plots).")

    print("=" * 70)
    print(f"  All outputs saved to: {out_dir}/")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# §12 TRAINING TIME REPORT
# ══════════════════════════════════════════════════════════════════════════════

from datetime import datetime as _dt

# Regex patterns for the timestamped log produced by the segmented trainer
_TS_RE      = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
_SEG_START  = re.compile(r"Training segment: up to iteration (\d+)/(\d+)")
_SEG_DONE   = re.compile(r"Segment complete: (\d+) iterations done")
_SNAP_RE    = re.compile(r"Snapshot saved:.*_iter(\d+)\.checkpoint")
_BCER_LINE  = re.compile(
    r"At iteration (\d+)/\d+/\d+,.*?BCER train=([\d.]+)%"
)
_EXIT_RE    = re.compile(r"Exit code: (\d+)")
_TS_FMT     = "%Y-%m-%d %H:%M:%S"


def _parse_ts(line):
    """Return datetime from a timestamped log line, or None."""
    m = _TS_RE.match(line)
    return _dt.strptime(m.group(1), _TS_FMT) if m else None


def parse_training_time_report(log_path):
    """
    Parse a training.log that was written by the segmented training script
    (with [YYYY-MM-DD HH:MM:SS] prefixes on every line).

    Returns a list of segment dicts:
      {
        'segment'    : int   – segment index (1-based)
        'iter_start' : int   – iteration at start of segment
        'iter_end'   : int   – iteration at end of segment
        'total_iters': int   – total planned iterations
        'start_ts'   : datetime
        'end_ts'     : datetime
        'wall_sec'   : float – elapsed wall-clock seconds
        'iters_done' : int   – iterations actually trained in this segment
        'iter_per_sec': float
        'best_bcer'  : float – lowest BCER seen in this segment
        'last_bcer'  : float – BCER at final iteration of the segment
        'exit_code'  : int
      }
    """
    segments   = []
    best_run   = []          # longest completed run across restarts
    current    = None
    prev_iter  = 0          # last iteration of the previous segment
    seg_idx    = 0
    last_bcer  = None       # most recent BCER in current segment
    best_bcer  = None
    seg_start_ts = None

    with open(log_path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip()
            ts   = _parse_ts(line)

            # ── Segment header ────────────────────────────────────────────
            m = _SEG_START.search(line)
            if m:
                iter_end    = int(m.group(1))
                total_iters = int(m.group(2))

                # ── Restart detection ─────────────────────────────────────
                # If the new segment targets an iteration ≤ what we've already
                # accumulated, the training script was re-launched from scratch
                # and its output was appended to the same log file.
                # Discard all previously collected segments so only this latest
                # run is reported (avoids bogus "20000→2000" entries).
                if iter_end <= prev_iter and segments:
                    # Save current run as best if it's the longest seen so far
                    if len(segments) >= len(best_run):
                        best_run = segments[:]
                    segments  = []
                    prev_iter = 0
                    seg_idx   = 0

                seg_idx    += 1
                current     = {
                    "segment"    : seg_idx,
                    "iter_start" : prev_iter,
                    "iter_end"   : iter_end,
                    "total_iters": total_iters,
                    "start_ts"   : ts,
                    "end_ts"     : None,
                    "wall_sec"   : None,
                    "iters_done" : None,
                    "iter_per_sec": None,
                    "best_bcer"  : None,
                    "last_bcer"  : None,
                    "exit_code"  : None,
                }
                seg_start_ts = ts
                last_bcer    = None
                best_bcer    = None
                continue

            if current is None:
                continue

            # ── BCER lines ────────────────────────────────────────────────
            m = _BCER_LINE.search(line)
            if m:
                bcer = float(m.group(2))
                last_bcer = bcer
                if best_bcer is None or bcer < best_bcer:
                    best_bcer = bcer
                continue

            # ── Exit code ─────────────────────────────────────────────────
            m = _EXIT_RE.search(line)
            if m:
                current["exit_code"] = int(m.group(1))
                continue

            # ── Segment complete ──────────────────────────────────────────
            m = _SEG_DONE.search(line)
            if m:
                end_ts  = ts
                start_ts = current["start_ts"]
                iters_done = current["iter_end"] - current["iter_start"]
                if start_ts and end_ts and end_ts > start_ts:
                    wall_sec = (end_ts - start_ts).total_seconds()
                else:
                    wall_sec = 0.0
                current["end_ts"]      = end_ts
                current["wall_sec"]    = wall_sec
                current["iters_done"]  = iters_done
                current["iter_per_sec"] = (
                    iters_done / wall_sec if wall_sec > 0 else 0.0
                )
                current["best_bcer"]   = best_bcer
                current["last_bcer"]   = last_bcer
                segments.append(current)
                prev_iter = current["iter_end"]
                current   = None

    # Return the longest run: the final one or a previously saved best_run
    return segments if len(segments) >= len(best_run) else best_run


def print_training_time_report(segments, model_name=""):
    """Print a formatted table of segment timing."""
    if not segments:
        print("  No segmented training data found in log.")
        return

    title = f"Training Time Report — {model_name}" if model_name else "Training Time Report"
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    hdr = (f"  {'Seg':>4}  {'Iters':>10}  {'Wall Time':>10}  "
           f"{'iter/s':>7}  {'Best BCER%':>11}  {'Last BCER%':>11}  {'OK':>4}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    total_wall  = 0.0
    total_iters = 0
    for s in segments:
        ws   = s["wall_sec"] or 0.0
        mm, ss = divmod(int(ws), 60)
        hh, mm = divmod(mm, 60)
        wt_str = f"{hh:02d}:{mm:02d}:{ss:02d}"
        total_wall  += ws
        total_iters += s["iters_done"] or 0
        best_s = f"{s['best_bcer']:.3f}" if s["best_bcer"] is not None else "  —"
        last_s = f"{s['last_bcer']:.3f}" if s["last_bcer"] is not None else "  —"
        ok_s   = "✓" if s["exit_code"] == 0 else ("✗" if s["exit_code"] is not None else "?")
        print(f"  {s['segment']:>4}  "
              f"{s['iter_start']:>5}→{s['iter_end']:<5}  "
              f"{wt_str:>10}  "
              f"{s['iter_per_sec']:>7.2f}  "
              f"{best_s:>11}  "
              f"{last_s:>11}  "
              f"{ok_s:>4}")

    print("  " + "-" * (len(hdr) - 2))
    tw_mm, tw_ss = divmod(int(total_wall), 60)
    tw_hh, tw_mm = divmod(tw_mm, 60)
    avg_ips = total_iters / total_wall if total_wall > 0 else 0.0
    print(f"  {'TOTAL':>4}  {total_iters:>10}  "
          f"{tw_hh:02d}:{tw_mm:02d}:{tw_ss:02d}  "
          f"{avg_ips:>7.2f}  (avg iter/s)")
    print("=" * 80)
    print()


def export_training_time_csv(segments, out_path):
    """Write segment timing to a CSV file."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Segment", "IterStart", "IterEnd", "TotalIters",
            "StartTime", "EndTime", "WallSeconds",
            "ItersDone", "IterPerSec",
            "BestBCER", "LastBCER", "ExitCode",
        ])
        for s in segments:
            writer.writerow([
                s["segment"],
                s["iter_start"],
                s["iter_end"],
                s["total_iters"],
                s["start_ts"].strftime(_TS_FMT) if s["start_ts"] else "",
                s["end_ts"].strftime(_TS_FMT)   if s["end_ts"]   else "",
                f"{s['wall_sec']:.1f}"           if s["wall_sec"] is not None else "",
                s["iters_done"]  if s["iters_done"]  is not None else "",
                f"{s['iter_per_sec']:.3f}"       if s["iter_per_sec"] is not None else "",
                f"{s['best_bcer']:.4f}"          if s["best_bcer"] is not None else "",
                f"{s['last_bcer']:.4f}"          if s["last_bcer"] is not None else "",
                s["exit_code"] if s["exit_code"] is not None else "",
            ])
    print(f"  ✓ Timing CSV  → {out_path}")


def plot_training_time(segments, model_name, out_path):
    """
    Dual-axis chart:
      Left  — bar chart of wall-clock seconds per segment
      Right — BCER (best in segment) overlaid as a line
    """
    if not segments:
        return

    seg_labels = [f"{s['iter_start']}→\n{s['iter_end']}" for s in segments]
    wall_secs  = [s["wall_sec"]   or 0.0  for s in segments]
    iter_ps    = [s["iter_per_sec"] or 0.0 for s in segments]
    best_bcer  = [s["best_bcer"]  for s in segments]
    x          = np.arange(len(segments))

    fig, ax1 = plt.subplots(figsize=(max(10, len(segments) * 0.9), 5))
    fig.patch.set_facecolor(STYLE_BG)
    ax1.set_facecolor(STYLE_AX)

    # ── Bar: wall time per segment ─────────────────────────────────────────
    bars = ax1.bar(x, wall_secs, color=PALETTE[0], alpha=0.75,
                   zorder=2, label="Wall time (s)")
    for bar, ips in zip(bars, iter_ps):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + max(wall_secs) * 0.01,
                 f"{ips:.2f}\nit/s", ha="center", va="bottom",
                 fontsize=7, color=STYLE_TEXT, fontweight="bold")

    ax1.set_xlabel("Iteration Range", fontsize=LABEL_FS, color=STYLE_TEXT)
    ax1.set_ylabel("Wall-clock Time (s)", fontsize=LABEL_FS, color=PALETTE[0])
    ax1.set_xticks(x)
    ax1.set_xticklabels(seg_labels, fontsize=8, color=STYLE_TEXT)
    ax1.tick_params(axis="y", labelcolor=PALETTE[0])
    ax1.spines[:].set_color(STYLE_GRID)
    ax1.grid(axis="y", color=STYLE_GRID, linewidth=0.6, linestyle="--", zorder=1)

    # ── Line: best BCER per segment ────────────────────────────────────────
    has_bcer = any(v is not None for v in best_bcer)
    if has_bcer:
        ax2 = ax1.twinx()
        ax2.set_facecolor("none")
        bcer_y = [v if v is not None else float("nan") for v in best_bcer]
        ax2.plot(x, bcer_y, color=PALETTE[3], linewidth=LINE_W,
                 marker="o", markersize=MARKER_SZ, label="Best BCER (%)")
        ax2.set_ylabel("Best BCER (%)", fontsize=LABEL_FS, color=PALETTE[3])
        ax2.tick_params(axis="y", labelcolor=PALETTE[3])
        ax2.spines[:].set_color(STYLE_GRID)
        # Combined legend
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=LEGEND_FS,
                   facecolor=STYLE_AX, edgecolor=STYLE_GRID,
                   labelcolor=STYLE_TEXT)
    else:
        ax1.legend(fontsize=LEGEND_FS, facecolor=STYLE_AX,
                   edgecolor=STYLE_GRID, labelcolor=STYLE_TEXT)

    ax1.set_title(
        f"Training Time per Segment — {model_name}",
        fontsize=TITLE_FS, color=STYLE_TEXT, pad=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Timing plot → {out_path}")


def run_timereport(data_dir="tesseract_data", out_dir=PLOTS_DIR, model_name=None):
    """
    Main entry for the training time report.
    If model_name is given, only that model is processed;
    otherwise all models found under data_dir are processed.
    """
    script_dir = Path(__file__).parent.resolve()
    data_path  = script_dir / data_dir
    out_path   = script_dir / out_dir
    out_path.mkdir(parents=True, exist_ok=True)

    # Collect models to process
    if model_name:
        log_file = data_path / model_name / "training.log"
        if not log_file.exists():
            print(f"  ERROR: {log_file} not found.")
            return
        model_list = [(model_name, str(log_file))]
    else:
        model_list = [
            (m.name, str(m / "training.log"))
            for m in sorted(data_path.iterdir())
            if m.is_dir() and (m / "training.log").exists()
        ]
        if not model_list:
            print(f"  No models with training.log found in {data_path}")
            return

    for mname, log_path in model_list:
        print(f"\n  Parsing: {log_path}")
        segments = parse_training_time_report(log_path)
        if not segments:
            print(f"  [{mname}] No timestamped segment data found.")
            print("  Make sure the training.log was written by the segmented")
            print("  training script (with [YYYY-MM-DD HH:MM:SS] timestamps).")
            continue

        print_training_time_report(segments, model_name=mname)

        # CSV
        csv_path = str(out_path / f"report_training_time_{mname}.csv")
        export_training_time_csv(segments, csv_path)

        # Plot
        plot_path = str(out_path / f"plot_training_time_{mname}.png")
        plot_training_time(segments, mname, plot_path)


# ══════════════════════════════════════════════════════════════════════════════
# §13 CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Tesseract training tools: plot logs, compare models, report timing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  plots              Generate training / eval plots from training.log files  (default)
  compare            Run model comparison on a test image set
  metric_explanation Generate CoER + CER/WER alignment explanation diagrams
  inspection         Generate sample inspection grids showing prediction vs GT
  timereport         Parse timestamps in training.log and report training time
  all                Run plots + compare + timereport

Examples:
  python tesseract_tools.py plots
  python tesseract_tools.py plots --data_dir data --out_dir plots
  python tesseract_tools.py compare
  python tesseract_tools.py metric_explanation
  python tesseract_tools.py metric_explanation --custom-gt data/lao-ground-truth-test/sample_001.gt.txt
  python tesseract_tools.py metric_explanation --custom-gt /path/to/my_text.gt.txt --compare_dir output/
  python tesseract_tools.py inspection
  python tesseract_tools.py inspection --num-samples 10 --min_length 10
  python tesseract_tools.py timereport
  python tesseract_tools.py timereport --model_name lao48cnn32
  python tesseract_tools.py all
""",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="plots",
        choices=["plots", "compare", "metric_explanation", "timereport", "all"],
        help="Which pipeline to run (default: plots)",
    )
    parser.add_argument(
        "--data_dir", default="data",
        help="Root dir with model sub-folders containing training.log (default: data)",
    )
    parser.add_argument(
        "--tessdata_dir", default="tesseract_data",
        help="Dir containing compiled .traineddata files (default: tesseract_data)",
    )
    parser.add_argument(
        "--out_dir",  default=PLOTS_DIR,
        help=f"Output directory for training plots (default: {PLOTS_DIR})",
    )
    parser.add_argument(
        "--compare_dir", default=COMPARE_DIR,
        help=f"Output directory for comparison results (default: {COMPARE_DIR})",
    )
    parser.add_argument(
        "--model_name", default=None,
        help="(timereport) Name of a single model to report. Omit to report all models.",
    )
    parser.add_argument(
        "--custom-gt", dest="custom_gt", default=None, metavar="FILE.gt.txt",
        help="(metric_explanation) Path to a .gt.txt file whose text is used as the "
             "GT sample. If a paired image (same stem, .png/.jpg/…) exists beside it, "
             "Tesseract will be run on that image for a real prediction; otherwise the "
             "GT text itself is used as the prediction (CoER breakdown only).",
    )
    parser.add_argument(
        "--num-samples", dest="num_samples", type=int, default=5,
        help="(metric_explanation) Number of GT samples to select from IMAGE_FOLDER "
             "when running in default mode (default: 5).",
    )
    parser.add_argument(
        "--min_length", dest="min_length", type=int, default=3,
        help="(metric_explanation) Minimum GT text length to be selected (default: 3).",
    )
    parser.add_argument(
        "--max_length", dest="max_length", type=int, default=None,
        help="(metric_explanation) Maximum GT text length to be selected "
             "(default: no upper limit). Use to cap very long lines.",
    )
    parser.add_argument(
        "--models", dest="models", nargs="+", default=None, metavar="MODEL",
        help="(metric_explanation) Explicit list of model names to evaluate. "
             "Overrides CUSTOM_MODEL_NAMES. Example: --models LaoCRNN-I64K5C16 I64K2C32",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating training log plots and comparison plots."
    )
    parser.add_argument(
        "--realworld_dir", "--real-world-data", dest="realworld_dir", default=None,
        help="Optional folder containing real-world image+.gt.txt pairs to evaluate Tesseract models on.",
    )
    args = parser.parse_args()

    if args.command in ("plots", "all"):
        if args.no_plots:
            print("  Skipping training log plots (--no-plots).")
        else:
            run_plots(data_dir=args.data_dir, out_dir=args.out_dir)

    if args.command in ("compare", "all"):
        run_compare(out_dir=args.compare_dir,
                    tessdata_dir=args.tessdata_dir,
                    logs_dir=args.data_dir,
                    skip_plots=args.no_plots,
                    model_names=args.models,
                    realworld_dir=args.realworld_dir)

    if args.command == "metric_explanation":
        run_metric_explanation(
            out_dir=args.compare_dir,
            tessdata_dir=args.tessdata_dir,
            image_folder=args.realworld_dir if args.realworld_dir else IMAGE_FOLDER,
            custom_gt_file=args.custom_gt,
            num_samples=args.num_samples,
            min_length=args.min_length,
            max_length=args.max_length,
            mode="metric_explanation",
            model_names=args.models,
        )

    if args.command == "inspection":
        plot_tesseract_cnn_inspection(
            tessdata_dir=args.tessdata_dir,
            image_folder=args.realworld_dir if args.realworld_dir else IMAGE_FOLDER,
            out_dir=args.compare_dir,
            num_samples=args.num_samples,
            min_length=args.min_length
        )

    if args.command in ("timereport", "all"):
        run_timereport(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            model_name=args.model_name,
        )


if __name__ == "__main__":
    main()
