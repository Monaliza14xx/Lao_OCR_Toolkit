"""
plot_training.py
────────────────
Generate all training/evaluation plots from Tesseract LSTM training logs.
Now upgraded with the "Premium" dashboard style and aesthetics.

Output plots
────────────
  plot_loss_combined.png   – Mean-RMS (≈loss) for all models on one axes
  plot_acc_combined.png    – Character Accuracy (100-BCER) for all models
  training_plot.png        – Full 6-panel dashboard (per model subdirectory)
  plot_train_loss.png      – Mean-RMS per model (subplots)
  plot_eval_loss.png       – BCER eval per model (sparse checkpoints)
  plot_train_acc.png       – Char Accuracy (train) per model (subplots)
  plot_eval_acc.png        – Char Accuracy (eval) per model (sparse checkpoints)

Usage
─────
  python experiments/plot_training.py                   # auto-discover under data/
  python experiments/plot_training.py --data_dir data   # explicit data root
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: tesstrain_core, tesseract_tools

import os
import re
import glob
import argparse
from pathlib import Path
from tesstrain_core.plotting import DashboardPlotter, plot_multi_model

# ──────────────────────────────────────────────────────────────────────────────
# Regex patterns
# ──────────────────────────────────────────────────────────────────────────────
TRAIN_RE = re.compile(
    r"At iteration (\d+)/\d+/\d+, "
    r"mean rms=([\d.]+)%, "
    r"delta=([\d.]+)%, "
    r"BCER train=([\d.]+)%, "
    r"BWER train=([\d.]+)%"
)
EVAL_FILE_RE = re.compile(
    r"[\w]+_([\d.]+)_(\d+)_(\d+)\.eval\.log$"
)
EVAL_FILE_RE_NEW = re.compile(r"[\w]+_iter(\d+)\.eval\.log$")
EVAL_LOG_RE = re.compile(
    r"BCER eval=([\d.]+),\s+BWER eval=([\d.]+)"
)

# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_tess_data(log_path, eval_dir):
    """
    Load Tesseract log and eval data into a format compatible with DashboardPlotter.
    Returns: list of dicts (one per iteration/checkpoint).
    """
    entries = []
    
    # 1. Load training.log
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                m = TRAIN_RE.search(line)
                if m:
                    entries.append({
                        'iteration': int(m.group(1)),
                        'train_loss': float(m.group(2)) / 100.0,  # Convert % -> 0-1
                        'train_cer': float(m.group(4)) / 100.0,
                        'train_wer': float(m.group(5)) / 100.0,
                        'train_acc': (100.0 - float(m.group(4))) / 100.0,
                    })

    # 2. Load eval logs
    if eval_dir and os.path.exists(eval_dir):
        for fpath in glob.glob(os.path.join(eval_dir, "*.eval.log")):
            fname = os.path.basename(fpath)
            it = -1
            m_it = EVAL_FILE_RE_NEW.search(fname) or EVAL_FILE_RE.search(fname)
            if m_it:
                # if m_it has 3 groups (old format), second is iteration
                it = int(m_it.group(1)) if m_it.lastindex == 1 else int(m_it.group(2))

            if it >= 0:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        m_l = EVAL_LOG_RE.search(line)
                        if m_l:
                             # Find or create entry for this iteration
                             entry = next((e for e in entries if e['iteration'] == it), None)
                             if not entry:
                                 entry = {'iteration': it}
                                 entries.append(entry)
                             entry['eval_loss'] = float(m_l.group(1)) / 100.0
                             entry['eval_cer'] = float(m_l.group(1)) / 100.0
                             entry['eval_wer'] = float(m_l.group(2)) / 100.0
                             entry['eval_acc'] = (100.0 - float(m_l.group(1))) / 100.0

    entries.sort(key=lambda x: x['iteration'])
    return entries

def discover_models(data_dir):
    models = []
    for model_dir in sorted(Path(data_dir).iterdir()):
        if not model_dir.is_dir(): continue
        log_path = model_dir / "training.log"
        if not log_path.exists(): continue
        eval_dir = model_dir / "eval"
        models.append({
            'name': model_dir.name,
            'log': str(log_path),
            'eval': str(eval_dir) if eval_dir.is_dir() else None,
            'path': str(model_dir)
        })
    return models

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate premium training plots for Tesseract models.")
    parser.add_argument("--data_dir", default="data", help="Root data directory")
    parser.add_argument("--out_dir", default="plots", help="Global plots output directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovering Tesseract models in: {data_dir}\n")
    models = discover_models(data_dir)
    if not models:
        print("No models found. Exiting.")
        return

    all_model_data = []

    for m in models:
        print(f"Processing [{m['name']}]...")
        entries = load_tess_data(m['log'], m['eval'])
        if not entries:
            print(f"  Warning: No valid log entries for {m['name']}")
            continue
        
        # 1. Generate per-model dashboard in its directory
        plotter = DashboardPlotter(m['name'], m['path'])
        plotter.generate_dashboard(entries)
        plotter.generate_premium_charts(entries)
        
        all_model_data.append((m['name'], entries))

    # 2. Generate combined plots in global out_dir
    print(f"\nGenerating combined plots in: {out_dir}")
    
    # Loss (Mean RMS / BCER)
    loss_series = []
    for name, entries in all_model_data:
        train_loss = [(e['iteration'], e['train_loss']) for e in entries if 'train_loss' in e]
        eval_loss  = [(e['iteration'], e['eval_loss']) for e in entries if 'eval_loss' in e]
        if train_loss: loss_series.append((f"{name} (Train)", train_loss, False))
        if eval_loss:  loss_series.append((f"{name} (Eval)", eval_loss, True))
    
    plot_multi_model("Combined Training/Eval Loss", "Loss", "plot_loss_combined.png", 
                     loss_series, str(out_dir), log_y=True)

    # Accuracy
    acc_series = []
    for name, entries in all_model_data:
        train_acc = [(e['iteration'], e['train_acc']) for e in entries if 'train_acc' in e]
        eval_acc  = [(e['iteration'], e['eval_acc']) for e in entries if 'eval_acc' in e]
        if train_acc: acc_series.append((f"{name} (Train)", train_acc, False))
        if eval_acc:  acc_series.append((f"{name} (Eval)", eval_acc, True))
    
    plot_multi_model("Combined Character Accuracy", "Accuracy", "plot_acc_combined.png", 
                     acc_series, str(out_dir), ylim=(0, 1.05))

    print("\nAll plots generated successfully!")

if __name__ == "__main__":
    main()
