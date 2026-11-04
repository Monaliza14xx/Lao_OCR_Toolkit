"""
log_to_tensorboard.py
──────────────────────────────────────────────────────────
Export Tesseract LSTM training.log + eval/*.eval.log to
TensorBoard event files so you can compare train vs eval
curves for every model in one dashboard.

Scalars written per model
─────────────────────────
  Loss/Mean_RMS          (train, every checkpoint)
  Loss/Delta             (train, every checkpoint)
  Train/BCER             (train BCER%, every checkpoint)
  Train/BWER             (train BWER%, every checkpoint)
  Train/Accuracy         (100 - BCER train)
  Train/Skip_Ratio       (train, every checkpoint)
  Eval/BCER              (eval BCER%, sparse – eval checkpoints only)
  Eval/BWER              (eval BWER%, sparse – eval checkpoints only)
  Eval/Accuracy          (100 - BCER eval)

Usage
─────
  # Auto-discover all models under data/
  python log_to_tensorboard.py

  # Specific models
  python log_to_tensorboard.py --logs data/lao48cnn16/training.log data/lao48cnn32/training.log

  # Then open TensorBoard
  tensorboard --logdir=runs
"""
import os
import re
import glob
import argparse
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboard.summary.writer.event_file_writer import EventFileWriter  # noqa
        from tensorboardX import SummaryWriter  # fallback
    except ImportError:
        print("Error: tensorboard writer not found.")
        print("Install with:  pip install torch tensorboard")
        print("          or:  pip install tensorboardX tensorboard")
        exit(1)

# ── Regex patterns ──────────────────────────────────────────────────────────
# Handles both plain and timestamped lines:
#   [2026-04-06 18:19:55] At iteration 100/100/100, ...
TRAIN_RE = re.compile(
    r"At iteration (\d+)/\d+/\d+, "
    r"mean rms=([\d.]+)%, "
    r"delta=([\d.]+)%, "
    r"BCER train=([\d.]+)%, "
    r"BWER train=([\d.]+)%, "
    r"skip ratio=([\d.]+)%"
)

# eval/*.eval.log content:
#   BCER eval=28.427, BWER eval=69.355
EVAL_LOG_RE = re.compile(r"BCER eval=([\d.]+),\s+BWER eval=([\d.]+)")

# eval filename formats:
#   OLD: lao48cnn16_28.205_19545_20000.eval.log  → iteration = group(2)
#   NEW: lao48cnn32_iter002000.eval.log          → iteration = group(1)
EVAL_FILE_OLD = re.compile(r"[\w]+_[\d.]+_(\d+)_\d+\.eval\.log$")
EVAL_FILE_NEW = re.compile(r"[\w]+_iter(\d+)\.eval\.log$")


def parse_eval_logs(eval_dir):
    """Return sorted list of (iteration, bcer, bwer) from eval/*.eval.log."""
    rows = []
    for fpath in glob.glob(os.path.join(eval_dir, "*.eval.log")):
        fname = os.path.basename(fpath)
        m = EVAL_FILE_NEW.search(fname)
        if m:
            iteration = int(m.group(1))
        else:
            m = EVAL_FILE_OLD.search(fname)
            if not m:
                continue
            iteration = int(m.group(1))

        with open(fpath, "r", encoding="utf-8") as fh:
            for line in fh:
                em = EVAL_LOG_RE.search(line)
                if em:
                    rows.append((iteration, float(em.group(1)), float(em.group(2))))
    rows.sort(key=lambda r: r[0])
    return rows


def export_model(log_path, out_dir):
    """Parse one model's training.log + eval/ and write TensorBoard events."""
    log_path = Path(log_path).resolve()
    if not log_path.exists():
        print(f"  [SKIP] File not found: {log_path}")
        return

    model_name = log_path.parent.name
    eval_dir   = log_path.parent / "eval"
    tb_dir     = Path(out_dir) / model_name
    tb_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(tb_dir))

    # ── Training metrics ────────────────────────────────────────────────────
    print(f"  [{model_name}] Parsing training log …")
    train_count = 0
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            m = TRAIN_RE.search(line)
            if not m:
                continue
            it         = int(m.group(1))
            mean_rms   = float(m.group(2))
            delta      = float(m.group(3))
            bcer_train = float(m.group(4))
            bwer_train = float(m.group(5))
            skip_ratio = float(m.group(6))

            writer.add_scalar("Loss/Mean_RMS",    mean_rms,          it)
            writer.add_scalar("Loss/Delta",        delta,             it)
            writer.add_scalar("Train/BCER",        bcer_train,        it)
            writer.add_scalar("Train/BWER",        bwer_train,        it)
            writer.add_scalar("Train/Accuracy",    100.0 - bcer_train, it)
            writer.add_scalar("Train/Skip_Ratio",  skip_ratio,        it)
            train_count += 1

    print(f"    → {train_count} training steps written")

    # ── Eval metrics ────────────────────────────────────────────────────────
    eval_count = 0
    if eval_dir.exists():
        eval_rows = parse_eval_logs(str(eval_dir))
        for it, bcer_eval, bwer_eval in eval_rows:
            writer.add_scalar("Eval/BCER",     bcer_eval,           it)
            writer.add_scalar("Eval/BWER",     bwer_eval,           it)
            writer.add_scalar("Eval/Accuracy", 100.0 - bcer_eval,   it)
            eval_count += 1
        print(f"    → {eval_count} eval checkpoints written")
    else:
        print(f"    → No eval/ directory found, skipping eval metrics")

    writer.close()
    print(f"    → Events saved to: {tb_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Tesseract training.log + eval logs to TensorBoard.")
    parser.add_argument(
        "--logs", nargs="+", default=None,
        help="Paths to training.log files. If omitted, auto-discovers all "
             "models under data/.")
    parser.add_argument(
        "--data_dir", default="tesseract_data",
        help="Root data directory for auto-discovery (default: tesseract_data)")
    parser.add_argument(
        "--out_dir", default="runs",
        help="Output directory for TensorBoard event files (default: runs)")
    args = parser.parse_args()

    # Always work relative to this script's directory
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)

    # Resolve log list
    if args.logs:
        log_paths = args.logs
    else:
        data_path = script_dir / args.data_dir
        log_paths = sorted(str(p) for p in data_path.glob("*/training.log"))
        if not log_paths:
            print(f"No training.log files found under {data_path}")
            exit(1)
        print(f"Auto-discovered {len(log_paths)} model(s) under {data_path}\n")

    for lp in log_paths:
        export_model(lp, str(script_dir / args.out_dir))
        print()

    print("=" * 60)
    print("All done! Launch TensorBoard with:")
    print(f"  tensorboard --logdir={script_dir / args.out_dir}")
    print("Then open: http://localhost:6006/")
    print()
    print("In TensorBoard, group by tag to see Train vs Eval side-by-side.")
    print("Tip: pin 'Train/BCER' and 'Eval/BCER' on the same chart with")
    print("     the 'Tag regex' filter: ^(Train|Eval)/BCER$")
