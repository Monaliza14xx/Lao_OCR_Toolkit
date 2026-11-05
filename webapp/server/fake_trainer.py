"""Stand-in for `torch_train.py training`: no GPU, no data, writes a realistic log.csv.

Used by the server tests and for UI development:
    LAOOCR_TRAIN_SCRIPT=webapp/server/fake_trainer.py python -m uvicorn webapp.server.app:create_app --factory

Env: FAKE_TRAIN_STEP_SEC (sleep per debug interval, default 0.2), FAKE_TRAIN_FAIL=1 (exit 1 halfway).
"""
import argparse
import csv
import math
import os
import random
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tesstrain_core.lao_classes import LAO_COMPONENT_MAP  # noqa: E402

HEADER = (['iteration', 'timestamp', 'epoch', 'train_loss', 'eval_loss', 'train_cer', 'train_wer', 'train_acc',
           'eval_cer', 'eval_wer', 'eval_acc', 'mean_rms', 'learning_rate', 'gpu_mem_gb', 'samples_per_sec',
           'coer_source'] + [f'coer_{k}' for k in list(LAO_COMPONENT_MAP) + ['other']])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command')
    ap.add_argument('--log_file', required=True)
    ap.add_argument('--max_iterations', type=int, default=1000)
    ap.add_argument('--debug_interval', type=int, default=100)
    ap.add_argument('--learning_rate', type=float, default=0.002)
    ap.add_argument('--model_name', default='fake')
    args, _ = ap.parse_known_args()

    step = float(os.environ.get('FAKE_TRAIN_STEP_SEC', '0.2'))
    fail = os.environ.get('FAKE_TRAIN_FAIL') == '1'
    rng = random.Random(0)
    print('=' * 70)
    print('  Fake trainer (no GPU)')
    print(f'  Model:           {args.model_name}')
    print(f'  Max Iterations:  {args.max_iterations}')
    print('=' * 70, flush=True)

    with open(args.log_file, 'w', newline='') as f:
        csv.writer(f).writerow(HEADER)
    for it in range(args.debug_interval, args.max_iterations + 1, args.debug_interval):
        time.sleep(step)
        t = it / args.max_iterations
        cer = 0.9 * math.exp(-4 * t) + 0.02 + rng.uniform(0, 0.01)
        row = {'iteration': it, 'timestamp': datetime.now().isoformat(), 'epoch': it // 500,
               'train_loss': round(3.5 * math.exp(-3 * t) + 0.1, 6), 'train_cer': round(cer, 6),
               'train_wer': round(min(1, cer * 2.5), 6), 'train_acc': round(1 - cer, 6),
               'learning_rate': args.learning_rate, 'gpu_mem_gb': 1.2, 'samples_per_sec': 850,
               'mean_rms': 0.3}
        if (it // args.debug_interval) % 5 == 0:
            row.update(eval_loss=round(3.8 * math.exp(-2.8 * t) + 0.15, 6), eval_cer=round(cer * 1.2, 6),
                       eval_wer=round(min(1, cer * 3), 6), eval_acc=round(1 - cer * 1.2, 6), coer_source='eval')
            for i, k in enumerate(list(LAO_COMPONENT_MAP) + ['other']):
                row[f'coer_{k}'] = round(cer * (0.5 + (i % 4) * 0.4), 6)
        with open(args.log_file, 'a', newline='') as f:
            csv.writer(f).writerow([row.get(h, '') for h in HEADER])
        print(f'  [{it:>6}] loss={row["train_loss"]:.4f}  CER={cer * 100:.2f}%', flush=True)
        if fail and t >= 0.5:
            print('RuntimeError: fake failure', file=sys.stderr, flush=True)
            sys.exit(1)
    print('  Training complete.', flush=True)


if __name__ == '__main__':
    main()
