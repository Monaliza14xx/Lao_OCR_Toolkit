#!/usr/bin/env python3
"""Step 0 part-swap configurations (EXPERIMENT_PLAN.md §2, §4): one module of B0 changed at a time.

Run from the repository root:

  python experiments/step0_specs.py                     # table: name, params, rows, frames, spec
  python experiments/step0_specs.py --commands          # torch_train.py commands, seeds 1 2 3
  python experiments/step0_specs.py --commands --only B0 S-Atys --seeds 1 2 3 4 5 \\
      --extra "--ground_truth_dir data/lao-ground-truth --output_root data/step0"

Every spec is built with tesstrain_core.model.build_model before anything is printed, so a
spec that doesn't build fails here, not 3 hours into a Colab run.
"""
import argparse
import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from tesstrain_core.model import build_model, output_lengths  # noqa: E402

NUM_CLASSES = 80          # 79 characters + CTC blank, as in the thesis data
HEAD = '1,64,0,1'          # thesis input: grayscale, height 64, variable width
B0_CNN = 'Ct5,5,16 Mp3,3'
B0_SQUASH = 'Lfys64'
B0_RNN = 'Lfx96 Lrx96 Lfx384'


def spec(cnn=B0_CNN, squash=B0_SQUASH, rnn=B0_RNN):
    return '[' + ' '.join(p for p in (HEAD, cnn, squash, rnn, 'O1c###') if p) + ']'


# (name, group, spec, what changes vs B0)
CONFIGS = [
    ('B0',        'baseline', spec(), 'thesis I64K5C16'),
    # height collapse ("squash")
    ('S-flatten', 'squash', spec(squash=''), 'no collapse: rows × channels flattened per column'),
    ('S-Atys64',  'squash', spec(squash='Atys64'), 'additive attention over rows'),
    ('S-Matys64', 'squash', spec(squash='Matys64'), '4-head attention over rows'),
    ('S-Hybys64', 'squash', spec(squash='Hybys64'), 'vertical BiLSTM + attention'),
    # sequence model
    ('R-none',    'rnn', spec(rnn=''), 'no sequence model'),
    ('R-fwd',     'rnn', spec(rnn='Lfx384'), 'one forward-only LSTM'),
    ('R-BiLSTM',  'rnn', spec(rnn='Lbx96 Lbx384'), 'true BiLSTM (B0 is fwd → rev → fwd)'),
    # CNN; time resolution (T = W/3) and rows at the collapse kept as in B0 where possible
    ('C-K3',      'cnn', spec(cnn='Ct3,3,16 Mp3,3'), '3×3 kernel'),
    ('C-2conv',   'cnn', spec(cnn='Ct5,5,16 Mp3,3 Ct5,5,32'), 'second conv after the pool'),
    ('C-Lcn4',    'cnn', spec(cnn='Lcn4,24,1 Mp3,3 Lcn4,48,2 Lcn4,64,1,2,1'),
     'LCNetV4-style backbone, ~1M params total'),
    # CTC time resolution (x-stride of the pool), CNN otherwise B0
    ('X-sx1',     'xstride', spec(cnn='Ct5,5,16 Mpa3,3,3,1,0,1'), 'x-stride 1: T = W'),
    ('X-sx2',     'xstride', spec(cnn='Ct5,5,16 Mpa3,3,3,2,0,0'), 'x-stride 2: T ≈ W/2'),
]

DEFAULT_TRAIN_ARGS = ('--batch_size 64 --learning_rate 0.002 --max_iterations 50000 '
                      '--lr_scheduler plateau')


def describe(net_spec, width=512):
    """(params, rows reaching the height collapse, output frames for a line of `width` px)."""
    model = build_model(net_spec.replace('###', str(NUM_CLASSES)), NUM_CLASSES).eval()
    params = sum(p.numel() for p in model.parameters())
    x = torch.zeros(1, 1, int(HEAD.split(',')[1]), width)
    rows = None
    with torch.no_grad():
        for layer in model:
            if x.dim() == 4 and rows is None and not _is_cnn(layer):
                rows = x.shape[2]
            x = layer(x)
    frames = int(output_lengths(model, torch.tensor([width]))[0])
    assert x.shape[-1] == NUM_CLASSES and x.shape[1] == frames, (net_spec, x.shape, frames)
    return params, rows, frames


def _is_cnn(layer):
    name = layer.__class__.__name__
    return isinstance(layer, (torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.BatchNorm2d, torch.nn.Tanh,
                              torch.nn.ReLU, torch.nn.Dropout2d)) or name == 'Lcn4Stage'


def table(configs):
    rows = [describe(s) for _, _, s, _ in configs]
    base = rows[0][0] if configs and configs[0][0] == 'B0' else None
    lines = [f"{'name':10s} {'group':9s} {'params':>10s} {'vs B0':>7s} {'rows':>5s} {'T@512':>6s}  spec",
             '-' * 110]
    for (name, group, s, _), (params, h, t) in zip(configs, rows):
        rel = f'{params / base:6.2f}×' if base else ''
        lines.append(f'{name:10s} {group:9s} {params:>10,} {rel:>7s} {h if h is not None else "-":>5} {t:>6}  {s}')
    lines.append('')
    lines.append(f'params at {NUM_CLASSES} classes; rows = feature-map height entering the collapse '
                 f'(64 px input); T@512 = CTC frames for a 512 px line')
    return '\n'.join(lines)


def commands(configs, seeds, train_args, extra, output_root, script='torch_train.py'):
    out = []
    for name, _, s, _ in configs:
        for seed in seeds:
            run = f'step0-{name}-s{seed}'
            parts = ['python', script, 'training', '--model_name', run, '--net_spec', s, '--seed', str(seed)]
            if output_root:
                parts += ['--output_dir', os.path.join(output_root, run)]
            out.append(' '.join(shlex.quote(p) for p in parts) + f' {train_args} {extra}'.rstrip())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--commands', action='store_true', help='print torch_train.py training commands')
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3], help='seeds (default: 1 2 3)')
    ap.add_argument('--only', nargs='+', default=None, metavar='NAME',
                    help='names or groups to include (e.g. B0 squash C-Lcn4)')
    ap.add_argument('--train_args', default=DEFAULT_TRAIN_ARGS,
                    help=f'shared recipe for every run (default: "{DEFAULT_TRAIN_ARGS}")')
    ap.add_argument('--extra', default='', help='appended to every command (data paths, --padding legacy, ...)')
    ap.add_argument('--output_root', default='', help='put each run in OUTPUT_ROOT/<run name>')
    a = ap.parse_args(argv)

    configs = CONFIGS
    if a.only:
        configs = [c for c in CONFIGS if c[0] in a.only or c[1] in a.only]
        unknown = set(a.only) - {c[0] for c in CONFIGS} - {c[1] for c in CONFIGS}
        if unknown:
            ap.error(f'unknown name(s): {", ".join(sorted(unknown))}')

    if a.commands:
        for name, *_ in configs:
            describe(dict((c[0], c[2]) for c in CONFIGS)[name])     # fail early if a spec doesn't build
        print('\n'.join(commands(configs, a.seeds, a.train_args, a.extra, a.output_root)))
    else:
        print(table(configs))


if __name__ == '__main__':
    main()
