"""Console progress report printed by the trainer every --debug_interval iterations.

Example (an iteration where validation ran):

    ━━ iter 500/10,000 (5.0%) · epoch 2 · 312 lines/s · elapsed 2m40s · ETA 50m45s ━━
                  loss      CER      WER   line acc
      train     2.1456   45.21%   78.34%     12.50%   (avg loss 2.3102, 1 batch)
      eval      2.4012   48.10%   80.20%     10.94%   ★ best so far
      CoER     tone 12.4% · upper 8.1% · lower 3.0% · front 1.2% · rear 2.1%
               cons 5.2% · special 0.0% · Lao num 10.0% · digit 2.0% · space 3.5%
               sym 4.0%
               worst: tone marks
      model    weight RMS 0.0435 · lr 1e-04 · GPU 2.10 GB (peak 3.80 GB)
      samples  [1] ✗ CER 3.0%
                 GT   ສາທາລະນະລັດ ປະຊາທິປະໄຕ ປະຊາຊົນລາວ
                 Pred ສາທາລະນະລັດ ປະຊາ⟦ທິ→ທີ⟧ປະໄຕ ປະຊາຊົນລາວ

CoER (component error rate) is the error rate per Lao character class:
errors on characters of that class / characters of that class in the ground truth.
Eval CoER is shown every 5th report; in between, "CoER*" comes from the one
training batch just used and is noisy.

Sample diffs: ⟦GT→Pred⟧ wrong, ⟦−GT⟧ missing, ⟦+Pred⟧ extra.
"""

import difflib
import re
import unicodedata

from .metrics import LAO_CLASSES, edit_distance, normalize_lao_text

WIDTH = 78

# (category key in LAO_COMPONENT_MAP, short label, long label), in display order
COER_LABELS = [
    ('tone',         'tone',     'tone marks'),
    ('upper_vowel',  'upper',    'upper vowels'),
    ('lower_vowel',  'lower',    'lower vowels'),
    ('before_vowel', 'front',    'front vowels'),
    ('rear_vowel',   'rear',     'rear vowels'),
    ('consonant',    'cons',     'consonants'),
    ('special',      'special',  'special (ຼ)'),
    ('lao_number',   'Lao num',  'Lao digits'),
    ('digit',        'digit',    'Arabic digits'),
    ('latin',        'latin',    'Latin letters'),
    ('space',        'space',    'spaces'),
    ('symbol',       'sym',      'symbols'),
    ('other',        'other',    'other characters'),
]
COER_KEYS = [k for k, _, _ in COER_LABELS]

_CLUSTER = re.compile(r'.[̀-ͯັິ-ຼ່-ໍ]*', re.S)


def _clusters(s):
    """Split into base character + its combining marks, so a diff never leaves a lone mark."""
    return _CLUSTER.findall(unicodedata.normalize('NFC', s))


def mark_diff(gt, pred):
    """Pred text with each difference from GT marked: ⟦GT→Pred⟧ wrong, ⟦−GT⟧ missing, ⟦+Pred⟧ extra."""
    g, p = _clusters(gt), _clusters(pred)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, g, p, autojunk=False).get_opcodes():
        want, got = ''.join(g[i1:i2]), ''.join(p[j1:j2])
        if op == 'equal':
            out.append(got)
        elif op == 'delete':
            out.append(f'⟦−{want}⟧')
        elif op == 'insert':
            out.append(f'⟦+{got}⟧')
        else:
            out.append(f'⟦{want}→{got}⟧')
    return ''.join(out)


def _dur(sec):
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f'{h}h{m:02d}m' if h else f'{m}m{s:02d}s'


def _pct(x):
    return f'{x * 100:6.2f}%' if x is not None else '      –'


def _clip(s, n=70):
    return s if len(s) <= n else s[:n - 1] + '…'


def format_coer(coer):
    """'tone 12.4% · upper 8.1% …' for the classes present, plus the worst Lao class(es)."""
    if not coer:
        return None, None
    parts = [f'{short} {coer[key] * 100:.1f}%' for key, short, _ in COER_LABELS if key in coer]
    rates = {k: coer[k] for k in LAO_CLASSES if k in coer}
    if not rates or max(rates.values()) == 0 or max(rates.values()) == min(rates.values()):
        return parts, None
    top = max(rates.values())
    names = [long_ for key, _, long_ in COER_LABELS if rates.get(key) == top]
    return parts, ', '.join(names)


def _wrap(parts, indent, width=WIDTH):
    lines, cur = [], ''
    for p in parts:
        piece = p if not cur else ' · ' + p
        if cur and indent + len(cur) + len(piece) > width:
            lines.append(cur)
            cur = p
        else:
            cur += piece
    if cur:
        lines.append(cur)
    return lines


def render_progress(*, iteration, max_iter, epoch, sps, elapsed, iters_done,
                    loss, avg_loss, train, evaluation=None, coer=None, coer_source='eval',
                    is_best=False, mean_rms=None, lr=None, gpu_mem=None, gpu_peak=None,
                    samples=()):
    """Return the multi-line report.

    train / evaluation: dicts with cer, wer, acc (and evaluation also loss).
    coer: {category: rate} from compute_metrics.
    samples: [(gt, pred), ...]
    """
    rate = iters_done / elapsed if elapsed > 0 else 0
    eta = (max_iter - iteration) / rate if rate > 0 else 0
    head = (f' iter {iteration:,}/{max_iter:,} ({iteration / max_iter:.1%}) · epoch {epoch} · '
            f'{sps:.0f} lines/s · elapsed {_dur(elapsed)} · ETA {_dur(eta)} ')
    out = ['', '━━' + head + '━' * max(2, WIDTH - len(head) - 2)]

    out.append(f"  {'':7s}{'loss':>9s}{'CER':>9s}{'WER':>9s}{'line acc':>11s}")
    out.append(f"  {'train':7s}{loss:9.4f}{_pct(train['cer']):>9s}{_pct(train['wer']):>9s}"
               f"{_pct(train['acc']):>11s}   (avg loss {avg_loss:.4f}, 1 batch)")
    if evaluation is not None:
        out.append(f"  {'eval':7s}{evaluation['loss']:9.4f}{_pct(evaluation['cer']):>9s}"
                   f"{_pct(evaluation['wer']):>9s}{_pct(evaluation['acc']):>11s}"
                   + ('   ★ best so far' if is_best else ''))
        if evaluation.get('cer_strict') is not None:
            out.append(f"  {'':7s}  strict CER {evaluation['cer_strict'] * 100:.2f}%"
                       f" · CluER {evaluation['cluer'] * 100:.2f}%   (CER above is lenient)")
    elif is_best:
        out[-1] += '  ★'

    parts, worst = format_coer(coer)
    if parts:
        label = 'CoER' if coer_source == 'eval' else 'CoER*'
        rows = _wrap(parts, indent=11)
        out.append(f'  {label:7s}  ' + rows[0])
        out += [' ' * 11 + r for r in rows[1:]]
        note = f'worst: {worst}' if worst else ('no component errors' if not any(coer.values())
                                                   else 'all classes equally wrong')
        if coer_source != 'eval':
            note += '   (* train batch only; eval CoER every 5th report)'
        out.append(' ' * 11 + note)

    sysinfo = []
    if mean_rms is not None:
        sysinfo.append(f'weight RMS {mean_rms:.4f}')
    if lr is not None:
        sysinfo.append(f'lr {lr:.0e}')
    if gpu_mem:
        sysinfo.append(f'GPU {gpu_mem:.2f} GB' + (f' (peak {gpu_peak:.2f} GB)' if gpu_peak else ''))
    if sysinfo:
        out.append(f"  {'model':7s}  " + ' · '.join(sysinfo))

    for n, (gt, pred) in enumerate(samples):
        g, p = normalize_lao_text(gt), normalize_lao_text(pred)
        cer = edit_distance(p, g) / max(len(g), 1)
        tag = '✓ exact' if p == g else f'✗ CER {cer:.1%}'
        out.append(f"  {'samples' if n == 0 else '':7s}  [{n + 1}] {tag}")
        out.append(f"             GT   {_clip(gt)}")
        if p != g:
            out.append(f"             Pred {_clip(mark_diff(gt, pred)) if p else '(empty)'}")
    out.append('─' * WIDTH)
    return '\n'.join(out)
