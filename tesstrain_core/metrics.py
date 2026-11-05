import csv
import os
import random
import math
from datetime import datetime
import torch

# Lao character classes for CoER live in lao_classes.py (shared with the standalone scripts)
from .lao_classes import LAO_COMPONENT_MAP, LAO_CLASSES, get_component_type, fold_sara_am, split_clusters  # noqa: F401

import unicodedata

def normalize_lao_text(text):
    """Normalize Lao text by stripping invisible chars, standardizing clusters and roman numerals."""
    if not text: return ""
    # Standard Unicode Normalization
    text = unicodedata.normalize('NFC', text)
    text = fold_sara_am(text)   # ໍ + າ → ຳ (same glyphs, standard spelling)
    # Strip invisible characters
    for s in ['\u200b', '\u200c', '\u200d', '\ufeff']:
        text = text.replace(s, '')
    
    # Standardize Latin/Roman confusions (Common in Lao formal text)
    text = text.replace('ll', 'II').replace('lX', 'IX').replace('1X', 'IX')
    
    # Standardize Thai-to-Lao Vowel Confusions (Visually identical)
    text = text.replace('\u0E40', '\u0EC0') # เ -> ເ
    text = text.replace('\u0E41', '\u0EC1') # ແ ->ແ
    text = text.replace('\u0E42', '\u0EC2') # โ -> ໂ
    text = text.replace('\u0E43', '\u0EC3') # ใ -> ໃ
    text = text.replace('\u0E44', '\u0EC4') # ไ -> ໄ

    # Terminal Punctuation & Space Lenience
    text = text.strip()
    while text and text[-1] in ',. ':
        text = text[:-1]
    
    return text


LAO_SIMILAR_PAIRS = [
    ('ໃ', 'ໄ'), # Vowels (mai muan vs mai may)
    ('ດ', 'ຕ'), # Visually similar consonants
    ('ນ', 'ມ'),
    ('ບ', 'ປ'),
    ('ກ', 'ຖ'),
    ('ຜ', 'ຝ'),
    ('ພ', 'ຟ'),
]

def get_edit_cost(c1, c2):
    """Return substitution cost for two characters."""
    if c1 == c2: return 0
    # Check for known similar pairs (0.5 cost instead of 1.0)
    for p1, p2 in LAO_SIMILAR_PAIRS:
        if (c1 == p1 and c2 == p2) or (c1 == p2 and c2 == p1):
            return 0.5
    return 1.0


def edit_distance(s1, s2):
    if isinstance(s1, str): s1 = normalize_lao_text(s1)
    if isinstance(s2, str): s2 = normalize_lao_text(s2)
    if len(s1) < len(s2): return edit_distance(s2, s1)
    if not s2: return len(s1)
    prev = list(range(len(s2) + 1))
    for c1 in s1:
        cur = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            cost = get_edit_cost(c1, c2)
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + cost))
        prev = cur
    return prev[-1]

def levenshtein(a, b):
    """Plain edit distance (unit costs, no normalisation) between two strings or lists."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def strict_cer_counts(pred, gt):
    """(errors, GT characters) for strict CER: plain Levenshtein on the raw strings (P0.2)."""
    return levenshtein(pred, gt), len(gt)


def cluster_error_counts(pred, gt):
    """(errors, GT clusters) for CluER: Levenshtein over clusters (consonant + its marks = 1 unit).

    A missed tone mark costs 1 of ~15 characters in CER but 1 of ~8 clusters here, so errors
    in the stacked tiers are not hidden by the many easy baseline characters.
    """
    g = split_clusters(gt)
    return levenshtein(split_clusters(pred), g), len(g)


def ctc_greedy_decode(output):
    """Greedy CTC decode: [T, C] → list of label indices."""
    preds = output.argmax(dim=1).cpu().tolist()
    decoded, prev = [], -1
    for p in preds:
        if p != 0 and p != prev:
            decoded.append(p)
        prev = p
    return decoded

def ctc_beam_decode(output, beam_width=10, dawg_decoder=None):
    """CTC beam search decode: [T, C] → list of label indices."""
    log_probs = torch.nn.functional.log_softmax(output, dim=1).cpu()
    T, C = log_probs.shape
    BLANK = 0

    beams = [(0.0, ())]
    for t in range(T):
        new_beams = {}
        for log_p, prefix in beams:
            for c in range(C):
                c_lp = log_probs[t, c].item()
                new_lp = log_p + c_lp

                if c == BLANK:
                    key = prefix
                elif prefix and prefix[-1] == c:
                    key = prefix
                else:
                    key = prefix + (c,)

                if dawg_decoder and key != prefix:
                    dawg_bonus = dawg_decoder.score_prefix(key)
                    new_lp += dawg_bonus

                if key not in new_beams or new_beams[key] < new_lp:
                    new_beams[key] = new_lp

        beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        beams = [(lp, pf) for pf, lp in beams]

    if not beams:
        return []

    if dawg_decoder:
        scored = []
        for lp, prefix in beams:
            final_bonus = dawg_decoder.score_final(prefix)
            scored.append((lp + final_bonus, prefix))
        scored.sort(key=lambda x: x[0], reverse=True)
        return list(scored[0][1])

    return list(beams[0][1])

def ctc_decode(output, beam_width=0, dawg_decoder=None):
    if beam_width <= 1:
        return ctc_greedy_decode(output)
    return ctc_beam_decode(output, beam_width=beam_width, dawg_decoder=dawg_decoder)

def align_strings(pred, truth):
    """Align strings and identify specific edit operations (mat, sub, ins, del)."""
    if isinstance(pred, str): pred = normalize_lao_text(pred)
    if isinstance(truth, str): truth = normalize_lao_text(truth)
    n, m = len(pred), len(truth)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    ops = [[None] * (m + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        dp[i][0] = i
        ops[i][0] = ('ins', pred[i-1], '')
    for j in range(1, m + 1):
        dp[0][j] = j
        ops[0][j] = ('del', '', truth[j-1])
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = get_edit_cost(pred[i-1], truth[j-1])
            if cost == 0:
                cop = 'mat'
            else:
                cop = 'sub'
                
            choices = [
                (dp[i-1][j-1] + cost, (cop, pred[i-1], truth[j-1])),
                (dp[i-1][j] + 1, ('ins', pred[i-1], '')),
                (dp[i][j-1] + 1, ('del', '', truth[j-1]))
            ]
            best_cost, best_op = min(choices, key=lambda x: x[0])
            dp[i][j] = best_cost
            ops[i][j] = best_op
            
    i, j = n, m
    edits = []
    while i > 0 or j > 0:
        op_t = ops[i][j]
        op_type = op_t[0]
        edits.append(op_t)
        if op_type in ('sub', 'mat'):
            i -= 1; j -= 1
        elif op_type == 'ins':
            i -= 1
        elif op_type == 'del':
            j -= 1
    return edits[::-1]

def compute_component_metrics(edits):
    """Calculate error stats per component category."""
    stats = {cat: {'total': 0, 'errors': 0} for cat in LAO_COMPONENT_MAP.keys()}
    stats['other'] = {'total': 0, 'errors': 0}
    
    for op, pred_c, truth_c in edits:
        if op == 'mat':
            cat = get_component_type(truth_c)
            stats[cat]['total'] += 1
        elif op == 'sub':
            cat = get_component_type(truth_c)
            stats[cat]['total'] += 1
            stats[cat]['errors'] += 1
        elif op == 'del':
            cat = get_component_type(truth_c)
            stats[cat]['total'] += 1
            stats[cat]['errors'] += 1
        elif op == 'ins':
            cat = get_component_type(pred_c)
            # Insertions don't have a 'total' in truth, so we only count errors
            stats[cat]['errors'] += 1
            
    return stats

def compute_metrics(model, loader, dataset, device, max_batches=50, collect_samples=0,
                    use_real_lengths=True, details=None):
    """CER / WER / line accuracy / CoER of a model over (part of) a loader.

    use_real_lengths: decode only the frames that belong to each line (P0.1), so a line's
    prediction doesn't depend on the padding added for wider lines in its batch.
    False reproduces the old behaviour (decode the whole padded width).
    details: optional dict, filled with 'cer_strict' (plain Levenshtein, no normalisation),
    'cluer' (cluster error rate) and the counts behind them. The returned CER stays the
    lenient one, so existing callers are unchanged.
    """
    from .model import output_lengths
    model.eval()
    total_char_dist, total_chars = 0, 0
    total_wer_dist, total_wer_words = 0, 0
    total_correct, total_samples = 0, 0
    coer_stats = {cat: {'total': 0, 'errors': 0} for cat in LAO_COMPONENT_MAP.keys()}
    coer_stats['other'] = {'total': 0, 'errors': 0}
    
    strict_err = strict_n = clu_err = clu_n = 0
    all_pairs = []
    batches = 0
    with torch.no_grad():
        for imgs, labels, ilens, tlens in loader:
            if batches >= max_batches: break
            imgs = imgs.to(device)
            out = model(imgs, ilens) if use_real_lengths else model(imgs)
            if out.dim() == 4:
                B, C, H, W = out.shape
                out = out.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)
            T = out.shape[1]
            olens = (output_lengths(model, ilens, T) if use_real_lengths
                     else torch.full((out.shape[0],), T, dtype=torch.long))
            off = 0
            for i in range(out.shape[0]):
                dec = ctc_greedy_decode(out[i, :olens[i].item()])
                gt = labels[off:off + tlens[i].item()].cpu().tolist()
                off += tlens[i].item()
                pred_s = dataset.decode_labels(dec)
                gt_s = dataset.decode_labels(gt)
                
                # CER (Character)
                total_char_dist += edit_distance(pred_s, gt_s)
                total_chars += max(len(gt_s), 1)
                
                # Strict CER and CluER (P0.2)
                e, n = strict_cer_counts(pred_s, gt_s)
                strict_err += e; strict_n += max(n, 1)
                e, n = cluster_error_counts(pred_s, gt_s)
                clu_err += e; clu_n += max(n, 1)

                # WER (Word)
                total_wer_dist += edit_distance(pred_s.split(), gt_s.split())
                total_wer_words += len(gt_s.split())
                
                # Component Error Rate (CoER)
                edits = align_strings(pred_s, gt_s)
                batch_coer = compute_component_metrics(edits)
                for cat in coer_stats:
                    coer_stats[cat]['total'] += batch_coer[cat]['total']
                    coer_stats[cat]['errors'] += batch_coer[cat]['errors']
                
                total_samples += 1
                if pred_s == gt_s:
                    total_correct += 1
                if collect_samples > 0 or collect_samples == -1:
                    all_pairs.append((gt_s, pred_s))
            batches += 1
    model.train()
    cer = total_char_dist / max(total_chars, 1)
    wer = total_wer_dist / max(total_wer_words, 1)
    acc = total_correct / max(total_samples, 1)
    if details is not None:
        details.update(cer_strict=strict_err / max(strict_n, 1), cluer=clu_err / max(clu_n, 1),
                       strict_errors=strict_err, chars=strict_n, cluster_errors=clu_err,
                       clusters=clu_n, lines=total_samples)
    
    # Calculate CoER results
    coer_results = {}
    for cat, s in coer_stats.items():
        if s['total'] > 0:   # undefined for a class absent from the ground truth
            coer_results[cat] = s['errors'] / s['total']
            
    samples = []
    if collect_samples > 0 and all_pairs:
        samples = random.sample(all_pairs, min(collect_samples, len(all_pairs)))
    elif collect_samples == -1:
        samples = all_pairs
    return cer, wer, acc, coer_results, samples

def compute_mean_rms(model):
    total_sq, total_n = 0.0, 0
    with torch.no_grad():
        for p in model.parameters():
            total_sq += (p ** 2).sum().item()
            total_n += p.numel()
    return math.sqrt(total_sq / max(total_n, 1))

class TrainingLogger:
    HEADER = ['iteration', 'timestamp', 'epoch', 'train_loss', 'eval_loss',
              'train_cer', 'train_wer', 'train_acc',
              'eval_cer', 'eval_wer', 'eval_acc',
              'mean_rms', 'learning_rate', 'gpu_mem_gb', 'samples_per_sec',
              'coer_source'] + [f'coer_{k}' for k in list(LAO_COMPONENT_MAP) + ['other']] + [
              'train_cer_strict', 'train_cluer', 'eval_cer_strict', 'eval_cluer']

    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        with open(log_file, 'w', newline='') as f:
            csv.writer(f).writerow(self.HEADER)

    def log(self, **kw):
        kw.setdefault('timestamp', datetime.now().isoformat())
        with open(self.log_file, 'a', newline='') as f:
            csv.writer(f).writerow([kw.get(h, '') for h in self.HEADER])

    @staticmethod
    def load(log_file):
        entries = []
        with open(log_file) as f:
            for row in csv.DictReader(f):
                for k in ['iteration', 'epoch']:
                    if row.get(k): row[k] = int(row[k])
                for k in ['train_loss', 'eval_loss', 'train_cer', 'eval_cer',
                           'train_wer', 'eval_wer', 'train_acc', 'eval_acc', 
                           'learning_rate', 'gpu_mem_gb', 'samples_per_sec', 'mean_rms',
                           'train_cer_strict', 'train_cluer', 'eval_cer_strict', 'eval_cluer']:
                    if row.get(k) is not None and row[k] != '':
                        row[k] = float(row[k])
                entries.append(row)
        return entries
