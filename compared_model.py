"""
compare_tesseract.py
─────────────────────────────────────────────────────────────────────────────
Manual comparison tool for compiled Tesseract .traineddata models.

Usage:
    python compare_tesseract.py

Requirements:
    pip install pytesseract pillow matplotlib numpy editdistance

Make sure your .traineddata files are placed in CUSTOM_TESSDATA_DIR:
    ./model/lao48cnn16.traineddata
    ./model/lao48cnn32.traineddata
    ...
─────────────────────────────────────────────────────────────────────────────
"""

import os
import csv
import glob
import random
import time
from collections import Counter, defaultdict

import pytesseract
from PIL import Image

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib.patches import FancyBboxPatch
    import numpy as np
    HAS_MATPLOT = True
except ImportError:
    HAS_MATPLOT = False
    print("  [Warning] matplotlib/numpy not found — plots will be skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# === CONFIG ===
# ─────────────────────────────────────────────────────────────────────────────
BASE_TEST_DIR        = '/content/output/eval'
IMAGE_FOLDER         = '/path/to/your/test/images'   # UPDATE THIS
CUSTOM_TESSDATA_DIR  = 'tesseract_data'
TESSERACT_PATH       = 'tesseract'
OUTPUT_DIR           = 'data/tesseract_comparison'
LAO_FONT_PATH        = 'tesstrain_core/font/phetsarath_ot.ttf'

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

CUSTOM_MODEL_NAMES = [
    'lao48cnn16',
    'lao48cnn32',
    'lao48cnn16t16',
    'lao48cnn16t32',
]

# Tesseract PSM and OEM settings (6 = single text line, 1 = LSTM)
TESS_CONFIG = '--psm 7 --oem 1'


# ─────────────────────────────────────────────────────────────────────────────
# Lao Component Map (mirrors metrics.py)
# ─────────────────────────────────────────────────────────────────────────────
LAO_COMPONENT_MAP = {
    'tone':         ['່', '້', '໊', '໋'],
    'consonant':    ['ກ','ຂ','ຄ','ງ','ຈ','ຊ','ຍ','ດ','ຕ','ຖ','ທ','ນ','ບ','ປ',
                     'ຜ','ຝ','ພ','ຟ','ມ','ຢ','ຣ','ລ','ວ','ສ','ຫ','ອ','ຮ','ໜ','ໝ','ຽ',
                     'ໆ', 'ຯ'],
    'lao_number':   ['໐','໑','໒','໓','໔','໕','໖','໗','໘','໙'],
    'digit':        ['0','1','2','3','4','5','6','7','8','9'],
    'symbol':       list('.,"?/!:;- +*=()[]{}\'"@#$%^&_|\\<>«»—…""\'\'•·°±×÷₭¥€£$'),
    'before_vowel': ['ແ','ໂ','ເ','ໄ','ໃ'],
    'rear_vowel':   ['ະ','າ','ຳ'],
    'upper_vowel':  ['ັ','ິ','ີ','ົ','ຶ','ື','໌','ໍ'],
    'lower_vowel':  ['ຸ','ູ'],
    'special':      ['ຼ'],
}

_CHAR_TO_CAT = {}
for _cat, _chars in LAO_COMPONENT_MAP.items():
    for _ch in _chars:
        _CHAR_TO_CAT[_ch] = _cat


def get_component_type(ch: str) -> str:
    return _CHAR_TO_CAT.get(ch, 'other')


# ─────────────────────────────────────────────────────────────────────────────
# Edit distance utilities
# ─────────────────────────────────────────────────────────────────────────────
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
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if pred[i-1] == truth[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if pred[i-1]==truth[j-1] else 1):
            op = 'mat' if pred[i-1]==truth[j-1] else 'sub'
            ops.append((op, pred[i-1], truth[j-1]))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(('ins', pred[i-1], None))
            i -= 1
        else:
            ops.append(('del', None, truth[j-1]))
            j -= 1
    return list(reversed(ops))


# ─────────────────────────────────────────────────────────────────────────────
# Test pair discovery
# ─────────────────────────────────────────────────────────────────────────────
def find_test_pairs(image_folder: str):
    """Find (image_path, ground_truth) pairs.

    Assumes each image has a matching .gt.txt file in the same folder,
    e.g.  line001.png  +  line001.gt.txt
    """
    pairs = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff'):
        for img_path in sorted(glob.glob(os.path.join(image_folder, ext))):
            base = os.path.splitext(img_path)[0]
            gt_path = base + '.gt.txt'
            if os.path.exists(gt_path):
                with open(gt_path, encoding='utf-8') as f:
                    gt = f.read().strip()
                pairs.append((img_path, gt))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Tesseract inference
# ─────────────────────────────────────────────────────────────────────────────
def run_tesseract(image_path: str, model_name: str) -> str:
    """Run Tesseract with the given custom model and return the predicted text."""
    config = f'--tessdata-dir "{CUSTOM_TESSDATA_DIR}" -l {model_name} {TESS_CONFIG}'
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, config=config)
        return text.strip()
    except Exception as e:
        return ''


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(predictions: dict, test_pairs: list):
    """
    predictions: {model_name: {img_path: pred_str}}
    Returns: {model_name: {cer, wer, coer, acc, eval_time, char_stats, mistake_counter}}
    """
    results = {}
    for model_name, pred_map in predictions.items():
        total_cer_num   = 0
        total_cer_den   = 0
        total_wer_dist  = 0
        total_wer_words = 0
        coer_stats = {cat: {'total': 0, 'errors': 0} for cat in LAO_COMPONENT_MAP}
        coer_stats['other'] = {'total': 0, 'errors': 0}
        char_stats  = defaultdict(lambda: {'total': 0, 'correct': 0, 'errors': 0})
        mistake_ctr = Counter()
        t0 = time.time()

        for img_path, truth in test_pairs:
            pred = pred_map.get(img_path, '')

            # CER
            d = edit_distance(pred, truth)
            total_cer_num += d
            total_cer_den += max(len(truth), 1)

            # WER
            tw = truth.split()
            pw = pred.split()
            total_wer_dist  += edit_distance(pw, tw)
            total_wer_words += max(len(tw), 1)

            # Alignment for CoER + character stats
            edits = align_strings(pred, truth)
            for op, p_c, t_c in edits:
                ch = t_c if t_c else p_c
                cat = get_component_type(ch)

                if t_c:
                    char_stats[t_c]['total'] += 1
                    coer_stats[cat]['total'] += 1

                if op != 'mat':
                    if t_c:
                        char_stats[t_c]['errors'] += 1
                        coer_stats[cat]['errors'] += 1
                    mistake_ctr[(op, p_c, t_c)] += 1
                else:
                    if t_c:
                        char_stats[t_c]['correct'] += 1

        eval_time = time.time() - t0
        cer = total_cer_num / max(total_cer_den, 1)
        wer = total_wer_dist / max(total_wer_words, 1)
        coer = {cat: s['errors'] / max(s['total'], 1)
                for cat, s in coer_stats.items() if s['total'] > 0}
        coer_avg = sum(coer.values()) / max(len(coer), 1)
        acc = max(0.0, 1.0 - (cer + wer + coer_avg) / 3.0)

        results[model_name] = {
            'cer': cer, 'wer': wer, 'coer': coer, 'acc': acc,
            'eval_time': eval_time,
            'char_stats': char_stats,
            'mistake_ctr': mistake_ctr,
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
COER_COLS = [
    ('tone',         'Tone Mark'),
    ('upper_vowel',  'Upper Vowel'),
    ('consonant',    'Consonant'),
    ('before_vowel', 'Before Vowel'),
    ('rear_vowel',   'After Vowel'),
    ('lower_vowel',  'Lower Vowel'),
    ('special',      'Special'),
    ('symbol',       'Symbol'),
    ('digit',        'Digit'),
    ('lao_number',   'Lao Number'),
]


def print_summary(results: dict, models: list):
    print("\n  ==== Tesseract Model Comparison Results ====")
    header = f"  {'Model':<22} {'CER (%)':>10} {'WER (%)':>10} {'Accuracy (%)':>13} {'Time (s)':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in models:
        r = results.get(m)
        if r:
            print(f"  {m:<22} {r['cer']*100:>10.2f} {r['wer']*100:>10.2f} {r['acc']*100:>13.2f} {r['eval_time']:>10.1f}")
        else:
            print(f"  {m:<22} {'--':>10} {'--':>10} {'--':>13} {'--':>10}")
    print()


def export_csv_report(results: dict, models: list, out_dir: str):
    # Global metrics
    csv_path = os.path.join(out_dir, 'report.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['Model', 'CER (%)', 'WER (%)', 'Accuracy (%)', 'Eval Time (s)']
        for _, col_name in COER_COLS:
            header.append(f'CoER {col_name} (%)')
        writer.writerow(header)
        for m in models:
            r = results.get(m)
            if not r: continue
            row = [m, f"{r['cer']*100:.2f}", f"{r['wer']*100:.2f}",
                   f"{r['acc']*100:.2f}", f"{r['eval_time']:.1f}"]
            for cat, _ in COER_COLS:
                row.append(f"{r['coer'].get(cat, 0)*100:.2f}")
            writer.writerow(row)
    print(f"  ✓ Global CSV saved to: {csv_path}")

    # Character-level report
    all_chars = sorted(set(
        ch for m in models if results.get(m)
        for ch in results[m]['char_stats'].keys()
    ))
    char_csv = os.path.join(out_dir, 'report_characters.csv')
    with open(char_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['Character', 'Category']
        for m in models:
            header += [f'{m} GT Occurrences', f'{m} Correct', f'{m} Errors', f'{m} Error Rate (%)']
        writer.writerow(header)
        for ch in all_chars:
            cat = get_component_type(ch)
            row = [ch, cat]
            for m in models:
                cs = results[m]['char_stats'].get(ch, {'total': 0, 'correct': 0, 'errors': 0}) if results.get(m) else {}
                t = cs.get('total', 0)
                c = cs.get('correct', 0)
                e = cs.get('errors', 0)
                rate = (e / t * 100) if t > 0 else 0.0
                row += [t, c, e, f"{rate:.1f}"]
            writer.writerow(row)
    print(f"  ✓ Character CSV saved to: {char_csv}")

    # Mistakes report
    mistakes_csv = os.path.join(out_dir, 'report_mistakes.csv')
    with open(mistakes_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Operation', 'GT Char', 'Pred Char', 'Count'])
        for m in models:
            if not results.get(m): continue
            ctr = results[m]['mistake_ctr']
            errors = [(op, p, t, cnt) for (op, p, t), cnt in ctr.items() if op != 'mat']
            errors.sort(key=lambda x: x[3], reverse=True)
            for op, p, t, cnt in errors:
                writer.writerow([m, op, t or '', p or '', cnt])
    print(f"  ✓ Mistakes CSV saved to: {mistakes_csv}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
def _lao_font(size=12):
    if os.path.exists(LAO_FONT_PATH):
        return fm.FontProperties(fname=LAO_FONT_PATH, size=size)
    return fm.FontProperties(family='monospace', size=size)


def plot_global_bar(results: dict, models: list, out_dir: str):
    """Grouped bar chart comparing CER, WER, Avg CoER and Accuracy per model."""
    valid = [m for m in models if results.get(m)]
    if not valid:
        return

    colors = plt.get_cmap('tab10').colors
    x     = np.arange(len(valid))
    width = 0.2

    cer_v  = [results[m]['cer'] * 100 for m in valid]
    wer_v  = [results[m]['wer'] * 100 for m in valid]
    coer_v = [(sum(results[m]['coer'].values()) / max(len(results[m]['coer']), 1)) * 100
              if results[m]['coer'] else 0.0 for m in valid]
    acc_v  = [results[m]['acc'] * 100 for m in valid]

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width*1.5, cer_v,  width, label='CER (%)',      color=colors[0])
    rects2 = ax.bar(x - width*0.5, wer_v,  width, label='WER (%)',      color=colors[1])
    rects3 = ax.bar(x + width*0.5, coer_v, width, label='Avg CoER (%)', color=colors[2])
    rects4 = ax.bar(x + width*1.5, acc_v,  width, label='Accuracy (%)', color=colors[3])

    for rects in [rects1, rects2, rects3, rects4]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9,
                        fontweight='bold', rotation=90 if h > 0 else 0)

    ax.set_ylabel('Percentage (%)', fontweight='bold', fontsize=12)
    ax.set_title('Tesseract Model — Global Performance', fontweight='bold', fontsize=14, pad=16)
    ax.set_xticks(x)
    ax.set_xticklabels(valid, fontweight='bold', fontsize=11)
    ax.legend(fontsize=11, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    path = os.path.join(out_dir, 'plot_global_bar.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Global bar chart saved to: {path}")



def plot_coer_bar(results: dict, models: list, out_dir: str):
    valid = [m for m in models if results.get(m)]
    if not valid: return

    cats   = [c for c, _ in COER_COLS if any(results[m]['coer'].get(c) is not None for m in valid)]
    
    # Sort categories from 'less to more' based on average error rate
    def cat_avg_error(c):
        vals = [results[m]['coer'].get(c, 0) for m in valid]
        return sum(vals) / len(vals) if vals else 0
        
    cats.sort(key=cat_avg_error)
    
    labels = [dict(COER_COLS)[c] for c in cats]
    colors = plt.get_cmap('tab10').colors
    x      = np.arange(len(cats))
    width  = 0.8 / max(len(valid), 1)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, m in enumerate(valid):
        vals = [results[m]['coer'].get(c, 0) * 100 for c in cats]
        offset = (i - len(valid) / 2 + 0.5) * width
        rects = ax.bar(x + offset, vals, width, label=m, color=colors[i % len(colors)])
        for rect in rects:
            h = rect.get_height()
            if h > 0.5:
                ax.annotate(f'{h:.1f}', xy=(rect.get_x() + rect.get_width()/2, h),
                            xytext=(0, 2), textcoords='offset points',
                            ha='center', va='bottom', fontsize=7, fontweight='bold', rotation=90)

    ax.set_ylabel('Error Rate (%)', fontweight='bold')
    ax.set_title('Component-Level Error Rate (CoER)', fontweight='bold', fontsize=14, pad=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    path = os.path.join(out_dir, 'plot_coer_bar.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ CoER bar chart saved to: {path}")


def plot_demo_grid(predictions: dict, test_pairs: list, models: list, out_dir: str):
    """Pick a random sample, show the image + GT/Pred alignment for each model."""
    if not test_pairs: return
    valid = [m for m in models if m in predictions]
    if not valid: return

    # Pick a sample with at least one error
    demo_pair = None
    candidates = [(img, gt) for img, gt in test_pairs
                  if any(predictions[m].get(img, gt) != gt for m in valid)]
    if candidates:
        demo_pair = random.choice(candidates)
    else:
        demo_pair = random.choice(test_pairs)

    img_path, gt = demo_pair
    try:
        pil_img = Image.open(img_path)
    except Exception:
        return

    fp_sm = _lao_font(11)
    op_colors = {'mat': '#2ca02c', 'sub': '#d62728', 'del': '#e08000', 'ins': '#9467bd'}
    op_labels = {'mat': 'MATCH', 'sub': 'SUB', 'del': 'DEL', 'ins': 'INS'}
    MAX_COLS = 20
    CELL_W   = 14.0 / MAX_COLS
    CHAR_H   = 0.52
    ARROW_H  = 0.38
    LABEL_H  = 0.30

    img_w_px, img_h_px = pil_img.size
    IMG_H = max(14.0 * (img_h_px / max(img_w_px, 1)), 0.8)
    panel_h = 0.40 + IMG_H + 0.10 + CHAR_H + ARROW_H + CHAR_H + LABEL_H + 0.15

    fig, axs = plt.subplots(len(valid), 1, figsize=(14, panel_h * len(valid) + 0.5))
    if len(valid) == 1: axs = [axs]
    fig.suptitle('Tesseract Random Sample Comparison', fontsize=14, fontweight='bold', y=1.0)

    for i, model_name in enumerate(valid):
        pred_str = predictions[model_name].get(img_path, '')
        edits    = align_strings(pred_str, gt)[:MAX_COLS]
        total_h  = panel_h
        ax = axs[i]
        ax.set_xlim(0, 14); ax.set_ylim(0, total_h); ax.axis('off')
        cy = total_h - 0.05
        ax.text(0.15, cy, f"Model: {model_name}   |   GT: {gt}   |   Pred: {pred_str}",
                fontsize=10, fontweight='bold', va='top', color='#222222', clip_on=False)
        cy -= 0.38
        inset = ax.inset_axes([0.0, (cy - IMG_H) / total_h, 1.0, IMG_H / total_h])
        inset.imshow(pil_img, cmap='gray', aspect='auto',
                     extent=[0, img_w_px, 0, img_h_px])
        inset.set_xlim(0, img_w_px); inset.set_ylim(0, img_h_px); inset.axis('off')
        cy -= IMG_H + 0.10
        TRUTH_Y = cy - CHAR_H / 2
        ARROW_Y = cy - CHAR_H - ARROW_H / 2
        PRED_Y  = cy - CHAR_H - ARROW_H - CHAR_H / 2
        LBL_Y   = cy - CHAR_H - ARROW_H - CHAR_H - LABEL_H / 2
        ax.text(-0.1, TRUTH_Y, "GT:",   fontsize=8, fontweight='bold', ha='right', va='center', clip_on=False)
        ax.text(-0.1, PRED_Y,  "Pred:", fontsize=8, fontweight='bold', ha='right', va='center', clip_on=False)
        for col, (op, p_c, t_c) in enumerate(edits):
            color = op_colors.get(op, '#888888')
            cx = col * CELL_W + CELL_W / 2
            if t_c:
                ax.add_patch(FancyBboxPatch((cx - CELL_W*0.44, TRUTH_Y - CHAR_H*0.5),
                                            CELL_W*0.88, CHAR_H, boxstyle="round,pad=0.03",
                                            fc='white', ec=color, lw=1.8, clip_on=False))
                ax.text(cx, TRUTH_Y, t_c, fontproperties=fp_sm, ha='center', va='center', clip_on=False)
            ax.text(cx, ARROW_Y, '↓' if op != 'ins' else '↑',
                    fontsize=10, ha='center', va='center', color=color, fontweight='bold', clip_on=False)
            if p_c:
                ax.add_patch(FancyBboxPatch((cx - CELL_W*0.44, PRED_Y - CHAR_H*0.5),
                                            CELL_W*0.88, CHAR_H, boxstyle="round,pad=0.03",
                                            fc='white', ec=color, lw=1.8, clip_on=False))
                ax.text(cx, PRED_Y, p_c, fontproperties=fp_sm, ha='center', va='center', clip_on=False)
            ax.text(cx, LBL_Y, op_labels.get(op, op),
                    fontsize=6, ha='center', va='center', color=color, fontweight='bold', clip_on=False)

    plt.tight_layout(pad=0.6, h_pad=0.8)
    path = os.path.join(out_dir, 'plot_demo_grid.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Demo grid saved to: {path}")


def plot_metric_explanation(predictions: dict, test_pairs: list,
                             models: list, out_dir: str):
    """One metric-explanation diagram per model, all using the same 2 GT samples.

    Each diagram shows:
      • CoER section  — every GT character colour-coded by linguistic category
      • CER/WER section — character-level alignment grid (GT → Pred arrows)
    Saved as plot_metric_explanation_{model}.png.
    """
    valid = [m for m in models if m in predictions]
    if not valid or not test_pairs:
        return

    lao_font_path = LAO_FONT_PATH
    lao_font_lg = fm.FontProperties(fname=lao_font_path, size=16) if os.path.exists(lao_font_path) else fm.FontProperties(family='monospace', size=16)
    lao_font_sm = fm.FontProperties(fname=lao_font_path, size=13) if os.path.exists(lao_font_path) else fm.FontProperties(family='monospace', size=13)

    cat_colors = {
        'consonant':    '#1f77b4', 'tone':        '#d62728',
        'before_vowel': '#2ca02c', 'rear_vowel':  '#9467bd',
        'upper_vowel':  '#ff7f0e', 'lower_vowel': '#8c564b',
        'symbol':       '#7f7f7f', 'lao_number':  '#17becf',
        'digit':        '#bcbd22', 'special':     '#e377c2',
        'other':        '#aaaaaa',
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

    # ── Pick 2 shared GT image paths once (same across all models) ─────────
    _ref = valid[0]
    _err = [(f, t) for f, t in test_pairs
            if predictions[_ref].get(f, t) != t and 3 <= len(t) <= 25]
    _shared = random.sample(_err, 2) if len(_err) >= 2 else _err[:]
    if len(_shared) < 2:
        _pool = [(f, t) for f, t in test_pairs
                 if (f, t) not in _shared and 3 <= len(t) <= 25]
        random.shuffle(_pool)
        _shared += _pool[:2 - len(_shared)]
    if not _shared and test_pairs:
        _shared = [test_pairs[0]]
    if not _shared:
        print("  [Metric Explanation] No test pairs available.")
        return

    def _sample_h(truth, pred):
        cc  = list(truth)[:MAX_COLS * 3]
        n_c = len([cc[i:i+MAX_COLS] for i in range(0, len(cc), MAX_COLS)])
        ed  = align_strings(pred, truth)
        n_e = len([ed[i:i+MAX_COLS] for i in range(0, len(ed[:MAX_COLS*3]), MAX_COLS)])
        return (HDR_H + n_c*(CHAR_H+LABEL_H+ROW_GAP)
                + SEP_H + HDR_H + n_e*EDIT_ROW_H + FORMULA_H)

    n_leg_rows     = (len(cat_colors) - 1) // 5 + 1
    legend_block_h = 0.44 + n_leg_rows * 0.30

    for demo_m in valid:
        try:
            demo_samples = [(truth, predictions[demo_m].get(img_f, truth))
                            for img_f, truth in _shared]
            if not demo_samples:
                continue

            fig_h = (sum(_sample_h(t, p) for t, p in demo_samples)
                     + legend_block_h
                     + (len(demo_samples) - 1) * SEP_H + 0.5)

            fig, ax = plt.subplots(1, 1, figsize=(14, fig_h))
            ax.axis('off')
            ax.set_xlim(0, 14)
            ax.set_ylim(0, fig_h)
            ax.set_title(f'Model: {demo_m}', fontsize=12, fontweight='bold',
                         loc='right', pad=6, color='#555555')
            cy = fig_h - 0.15

            for s_idx, (demo_truth, demo_pred) in enumerate(demo_samples):
                coer_chars = list(demo_truth)[:MAX_COLS * 3]
                coer_rows  = [coer_chars[i:i+MAX_COLS]
                              for i in range(0, len(coer_chars), MAX_COLS)]
                edits      = align_strings(demo_pred, demo_truth)
                edit_rows  = [edits[i:i+MAX_COLS]
                              for i in range(0, len(edits[:MAX_COLS*3]), MAX_COLS)]

                lbl = f'Sample {s_idx+1}  —  CoER:  GT = "{demo_truth}"'
                ax.text(0.15, cy, lbl, fontsize=13, fontweight='bold',
                        va='top', color='#222222', clip_on=False)
                cy -= HDR_H

                for row in coer_rows:
                    char_cy  = cy - CHAR_H / 2
                    label_cy = cy - CHAR_H - LABEL_H / 2
                    for col, ch in enumerate(row):
                        cat   = get_component_type(ch)
                        color = cat_colors.get(cat, '#aaaaaa')
                        cx    = col * CELL_W + CELL_W / 2
                        ax.add_patch(FancyBboxPatch(
                            (cx-CELL_W*0.42, char_cy-CHAR_H*0.5),
                            CELL_W*0.84, CHAR_H, boxstyle='round,pad=0.04',
                            fc=color+'22', ec=color, lw=1.8, clip_on=False))
                        ax.text(cx, char_cy, ch, fontproperties=lao_font_lg,
                                ha='center', va='center', color='#111111', clip_on=False)
                        ax.text(cx, label_cy, cat.replace('_', '\n'),
                                fontsize=6.5, ha='center', va='center',
                                color=color, fontweight='bold',
                                linespacing=1.15, clip_on=False)
                    cy -= CHAR_H + LABEL_H + ROW_GAP

                if s_idx == 0:
                    cy -= 0.05
                    ax.plot([0.1, 13.9], [cy, cy], color='#cccccc', lw=0.8)
                    cy -= 0.22
                    for li, (cat, color) in enumerate(cat_colors.items()):
                        lx = (li % 5) * (14.0 / 5) + 0.25
                        ly = cy - (li // 5) * 0.30
                        ax.add_patch(FancyBboxPatch(
                            (lx-0.18, ly-0.09), 0.18, 0.18,
                            boxstyle='round,pad=0.02',
                            fc=color+'44', ec=color, lw=1, clip_on=False))
                        ax.text(lx+0.10, ly, cat.replace('_', ' '),
                                fontsize=7.5, va='center', color=color, fontweight='bold')
                    cy -= n_leg_rows * 0.30 + 0.22

                ax.plot([0.1, 13.9], [cy, cy], color='#cccccc', lw=0.8)
                cy -= SEP_H

                lbl2 = f'Sample {s_idx+1}  —  CER & WER Alignment'
                ax.text(0.15, cy, lbl2, fontsize=13, fontweight='bold',
                        va='top', color='#222222', clip_on=False)
                cy -= HDR_H

                op_colors = {'mat': '#2ca02c', 'sub': '#d62728',
                             'del': '#e08000', 'ins': '#9467bd'}
                op_labels = {'mat': 'MATCH', 'sub': 'SUB',
                             'del': 'DEL',   'ins': 'INS'}

                for row_edits in edit_rows:
                    TRUTH_Y = cy - 0.40
                    ARROW_Y = cy - 0.90
                    PRED_Y  = cy - 1.40
                    OPLBL_Y = cy - 1.78
                    ax.text(-0.05, TRUTH_Y, 'GT:',   fontsize=9, fontweight='bold',
                            ha='right', va='center', clip_on=False)
                    ax.text(-0.05, PRED_Y,  'Pred:', fontsize=9, fontweight='bold',
                            ha='right', va='center', clip_on=False)
                    for col, (op, p_c, t_c) in enumerate(row_edits):
                        color = op_colors.get(op, '#888888')
                        cx = col * CELL_W + CELL_W / 2
                        if t_c:
                            ax.add_patch(FancyBboxPatch(
                                (cx-CELL_W*0.42, TRUTH_Y-0.28),
                                CELL_W*0.84, 0.56, boxstyle='round,pad=0.04',
                                fc='white', ec=color, lw=2, clip_on=False))
                            ax.text(cx, TRUTH_Y, t_c, fontproperties=lao_font_sm,
                                    ha='center', va='center', clip_on=False)
                        ax.text(cx, ARROW_Y, '↓' if op != 'ins' else '↑',
                                fontsize=13, ha='center', va='center',
                                color=color, fontweight='bold', clip_on=False)
                        if p_c:
                            ax.add_patch(FancyBboxPatch(
                                (cx-CELL_W*0.42, PRED_Y-0.28),
                                CELL_W*0.84, 0.56, boxstyle='round,pad=0.04',
                                fc='white', ec=color, lw=2, clip_on=False))
                            ax.text(cx, PRED_Y, p_c, fontproperties=lao_font_sm,
                                    ha='center', va='center', clip_on=False)
                        ax.text(cx, OPLBL_Y, op_labels.get(op, op),
                                fontsize=7, ha='center', va='top',
                                color=color, fontweight='bold', clip_on=False)
                    cy -= EDIT_ROW_H

                cer_val = edit_distance(demo_pred, demo_truth) / max(len(demo_truth), 1)
                wer_val = (edit_distance(demo_pred.split(), demo_truth.split())
                           / max(len(demo_truth.split()), 1))
                formula = (f'Results:  CER = {cer_val*100:.1f}%  |  WER = {wer_val*100:.1f}%\n'
                           'Formula:  CER = (Sub + Del + Ins) / Total GT Chars')
                ax.text(0.2, cy - 0.05, formula, fontsize=9, style='italic', va='top',
                        bbox=dict(fc='#fffde7', ec='#f0c000',
                                  boxstyle='round,pad=0.5', lw=1.2),
                        clip_on=False)
                cy -= FORMULA_H + 0.15

                if s_idx < len(demo_samples) - 1:
                    ax.plot([0.1, 13.9], [cy, cy],
                            color='#888888', lw=1.5, linestyle='--')
                    cy -= SEP_H

            plt.subplots_adjust(left=0.06, right=0.98, top=0.99, bottom=0.01)
            expl_path = os.path.join(out_dir, f'plot_metric_explanation_{demo_m}.png')
            plt.savefig(expl_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  ✓ Metric Explanation [{demo_m}] → {expl_path}")
        except Exception as _me:
            print(f"  [Metric Explanation Error — {demo_m}] {_me}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Tesseract Model Comparison")
    print("=" * 70)
    print(f"  Models       : {', '.join(CUSTOM_MODEL_NAMES)}")
    print(f"  Tessdata dir : {CUSTOM_TESSDATA_DIR}")
    print(f"  Image folder : {IMAGE_FOLDER}")
    print(f"  Output dir   : {OUTPUT_DIR}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Validate tessdata ─────────────────────────────────────────
    missing = [m for m in CUSTOM_MODEL_NAMES
               if not os.path.exists(os.path.join(CUSTOM_TESSDATA_DIR, f'{m}.traineddata'))]
    if missing:
        print(f"  [Warning] Missing .traineddata files: {missing}")
        print(f"  Place them in:  {CUSTOM_TESSDATA_DIR}/")

    # ── Find test pairs ───────────────────────────────────────────
    test_pairs = find_test_pairs(IMAGE_FOLDER)
    if not test_pairs:
        print(f"  ERROR: No image+ground-truth pairs found in {IMAGE_FOLDER}")
        print("  Each image needs a matching .gt.txt file (same base name).")
        return
    print(f"  Found {len(test_pairs)} test pairs.\n")

    # ── Run inference ─────────────────────────────────────────────
    predictions = {}   # {model: {img_path: pred_str}}
    for model_name in CUSTOM_MODEL_NAMES:
        td_path = os.path.join(CUSTOM_TESSDATA_DIR, f'{model_name}.traineddata')
        if not os.path.exists(td_path):
            print(f"  [Skip] {model_name} — .traineddata not found.")
            continue
        print(f"  Running {model_name} on {len(test_pairs)} images...")
        pred_map = {}
        t0 = time.time()
        for idx, (img_path, _) in enumerate(test_pairs):
            pred_map[img_path] = run_tesseract(img_path, model_name)
            if (idx + 1) % 100 == 0:
                print(f"    {idx+1}/{len(test_pairs)} done  ({time.time()-t0:.1f}s)")
        predictions[model_name] = pred_map
        print(f"    Finished {model_name} in {time.time()-t0:.1f}s")

    if not predictions:
        print("  ERROR: No models produced predictions.")
        return

    # ── Compute metrics ───────────────────────────────────────────
    print("\n  Computing metrics...")
    results = compute_metrics(predictions, test_pairs)

    # ── Print summary ─────────────────────────────────────────────
    print_summary(results, list(predictions.keys()))

    # ── CoER summary ──────────────────────────────────────────────
    print("  ==== Component-Level Error Rate (CoER) ====")
    col_names = [name for _, name in COER_COLS]
    header = f"  {'Model':<22} " + " ".join(f"{n[:10]:>11}" for n in col_names)
    print(header)
    for m in predictions:
        r = results.get(m)
        if not r: continue
        row = f"  {m:<22} "
        for cat, _ in COER_COLS:
            v = r['coer'].get(cat, 0) * 100
            row += f"{v:>11.2f}"
        print(row)
    print()

    # ── Top 3 mistakes ────────────────────────────────────────────
    print("  ==== Top 3 Mistakes per Model ====")
    for m in predictions:
        r = results.get(m)
        if not r: continue
        errors = [(op, p, t, cnt) for (op, p, t), cnt in r['mistake_ctr'].items() if op != 'mat']
        errors.sort(key=lambda x: x[3], reverse=True)
        parts = []
        for op, p, t, cnt in errors[:3]:
            if op == 'sub': parts.append(f"'{t}'→'{p}' ({cnt})")
            elif op == 'del': parts.append(f"Missed '{t}' ({cnt})")
            elif op == 'ins': parts.append(f"Extra '{p}' ({cnt})")
        print(f"  [{m}] {', '.join(parts) or 'None'}")
    print()

    # ── Export CSVs ───────────────────────────────────────────────
    export_csv_report(results, list(predictions.keys()), OUTPUT_DIR)

    # ── Plots ─────────────────────────────────────────────────────
    if HAS_MATPLOT:
        print("  Generating plots...")
        plot_global_bar(results, list(predictions.keys()), OUTPUT_DIR)
        plot_coer_bar(results, list(predictions.keys()), OUTPUT_DIR)
        plot_demo_grid(predictions, test_pairs, list(predictions.keys()), OUTPUT_DIR)
        plot_metric_explanation(predictions, test_pairs, list(predictions.keys()), OUTPUT_DIR)
        print()

    print("=" * 70)
    print(f"  All outputs saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
