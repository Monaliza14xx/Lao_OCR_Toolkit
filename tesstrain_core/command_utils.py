import glob
import os
import random
import shutil
import subprocess
import sys
import json
import unicodedata
from collections import Counter
from .utils import resolve_paths
from .metrics import TrainingLogger


def cmd_charset_stats(args):
    """Analyze character frequencies from .gt.txt ground truth files."""
    gt_dir = getattr(args, 'ground_truth_dir', '') or ''
    if not gt_dir:
        # Fallback: use data_dir/model_name-ground-truth
        data_dir = getattr(args, 'data_dir', 'data')
        model_name = getattr(args, 'model_name', 'foo')
        gt_dir = os.path.join(data_dir, f'{model_name}-ground-truth')

    gt_files = sorted(glob.glob(os.path.join(gt_dir, '**', '*.gt.txt'), recursive=True))
    if not gt_files:
        print(f"  Error: No .gt.txt files found in {gt_dir}")
        sys.exit(1)

    # Count characters
    char_counter = Counter()
    total_lines = 0
    total_files = len(gt_files)
    for fp in gt_files:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    text = line.strip()
                    if text:
                        char_counter.update(text)
                        total_lines += 1
        except Exception as e:
            print(f"  Warning: {fp}: {e}")

    total_chars = sum(char_counter.values())
    unique_chars = len(char_counter)

    if total_chars == 0:
        print("  No characters found!")
        return

    # Sort by frequency (descending)
    sorted_chars = char_counter.most_common()

    # Categorize
    categories = Counter()
    for ch, _ in sorted_chars:
        cat = unicodedata.category(ch)
        categories[cat] += char_counter[ch]

    # Print header
    print(f"\n  ╔══════════════════════════════════════════════════════════════════════╗")
    print(f"  ║  Character Analytics — {gt_dir}")
    print(f"  ╠══════════════════════════════════════════════════════════════════════╣")
    print(f"  ║  Files: {total_files:,}    Lines: {total_lines:,}    "
          f"Total chars: {total_chars:,}    Unique: {unique_chars}")
    print(f"  ╚══════════════════════════════════════════════════════════════════════╝\n")

    # Print table header
    print(f"  {'#':>5}  {'Char':>6}  {'Unicode':>10}  {'Count':>10}  {'%':>7}  {'Category':<20}  Name")
    print(f"  {'─'*5}  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*7}  {'─'*20}  {'─'*30}")

    for rank, (ch, count) in enumerate(sorted_chars, 1):
        pct = count / total_chars * 100
        codepoint = f"U+{ord(ch):04X}"

        # Display character (handle whitespace/control chars)
        if ch == ' ':
            display = '␣'
        elif ch == '\t':
            display = '⇥'
        elif unicodedata.category(ch).startswith('C'):
            display = '·'
        else:
            display = ch

        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = f"<{unicodedata.category(ch)}>"

        cat_full = unicodedata.category(ch)
        cat_desc = {
            'Lu': 'Letter, uppercase',
            'Ll': 'Letter, lowercase',
            'Lt': 'Letter, titlecase',
            'Lm': 'Letter, modifier',
            'Lo': 'Letter, other',
            'Mn': 'Mark, nonspacing',
            'Mc': 'Mark, combining',
            'Me': 'Mark, enclosing',
            'Nd': 'Number, digit',
            'Nl': 'Number, letter',
            'No': 'Number, other',
            'Pc': 'Punct, connector',
            'Pd': 'Punct, dash',
            'Ps': 'Punct, open',
            'Pe': 'Punct, close',
            'Pi': 'Punct, initial',
            'Pf': 'Punct, final',
            'Po': 'Punct, other',
            'Sm': 'Symbol, math',
            'Sc': 'Symbol, currency',
            'Sk': 'Symbol, modifier',
            'So': 'Symbol, other',
            'Zs': 'Separator, space',
            'Zl': 'Separator, line',
            'Zp': 'Separator, paragraph',
            'Cc': 'Control',
            'Cf': 'Format',
        }.get(cat_full, cat_full)

        print(f"  {rank:>5}  {display:>6}  {codepoint:>10}  {count:>10,}  {pct:>6.2f}%  {cat_desc:<20}  {name}")

    # Category summary
    print(f"\n  ── Category Summary ──────────────────────────────")
    cat_sorted = sorted(categories.items(), key=lambda x: -x[1])
    for cat, count in cat_sorted:
        pct = count / total_chars * 100
        cat_desc = {
            'Lu': 'Uppercase letters', 'Ll': 'Lowercase letters',
            'Lo': 'Other letters', 'Mn': 'Nonspacing marks',
            'Mc': 'Combining marks', 'Nd': 'Digits',
            'Po': 'Punctuation', 'Zs': 'Spaces',
            'Pd': 'Dashes', 'Ps': 'Open brackets',
            'Pe': 'Close brackets', 'Sc': 'Currency',
            'Sm': 'Math symbols', 'So': 'Other symbols',
            'Cc': 'Control chars', 'Cf': 'Format chars',
        }.get(cat, cat)
        print(f"  {cat_desc:<25}  {count:>10,}  ({pct:5.1f}%)")

    print(f"\n  Total: {total_chars:,} characters, {unique_chars} unique\n")

    # Save to CSV
    output_dir = getattr(args, 'output_dir', '') or os.path.join(
        getattr(args, 'data_dir', 'data'), getattr(args, 'model_name', 'foo'))
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'charset_stats.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('rank,char,unicode,count,percentage,category,name\n')
        for rank, (ch, count) in enumerate(sorted_chars, 1):
            pct = count / total_chars * 100
            codepoint = f"U+{ord(ch):04X}"
            try:
                name = unicodedata.name(ch).replace(',', ' ')
            except ValueError:
                name = unicodedata.category(ch)
            # Escape the char for CSV
            escaped = ch.replace('"', '""')
            f.write(f'{rank},"{escaped}",{codepoint},{count},{pct:.4f},{unicodedata.category(ch)},{name}\n')

    print(f"  ✓ Saved: {csv_path}\n")

def cmd_unicharset(args):
    """Extract unicharset from .gt.txt ground truth files."""
    p = resolve_paths(args)
    gt_files = sorted(glob.glob(os.path.join(p.ground_truth_dir, '**', '*.gt.txt'), recursive=True))
    if not gt_files:
        print(f"Error: No .gt.txt files in {p.ground_truth_dir}")
        sys.exit(1)

    print(f"  Found {len(gt_files)} ground truth files")

    tool = shutil.which('unicharset_extractor')
    if tool and len(gt_files) < 500:
        cmd = [tool, '--output_unicharset', p.unicharset_file, '--norm_mode', '1'] + gt_files
        print(f"  Running unicharset_extractor...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✓ Unicharset: {p.unicharset_file}")
            return
        print(f"  Warning: unicharset_extractor failed, using Python fallback")
    elif tool and len(gt_files) >= 500:
        import tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as tmp:
            tmp.write('\n'.join(gt_files) + '\n')
            tmp_path = tmp.name
        try:
            xargs_cmd = f"cat {tmp_path} | xargs {tool} --output_unicharset {p.unicharset_file} --norm_mode 1"
            print(f"  Running unicharset_extractor via xargs ({len(gt_files)} files)...")
            result = subprocess.run(xargs_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(p.unicharset_file):
                os.unlink(tmp_path)
                print(f"  ✓ Unicharset: {p.unicharset_file}")
                return
        except Exception:
            pass
        try: os.unlink(tmp_path)
        except: pass
        print(f"  Warning: unicharset_extractor failed, using Python fallback")

    all_chars = set()
    for gt_file in gt_files:
        try:
            with open(gt_file, 'r', encoding='utf-8') as f:
                all_chars.update(f.read().strip())
        except Exception:
            pass

    all_chars.discard('\n')
    all_chars.discard('\r')
    sorted_chars = sorted(all_chars)

    with open(p.unicharset_file, 'w', encoding='utf-8') as f:
        f.write(f"{len(sorted_chars) + 1}\n")
        f.write("NULL 0 0,255,0,255,0,0,0,0,0,0 NULL 1\n")
        for i, ch in enumerate(sorted_chars):
            props = "0,255,0,255,0,0,0,0,0,0"
            if ch.isalpha():
                props = "1,255,0,255,0,0,0,0,0,0"
            elif ch.isdigit():
                props = "0,255,0,255,1,0,0,0,0,0"
            f.write(f"{ch} {props} {ch} {i + 2}\n")

    charset = {c: i + 1 for i, c in enumerate(sorted_chars)}
    with open(os.path.join(p.output_dir, 'charset.json'), 'w') as f:
        json.dump(charset, f, ensure_ascii=False, indent=2)

    print(f"  ✓ Unicharset: {p.unicharset_file} ({len(sorted_chars)} chars)")
    print(f"  ✓ Charset JSON: {os.path.join(p.output_dir, 'charset.json')}")

def cmd_lists(args):
    """Create train/eval/valid .lstmf file lists."""
    p = resolve_paths(args)
    lstmf_files = sorted(glob.glob(os.path.join(p.ground_truth_dir, '**', '*.lstmf'), recursive=True))
    if not lstmf_files:
        print(f"Error: No .lstmf files in {p.ground_truth_dir}")
        sys.exit(1)

    random.seed(args.random_seed)
    random.shuffle(lstmf_files)
    n = len(lstmf_files)

    ratio = getattr(args, 'ratio', None)

    if ratio and len(ratio) == 3:
        # 3-way split: train / eval / valid (percentages)
        r_train, r_eval, r_valid = ratio
        total = r_train + r_eval + r_valid
        if total != 100:
            # normalize to 100
            r_train = r_train / total * 100
            r_eval  = r_eval  / total * 100
            r_valid = r_valid / total * 100
            print(f"  Note: ratios normalized to 100 ({r_train:.1f}/{r_eval:.1f}/{r_valid:.1f})")

        i_train = int(n * r_train / 100)
        i_eval  = int(n * (r_train + r_eval) / 100)

        train_files = lstmf_files[:i_train]
        eval_files  = lstmf_files[i_train:i_eval]
        valid_files = lstmf_files[i_eval:]

        with open(p.train_list, 'w') as f:
            f.write('\n'.join(train_files) + '\n')
        with open(p.val_list, 'w') as f:
            f.write('\n'.join(eval_files) + '\n')
        with open(p.test_list, 'w') as f:
            f.write('\n'.join(valid_files) + '\n')

        print(f"  Total:  {n} files")
        print(f"  Train:  {p.train_list}  ({len(train_files)} files, {r_train:.0f}%)")
        print(f"  Val:    {p.val_list}    ({len(eval_files)} files, {r_eval:.0f}%)")
        print(f"  Test:   {p.test_list}   ({len(valid_files)} files, {r_valid:.0f}%)")

    elif ratio and len(ratio) == 2:
        # 2-way split: train / eval
        r_train, r_eval = ratio
        total = r_train + r_eval
        i_train = int(n * r_train / total)
        train_files = lstmf_files[:i_train]
        eval_files  = lstmf_files[i_train:]

        with open(p.train_list, 'w') as f:
            f.write('\n'.join(train_files) + '\n')
        with open(p.val_list, 'w') as f:
            f.write('\n'.join(eval_files) + '\n')

        print(f"  Total:  {n} files")
        print(f"  Train:  {p.train_list}  ({len(train_files)} files, {r_train/total*100:.0f}%)")
        print(f"  Val:    {p.val_list}    ({len(eval_files)} files, {r_eval/total*100:.0f}%)")

    else:
        # Legacy: use --ratio_train float (default 0.90)
        split = int(n * args.ratio_train)
        train_files = lstmf_files[:split]
        eval_files  = lstmf_files[split:]

        with open(p.train_list, 'w') as f:
            f.write('\n'.join(train_files) + '\n')
        with open(p.val_list, 'w') as f:
            f.write('\n'.join(eval_files) + '\n')

        print(f"  Train:  {p.train_list}  ({len(train_files)} files)")
        print(f"  Val:    {p.val_list}    ({len(eval_files)} files)")




def cmd_proto_model(args):
    """Build proto .traineddata file."""
    p = resolve_paths(args)

    tool = shutil.which('combine_lang_model')
    if tool:
        cmd = [tool,
               '--input_unicharset', p.unicharset_file,
               '--script_dir', getattr(args, 'tessdata', '') or '/usr/share/tessdata',
               '--output_dir', p.output_dir,
               '--lang', args.model_name]

        if os.path.exists(p.wordlist_file):
            cmd += ['--words', p.wordlist_file]
        if os.path.exists(p.numbers_file):
            cmd += ['--numbers', p.numbers_file]
        if os.path.exists(p.punc_file):
            cmd += ['--puncs', p.punc_file]

        print(f"  Running combine_lang_model...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            expected = os.path.join(p.output_dir, f'{args.model_name}.traineddata')
            if os.path.exists(expected):
                print(f"  ✓ Proto model: {expected}")
                return
        print(f"  Warning: combine_lang_model failed: {result.stderr}")
        print(f"  Continuing without proto model (pure PyTorch training still works)")
    else:
        print("  combine_lang_model not found. Proto model is optional for PyTorch training.")
        print("  Install tesseract-ocr-training-tools to enable it.")

def cmd_plot(args):
    """Generate TensorBoard-style training dashboard with 6 panels."""
    try:
        from .plotting import DashboardPlotter
    except ImportError:
        print("Error: Could not import plotting utilities.")
        return

    p = resolve_paths(args)
    log_file = p.log_file
    if not os.path.exists(log_file):
        print(f"No log file found at {log_file}")
        return

    entries = TrainingLogger.load(log_file)
    if not entries:
        print("No log entries found!")
        return

    # Detect epoch boundaries
    epoch_boundaries = []
    prev_epoch = None
    for e in entries:
        ep = e.get('epoch')
        if ep is not None and ep != '' and ep != prev_epoch:
            epoch_boundaries.append((e['iteration'], ep))
            prev_epoch = ep

    # Run plotter
    plotter = DashboardPlotter(args.model_name, p.output_dir)
    dashboard_path = plotter.generate_dashboard(entries, epoch_boundaries)
    print(f"  ✓ Dashboard saved: {dashboard_path}")
    
    premium_paths = plotter.generate_premium_charts(entries, epoch_boundaries)
    print(f"  ✓ Individual plots: {len(premium_paths)} figures saved to {p.output_dir}/")
    for path in premium_paths:
        print(f"      • {os.path.basename(path)}")


def cmd_clean(args):
    """Clean all generated files."""
    p = resolve_paths(args)
    if not os.path.exists(p.output_dir):
        print("Nothing to clean.")
        return

    print(f"  Cleaning: {p.output_dir}")
    exts = ['*.pt', '*.log', '*.json', '*.onnx',
            'unicharset', 'charset.json', 'list.train', 'list.eval',
            'config.json', 'training_plot.png', 'plot_*.png',
            'charset_stats.csv']

    removed = 0
    for ext in exts:
        for f in glob.glob(os.path.join(p.output_dir, ext)):
            os.remove(f)
            removed += 1

    print(f"  ✓ Removed {removed} files")

