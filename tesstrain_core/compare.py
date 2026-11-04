import os
import glob
import random
from collections import Counter
import torch
import numpy as np
from PIL import Image, ImageDraw
from tesstrain_core.model import build_model
from tesstrain_core.recognize import recognize_line
from tesstrain_core.metrics import TrainingLogger, edit_distance, align_strings, get_component_type, LAO_COMPONENT_MAP, compute_component_metrics
from tesstrain_core.plotting import DashboardPlotter

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator  # noqa: F401
except ImportError:
    pass

def add_projection_profile_to_image(pil_img):
    """Augment image by adding its vertical projection profile underneath it."""
    # Convert image to grayscale (L)
    gray_img = pil_img.convert("L")
    w, h = gray_img.size
    
    # Check if image is light background or dark background
    # Calculate average pixel value of the image
    stat = np.mean(np.array(gray_img))
    is_light_bg = stat > 127
    
    # Determine what threshold to use for text pixels
    if is_light_bg:
        # Light background: text pixels are dark (< 128)
        text_threshold = lambda val: val < 128
        bg_color = 255
        text_color = 0
    else:
        # Dark background: text pixels are light (>= 128)
        text_threshold = lambda val: val >= 128
        bg_color = 0
        text_color = 255

    # Calculate vertical projection profile (count text pixels)
    pixels = gray_img.load()
    projection = []
    for x in range(w):
        text_count = 0
        for y in range(h):
            if text_threshold(pixels[x, y]):
                text_count += 1
        projection.append(text_count)
        
    # The height of the projection profile matches the text image height
    proj_h = h
    divider_h = 2
    
    # Create combined image with background color matching the original
    combined_img = Image.new("L", (w, h + divider_h + proj_h), color=bg_color)
    
    # Paste original image at the top
    combined_img.paste(gray_img, (0, 0))
    
    combined_draw = ImageDraw.Draw(combined_img)
    # Draw horizontal divider line (using text color)
    divider_y = h
    combined_draw.line([(0, divider_y), (w, divider_y)], fill=text_color, width=divider_h)
    
    # Draw vertical projection lines downwards
    start_y = divider_y + divider_h
    for x in range(w):
        length = projection[x]
        if length > 0:
            combined_draw.line([(x, start_y), (x, start_y + length)], fill=text_color, width=1)
            
    return combined_img

def cmd_compare(args):
    print("======================================================================")
    print("  Tesseract PyTorch Model Comparison")
    print("======================================================================")
    
    models = list(getattr(args, 'models', []))
    tess_dir = getattr(args, 'tess_models_dir', '')
    tess_models = {}
    
    if tess_dir and os.path.isdir(tess_dir):
        print(f"  Scanning {tess_dir} for .traineddata models...")
        for t_file in sorted(glob.glob(os.path.join(tess_dir, '*.traineddata'))):
            t_name = os.path.basename(t_file).replace('.traineddata', '') + '_Tess'
            tess_models[t_name] = t_file
            if t_name not in models:
                models.append(t_name)

    if not models:
        print("  ERROR: Please specify at least one model with --models or --tess_models_dir")
        return
        
    custom_dir = getattr(args, 'custom_img_dir', '')
    realworld_dir = getattr(args, 'custom_realworld_img_dir', '')
    out_dir = getattr(args, 'output_dir', 'data/comparison')
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"  Models to compare: {', '.join(models)}")
    print(f"  Output directory:  {out_dir}")
    print()
    
    def safe_float(v):
        if v is None or v == '':
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    def format_val(v, fmt):
        fv = safe_float(v)
        return f"{fv:{fmt}}" if fv is not None else "--"

    def format_pct(v, fmt):
        fv = safe_float(v)
        return f"{fv * 100:{fmt}}%" if fv is not None else "--"

    def format_csv_val(v, fmt):
        fv = safe_float(v)
        return f"{fv:{fmt}}" if fv is not None else ""
    
    # -------------------------------------------------------------
    # Load training logs and match iterations with checkpoints
    # -------------------------------------------------------------
    model_metrics = {}
    for model_name in models:
        log_file = f'data/{model_name}/training.log'
        entries = []
        if os.path.exists(log_file):
            try:
                entries = TrainingLogger.load(log_file)
            except Exception as e:
                print(f"  [Warning] Failed to load log for {model_name}: {e}")
                
        # Determine checkpoint and iteration
        ckpt_path_best = f'data/{model_name}/{model_name}_best.pt'
        ckpt_path_final = f'data/{model_name}/{model_name}_final.pt'
        
        used_best = os.path.exists(ckpt_path_best)
        ckpt_path = ckpt_path_best if used_best else ckpt_path_final
        
        ckpt_it = None
        if os.path.exists(ckpt_path):
            try:
                # Load on CPU just to read metadata
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                ckpt_it = ckpt.get('iteration')
            except Exception as e:
                print(f"  [Warning] Failed to load checkpoint {ckpt_path} to read iteration: {e}")
                
        selected_entry = None
        eval_entry = None
        if entries:
            if ckpt_it is not None:
                for e in entries:
                    if e.get('iteration') == ckpt_it:
                        selected_entry = e
                        break
            if selected_entry is None:
                if used_best:
                    valid_eval = [e for e in entries if e.get('eval_cer') is not None and e.get('eval_cer') != '']
                    if valid_eval:
                        selected_entry = min(valid_eval, key=lambda e: e['eval_cer'])
                    else:
                        valid_train = [e for e in entries if e.get('train_cer') is not None and e.get('train_cer') != '']
                        if valid_train:
                            selected_entry = min(valid_train, key=lambda e: e['train_cer'])
                if selected_entry is None:
                    selected_entry = entries[-1]
            
            # Find the closest evaluation metrics before/at or after selected entry's iteration
            sel_it = selected_entry.get('iteration')
            if sel_it is not None:
                for e in entries:
                    if e.get('iteration') == sel_it:
                        if e.get('eval_cer') is not None and e.get('eval_cer') != '':
                            eval_entry = e
                            break
                if eval_entry is None:
                    candidates = [e for e in entries if e.get('iteration') is not None and e.get('iteration') <= sel_it and e.get('eval_cer') is not None and e.get('eval_cer') != '']
                    if candidates:
                        eval_entry = candidates[-1]
                if eval_entry is None:
                    candidates = [e for e in entries if e.get('iteration') is not None and e.get('iteration') > sel_it and e.get('eval_cer') is not None and e.get('eval_cer') != '']
                    if candidates:
                        eval_entry = candidates[0]
            if eval_entry is None:
                candidates = [e for e in entries if e.get('eval_cer') is not None and e.get('eval_cer') != '']
                if candidates:
                    eval_entry = min(candidates, key=lambda e: e['eval_cer'])
                    
        # Calculate training time from the log
        train_time = 0.0
        if len(entries) >= 2 and 'timestamp' in entries[0]:
            try:
                from datetime import datetime
                total_s = 0.0
                limit_it = selected_entry.get('iteration') if selected_entry else None
                for i in range(1, len(entries)):
                    if limit_it is not None and entries[i].get('iteration', 9999999) > limit_it:
                        break
                    t1 = datetime.fromisoformat(entries[i-1]['timestamp'])
                    t2 = datetime.fromisoformat(entries[i]['timestamp'])
                    diff = (t2 - t1).total_seconds()
                    if 0 <= diff < 7200:
                        total_s += diff
                train_time = total_s
            except Exception as e:
                print(f"  ⚠ Could not read training time: {e}")
                
        model_metrics[model_name] = {
            'checkpoint_type': 'best' if used_best else ('final' if os.path.exists(ckpt_path_final) else 'none'),
            'iteration': selected_entry.get('iteration') if selected_entry else None,
            'epoch': selected_entry.get('epoch') if selected_entry else None,
            'train_loss': selected_entry.get('train_loss') if selected_entry else None,
            'train_cer': selected_entry.get('train_cer') if selected_entry else None,
            'train_wer': selected_entry.get('train_wer') if selected_entry else None,
            'train_acc': selected_entry.get('train_acc') if selected_entry else None,
            'eval_iteration': eval_entry.get('iteration') if eval_entry else None,
            'eval_epoch': eval_entry.get('epoch') if eval_entry else None,
            'eval_loss': eval_entry.get('eval_loss') if eval_entry else None,
            'eval_cer': eval_entry.get('eval_cer') if eval_entry else None,
            'eval_wer': eval_entry.get('eval_wer') if eval_entry else None,
            'eval_acc': eval_entry.get('eval_acc') if eval_entry else None,
            'train_time': train_time,
            'test': None
        }
    
    # -------------------------------------------------------------
    # 0. Export to TensorBoard
    # -------------------------------------------------------------
    print("  Exporting metrics to TensorBoard...")
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(out_dir, "tensorboard_logs")

        # Map CSV column names → TensorBoard tag  (Group/Metric)
        # TensorBoard groups all tags that share the same prefix before '/'.
        # These names match training.py so live runs and post-hoc CSV exports
        # appear in the same groups in the TensorBoard UI.
        _TAG_MAP = {
            # ── Train ─────────────────────────────────────────────────────
            'train_loss':       'Train/Loss',
            'train_cer':        'Train/CER',
            'train_wer':        'Train/WER',
            'train_acc':        'Train/Accuracy',
            # ── Eval ──────────────────────────────────────────────────────
            'eval_loss':        'Eval/Loss',
            'eval_cer':         'Eval/CER',
            'eval_wer':         'Eval/WER',
            'eval_acc':         'Eval/Accuracy',
            # ── Model internals ───────────────────────────────────────────
            'mean_rms':         'Model/Mean_RMS',
            # ── System / hardware ─────────────────────────────────────────
            'learning_rate':    'System/Learning_Rate',
            'gpu_mem_gb':       'System/GPU_Memory_GB',
            'samples_per_sec':  'System/Samples_per_Sec',
        }

        for model_name in models:
            log_file = f'data/{model_name}/training.log'
            if not os.path.exists(log_file):
                continue
            entries = TrainingLogger.load(log_file)
            if not entries:
                continue

            writer = SummaryWriter(os.path.join(tb_dir, model_name))
            for e in entries:
                step = e.get('iteration')
                if step is None or step == '':
                    continue
                for k, v in e.items():
                    if k in ['iteration', 'epoch', 'timestamp']:
                        continue
                    if not isinstance(v, (int, float)):
                        continue
                    tag = _TAG_MAP.get(k, k)   # fall back to raw name if unknown
                    writer.add_scalar(tag, v, step)
            writer.close()

            # Generate Dashboard Plot
            try:
                plotter = DashboardPlotter(model_name, out_dir)
                plotter.generate_dashboard(entries)
                print(f"  ✓ Dashboard saved to: {out_dir}/plot_dashboard_{model_name}.png")
            except Exception as de:
                print(f"  [Warning] Dashboard failed for {model_name}: {de}")

        print(f"  ✓ TensorBoard logs saved to: {tb_dir}")
        print(f"  (Run 'tensorboard --logdir={tb_dir}' to view)\n")
    except ImportError:
        print("  Notice: tensorboard not installed. Skipping TensorBoard export.\n")


    # -------------------------------------------------------------
    # 1. Plotting training logs
    # -------------------------------------------------------------
    has_matplot = 'plt' in globals() and not getattr(args, 'no_plots', False)
    if not has_matplot:
        if getattr(args, 'no_plots', False):
            print("  Notice: Skipping plots (--no-plots).")
        else:
            print("  Notice: pip install matplotlib to generate comparison plots.")
    else:
        # We will plot metrics as separate figures
        metrics_to_plot = [
            ('cer', 'Character Error Rate (CER)', 'CER', 'train_cer', 'eval_cer', True, False),
            ('wer', 'Word Error Rate (WER)', 'WER', 'train_wer', 'eval_wer', True, False),
            ('loss_combined', 'CTC Loss (Combined)', 'Loss', 'train_loss', 'eval_loss', True, True),
            ('acc_combined', 'Exact-Match Accuracy (Combined)', 'Accuracy', 'train_acc', 'eval_acc', False, False),
            ('train_loss', 'CTC Loss (Train)', 'Loss', 'train_loss', None, True, True),
            ('eval_loss', 'CTC Loss (Eval)', 'Loss', None, 'eval_loss', True, True),
            ('train_acc', 'Accuracy (Train)', 'Accuracy', 'train_acc', None, False, False),
            ('eval_acc', 'Accuracy (Eval)', 'Accuracy', None, 'eval_acc', False, False),
            ('train_wer', 'Word Error Rate (Train)', 'WER', 'train_wer', None, True, False),
            ('eval_wer', 'Word Error Rate (Eval)', 'WER', None, 'eval_wer', True, False)
        ]
        
        cmap = plt.get_cmap('tab10')
        
        from matplotlib.lines import Line2D
        
        for file_suffix, title, ylabel, train_key, val_key, plot_min, is_log in metrics_to_plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(title, fontweight='bold', fontsize=16, pad=15)
            ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
            ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
            
            # Finer academic grid
            ax.grid(True, which='major', color='#dddddd', linestyle='-', linewidth=1.0)
            ax.grid(True, which='minor', color='#eeeeee', linestyle=':', linewidth=0.8)
            ax.minorticks_on()
            
            # Thicker axis lines
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            
            ax.tick_params(axis='both', which='major', labelsize=12)            
            if is_log:
                ax.set_yscale('log')
            elif file_suffix == 'acc':
                ax.set_ylim(-0.05, 1.05)
            else:
                ax.set_ylim(bottom=0)
                
            plotted_any = False
            
            for idx, model_name in enumerate(models):
                log_file = f'data/{model_name}/training.log'
                if not os.path.exists(log_file):
                    continue
                    
                entries = TrainingLogger.load(log_file)
                if not entries:
                    continue
                    
                iters = [e['iteration'] for e in entries]
                color = cmap(idx % 10)
                
                train_data = []
                if train_key:
                    train_data = [(i, e[train_key]) for i, e in zip(iters, entries) if e.get(train_key) is not None and e.get(train_key) != '']
                val_data = []
                if val_key:
                    val_data = [(i, e[val_key]) for i, e in zip(iters, entries) if e.get(val_key) is not None and e.get(val_key) != '']
                
                if train_data:
                    train_xs, train_smoothed, _ = _rolling_mean_std(train_data, 5)
                    ax.plot(train_xs, train_smoothed, color=color, lw=1.5, linestyle='--', alpha=0.7)
                    plotted_any = True
                    
                    # Annotate if we ONLY have train data for this plot
                    if not val_key:
                        if plot_min:
                            bx, by = min(train_data, key=lambda p: p[1])
                        else:
                            bx, by = max(train_data, key=lambda p: p[1])
                        ax.plot(bx, by, marker='*', markersize=10, color=color, markeredgecolor='white', markeredgewidth=1.0, zorder=10)
                        y_offset = 12 if idx % 2 == 0 else -18
                        if not plot_min: y_offset = -y_offset
                        ax.annotate(f'{by:.3f}', xy=(bx, by), xytext=(0, y_offset), 
                                    textcoords='offset points', color=color, fontsize=10, 
                                    ha='center', fontweight='bold')

                if val_data:
                    # plot raw faint
                    val_xs, raw_val_ys = zip(*val_data)
                    ax.plot(val_xs, raw_val_ys, color=color, alpha=0.15, lw=1)
                    
                    # plot smoothed solid and variance
                    val_xs, val_smoothed, val_std = _rolling_mean_std(val_data, 9)
                    ax.plot(val_xs, val_smoothed, color=color, lw=2.5, label=f'{model_name}')
                    
                    # Variance shading (± 1 standard deviation)
                    val_smoothed_np = np.array(val_smoothed)
                    val_std_np = np.array(val_std)
                    ax.fill_between(val_xs, val_smoothed_np - val_std_np, val_smoothed_np + val_std_np, color=color, alpha=0.1)
                    
                    # Annotate best
                    if plot_min:
                        bx, by = min(val_data, key=lambda p: p[1])
                    else:
                        bx, by = max(val_data, key=lambda p: p[1])
                    
                    # Add star marker at the peak/trough
                    ax.plot(bx, by, marker='*', markersize=14, color=color, markeredgecolor='white', markeredgewidth=1.0, zorder=10)
                    
                    # Stagger annotations slightly based on idx to prevent overlap
                    y_offset = 12 if idx % 2 == 0 else -18
                    if not plot_min:
                        y_offset = -y_offset
                        
                    ax.annotate(f'{by:.3f}', xy=(bx, by), xytext=(0, y_offset), 
                                textcoords='offset points', color=color, fontsize=10, 
                                ha='center', fontweight='bold')
                    plotted_any = True

            if plotted_any:
                # Add Legend
                handles, labels = ax.get_legend_handles_labels()
                # Append line style explanations
                handles.extend([Line2D([0], [0], color='gray', lw=2),
                                Line2D([0], [0], color='gray', lw=1.5, linestyle='--')])
                labels.extend(['(Validation)', '(Training)'])
                
                ax.legend(handles, labels, loc='best')
                
                plt.tight_layout()
                out_plot = os.path.join(out_dir, f'plot_{file_suffix}.png')
                plt.savefig(out_plot, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
                print(f"  ✓ Plot saved to: {out_plot}")
            
            plt.close(fig)
            
            if not plotted_any:
                print(f"  Warning: No continuous metric data found for {title}.")

    # -------------------------------------------------------------
    # 2. Custom Image Directory Evaluation & Deep Analytics
    # -------------------------------------------------------------
    results = None
    realworld_results = None
    error_analysis = {}
    demo_predictions = {}
    all_predictions = {}
    test_pairs = []
    demo_img_path = None
    demo_gt = None

    def run_evaluation(test_dir, desc_name):
        print(f"\n  Running Custom Evaluation ({desc_name}) on {test_dir}...")
        if not os.path.isdir(test_dir):
            print(f"  ERROR: Directory {test_dir} not found.")
            return None, None, None, None, None, None, None

        gt_files = sorted(glob.glob(os.path.join(test_dir, '**', '*.gt.txt'), recursive=True))
        if not gt_files:
            print(f"  Warning: No .gt.txt files found in {test_dir}")
            return None, None, None, None, None, None, None
            
        test_pairs = []
        for gt_f in gt_files:
            base = "".join(gt_f.rsplit('.gt.txt', 1))
            img_f = None
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bin.png']:
                if os.path.exists(base + ext):
                    img_f = base + ext
                    break
            if img_f:
                with open(gt_f, 'r', encoding='utf-8') as f:
                    truth = f.read().strip()
                if truth:
                    test_pairs.append((img_f, truth))
                    
        if not test_pairs:
            print(f"  Warning: No matching image/text pairs found in {test_dir}.")
            return None, None, None, None, None, None, None
            
        print(f"  Found {len(test_pairs)} test pairs.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        demo_idx = random.randint(0, len(test_pairs) - 1)
        d_img_path = test_pairs[demo_idx][0]
        d_gt = test_pairs[demo_idx][1]
        d_predictions = {}
        
        eval_results = {}
        err_analysis = {}
        all_preds = {m: {} for m in models}
        
        for model_name in models:
            if model_name in tess_models:
                import time
                try:
                    import pytesseract
                    start_eval_time = time.time()
                    total_cer = []
                    total_wer_dist = 0
                    total_wer_words = 0
                    conf_counter = Counter()
                    coer_stats = {cat: {'total': 0, 'errors': 0} for cat in LAO_COMPONENT_MAP.keys()}
                    coer_stats['other'] = {'total': 0, 'errors': 0}

                    traineddata_path = tess_models[model_name]
                    t_dir = os.path.dirname(traineddata_path)
                    t_model_name = os.path.basename(traineddata_path).replace(".traineddata", "")
                    config = f'--tessdata-dir "{t_dir}" -l {t_model_name} --psm 7 --oem 1'
                    
                    try:
                        from tqdm import tqdm
                        iterator = tqdm(test_pairs, desc=f"  Evaluating {model_name}", leave=False, dynamic_ncols=True)
                    except ImportError:
                        iterator = test_pairs
                        
                    for img_f, truth in iterator:
                        try:
                            pil = Image.open(img_f).convert('L')
                            pred = pytesseract.image_to_string(pil, config=config).strip()
                        except Exception as e:
                            print(f"  [Error Tesseract {model_name} on {img_f}] {e}")
                            pred = ""
                        
                        all_preds[model_name][img_f] = pred
                        
                        if img_f == d_img_path:
                            d_predictions[model_name] = pred
                        
                        cer = edit_distance(pred, truth) / max(len(truth), 1)
                        total_cer.append(cer)
                        
                        pred_words = pred.split()
                        truth_words = truth.split()
                        total_wer_dist += edit_distance(pred_words, truth_words)
                        total_wer_words += len(truth_words)
                        
                        edits = align_strings(pred, truth)
                        batch_coer = compute_component_metrics(edits)
                        for cat in coer_stats:
                            coer_stats[cat]['total'] += batch_coer[cat]['total']
                            coer_stats[cat]['errors'] += batch_coer[cat]['errors']
                        
                        for err in edits:
                            conf_counter[err] += 1
                            
                    end_eval_time = time.time()
                    total_eval_time = end_eval_time - start_eval_time
                        
                    avg_cer = sum(total_cer) / len(total_cer) if total_cer else 0.0
                    wer = total_wer_dist / max(total_wer_words, 1)
                    
                    coer_results = {}
                    for cat, s in coer_stats.items():
                         if s['total'] > 0:
                             coer_results[cat] = s['errors'] / s['total']
                              
                    avg_coer = sum(coer_results.values()) / max(len(coer_results), 1) if coer_results else 0.0
                    acc = max(0.0, 1.0 - ((avg_cer + wer + avg_coer) / 3.0))
                    
                    eval_results[model_name] = {
                        'cer': avg_cer,
                        'wer': wer,
                        'acc': acc,
                        'time': total_eval_time,
                        'train_time': 0.0,
                        'coer': coer_results
                    }
                    err_analysis[model_name] = conf_counter
                except ImportError:
                    print(f"  [Skipping {model_name}] pytesseract is not installed.")
                    eval_results[model_name] = None
                except Exception as e:
                    print(f"  [Error running Tesseract {model_name}] {e}")
                    eval_results[model_name] = None
                continue

            ckpt_path = f'data/{model_name}/{model_name}_best.pt'
            if not os.path.exists(ckpt_path):
                ckpt_path = f'data/{model_name}/{model_name}_final.pt'
                if not os.path.exists(ckpt_path):
                    print(f"  [Skipping {model_name}] No checkpoint found at {ckpt_path}")
                    eval_results[model_name] = None
                    continue
                    
            try:
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                spec = ckpt['net_spec'].replace('###', str(ckpt['num_classes']))
                model = build_model(spec, ckpt['num_classes']).to(device)
                model.load_state_dict(ckpt['model_state_dict'])
                model.eval()
                
                inv_charset = {v: k for k, v in ckpt['charset'].items()}
                
                target_h = 48
                import re
                m = re.match(r'\[?\d+,(\d+),', ckpt.get('net_spec', ''))
                if m: target_h = int(m.group(1))
                
                import time
                total_cer = []
                total_wer_dist = 0
                total_wer_words = 0
                conf_counter = Counter()
                coer_stats = {cat: {'total': 0, 'errors': 0} for cat in LAO_COMPONENT_MAP.keys()}
                coer_stats['other'] = {'total': 0, 'errors': 0}
                
                start_eval_time = time.time()
                
                try:
                    from tqdm import tqdm
                    iterator = tqdm(test_pairs, desc=f"  Evaluating {model_name}", leave=False, dynamic_ncols=True)
                except ImportError:
                    iterator = test_pairs
                    
                for img_f, truth in iterator:
                    pil = Image.open(img_f).convert('L')
                    pred = recognize_line(model, pil, target_h, device, inv_charset, beam_width=0, dawg_decoder=None)
                    
                    all_preds[model_name][img_f] = pred
                    
                    if img_f == d_img_path:
                        d_predictions[model_name] = pred
                    
                    cer = edit_distance(pred, truth) / max(len(truth), 1)
                    total_cer.append(cer)
                    
                    pred_words = pred.split()
                    truth_words = truth.split()
                    total_wer_dist += edit_distance(pred_words, truth_words)
                    total_wer_words += len(truth_words)
                    
                    edits = align_strings(pred, truth)
                    batch_coer = compute_component_metrics(edits)
                    for cat in coer_stats:
                        coer_stats[cat]['total'] += batch_coer[cat]['total']
                        coer_stats[cat]['errors'] += batch_coer[cat]['errors']
                    
                    for err in edits:
                        conf_counter[err] += 1
                        
                end_eval_time = time.time()
                total_eval_time = end_eval_time - start_eval_time
                    
                avg_cer = sum(total_cer) / len(total_cer)
                wer = total_wer_dist / max(total_wer_words, 1)
                
                coer_results = {}
                for cat, s in coer_stats.items():
                     if s['total'] > 0:
                         coer_results[cat] = s['errors'] / s['total']
                          
                avg_coer = sum(coer_results.values()) / max(len(coer_results), 1) if coer_results else 0.0
                acc = max(0.0, 1.0 - ((avg_cer + wer + avg_coer) / 3.0))
                
                train_time = 0.0
                log_file = f'data/{model_name}/training.log'
                if os.path.exists(log_file):
                    from datetime import datetime
                    entries = TrainingLogger.load(log_file)
                    if len(entries) >= 2 and 'timestamp' in entries[0] and 'timestamp' in entries[-1]:
                        try:
                            total_s = 0.0
                            for i in range(1, len(entries)):
                                t1 = datetime.fromisoformat(entries[i-1]['timestamp'])
                                t2 = datetime.fromisoformat(entries[i]['timestamp'])
                                diff = (t2 - t1).total_seconds()
                                if 0 <= diff < 7200:
                                    total_s += diff
                            train_time = total_s
                        except Exception as e:
                            print(f"  ⚠ Could not read training time: {e}")
                          
                eval_results[model_name] = {
                    'cer': avg_cer,
                    'wer': wer,
                    'acc': acc,
                    'time': total_eval_time,
                    'train_time': train_time,
                    'coer': coer_results
                }
                err_analysis[model_name] = conf_counter
            except Exception as e:
                print(f"  [Error running {model_name}] {e}")
                eval_results[model_name] = None
        return eval_results, err_analysis, d_predictions, d_img_path, d_gt, all_preds, test_pairs

    if custom_dir:
        res_dict, err_an, demo_preds, demo_path, d_gt, all_preds_custom, test_pairs_custom = run_evaluation(custom_dir, "Custom Test Set")
        if res_dict:
            results = res_dict
            error_analysis = err_an
            demo_predictions = demo_preds
            demo_img_path = demo_path
            demo_gt = d_gt
            all_predictions = all_preds_custom
            test_pairs = test_pairs_custom
            for model_name in models:
                if model_name in results and results[model_name]:
                    model_metrics[model_name]['test'] = results[model_name]

    if realworld_dir:
        res_dict_rw, err_an_rw, demo_preds_rw, demo_path_rw, d_gt_rw, all_preds_rw, test_pairs_rw = run_evaluation(realworld_dir, "Real-world Test Set")
        if res_dict_rw:
            realworld_results = res_dict_rw
            if not results:
                error_analysis = err_an_rw
                demo_predictions = demo_preds_rw
                demo_img_path = demo_path_rw
                demo_gt = d_gt_rw
                all_predictions = all_preds_rw
                test_pairs = test_pairs_rw
            for model_name in models:
                if model_name in realworld_results and realworld_results[model_name]:
                    model_metrics[model_name]['realworld'] = realworld_results[model_name]

        # Print Separated Results by Splits
        print("\n  ==========================================================================================")
        print("    COMPARISON RESULTS: SEPARATED BY SPLITS")
        print("  ==========================================================================================")

        # Sort models for each table/split
        sorted_train_models = sorted(
            models,
            key=lambda name: safe_float(model_metrics.get(name, {}).get('train_acc')) if safe_float(model_metrics.get(name, {}).get('train_acc')) is not None else -1.0,
            reverse=True
        )
        sorted_eval_models = sorted(
            models,
            key=lambda name: safe_float(model_metrics.get(name, {}).get('eval_acc')) if safe_float(model_metrics.get(name, {}).get('eval_acc')) is not None else -1.0,
            reverse=True
        )
        
        sorted_test_models = []
        if results is not None:
            sorted_test_models = sorted(
                models,
                key=lambda name: safe_float(results.get(name, {}).get('acc')) if results.get(name) is not None and safe_float(results.get(name, {}).get('acc')) is not None else -1.0,
                reverse=True
            )
            
        sorted_rw_models = []
        if realworld_results is not None:
            sorted_rw_models = sorted(
                models,
                key=lambda name: safe_float(realworld_results.get(name, {}).get('acc')) if realworld_results.get(name) is not None and safe_float(realworld_results.get(name, {}).get('acc')) is not None else -1.0,
                reverse=True
            )

        sorted_coer_test_models = []
        if results is not None:
            sorted_coer_test_models = sorted(
                models,
                key=lambda name: (
                    safe_float(results.get(name, {}).get('coer', {}).get('tone'))
                    if results.get(name) is not None and safe_float(results.get(name, {}).get('coer', {}).get('tone')) is not None
                    else 999.0
                )
            )

        sorted_coer_rw_models = []
        if realworld_results is not None:
            sorted_coer_rw_models = sorted(
                models,
                key=lambda name: (
                    safe_float(realworld_results.get(name, {}).get('coer', {}).get('tone'))
                    if realworld_results.get(name) is not None and safe_float(realworld_results.get(name, {}).get('coer', {}).get('tone')) is not None
                    else 999.0
                )
            )

        # 1. Training Split
        print("\n  ==== 🏋️ Training Split (from Logs) ====")
        print("  | Model                | Iteration | Epoch | Loss     | CER (%)  | WER (%)  | Accuracy |")
        print("  |----------------------|-----------|-------|----------|----------|----------|----------|")
        for model_name in sorted_train_models:
            m = model_metrics.get(model_name, {})
            it_str = f"{m.get('iteration')}" if m.get('iteration') is not None else "--"
            ep_str = f"{m.get('epoch')}" if m.get('epoch') is not None else "--"
            loss_str = format_val(m.get('train_loss'), ".4f")
            cer_str = format_pct(m.get('train_cer'), ".2f")
            wer_str = format_pct(m.get('train_wer'), ".2f")
            acc_str = format_pct(m.get('train_acc'), ".2f")
            print(f"  | {model_name:<20} | {it_str:>9} | {ep_str:>5} | {loss_str:>8} | {cer_str:>8} | {wer_str:>8} | {acc_str:>8} |")

        # 2. Eval Split
        print("\n  ==== 🧪 Eval Split (from Logs) ====")
        print("  | Model                | Iteration | Epoch | Loss     | CER (%)  | WER (%)  | Accuracy |")
        print("  |----------------------|-----------|-------|----------|----------|----------|----------|")
        for model_name in sorted_eval_models:
            m = model_metrics.get(model_name, {})
            it_str = f"{m.get('eval_iteration')}" if m.get('eval_iteration') is not None else "--"
            ep_str = f"{m.get('eval_epoch')}" if m.get('eval_epoch') is not None else "--"
            loss_str = format_val(m.get('eval_loss'), ".4f")
            cer_str = format_pct(m.get('eval_cer'), ".2f")
            wer_str = format_pct(m.get('eval_wer'), ".2f")
            acc_str = format_pct(m.get('eval_acc'), ".2f")
            print(f"  | {model_name:<20} | {it_str:>9} | {ep_str:>5} | {loss_str:>8} | {cer_str:>8} | {wer_str:>8} | {acc_str:>8} |")

        # 3. Testing Split
        if results is not None:
            print(f"\n  ==== 🎯 Testing Split (Custom Evaluation on: {custom_dir}) ====")
            print("  | Model                | Avg CER  | Avg WER  | Accuracy | Eval Time | Train Time |")
            print("  |----------------------|----------|----------|----------|-----------|------------|")
            for model_name in sorted_test_models:
                res = results.get(model_name)
                if res is not None:
                    cer_str = f"{res['cer']*100:.2f}%"
                    wer_str = f"{res['wer']*100:.2f}%"
                    acc_str = f"{res['acc']*100:.2f}%"
                    eval_time_str = f"{res['time']:.2f}s"
                    train_h = res['train_time'] / 3600.0 if res.get('train_time', 0) > 0 else 0.0
                    train_time_str = f"{train_h:.2f}h"
                    print(f"  | {model_name:<20} | {cer_str:>8} | {wer_str:>8} | {acc_str:>8} | {eval_time_str:>9} | {train_time_str:>10} |")
                else:
                    print(f"  | {model_name:<20} | {'--':>8} | {'--':>8} | {'--':>8} | {'--':>9} | {'--':>10} |")

        # 4. Real-world Testing Split
        if realworld_results is not None:
            print(f"\n  ==== 🌎 Real-world Testing Split (Custom Evaluation on: {realworld_dir}) ====")
            print("  | Model                | Avg CER  | Avg WER  | Accuracy | Eval Time | Train Time |")
            print("  |----------------------|----------|----------|----------|-----------|------------|")
            for model_name in sorted_rw_models:
                res = realworld_results.get(model_name)
                if res is not None:
                    cer_str = f"{res['cer']*100:.2f}%"
                    wer_str = f"{res['wer']*100:.2f}%"
                    acc_str = f"{res['acc']*100:.2f}%"
                    eval_time_str = f"{res['time']:.2f}s"
                    train_h = res['train_time'] / 3600.0 if res.get('train_time', 0) > 0 else 0.0
                    train_time_str = f"{train_h:.2f}h"
                    print(f"  | {model_name:<20} | {cer_str:>8} | {wer_str:>8} | {acc_str:>8} | {eval_time_str:>9} | {train_time_str:>10} |")
                else:
                    print(f"  | {model_name:<20} | {'--':>8} | {'--':>8} | {'--':>8} | {'--':>9} | {'--':>10} |")

        # Print CoER
        coer_cols = [('tone', 'Tone Mark'), ('upper_vowel', 'Upper Vowel'), ('consonant', 'Consonant'),
                     ('before_vowel', 'Before Vowel'), ('rear_vowel', 'After Vowel'), ('lower_vowel', 'Lower Vowel'),
                     ('special', 'Special'), ('lao_number', 'Lao Number'), ('digit', 'Digit'), ('latin', 'Latin'),
                     ('space', 'Space'), ('symbol', 'Symbol')]
                     
        if results is not None:
            print(f"\n  ==== Component-Level Error Rate (CoER) - Custom Test Set ({custom_dir}) ====")
            header = "  | Model                | " + " | ".join(name for _, name in coer_cols) + " |"
            sep_line = "  |----------------------|" + "|".join("-" * (len(name) + 2) for _, name in coer_cols) + "|"
            print(header)
            print(sep_line)
            for model_name in sorted_coer_test_models:
                res = results.get(model_name)
                if res is not None:
                    row_str = f"  | {model_name:<20}"
                    coer = res['coer']
                    for cat, name in coer_cols:
                         val = coer.get(cat)
                         if val is not None:
                             row_str += f" | {val*100:>{len(name)-1}.2f}% "
                         else:
                             row_str += f" | {'--':>{len(name)}} "
                    row_str += "|"
                    print(row_str)

        if realworld_results is not None:
            print(f"\n  ==== Component-Level Error Rate (CoER) - Real-world Test Set ({realworld_dir}) ====")
            header = "  | Model                | " + " | ".join(name for _, name in coer_cols) + " |"
            sep_line = "  |----------------------|" + "|".join("-" * (len(name) + 2) for _, name in coer_cols) + "|"
            print(header)
            print(sep_line)
            for model_name in sorted_coer_rw_models:
                res = realworld_results.get(model_name)
                if res is not None:
                    row_str = f"  | {model_name:<20}"
                    coer = res['coer']
                    for cat, name in coer_cols:
                         val = coer.get(cat)
                         if val is not None:
                             row_str += f" | {val*100:>{len(name)-1}.2f}% "
                         else:
                             row_str += f" | {'--':>{len(name)}} "
                    row_str += "|"
                    print(row_str)
        # Generate Bar Charts for Custom Evaluation
        def generate_eval_bar_charts(eval_results, title_suffix, file_prefix):
            if not has_matplot or not eval_results or not any(r is not None for r in eval_results.values()):
                return
            print(f"  Generating Bar Charts for {title_suffix}...")
            try:
                valid_models = sorted(
                    [m for m in models if eval_results.get(m) is not None],
                    key=lambda m: safe_float(eval_results[m].get('acc')) if safe_float(eval_results[m].get('acc')) is not None else -1.0,
                    reverse=True
                )
                colors = plt.get_cmap('tab10').colors
                
                # 1. Global Metrics Bar Chart
                cer_vals  = [eval_results[m]['cer'] * 100 for m in valid_models]
                wer_vals  = [eval_results[m]['wer'] * 100 for m in valid_models]
                coer_vals = [(sum(eval_results[m]['coer'].values()) / max(len(eval_results[m]['coer']), 1)) * 100
                             if eval_results[m]['coer'] else 0.0 for m in valid_models]
                acc_vals  = [eval_results[m]['acc'] * 100 for m in valid_models]

                x     = np.arange(len(valid_models))
                width = 0.2
                fig, ax = plt.subplots(figsize=(12, 6))

                rects1 = ax.bar(x - width*1.5, cer_vals,  width, label='CER (%)',      color=colors[0])
                rects2 = ax.bar(x - width*0.5, wer_vals,  width, label='WER (%)',      color=colors[1])
                rects3 = ax.bar(x + width*0.5, coer_vals, width, label='Avg CoER (%)', color=colors[2])
                rects4 = ax.bar(x + width*1.5, acc_vals,  width, label='Accuracy (%)', color=colors[3])

                for rects in [rects1, rects2, rects3, rects4]:
                    for rect in rects:
                        h = rect.get_height()
                        ax.annotate(f'{h:.1f}',
                                    xy=(rect.get_x() + rect.get_width() / 2, h),
                                    xytext=(0, 3), textcoords='offset points',
                                    ha='center', va='bottom', fontsize=9,
                                    fontweight='bold', rotation=90 if h > 0 else 0)

                ax.set_ylabel('Percentage (%)', fontweight='bold', fontsize=12)
                ax.set_title(f'Global Performance Summary ({title_suffix})', fontweight='bold', fontsize=16, pad=20)
                ax.set_xticks(x)
                ax.set_xticklabels(valid_models, fontweight='bold', fontsize=11)
                ax.legend(fontsize=11, bbox_to_anchor=(1.01, 1), loc='upper left')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                global_plot_path = os.path.join(out_dir, f'plot_global_bar_{file_prefix}.png')
                plt.savefig(global_plot_path, dpi=200, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                print(f"  ✓ Global metrics bar chart saved to: {global_plot_path}")
                
                # 2. CoER Bar Chart 
                # Sort categories from 'less to more' based on aggregate error rate
                sorted_coer_cols = coer_cols.copy()
                def cat_avg_error_pt(cat_tuple):
                    c = cat_tuple[0]
                    vals = [eval_results[m]['coer'].get(c, 0) for m in valid_models]
                    return sum(vals) / len(vals) if vals else 0
                sorted_coer_cols.sort(key=cat_avg_error_pt)
                
                fig, ax = plt.subplots(figsize=(14, 7))
                x = np.arange(len(sorted_coer_cols))
                n_models = len(valid_models)
                total_width = 0.8
                width = total_width / max(n_models, 1)
                
                for i, model_name in enumerate(valid_models):
                    res = eval_results[model_name]
                    vals = [res['coer'].get(key, 0.0) * 100 for key, _ in sorted_coer_cols]
                    offset = x + (i - n_models / 2 + 0.5) * width
                    rects = ax.bar(offset, vals, width, label=model_name, color=colors[i % len(colors)])
                    
                    for rect in rects:
                        height = rect.get_height()
                        if height > 0:
                            ax.annotate(f'{height:.1f}',
                                        xy=(rect.get_x() + rect.get_width() / 2, height),
                                        xytext=(0, 2),
                                        textcoords="offset points",
                                        ha='center', va='bottom', fontsize=8, rotation=90)
                
                ax.set_ylabel('Error Rate (%)', fontweight='bold', fontsize=12)
                ax.set_title(f'Component-Level Error Rate (CoER) ({title_suffix})', fontweight='bold', fontsize=16, pad=20)
                ax.set_xticks(x)
                ax.set_xticklabels([m[1] for m in sorted_coer_cols], fontweight='bold', fontsize=11)
                ax.legend(fontsize=11)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                coer_plot_path = os.path.join(out_dir, f'plot_coer_bar_{file_prefix}.png')
                plt.savefig(coer_plot_path, dpi=200, bbox_inches='tight')
                plt.close(fig)
                print(f"  ✓ CoER bar chart saved to: {coer_plot_path}\n")
                
            except Exception as e:
                print(f"  [Bar Chart Plot Error for {title_suffix}] {e}\n")

        if results is not None:
            generate_eval_bar_charts(results, "Custom Test Set", "custom")
        if realworld_results is not None:
            generate_eval_bar_charts(realworld_results, "Real-world Test Set", "realworld")

        # Markdown Report Generation
        report_md = getattr(args, 'report', '')
        if not report_md:
            report_md = os.path.join(out_dir, 'report.md')
            
        with open(report_md, 'w', encoding='utf-8') as f:
            f.write("# 🧪 Model Comparison Analytics\n\n")
            
            # 1. Training Split
            f.write("## 🏋️ Training Split (from Logs)\n")
            f.write("| Model | Iteration | Epoch | Loss | CER | WER | Accuracy |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for model_name in sorted_train_models:
                m = model_metrics.get(model_name, {})
                it_str = f"{m.get('iteration')}" if m.get('iteration') is not None else "--"
                ep_str = f"{m.get('epoch')}" if m.get('epoch') is not None else "--"
                loss_str = format_val(m.get('train_loss'), ".4f")
                cer_str = format_pct(m.get('train_cer'), ".2f")
                wer_str = format_pct(m.get('train_wer'), ".2f")
                acc_str = format_pct(m.get('train_acc'), ".2f")
                f.write(f"| **{model_name}** | {it_str} | {ep_str} | {loss_str} | {cer_str} | {wer_str} | {acc_str} |\n")
            f.write("\n")
            
            # 2. Eval Split
            f.write("## 🧪 Eval Split (from Logs)\n")
            f.write("| Model | Iteration | Epoch | Loss | CER | WER | Accuracy |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for model_name in sorted_eval_models:
                m = model_metrics.get(model_name, {})
                it_str = f"{m.get('eval_iteration')}" if m.get('eval_iteration') is not None else "--"
                ep_str = f"{m.get('eval_epoch')}" if m.get('eval_epoch') is not None else "--"
                loss_str = format_val(m.get('eval_loss'), ".4f")
                cer_str = format_pct(m.get('eval_cer'), ".2f")
                wer_str = format_pct(m.get('eval_wer'), ".2f")
                acc_str = format_pct(m.get('eval_acc'), ".2f")
                f.write(f"| **{model_name}** | {it_str} | {ep_str} | {loss_str} | {cer_str} | {wer_str} | {acc_str} |\n")
            f.write("\n")
            
            # 3. Testing Split
            if results is not None:
                f.write(f"## 🎯 Testing Split (Custom Evaluation on: `{custom_dir}`)\n")
                f.write("| Model | Avg CER | Avg WER | Accuracy | Eval Time | Train Time |\n")
                f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
                for model_name in sorted_test_models:
                    res = results.get(model_name)
                    if res is not None:
                        cer_str = f"{res['cer']*100:.2f}%"
                        wer_str = f"{res['wer']*100:.2f}%"
                        acc_str = f"{res['acc']*100:.2f}%"
                        eval_time_str = f"{res['time']:.2f}s"
                        train_h = res['train_time'] / 3600.0 if res.get('train_time', 0) > 0 else 0.0
                        train_time_str = f"{train_h:.2f}h"
                        f.write(f"| **{model_name}** | {cer_str} | {wer_str} | {acc_str} | {eval_time_str} | {train_time_str} |\n")
                    else:
                        f.write(f"| **{model_name}** | -- | -- | -- | -- | -- |\n")
                f.write("\n")
                
            # 4. Real-world Testing Split
            if realworld_results is not None:
                f.write(f"## 🌎 Real-world Testing Split (Custom Evaluation on: `{realworld_dir}`)\n")
                f.write("| Model | Avg CER | Avg WER | Accuracy | Eval Time | Train Time |\n")
                f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
                for model_name in sorted_rw_models:
                    res = realworld_results.get(model_name)
                    if res is not None:
                        cer_str = f"{res['cer']*100:.2f}%"
                        wer_str = f"{res['wer']*100:.2f}%"
                        acc_str = f"{res['acc']*100:.2f}%"
                        eval_time_str = f"{res['time']:.2f}s"
                        train_h = res['train_time'] / 3600.0 if res.get('train_time', 0) > 0 else 0.0
                        train_time_str = f"{train_h:.2f}h"
                        f.write(f"| **{model_name}** | {cer_str} | {wer_str} | {acc_str} | {eval_time_str} | {train_time_str} |\n")
                    else:
                        f.write(f"| **{model_name}** | -- | -- | -- | -- | -- |\n")
                f.write("\n")
            
            # CoER table in markdown
            if results is not None:
                f.write(f"## 🧩 Component-Level Error Rate (CoER) - Custom Test Set ({custom_dir})\n")
                f.write("| Model | " + " | ".join(name for _, name in coer_cols) + " |\n")
                f.write("| :--- " + "".join("| :--- " for _ in coer_cols) + "|\n")
                for model_name in sorted_coer_test_models:
                    res = results.get(model_name)
                    if res is not None:
                        row_str = f"| **{model_name}**"
                        coer = res['coer']
                        for cat, name in coer_cols:
                            val = coer.get(cat)
                            row_str += f" | {val*100:.2f}%" if val is not None else " | --"
                        row_str += " |\n"
                        f.write(row_str)
                f.write("\n")

            if realworld_results is not None:
                f.write(f"## 🧩 Component-Level Error Rate (CoER) - Real-world Test Set ({realworld_dir})\n")
                f.write("| Model | " + " | ".join(name for _, name in coer_cols) + " |\n")
                f.write("| :--- " + "".join("| :--- " for _ in coer_cols) + "|\n")
                for model_name in sorted_coer_rw_models:
                    res = realworld_results.get(model_name)
                    if res is not None:
                        row_str = f"| **{model_name}**"
                        coer = res['coer']
                        for cat, name in coer_cols:
                            val = coer.get(cat)
                            row_str += f" | {val*100:.2f}%" if val is not None else " | --"
                        row_str += " |\n"
                        f.write(row_str)
                f.write("\n")
                    
        print(f"  ✓ Exported Markdown report to: {report_md}")
        
        # CSV Export
        import csv
        report_csv = os.path.join(out_dir, 'report.csv')
        with open(report_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            headers = [
                'Model', 
                'Train Iteration', 'Train Epoch', 'Train Loss', 'Train CER', 'Train WER', 'Train Accuracy',
                'Eval Iteration', 'Eval Epoch', 'Eval Loss', 'Eval CER', 'Eval WER', 'Eval Accuracy',
                'Train Time (h)'
            ]
            if results is not None:
                headers += ['Test CER', 'Test WER', 'Test Accuracy', 'Test Eval Time (s)']
            if realworld_results is not None:
                headers += ['Real-world Test CER', 'Real-world Test WER', 'Real-world Test Accuracy', 'Real-world Test Eval Time (s)']
            
            if results is not None:
                headers += [f"Test CoER: {name}" for _, name in coer_cols]
            if realworld_results is not None:
                headers += [f"Real-world CoER: {name}" for _, name in coer_cols]
                
            writer.writerow(headers)
            
            # Sort by best available accuracy metric
            def get_best_acc_metric(name):
                if realworld_results is not None and realworld_results.get(name) is not None:
                    val = safe_float(realworld_results[name].get('acc'))
                    if val is not None:
                        return (4, val)
                if results is not None and results.get(name) is not None:
                    val = safe_float(results[name].get('acc'))
                    if val is not None:
                        return (3, val)
                m = model_metrics.get(name, {})
                val = safe_float(m.get('eval_acc'))
                if val is not None:
                    return (2, val)
                val = safe_float(m.get('train_acc'))
                if val is not None:
                    return (1, val)
                return (0, -1.0)
                
            sorted_csv_models = sorted(
                models,
                key=get_best_acc_metric,
                reverse=True
            )
            
            for model_name in sorted_csv_models:
                m = model_metrics.get(model_name, {})
                res = results.get(model_name) if results is not None else None
                res_rw = realworld_results.get(model_name) if realworld_results is not None else None
                train_h = m.get('train_time', 0.0) / 3600.0 if m.get('train_time', 0) > 0 else 0.0
                
                row = [
                    model_name,
                    m.get('iteration', ''),
                    m.get('epoch', ''),
                    format_csv_val(m.get('train_loss'), ".6f"),
                    format_csv_val(m.get('train_cer'), ".6f"),
                    format_csv_val(m.get('train_wer'), ".6f"),
                    format_csv_val(m.get('train_acc'), ".6f"),
                    m.get('eval_iteration', ''),
                    m.get('eval_epoch', ''),
                    format_csv_val(m.get('eval_loss'), ".6f"),
                    format_csv_val(m.get('eval_cer'), ".6f"),
                    format_csv_val(m.get('eval_wer'), ".6f"),
                    format_csv_val(m.get('eval_acc'), ".6f"),
                    f"{train_h:.2f}"
                ]
                
                if results is not None:
                    if res is not None:
                        row += [
                            f"{res['cer']:.6f}",
                            f"{res['wer']:.6f}",
                            f"{res['acc']:.6f}",
                            f"{res['time']:.2f}"
                        ]
                    else:
                        row += ['', '', '', '']
                        
                if realworld_results is not None:
                    if res_rw is not None:
                        row += [
                            f"{res_rw['cer']:.6f}",
                            f"{res_rw['wer']:.6f}",
                            f"{res_rw['acc']:.6f}",
                            f"{res_rw['time']:.2f}"
                        ]
                    else:
                        row += ['', '', '', '']

                if results is not None:
                    if res is not None:
                        coer = res['coer']
                        for cat, _ in coer_cols:
                            val = coer.get(cat)
                            row.append(f"{val:.6f}" if val is not None else '')
                    else:
                        row += [''] * len(coer_cols)

                if realworld_results is not None:
                    if res_rw is not None:
                        coer_rw = res_rw['coer']
                        for cat, _ in coer_cols:
                            val = coer_rw.get(cat)
                            row.append(f"{val:.6f}" if val is not None else '')
                    else:
                        row += [''] * len(coer_cols)
                
                writer.writerow(row)
                    
        print(f"  ✓ Exported CSV report to: {report_csv}\n")
        
        # Specific Character Error Rate CSV Export
        if error_analysis:
            valid_models = [m for m in models if m in error_analysis]
            if valid_models:
                gt_counts = Counter()
                model_matches = {m: Counter() for m in valid_models}
                model_errors = {m: Counter() for m in valid_models}
                
                # Use the first valid model to determine Ground Truth occurrences
                ref_counter = error_analysis[valid_models[0]]
                for (op, p_c, t_c), count in ref_counter.items():
                    if op in ('mat', 'sub', 'del') and t_c:
                        gt_counts[t_c] += count
                        
                for m in valid_models:
                    counter = error_analysis[m]
                    for (op, p_c, t_c), count in counter.items():
                        if op == 'mat' and t_c:
                            model_matches[m][t_c] += count
                        if op in ('sub', 'del') and t_c:
                            model_errors[m][t_c] += count
                        if op == 'ins' and p_c:
                            model_errors[m][p_c] += count
                            
                all_chars = set(gt_counts.keys())
                for m in valid_models:
                    all_chars.update(model_errors[m].keys())
                all_chars = sorted(list(all_chars))
                
                char_csv = os.path.join(out_dir, 'report_characters.csv')
                with open(char_csv, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    headers = ['Character', 'GT Occurrences']
                    for m in valid_models:
                        headers.extend([f"[{m}] Correct", f"[{m}] Errors", f"[{m}] Error Rate (%)"])
                    writer.writerow(headers)
                    
                    for c in all_chars:
                        total = gt_counts.get(c, 0)
                        row = [c, total]
                        for m in valid_models:
                            correct = model_matches[m].get(c, 0)
                            errors = model_errors[m].get(c, 0)
                            
                            row.append(correct)
                            row.append(errors)
                            
                            if total > 0:
                                rate = (errors / total) * 100
                                row.append(f"{rate:.2f}%")
                            else:
                                if errors > 0:
                                    row.append(f"{errors} (inserts)")
                                else:
                                    row.append("0.00%")
                        writer.writerow(row)
                print(f"  ✓ Exported Character Error Rate CSV to: {char_csv}")
                
                # Common Mistakes CSV Export
                mistakes_csv = os.path.join(out_dir, 'report_mistakes.csv')
                with open(mistakes_csv, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Model', 'Operation', 'Truth Character', 'Predicted Character', 'Error Count'])
                    
                    for m in valid_models:
                        counter = error_analysis[m]
                        errors = [(op, p_c, t_c, cnt) for (op, p_c, t_c), cnt in counter.items() if op != 'mat']
                        errors.sort(key=lambda x: x[3], reverse=True)
                        for op, p, t, cnt in errors:
                            writer.writerow([m, op, t if t else '', p if p else '', cnt])
                print(f"  ✓ Exported Common Mistakes CSV to: {mistakes_csv}\n")
                
                print("  ==== Top 3 Mistakes by Model ====")
                for m in valid_models:
                    counter = error_analysis[m]
                    errors = [(op, p_c, t_c, cnt) for (op, p_c, t_c), cnt in counter.items() if op != 'mat']
                    errors.sort(key=lambda x: x[3], reverse=True)
                    mistakes_str = []
                    for op, p, t, cnt in errors[:3]:
                        if op == 'sub': mistakes_str.append(f"'{t}'->'{p}' ({cnt})")
                        elif op == 'del': mistakes_str.append(f"Missed '{t}' ({cnt})")
                        elif op == 'ins': mistakes_str.append(f"Extra '{p}' ({cnt})")
                    print(f"  [{m}] {', '.join(mistakes_str) if mistakes_str else 'None'}")
                print("  =================================\n")

        # -------------------------------------------------------------
        # 3. Grid Demo Plotting
        # -------------------------------------------------------------
        if has_matplot and demo_predictions:
            print("  Generating Prediction Grid Demo...")
            try:
                import matplotlib.font_manager as fm
                from matplotlib.patches import FancyBboxPatch
                lao_font_path = 'tesstrain_core/font/phetsarath_ot.ttf'
                fp_sm = fm.FontProperties(fname=lao_font_path, size=11) if os.path.exists(lao_font_path) else fm.FontProperties(family='monospace', size=11)

                def _draw_alignment_grid(ax, pil_img, gt_str, pred_str, title, cat_colors):
                    """Draw image + GT/Pred alignment cell rows in one Axes."""
                    MAX_COLS = 20
                    FIG_W = 14.0          # data-unit width of the axes
                    CELL_W = FIG_W / MAX_COLS
                    CHAR_H = 0.52
                    ARROW_H = 0.38
                    LABEL_H = 0.30

                    # Compute image display height so it keeps its natural aspect ratio
                    pil_img = add_projection_profile_to_image(pil_img)
                    img_w_px, img_h_px = pil_img.size
                    aspect_ratio = img_h_px / max(img_w_px, 1)
                    IMG_H = max(FIG_W * aspect_ratio, 0.8)   # at least 0.8 units tall

                    edits = align_strings(pred_str, gt_str)
                    edits_trim = edits[:MAX_COLS]

                    # Total panel height
                    total_h = 0.40 + IMG_H + 0.10 + CHAR_H + ARROW_H + CHAR_H + LABEL_H + 0.15
                    ax.set_xlim(0, FIG_W)
                    ax.set_ylim(0, total_h)
                    ax.axis('off')

                    cy = total_h - 0.05

                    # Title line
                    ax.text(0.15, cy, title, fontsize=10, fontweight='bold', va='top', color='#222222', clip_on=False)
                    cy -= 0.38

                    # Image inset — compute fraction coords from data coords
                    img_ax_h = IMG_H / total_h
                    img_ax_y = (cy - IMG_H) / total_h
                    inset = ax.inset_axes([0.0, img_ax_y, 1.0, img_ax_h])
                    inset.imshow(pil_img, cmap='gray', aspect='auto',
                                 extent=[0, img_w_px, 0, img_h_px])
                    inset.set_xlim(0, img_w_px)
                    inset.set_ylim(0, img_h_px)
                    inset.axis('off')
                    cy -= IMG_H + 0.10

                    # Truth row
                    TRUTH_Y = cy - CHAR_H / 2
                    ARROW_Y = cy - CHAR_H - ARROW_H / 2
                    PRED_Y  = cy - CHAR_H - ARROW_H - CHAR_H / 2
                    LBL_Y   = cy - CHAR_H - ARROW_H - CHAR_H - LABEL_H / 2

                    op_colors = {'mat': '#2ca02c', 'sub': '#d62728', 'del': '#e08000', 'ins': '#9467bd'}
                    op_labels = {'mat': 'MATCH', 'sub': 'SUB', 'del': 'DEL', 'ins': 'INS'}

                    ax.text(-0.1, TRUTH_Y, "GT:",   fontsize=8, fontweight='bold', ha='right', va='center', clip_on=False)
                    ax.text(-0.1, PRED_Y,  "Pred:", fontsize=8, fontweight='bold', ha='right', va='center', clip_on=False)

                    for col, (op, p_c, t_c) in enumerate(edits_trim):
                        color = op_colors.get(op, '#888888')
                        cx = col * CELL_W + CELL_W / 2

                        if t_c:
                            r = FancyBboxPatch((cx - CELL_W*0.44, TRUTH_Y - CHAR_H*0.5),
                                               CELL_W*0.88, CHAR_H,
                                               boxstyle="round,pad=0.03",
                                               fc='white', ec=color, lw=1.8, clip_on=False)
                            ax.add_patch(r)
                            ax.text(cx, TRUTH_Y, t_c, fontproperties=fp_sm, ha='center', va='center', clip_on=False)

                        ax.text(cx, ARROW_Y, '↓' if op != 'ins' else '↑',
                                fontsize=10, ha='center', va='center', color=color, fontweight='bold', clip_on=False)

                        if p_c:
                            r = FancyBboxPatch((cx - CELL_W*0.44, PRED_Y - CHAR_H*0.5),
                                               CELL_W*0.88, CHAR_H,
                                               boxstyle="round,pad=0.03",
                                               fc='white', ec=color, lw=1.8, clip_on=False)
                            ax.add_patch(r)
                            ax.text(cx, PRED_Y, p_c, fontproperties=fp_sm, ha='center', va='center', clip_on=False)

                        ax.text(cx, LBL_Y, op_labels.get(op, op),
                                fontsize=6, ha='center', va='center', color=color, fontweight='bold', clip_on=False)

                valid_models = [m for m in models if m in demo_predictions]
                if valid_models:
                    demo_pil = Image.open(demo_img_path)
                    demo_pil = add_projection_profile_to_image(demo_pil)
                    # Each model panel is ~4.2 inches tall
                    panel_h = 4.4
                    fig_h = len(valid_models) * panel_h + 0.5
                    fig, axs = plt.subplots(len(valid_models), 1, figsize=(14, fig_h))
                    if len(valid_models) == 1:
                        axs = [axs]

                    fig.suptitle('Random Sample OCR Comparison', fontsize=15, fontweight='bold', y=1.0)

                    for i, model_name in enumerate(valid_models):
                        pred = demo_predictions[model_name]
                        title = f"Model: {model_name}   |   GT: {demo_gt}   |   Pred: {pred}"
                        _draw_alignment_grid(axs[i], demo_pil, demo_gt, pred, title, {})

                    plt.tight_layout(pad=0.6, h_pad=0.8)
                    demo_plot_path = os.path.join(out_dir, 'plot_demo_grid.png')
                    plt.savefig(demo_plot_path, dpi=200, bbox_inches='tight', facecolor='white')
                    plt.close(fig)
                    print(f"  ✓ Demo predictive grid saved to: {demo_plot_path}\n")

                    print("  ==== Demo Predictions ====")
                    print(f"  Image: {demo_img_path}")
                    print(f"  GT:    {demo_gt}")
                    for model_name in valid_models:
                        print(f"  [{model_name}] Pred: {demo_predictions[model_name]}")
                    print("  ==========================\n")

            except Exception as e:
                print(f"  [Demo Grid Error] {e}\n")

        # -------------------------------------------------------------
        # 4. Worst CoER Category Demo Grid
        # -------------------------------------------------------------
        if has_matplot and all_predictions:
            print("  Generating Worst CoER Category Demo...")
            try:
                valid_models = [m for m in models if results.get(m) is not None]
                avg_coer_errors = {}
                for cat, _ in coer_cols:
                    cat_errors = [results[m]['coer'].get(cat, 0.0) for m in valid_models]
                    if cat_errors:
                        avg_coer_errors[cat] = sum(cat_errors) / len(cat_errors)

                if avg_coer_errors:
                    worst_cat = max(avg_coer_errors, key=avg_coer_errors.get)
                    worst_cat_name = dict(coer_cols).get(worst_cat, worst_cat)

                    target_chars = LAO_COMPONENT_MAP.get(worst_cat, [])
                    worst_cat_candidates = [pair for pair in test_pairs if any(c in pair[1] for c in target_chars)]

                    if worst_cat_candidates:
                        worst_demo_img_path, worst_demo_gt = random.choice(worst_cat_candidates)
                        worst_demo_pil = Image.open(worst_demo_img_path)
                        worst_demo_pil = add_projection_profile_to_image(worst_demo_pil)

                        import matplotlib.font_manager as fm
                        from matplotlib.patches import FancyBboxPatch
                        lao_font_path = 'tesstrain_core/font/phetsarath_ot.ttf'
                        fp_sm = fm.FontProperties(fname=lao_font_path, size=11) if os.path.exists(lao_font_path) else fm.FontProperties(family='monospace', size=11)

                        panel_h = 4.4
                        fig_h = len(valid_models) * panel_h + 0.6
                        fig, axs = plt.subplots(len(valid_models), 1, figsize=(14, fig_h))
                        if len(valid_models) == 1:
                            axs = [axs]

                        fig.suptitle(f'Worst Category OCR: {worst_cat_name}', fontsize=15, fontweight='bold', y=1.0)

                        op_colors = {'mat': '#2ca02c', 'sub': '#d62728', 'del': '#e08000', 'ins': '#9467bd'}
                        op_labels = {'mat': 'MATCH', 'sub': 'SUB', 'del': 'DEL', 'ins': 'INS'}
                        MAX_COLS = 20
                        CELL_W = 14.0 / MAX_COLS
                        CHAR_H = 0.52
                        ARROW_H = 0.38
                        LABEL_H = 0.30
                        FIG_W   = 14.0
                        CELL_W  = FIG_W / MAX_COLS

                        # Compute image display height from actual pixel dims
                        img_w_px, img_h_px = worst_demo_pil.size
                        aspect_ratio = img_h_px / max(img_w_px, 1)
                        IMG_H = max(FIG_W * aspect_ratio, 0.8)

                        for i, model_name in enumerate(valid_models):
                            pred_str = all_predictions[model_name].get(worst_demo_img_path, '')
                            edits_trim = align_strings(pred_str, worst_demo_gt)[:MAX_COLS]

                            total_h = 0.40 + IMG_H + 0.10 + CHAR_H + ARROW_H + CHAR_H + LABEL_H + 0.15
                            ax = axs[i]
                            ax.set_xlim(0, FIG_W)
                            ax.set_ylim(0, total_h)
                            ax.axis('off')

                            cy = total_h - 0.05
                            title = f"Model: {model_name}   |   GT: {worst_demo_gt}   |   Pred: {pred_str}"
                            ax.text(0.15, cy, title, fontsize=10, fontweight='bold', va='top', color='#222222', clip_on=False)
                            cy -= 0.38

                            img_ax_h = IMG_H / total_h
                            img_ax_y = (cy - IMG_H) / total_h
                            inset = ax.inset_axes([0.0, img_ax_y, 1.0, img_ax_h])
                            inset.imshow(worst_demo_pil, cmap='gray', aspect='auto',
                                         extent=[0, img_w_px, 0, img_h_px])
                            inset.set_xlim(0, img_w_px)
                            inset.set_ylim(0, img_h_px)
                            inset.axis('off')
                            cy -= IMG_H + 0.10

                            TRUTH_Y = cy - CHAR_H / 2
                            ARROW_Y = cy - CHAR_H - ARROW_H / 2
                            PRED_Y  = cy - CHAR_H - ARROW_H - CHAR_H / 2
                            LBL_Y   = cy - CHAR_H - ARROW_H - CHAR_H - LABEL_H / 2

                            ax.text(-0.1, TRUTH_Y, "GT:",   fontsize=8, fontweight='bold', ha='right', va='center', clip_on=False)
                            ax.text(-0.1, PRED_Y,  "Pred:", fontsize=8, fontweight='bold', ha='right', va='center', clip_on=False)

                            for col, (op, p_c, t_c) in enumerate(edits_trim):
                                color = op_colors.get(op, '#888888')
                                cx = col * CELL_W + CELL_W / 2

                                if t_c:
                                    r = FancyBboxPatch((cx - CELL_W*0.44, TRUTH_Y - CHAR_H*0.5),
                                                       CELL_W*0.88, CHAR_H,
                                                       boxstyle="round,pad=0.03",
                                                       fc='white', ec=color, lw=1.8, clip_on=False)
                                    ax.add_patch(r)
                                    ax.text(cx, TRUTH_Y, t_c, fontproperties=fp_sm, ha='center', va='center', clip_on=False)

                                ax.text(cx, ARROW_Y, '↓' if op != 'ins' else '↑',
                                        fontsize=10, ha='center', va='center', color=color, fontweight='bold', clip_on=False)

                                if p_c:
                                    r = FancyBboxPatch((cx - CELL_W*0.44, PRED_Y - CHAR_H*0.5),
                                                       CELL_W*0.88, CHAR_H,
                                                       boxstyle="round,pad=0.03",
                                                       fc='white', ec=color, lw=1.8, clip_on=False)
                                    ax.add_patch(r)
                                    ax.text(cx, PRED_Y, p_c, fontproperties=fp_sm, ha='center', va='center', clip_on=False)

                                ax.text(cx, LBL_Y, op_labels.get(op, op),
                                        fontsize=6, ha='center', va='center', color=color, fontweight='bold', clip_on=False)

                        plt.tight_layout(pad=0.6, h_pad=0.8)
                        worst_demo_plot_path = os.path.join(out_dir, 'plot_worst_coer_grid.png')
                        plt.savefig(worst_demo_plot_path, dpi=200, bbox_inches='tight', facecolor='white')
                        plt.close(fig)
                        print(f"  ✓ Worst category ({worst_cat_name}) grid saved to: {worst_demo_plot_path}\n")

                        print(f"  ==== Worst Category Demo: {worst_cat_name} ====")
                        print(f"  Image: {worst_demo_img_path}")
                        print(f"  GT:    {worst_demo_gt}")
                        for model_name in valid_models:
                            print(f"  [{model_name}] Pred: {all_predictions[model_name].get(worst_demo_img_path, '')}")
                        print("  ==========================\n")
            except Exception as e:
                print(f"  [Worst Category Grid Error] {e}\n")

        # -------------------------------------------------------------
        # 5. Metric Explanation Demonstration Plot (one file per model,
        #    same 2 GT samples shown across all models for easy comparison)
        # -------------------------------------------------------------
        if has_matplot and 'all_predictions' in locals() and all_predictions:
            print("  Generating Metric Explanation Diagrams...")
            try:
                import matplotlib.font_manager as fm
                from matplotlib.patches import FancyBboxPatch

                valid_models = [m for m in models if m in all_predictions]
                if not valid_models:
                    raise ValueError("No valid models for metric explanation.")

                lao_font_path = 'tesstrain_core/font/phetsarath_ot.ttf'
                lao_font_lg = (fm.FontProperties(fname=lao_font_path, size=16)
                               if os.path.exists(lao_font_path)
                               else fm.FontProperties(family='monospace', size=16))
                lao_font_sm = (fm.FontProperties(fname=lao_font_path, size=13)
                               if os.path.exists(lao_font_path)
                               else fm.FontProperties(family='monospace', size=13))

                cat_colors = {
                    'consonant':    '#1f77b4', 'tone':        '#d62728',
                    'before_vowel': '#2ca02c', 'rear_vowel':  '#9467bd',
                    'upper_vowel':  '#ff7f0e', 'lower_vowel': '#8c564b',
                    'symbol':       '#7f7f7f', 'lao_number':  '#17becf',
                    'digit':        '#bcbd22', 'special':     '#e377c2',
                    'latin':        '#393b79', 'space':       '#c7c7c7',
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

                # ── Pick 2 shared GT image paths (same across all models) ──
                _ref = valid_models[0]
                _err = [(f, t) for f, t in test_pairs
                        if all_predictions[_ref].get(f, t) != t and 3 <= len(t) <= 25]
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
                else:
                    def _sample_h(truth, pred):
                        cc  = list(truth)[:MAX_COLS * 3]
                        n_c = len([cc[i:i+MAX_COLS] for i in range(0, len(cc), MAX_COLS)])
                        ed  = align_strings(pred, truth)
                        n_e = len([ed[i:i+MAX_COLS] for i in range(0, len(ed[:MAX_COLS*3]), MAX_COLS)])
                        return (HDR_H + n_c*(CHAR_H+LABEL_H+ROW_GAP)
                                + SEP_H + HDR_H + n_e*EDIT_ROW_H + FORMULA_H)

                    n_leg_rows     = (len(cat_colors) - 1) // 5 + 1
                    legend_block_h = 0.44 + n_leg_rows * 0.30

                    for demo_m in valid_models:
                        try:
                            # Each model uses its own predictions for the shared GT paths
                            demo_samples = [(truth, all_predictions[demo_m].get(img_f, truth))
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

                                # CoER header
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
                                            CELL_W*0.84, CHAR_H,
                                            boxstyle='round,pad=0.04',
                                            fc=color+'22', ec=color, lw=1.8, clip_on=False))
                                        ax.text(cx, char_cy, ch, fontproperties=lao_font_lg,
                                                ha='center', va='center', color='#111111', clip_on=False)
                                        ax.text(cx, label_cy, cat.replace('_', '\n'),
                                                fontsize=6.5, ha='center', va='center',
                                                color=color, fontweight='bold',
                                                linespacing=1.15, clip_on=False)
                                    cy -= CHAR_H + LABEL_H + ROW_GAP

                                # Legend once (after first sample's CoER)
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
                                                fontsize=7.5, va='center', color=color,
                                                fontweight='bold')
                                    cy -= n_leg_rows * 0.30 + 0.22

                                ax.plot([0.1, 13.9], [cy, cy], color='#cccccc', lw=0.8)
                                cy -= SEP_H

                                # CER/WER header
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
                                    ax.text(-0.05, TRUTH_Y, 'GT:',   fontsize=9,
                                            fontweight='bold', ha='right', va='center', clip_on=False)
                                    ax.text(-0.05, PRED_Y,  'Pred:', fontsize=9,
                                            fontweight='bold', ha='right', va='center', clip_on=False)
                                    for col, (op, p_c, t_c) in enumerate(row_edits):
                                        color = op_colors.get(op, '#888888')
                                        cx = col * CELL_W + CELL_W / 2
                                        if t_c:
                                            ax.add_patch(FancyBboxPatch(
                                                (cx-CELL_W*0.42, TRUTH_Y-0.28),
                                                CELL_W*0.84, 0.56, boxstyle='round,pad=0.04',
                                                fc='white', ec=color, lw=2, clip_on=False))
                                            ax.text(cx, TRUTH_Y, t_c,
                                                    fontproperties=lao_font_sm,
                                                    ha='center', va='center', clip_on=False)
                                        ax.text(cx, ARROW_Y,
                                                '↓' if op != 'ins' else '↑',
                                                fontsize=13, ha='center', va='center',
                                                color=color, fontweight='bold', clip_on=False)
                                        if p_c:
                                            ax.add_patch(FancyBboxPatch(
                                                (cx-CELL_W*0.42, PRED_Y-0.28),
                                                CELL_W*0.84, 0.56, boxstyle='round,pad=0.04',
                                                fc='white', ec=color, lw=2, clip_on=False))
                                            ax.text(cx, PRED_Y, p_c,
                                                    fontproperties=lao_font_sm,
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
                                ax.text(0.2, cy - 0.05, formula, fontsize=9,
                                        style='italic', va='top',
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
            except Exception as e:
                print(f"  [Metric Explanation Error] {e}\n")


def _rolling_mean_std(data, window_size=5):
    """Returns rolling mean and std deviation for smooth plotting and shaded variance intervals."""

    if len(data) < window_size:
        xs = [d[0] for d in data]
        ys = [d[1] for d in data]
        return xs, ys, [0]*len(ys)

    xs = [d[0] for d in data]
    ys = [d[1] for d in data]
    
    smoothed_y = []
    std_y = []
    
    for i in range(len(ys)):
        start = max(0, i - window_size // 2)
        end = min(len(ys), i + window_size // 2 + 1)
        window = ys[start:end]
        
        mean = sum(window) / len(window)
        # Sample variance
        variance = sum((x - mean) ** 2 for x in window) / max(1, len(window) - 1)
        std = variance ** 0.5
        
        smoothed_y.append(mean)
        std_y.append(std)
        
    return xs, smoothed_y, std_y
