import os
import glob
import random
import re
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch

from tesstrain_core.model import build_model
from tesstrain_core.recognize import recognize_line
from tesstrain_core.metrics import align_strings

def add_projection_profile_to_image(pil_img):
    """Augment image by adding its vertical projection profile underneath it."""
    # Convert image to grayscale (L)
    gray_img = pil_img.convert("L")
    w, h = gray_img.size
    
    # Check if image is light background or dark background
    stat = np.mean(np.array(gray_img))
    is_light_bg = stat > 127
    
    # Determine threshold and colors
    if is_light_bg:
        text_threshold = lambda val: val < 128
        bg_color = 255
        text_color = 0
    else:
        text_threshold = lambda val: val >= 128
        bg_color = 0
        text_color = 255

    # Calculate vertical projection profile
    pixels = gray_img.load()
    projection = []
    for x in range(w):
        text_count = 0
        for y in range(h):
            if text_threshold(pixels[x, y]):
                text_count += 1
        projection.append(text_count)
        
    proj_h = h
    divider_h = 2
    
    combined_img = Image.new("L", (w, h + divider_h + proj_h), color=bg_color)
    combined_img.paste(gray_img, (0, 0))
    
    combined_draw = ImageDraw.Draw(combined_img)
    divider_y = h
    combined_draw.line([(0, divider_y), (w, divider_y)], fill=text_color, width=divider_h)
    
    start_y = divider_y + divider_h
    for x in range(w):
        length = projection[x]
        if length > 0:
            combined_draw.line([(x, start_y), (x, start_y + length)], fill=text_color, width=1)
            
    return combined_img

def cmd_projection(args):
    print("======================================================================")
    print("  Tesseract PyTorch Model Projection Analysis")
    print("======================================================================")

    models = list(getattr(args, 'models', []))
    if not models:
        print("  ERROR: Please specify at least one model with --models")
        return

    custom_dir = getattr(args, 'custom_img_dir', '')
    realworld_dir = getattr(args, 'custom_realworld_img_dir', '')
    out_dir = getattr(args, 'output_dir', 'data/projection')
    os.makedirs(out_dir, exist_ok=True)

    num_samples = getattr(args, 'num_samples', 5)
    min_length = getattr(args, 'min_length', 40)

    # Load fonts for matplotlib CJK/Lao characters
    lao_font_path = 'tesstrain_core/font/phetsarath_ot.ttf'
    fp_sm = fm.FontProperties(fname=lao_font_path, size=11) if os.path.exists(lao_font_path) else fm.FontProperties(family='monospace', size=11)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")

    # 1. Load checkpoints and models
    loaded_models = {}
    for model_name in models:
        ckpt_path = f'data/{model_name}/{model_name}_best.pt'
        if not os.path.exists(ckpt_path):
            ckpt_path = f'data/{model_name}/{model_name}_final.pt'
            if not os.path.exists(ckpt_path):
                print(f"  [Warning] No checkpoint found for {model_name}. Skipping.")
                continue
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            spec = ckpt['net_spec'].replace('###', str(ckpt['num_classes']))
            model = build_model(spec, ckpt['num_classes']).to(device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            
            inv_charset = {v: k for k, v in ckpt['charset'].items()}
            target_h = 48
            m = re.match(r'\[?\d+,(\d+),', ckpt.get('net_spec', ''))
            if m:
                target_h = int(m.group(1))

            loaded_models[model_name] = {
                'model': model,
                'target_h': target_h,
                'inv_charset': inv_charset
            }
            print(f"  ✓ Loaded model: {model_name}")
        except Exception as e:
            print(f"  [Error] Failed to load model {model_name}: {e}")

    if not loaded_models:
        print("  ERROR: No models loaded successfully.")
        return

    # Process each custom directory
    dirs_to_process = []
    if custom_dir:
        dirs_to_process.append((custom_dir, "custom"))
    if realworld_dir:
        dirs_to_process.append((realworld_dir, "realworld"))

    if not dirs_to_process:
        print("  ERROR: Please specify --custom_img_dir or --custom_realworld_img_dir")
        return

    for test_dir, prefix in dirs_to_process:
        print(f"\n  Processing directory: {test_dir}")
        gt_files = sorted(glob.glob(os.path.join(test_dir, '**', '*.gt.txt'), recursive=True))
        if not gt_files:
            print(f"  [Warning] No .gt.txt files found in {test_dir}")
            continue

        pairs = []
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
                    pairs.append((img_f, truth))

        # Filter by min_length
        filtered_pairs = [(img_f, truth) for img_f, truth in pairs if len(truth) >= min_length]
        print(f"  Found {len(pairs)} pairs total. {len(filtered_pairs)} meet min_length >= {min_length}")

        if not filtered_pairs:
            print(f"  No pairs meet the minimum length filter of {min_length} in {test_dir}")
            continue

        # Deterministic shuffle & select samples
        random.seed(42)
        random.shuffle(filtered_pairs)
        selected_pairs = filtered_pairs[:num_samples]
        print(f"  Selected {len(selected_pairs)} samples for projection visualization.")

        # For each selected sample
        for idx, (img_path, gt_text) in enumerate(selected_pairs):
            print(f"    - Processing sample {idx+1}/{len(selected_pairs)}: {os.path.basename(img_path)}")
            pil_img = Image.open(img_path).convert('L')
            
            # 1. Combined image + projection (only computed once per sample)
            augmented_img = add_projection_profile_to_image(pil_img)
            img_w_px, img_h_px = augmented_img.size
            
            # Define layout constants
            MAX_COLS = 30
            FIG_W = 16.0
            CELL_W = FIG_W / MAX_COLS
            CHAR_H = 0.55
            ARROW_H = 0.40
            LABEL_H = 0.35
            
            # Subplot height for image: preserve aspect ratio
            aspect_ratio = img_h_px / max(img_w_px, 1)
            IMG_H = max(FIG_W * aspect_ratio, 1.2)
            
            # Subplot height for the model's comparison:
            model_row_h = CHAR_H + ARROW_H + CHAR_H + LABEL_H + 0.40
            
            # Total figure dimensions for a single model plot
            total_figure_h = IMG_H + model_row_h + 1.0
            
            # For each model, generate and save a separate plot in its folder
            for model_name, m_info in loaded_models.items():
                # Create model output directory
                model_out_dir = os.path.join(out_dir, model_name)
                os.makedirs(model_out_dir, exist_ok=True)
                
                try:
                    pred_str = recognize_line(
                        m_info['model'], pil_img, m_info['target_h'],
                        device, m_info['inv_charset'], beam_width=0, dawg_decoder=None
                    )
                except Exception as e:
                    pred_str = f"[Error: {e}]"
                
                # Setup plot
                fig = plt.figure(figsize=(16, total_figure_h))
                
                # 1. Draw Augmented Image Subplot (at the top)
                img_frac = IMG_H / total_figure_h
                ax_img = fig.add_axes([0.05, 1.0 - img_frac - 0.05, 0.90, img_frac])
                ax_img.imshow(augmented_img, cmap='gray', aspect='auto', extent=[0, img_w_px, 0, img_h_px])
                ax_img.set_xlim(0, img_w_px)
                ax_img.set_ylim(0, img_h_px)
                ax_img.axis('off')
                ax_img.set_title(f"Sample {idx+1} | GT: {gt_text}", fontsize=14, fontweight='bold', pad=10)
                
                # 2. Draw Model Alignment Subplot (below)
                curr_y_top = 1.0 - img_frac - 0.08
                row_frac = model_row_h / total_figure_h
                
                edits = align_strings(pred_str, gt_text)
                edits_trim = edits[:MAX_COLS]
                
                ax_row = fig.add_axes([0.05, curr_y_top - row_frac, 0.90, row_frac])
                ax_row.set_xlim(0, FIG_W)
                ax_row.set_ylim(0, model_row_h)
                ax_row.axis('off')
                
                # Title / Model info
                ax_row.text(0.0, model_row_h - 0.10, f"Model: {model_name} (Pred: {pred_str})", 
                            fontsize=10, fontweight='bold', va='top', color='#333333')
                
                # Draw character boxes
                TRUTH_Y = model_row_h - 0.70
                ARROW_Y = TRUTH_Y - CHAR_H - 0.15
                PRED_Y  = ARROW_Y - ARROW_H - 0.15
                LBL_Y   = PRED_Y - CHAR_H - 0.15
                
                op_colors = {'mat': '#2ca02c', 'sub': '#d62728', 'del': '#e08000', 'ins': '#9467bd'}
                op_labels = {'mat': 'MATCH', 'sub': 'SUB', 'del': 'DEL', 'ins': 'INS'}
                
                ax_row.text(-0.25, TRUTH_Y, "GT:",   fontsize=8, fontweight='bold', ha='right', va='center')
                ax_row.text(-0.25, PRED_Y,  "Pred:", fontsize=8, fontweight='bold', ha='right', va='center')
                
                for col, (op, p_c, t_c) in enumerate(edits_trim):
                    color = op_colors.get(op, '#888888')
                    cx = col * CELL_W + CELL_W / 2
                    
                    # GT Character Box
                    if t_c:
                        r = FancyBboxPatch((cx - CELL_W*0.44, TRUTH_Y - CHAR_H*0.5),
                                           CELL_W*0.88, CHAR_H,
                                           boxstyle="round,pad=0.03",
                                           fc='white', ec=color, lw=1.5)
                        ax_row.add_patch(r)
                        ax_row.text(cx, TRUTH_Y, t_c, fontproperties=fp_sm, ha='center', va='center')
                        
                    # Alignment Indicator
                    ax_row.text(cx, ARROW_Y, '↓' if op != 'ins' else '↑',
                                fontsize=9, ha='center', va='center', color=color, fontweight='bold')
                    
                    # Predicted Character Box
                    if p_c:
                        r = FancyBboxPatch((cx - CELL_W*0.44, PRED_Y - CHAR_H*0.5),
                                           CELL_W*0.88, CHAR_H,
                                           boxstyle="round,pad=0.03",
                                           fc='white', ec=color, lw=1.5)
                        ax_row.add_patch(r)
                        ax_row.text(cx, PRED_Y, p_c, fontproperties=fp_sm, ha='center', va='center')
                        
                    # Edit Distance Operation Labels
                    ax_row.text(cx, LBL_Y, op_labels.get(op, op),
                                fontsize=6, ha='center', va='center', color=color, fontweight='bold')
                
                plt.tight_layout()
                out_filename = f"{prefix}_sample_{idx+1}.png"
                out_path = os.path.join(model_out_dir, out_filename)
                plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                print(f"    ✓ Projection plot for {model_name} saved to: {out_path}")

    print("\n  ✓ Finished generating projection analyses.\n")
