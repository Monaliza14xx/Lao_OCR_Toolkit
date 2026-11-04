import os
import sys
import glob
import random
import torch
import numpy as np
from PIL import Image
from argparse import Namespace

from tesstrain_core.utils import resolve_paths
from tesstrain_core.dataset import TesseractDataset
from tesstrain_core.model import (
    build_model, AttentionCollapseY, MultiHeadAttentionCollapseY,
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    pass

def _normalize_and_save(tensor: torch.Tensor, save_path: str, is_heatmap=False):
    """Normalize a 2D tensor to 0-255 uint8 and save as PNG."""
    arr = tensor.detach().cpu().numpy()
    if arr.size == 0:
        return
        
    # Min-max normalization per image
    amin, amax = arr.min(), arr.max()
    if amax > amin:
        arr = (arr - amin) / (amax - amin)
    else:
        arr = np.zeros_like(arr)
        
    arr = (arr * 255.0).astype(np.uint8)
    
    if is_heatmap:
        # For LSTMs, applying a simple colormap logic could be done, 
        # but grayscale is universally readable and avoids extra dependencies.
        # We can just save it as grayscale.
        pass
        
    img = Image.fromarray(arr, mode='L')
    img.save(save_path)


# ──────────────────────────────────────────────────────────────────────────────
#  Attention heatmap helpers
# ──────────────────────────────────────────────────────────────────────────────

def _capture_attention_weights(layer, x_4d: torch.Tensor):
    """Run x_4d through an attention collapse layer and return the captured
    attention weights alpha as a numpy array.

    Returns
    -------
    alpha_np : np.ndarray, shape (H, W)   for AttentionCollapseY
               np.ndarray, shape (num_heads, H, W) for MultiHeadAttentionCollapseY
    """
    captured = {}

    if isinstance(layer, AttentionCollapseY):
        def hook_fn(module, inp, out):
            # Recompute alpha inside the hook by re-running the scoring.
            # The hook fires after forward(), but we need alpha — so we re-derive
            # it from the saved pre-hook input (same as x_4d passed in).
            x = inp[0]          # (B*W, H, C)
            q = torch.tanh(module.query(x))
            k = torch.tanh(module.key(x))
            e = module.energy(q + k).squeeze(-1)   # (B*W, H)
            captured['alpha'] = torch.softmax(e, dim=-1).detach().cpu()

        handle = layer.energy.register_forward_hook(
            lambda m, i, o: None)   # placeholder, overridden below
        handle.remove()

        # Re-run with a proper hook on the whole layer
        def layer_hook(m, inp, out):
            x = inp[0]
            if x.dim() == 4:
                B, C, H, W = x.shape
                x_col = x.permute(0, 3, 2, 1).contiguous().view(B * W, H, C)
                q = torch.tanh(m.query(x_col))
                k = torch.tanh(m.key(x_col))
                e = m.energy(q + k).squeeze(-1)
                alpha = torch.softmax(e, dim=-1).detach().cpu()  # (B*W, H)
                B_orig = x.shape[0]
                W_orig = x.shape[3]
                captured['alpha'] = alpha.view(B_orig, W_orig, H)  # (B, W, H)

        h = layer.register_forward_hook(layer_hook)
        with torch.no_grad():
            layer(x_4d)
        h.remove()

        alpha = captured.get('alpha')           # (1, W, H)
        if alpha is None:
            return None
        return alpha[0].permute(1, 0).numpy()   # (H, W)

    elif isinstance(layer, MultiHeadAttentionCollapseY):
        def mha_hook(m, inp, out):
            x = inp[0]
            if x.dim() == 4:
                B, C, H, W = x.shape
                x_col = x.permute(0, 3, 2, 1).contiguous().view(B * W, H, C)
                BW = B * W
                kv = m.kv_proj(x_col)
                k, _ = kv.split(m.out_d, dim=-1)
                k = k.view(BW, H, m.num_heads, m.head_dim)
                cq = m.collapse_q.expand(BW, -1, -1, -1)
                attn = (cq * k).sum(dim=-1) / m.scale  # (BW, H, heads)
                alpha = torch.softmax(attn, dim=1).detach().cpu()  # (BW, H, heads)
                alpha = alpha.view(B, W, H, m.num_heads)           # (B, W, H, heads)
                captured['alpha'] = alpha

        h = layer.register_forward_hook(mha_hook)
        with torch.no_grad():
            layer(x_4d)
        h.remove()

        alpha = captured.get('alpha')                       # (1, W, H, heads)
        if alpha is None:
            return None
        # Return (num_heads, H, W)
        return alpha[0].permute(2, 1, 0).numpy()           # (heads, H, W)

    return None


def _save_attention_heatmaps(layer, x_4d: torch.Tensor,
                              out_dir: str, layer_name: str):
    """Extract and save attention weight heatmaps to out_dir.

    Saves:
      attention_heatmap.png                    — (H × W) single map or mean-across-heads
      attention_head_{n}.png                   — per-head maps (MultiHead only)
      attention_heads_overview.png             — annotated matplotlib figure (MultiHead only)
    """
    os.makedirs(out_dir, exist_ok=True)
    alpha = _capture_attention_weights(layer, x_4d)
    if alpha is None:
        return

    if alpha.ndim == 2:
        # AttentionCollapseY: (H, W)
        arr = alpha
        arr_u8 = np.clip((arr - arr.min()) / max(arr.max() - arr.min(), 1e-8) * 255,
                         0, 255).astype(np.uint8)
        Image.fromarray(arr_u8, mode='L').save(
            os.path.join(out_dir, 'attention_heatmap.png'))

    elif alpha.ndim == 3:
        # MultiHeadAttentionCollapseY: (num_heads, H, W)
        num_heads = alpha.shape[0]

        # Save per-head maps
        for h_idx in range(num_heads):
            arr = alpha[h_idx]
            arr_u8 = np.clip(
                (arr - arr.min()) / max(arr.max() - arr.min(), 1e-8) * 255,
                0, 255).astype(np.uint8)
            Image.fromarray(arr_u8, mode='L').save(
                os.path.join(out_dir, f'attention_head_{h_idx}.png'))

        # Save mean-across-heads
        mean_map = alpha.mean(axis=0)
        mean_u8 = np.clip(
            (mean_map - mean_map.min()) / max(mean_map.max() - mean_map.min(), 1e-8) * 255,
            0, 255).astype(np.uint8)
        Image.fromarray(mean_u8, mode='L').save(
            os.path.join(out_dir, 'attention_heatmap.png'))

        # Save an annotated matplotlib overview
        if 'plt' in globals():
            tier_labels = ['Lower Vowel', 'Consonant', 'Upper Vowel', 'Tone Mark']
            fig, axes = plt.subplots(1, num_heads + 1,
                                     figsize=(3.5 * (num_heads + 1), 2.5))
            for h_idx in range(num_heads):
                ax = axes[h_idx]
                ax.imshow(alpha[h_idx], cmap='viridis', aspect='auto',
                          vmin=0, vmax=alpha.max())
                label = tier_labels[h_idx] if h_idx < len(tier_labels) else f'Head {h_idx}'
                ax.set_title(f'Head {h_idx}\n({label})', fontsize=9)
                ax.set_xlabel('Column (→ time)', fontsize=7)
                if h_idx == 0:
                    ax.set_ylabel('Row (↓ height)', fontsize=7)
                else:
                    ax.set_yticks([])

            axes[num_heads].imshow(mean_map, cmap='viridis', aspect='auto',
                                   vmin=0, vmax=alpha.max())
            axes[num_heads].set_title('Mean\n(all heads)', fontsize=9)
            axes[num_heads].set_xlabel('Column (→ time)', fontsize=7)
            axes[num_heads].set_yticks([])

            plt.suptitle(f'{layer_name} — Vertical Attention Weights', fontsize=11,
                         fontweight='bold')
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, 'attention_heads_overview.png'),
                        dpi=200, bbox_inches='tight')
            plt.close(fig)


def _make_attention_overview_image(layer, x_4d: torch.Tensor,
                                   layer_name: str) -> np.ndarray | None:
    """Return a (H, W) uint8 numpy array of the mean attention weights,
    suitable for embedding in the inspection overview grid."""
    alpha = _capture_attention_weights(layer, x_4d)
    if alpha is None:
        return None

    if alpha.ndim == 2:
        arr = alpha
    elif alpha.ndim == 3:
        arr = alpha.mean(axis=0)
    else:
        return None

    arr_u8 = np.clip(
        (arr - arr.min()) / max(arr.max() - arr.min(), 1e-8) * 255,
        0, 255).astype(np.uint8)
    return arr_u8


def cmd_inspection(args: Namespace):
    print("======================================================================")
    print("  Tesseract PyTorch Model Inspection")
    print("======================================================================")
    
    p = resolve_paths(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device:       {device}")
    
    tessdata_dir = getattr(args, 'tessdata_dir', '')
    
    if tessdata_dir:
        tess_path = os.path.join(tessdata_dir, f"{args.model_name}.traineddata")
        if not os.path.exists(tess_path):
            print(f"  ERROR: Tesseract model not found: {tess_path}")
            return
            
        print(f"  Loading Tesseract: {tess_path}")
        from tesstrain_core.tesseract_transfer import load_tesseract_pretrained_state_dict
        
        # Infer net_spec from model name if not provided in args
        if hasattr(args, 'net_spec') and args.net_spec:
            net_spec = args.net_spec
            print(f"  Using provided SPEC: {net_spec}")
        else:
            mname = args.model_name
            h = 64 if "I64" in mname else 48
            if "K5C32" in mname: ct = "Ct5,5,32"
            elif "K5C16" in mname: ct = "Ct5,5,16"
            elif "K3C32" in mname: ct = "Ct3,3,32"
            else: ct = "Ct3,3,16"
            
            net_spec = f"[1,{h},0,1 {ct} Mp3,3 Lfys{h} Lfx96 Lrx96 Lfx256 O1c###]"
            print(f"  Inferred SPEC: {net_spec}")
        
        tess_state, tess_meta = load_tesseract_pretrained_state_dict(tess_path, net_spec)
        num_classes = list(tess_state.values())[-2].shape[0]
        model = build_model(net_spec, num_classes).to(device)
        model.load_state_dict(tess_state)
        model.eval()
        
        out_base = os.path.join(p.output_dir, 'inspection_tesseract')
    else:
        # 1. Find checkpoint
        ckpt_path = getattr(args, 'start_model', None)
        if not ckpt_path:
            ckpt_path = os.path.join(p.output_dir, f"{args.model_name}_best.pt")
            
        if not os.path.exists(ckpt_path):
            print(f"  ERROR: Checkpoint not found at {ckpt_path}")
            return
            
        print(f"  Loading PyTorch: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        
        net_spec = ckpt.get('net_spec', '[1,36,0,1 Ct3,3,16 Mp3,3 Lfys48 Lfx96 Lrx96 Lfx256 O1c###]')
        print(f"  NET_SPEC:     {net_spec}")
        
        # 2. Build model
        out_weight = ckpt['model_state_dict'].get(f"{len(ckpt['model_state_dict'])//2 - 1}.weight")
        if out_weight is not None:
            num_classes = out_weight.shape[0]
        else:
            num_classes = list(ckpt['model_state_dict'].values())[-2].shape[0]
            
        model = build_model(net_spec, num_classes).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        
        out_base = os.path.join(p.output_dir, 'inspection')
    
    # 3. Find files to inspect
    files = []
    if os.path.exists(p.val_list):
        with open(p.val_list) as f:
            files = [l.strip() for l in f if l.strip()]
    
    if not files:
        files = sorted(glob.glob(os.path.join(p.ground_truth_dir, '**', '*.lstmf'), recursive=True))
        
    if not files:
        print("  ERROR: No .lstmf files found to inspect.")
        return
        
    num_samples = getattr(args, 'num_samples', 10)
    min_length  = getattr(args, 'min_length',  3)
    seed        = getattr(args, 'seed',        None)
    
    dataset = TesseractDataset(files)
    valid_indices = [i for i, (_, text) in enumerate(dataset.samples) if len(text) >= min_length]
    if not valid_indices:
        print(f"  ERROR: No samples found matching min_length >= {min_length}")
        return
    
    rng = random.Random(seed)   # fixed seed → same samples across all models
    selected_indices = rng.sample(valid_indices, min(num_samples, len(valid_indices)))
    selected_indices.sort()     # keep a stable order for nicer numbered output
    
    samples = [files[i] for i in selected_indices]
    dataset = TesseractDataset(samples)
    print(f"  Seed:         {seed if seed is not None else 'random'}")
    print(f"  Inspecting:   {len(samples)} images")
    os.makedirs(out_base, exist_ok=True)
    print(f"  Output Dir:   {out_base}")
    
    # Store data for the overview grid plot
    # overview_data[sample_idx] = {'Input': img_np, 'Conv2d (1)': img_np, ...}
    overview_data = []
    
    # 4. Forward pass and export
    for i in range(len(samples)):
        img_tensor, _, _, _ = dataset[i]
        gt_text = dataset.samples[i][1]
        sample_dir = os.path.join(out_base, f"{i+1:02d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Write ground truth to text file
        with open(os.path.join(sample_dir, 'gt.txt'), 'w', encoding='utf-8') as f:
            f.write(gt_text + '\n')
            
        x = img_tensor.unsqueeze(0).to(device) # (1, 1, H, W)
        
        inv_charset = dataset.inv_charset if hasattr(dataset, 'inv_charset') else {}
        with torch.no_grad():
            log_probs = model(x)
        pred_text = _ctc_greedy_decode(log_probs, inv_charset)
        
        with open(os.path.join(sample_dir, 'pred.txt'), 'w', encoding='utf-8') as f:
            f.write(pred_text + '\n')
        
        print(f"\n  [{i+1}/{len(samples)}] '{gt_text}' -> {sample_dir}")
        
        sample_dict = {}
        sample_dict['GT'] = gt_text
        sample_dict['Pred'] = pred_text
        
        # Save input
        layer_dir = os.path.join(sample_dir, "00_Input")
        os.makedirs(layer_dir, exist_ok=True)
        
        # Invert colors so text is black on white for easier viewing (assume input is white on black typical for tesseract)
        img_np = x[0, 0].cpu().numpy()
        min_v, max_v = img_np.min(), img_np.max()
        if max_v > min_v:
            img_np = (img_np - min_v) / (max_v - min_v)
        out_img = np.array(255 - (img_np * 255.0), dtype=np.uint8)
        Image.fromarray(out_img, mode='L').save(os.path.join(layer_dir, "input.png"))
        
        sample_dict['Original Input'] = out_img
        
        # Forward through layers, tracking the previous 4-D tensor for
        # attention heatmap extraction (needs the pre-collapse CNN feature map).
        conv_cnt = 0
        pool_cnt = 0
        _prev_x = x  # will be updated each layer
        for layer_idx, layer in enumerate(model):
            _prev_x = x  # save the input to this layer
            x = layer(x)
            
            layer_name = layer.__class__.__name__
            layer_dir = os.path.join(sample_dir, f"{layer_idx+1:02d}_{layer_name}")
            os.makedirs(layer_dir, exist_ok=True)
            
            if x.dim() == 4:
                # CNN / MaxPool output: (1, C, H, W)
                C_dim = x.shape[1]
                for c in range(C_dim):
                    save_path = os.path.join(layer_dir, f"channel_{c:03d}.png")
                    _normalize_and_save(x[0, c], save_path)
                
                # Save mean activation for overview
                mean_act = x[0].mean(dim=0).detach().cpu().numpy()
                amin, amax = mean_act.min(), mean_act.max()
                if amax > amin:
                    mean_act = (mean_act - amin) / (amax - amin)
                mean_img = np.array(mean_act * 255.0, dtype=np.uint8)
                Image.fromarray(mean_img, mode='L').save(os.path.join(layer_dir, "mean_activation.png"))
                
                if layer_name == 'Conv2d':
                    conv_cnt += 1
                    if conv_cnt <= 3: # Keep first few convs for overview
                        sample_dict[f'Conv2d ({conv_cnt}) Mean'] = mean_img
                elif layer_name == 'MaxPool2d':
                    pool_cnt += 1
                    if pool_cnt <= 2: # Keep first few pools for overview
                        sample_dict[f'MaxPool ({pool_cnt}) Mean'] = mean_img

            elif x.dim() == 3:
                # LSTM / Linear / Attention-collapse output: (1, W, C)
                features = x[0].transpose(0, 1) # (C, W)
                save_path = os.path.join(layer_dir, "features.png")
                _normalize_and_save(features, save_path, is_heatmap=True)

                # ── Attention heatmap export ──────────────────────────────────
                # For AttentionCollapseY / MultiHeadAttentionCollapseY we
                # re-run just that layer with a hook to capture attention weights,
                # then save them as interpretable vertical-attention heatmaps.
                if isinstance(layer, (AttentionCollapseY, MultiHeadAttentionCollapseY)):
                    _save_attention_heatmaps(layer, _prev_x, layer_dir, layer_name)
                    # Also add a thumbnail to the overview
                    attn_heatmap_np = _make_attention_overview_image(
                        layer, _prev_x, layer_name)
                    if attn_heatmap_np is not None:
                        sample_dict[f'{layer_name} Attention'] = attn_heatmap_np

        overview_data.append(sample_dict)
                
    # 5. Build Matplotlib Overview Grid
    has_matplot = 'plt' in globals()
    if has_matplot and overview_data:
        print("\n  Generating academic overview grid plotting...")
        
        # Max 6 samples for width
        plot_samples = overview_data[:6]
        n_cols = len(plot_samples)
        
        # Get all layer keys from the first sample
        row_keys = [k for k in plot_samples[0].keys() if k not in ['GT', 'Pred']]
        n_rows = len(row_keys)
        
        import matplotlib.font_manager as fm
        font_path = os.path.join(os.path.dirname(__file__), 'font', 'phetsarath_ot.ttf')
        lao_font_title = fm.FontProperties(fname=font_path, size=9) if os.path.exists(font_path) else None
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 1.5))
        if n_cols == 1: axes = np.expand_dims(axes, axis=1)
        if n_rows == 1: axes = np.expand_dims(axes, axis=0)
        
        for col in range(n_cols):
            samp = plot_samples[col]
            
            gt = samp.get('GT', '')
            pred = samp.get('Pred', '')
            title = f"GT: {gt}\nPred: {pred}"
            
            for row, key in enumerate(row_keys):
                ax = axes[row, col]
                img = samp[key]
                ax.imshow(img, cmap='gray', aspect='auto')
                ax.set_xticks([])
                ax.set_yticks([])
                
                if row == 0:
                    if lao_font_title:
                        ax.set_title(title, fontproperties=lao_font_title, color='#1a1a2e')
                    else:
                        ax.set_title(title, fontsize=8, color='#1a1a2e')
                
                # Only put title on first column or top row
                if col == 0:
                    ax.set_ylabel(key, fontsize=12, fontweight='bold', rotation=0, labelpad=40, ha='right', va='center')
                    
        plt.tight_layout()
        grid_path = os.path.join(out_base, "inspection_overview_grid.png")
        plt.savefig(grid_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Overview grid saved to: {grid_path}")

    print(f"\n  Inspection complete. Results in {out_base}")


# ──────────────────────────────────────────────────────────────────────────────
#  Cross-Model Inspection: WHY one model beats another
#  Usage: call cmd_compare_inspection(args) where args has:
#    args.models        = ['lao48cnn3x3c16', 'lao48cnn3x3c32',
#                          'lao48cnn5x5c16', 'lao48cnn5x5c32']
#    args.model_labels  = ['A: 3x3/16ch', 'B: 3x3/32ch',
#                          'C: 5x5/16ch', 'D: 5x5/32ch']   (optional)
#    args.num_samples   = 6   (how many images to inspect)
#    args.output_dir    = 'data/comparison'
#    args.ground_truth_dir / args.val_list  (same as normal inspection)
# ──────────────────────────────────────────────────────────────────────────────

def _load_model_from_name(model_name: str, base_dir: str, device):
    """Load a checkpoint by model name, return (model, net_spec) or (None, None)."""
    ckpt_path = os.path.join(base_dir, model_name, f"{model_name}_best.pt")
    if not os.path.exists(ckpt_path):
        # fallback: scan for any .pt file
        candidates = glob.glob(os.path.join(base_dir, model_name, "*.pt"))
        if not candidates:
            print(f"  WARNING: No checkpoint found for {model_name}")
            return None, None
        ckpt_path = sorted(candidates)[-1]

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    net_spec = ckpt.get('net_spec',
                        '[1,36,0,1 Ct3,3,16 Mp3,3 Lfys48 Lfx96 Lrx96 Lfx256 O1c###]')
    num_classes = list(ckpt['model_state_dict'].values())[-2].shape[0]
    model = build_model(net_spec, num_classes).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  ✓ Loaded {model_name}  ({net_spec})")
    return model, net_spec


def _extract_cnn_activations(model, x: torch.Tensor):
    """
    Run x through model, stop at first non-4D layer.
    Returns list of (layer_name, mean_activation_2d_np) for CNN/Pool layers.
    Also returns the post-CNN feature map shape.
    """
    activations = []
    with torch.no_grad():
        for layer in model:
            x = layer(x)
            if x.dim() == 4:               # still CNN/Pool stage
                name = layer.__class__.__name__
                # Return the full (C, H, W) tensor rather than already taking the mean
                activations.append((name, x[0].cpu().numpy()))
            else:
                break                       # entered LSTM stage
    return activations


def _ctc_greedy_decode(log_probs: torch.Tensor, inv_charset: dict) -> str:
    """Simple greedy CTC decode: argmax → collapse repeats → remove blank."""
    indices = log_probs.argmax(dim=-1).view(-1).cpu().tolist()  # (T,)
    blank = 0  # PyTorch CTCLoss default blank is 0
    out, prev = [], -1
    for idx in indices:
        if idx != prev and idx != blank:
            ch = inv_charset.get(idx, '?')
            out.append(ch)
        prev = idx
    return ''.join(out)


def _highlight_errors(pred: str, gt: str) -> str:
    """Return a string where wrong chars are marked with [X]."""
    # Simple character-level diff using edit alignment
    result = []
    # Cheap alignment: pad shorter string
    for p_ch, g_ch in zip(pred.ljust(max(len(pred), len(gt))),
                           gt.ljust(max(len(pred), len(gt)))):
        if p_ch == g_ch:
            result.append(p_ch)
        else:
            result.append('✗')
    return ''.join(result)


def cmd_compare_inspection(args):
    """
    Produce one publication-quality comparison figure per ablation pair.
    Layout per figure:
        ┌─────────────────────────────┬─────────────────────────────┐
        │      Model A (name_a)       │      Model B (name_b)       │
        ├─────────┬─────────┬─────────┼─────────┬─────────┬─────────┤
        │  Input  │ Conv2d  │  Tanh   │  Input  │ Conv2d  │  Tanh   │
        │ [img 1] │ [heat]  │ [heat]  │ [img 1] │ [heat]  │ [heat]  │
        │ [img 2] │ ...     │ ...     │ [img 2] │ ...     │ ...     │
        └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

    Pairs come from --pairs MODEL_A:MODEL_B ... . Without --pairs, the 7 ablation
    pairs of the original experiments are used (baseline LaoCRNN-I48K3C16 against
    wider filters, larger kernels and 64 px input).
    """
    print("=" * 70)
    print("  Cross-Model Inspection: Ablation Comparison Grids")
    print("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # ── Config ────────────────────────────────────────────────────────────────
    base_dir = getattr(args, 'base_data_dir', 'data')
    out_dir  = getattr(args, 'output_dir', 'data/comparison')
    num_samp = getattr(args, 'num_samples', 5)
    seed     = getattr(args, 'seed', 42)
    min_len  = getattr(args, 'min_length', 3)
    os.makedirs(out_dir, exist_ok=True)

    # Default: the 7 ablation pairs of the original experiments, all against the
    # baseline LaoCRNN-I48K3C16 (48px / 3×3 / 16ch). --pairs replaces them.
    ABLATION_PAIRS = [
        # 1. Filter width: same res + kernel, double filters
        ('compare_filter',
         'Compare Filter  (16ch → 32ch)',
         'LaoCRNN-I48K3C16', 'LaoCRNN-I48K3C32',
         'I48 / 3×3 / 16ch  (baseline)', 'I48 / 3×3 / 32ch'),

        # 2. Kernel size: same res + filter, wider kernel
        ('compare_kernel',
         'Compare Kernel  (3×3 → 5×5)',
         'LaoCRNN-I48K3C16', 'LaoCRNN-I48K5C16',
         'I48 / 3×3 / 16ch  (baseline)', 'I48 / 5×5 / 16ch'),

        # 3. Kernel + Filter: wider kernel AND more filters
        ('compare_kernel_filter',
         'Compare Kernel + Filter  (3×3/16ch → 5×5/32ch)',
         'LaoCRNN-I48K3C16', 'LaoCRNN-I48K5C32',
         'I48 / 3×3 / 16ch  (baseline)', 'I48 / 5×5 / 32ch'),

        # 4. High resolution: same kernel + filter, taller input
        ('compare_highres',
         'Compare High Resolution  (48px → 64px)',
         'LaoCRNN-I48K3C16', 'LaoCRNN-I64K3C16',
         'I48 / 3×3 / 16ch  (baseline)', 'I64 / 3×3 / 16ch'),

        # 5. High res + Filter
        ('compare_highres_filter',
         'Compare High Resolution + Filter  (48px/16ch → 64px/32ch)',
         'LaoCRNN-I48K3C16', 'LaoCRNN-I64K3C32',
         'I48 / 3×3 / 16ch  (baseline)', 'I64 / 3×3 / 32ch'),

        # 6. High res + Kernel
        ('compare_highres_kernel',
         'Compare High Resolution + Kernel  (48px/3×3 → 64px/5×5)',
         'LaoCRNN-I48K3C16', 'LaoCRNN-I64K5C16',
         'I48 / 3×3 / 16ch  (baseline)', 'I64 / 5×5 / 16ch'),

        # 7. High res + Kernel + Filter (full scale-up)
        ('compare_highres_kernel_filter',
         'Compare High Resolution + Kernel + Filter  (full scale-up)',
         'LaoCRNN-I48K3C16', 'LaoCRNN-I64K5C32',
         'I48 / 3×3 / 16ch  (baseline)', 'I64 / 5×5 / 32ch'),
    ]
    user_pairs = getattr(args, 'pairs', None)
    if user_pairs:
        ABLATION_PAIRS = []
        for spec in user_pairs:
            if ':' not in spec:
                sys.exit(f"Error: --pairs expects MODEL_A:MODEL_B, got '{spec}'")
            a, b = spec.split(':', 1)
            ABLATION_PAIRS.append((f'compare_{a}_vs_{b}', f'{a}  vs  {b}', a, b, a, b))


    # ── Shared sample selection ────────────────────────────────────────────────
    val_list = getattr(args, 'val_list', '')
    gt_dir   = getattr(args, 'ground_truth_dir', '')
    if not gt_dir:
        p = resolve_paths(args)
        gt_dir   = p.ground_truth_dir
        val_list = p.val_list

    files = []
    if val_list and os.path.exists(val_list):
        with open(val_list) as fh:
            files = [l.strip() for l in fh if l.strip()]
    if not files:
        files = sorted(glob.glob(os.path.join(gt_dir, '**', '*.lstmf'), recursive=True))
    if not files:
        print("  ERROR: No .lstmf files found.")
        return

    dataset_all = TesseractDataset(files)
    valid_idx   = [i for i, (_, txt) in enumerate(dataset_all.samples) if len(txt) >= min_len]
    rng         = random.Random(seed)
    sel         = sorted(rng.sample(valid_idx, min(num_samp, len(valid_idx))))
    shared_files = [files[i] for i in sel]
    print(f"  Seed: {seed}  |  {len(shared_files)} shared samples")

    # ── Helper: extract conv & tanh activations ───────────────────────────────
    def _get_activations(model, x):
        """Return (conv_arr, tanh_arr) as full (C, H, W) float32 numpy arrays
        for the last Conv2d and last Tanh/ReLU layer in the CNN stage."""
        last_conv, last_tanh = None, None
        with torch.no_grad():
            for layer in model:
                x = layer(x)
                if x.dim() != 4:
                    break
                name = layer.__class__.__name__
                arr  = x[0].cpu().numpy()   # (C, H, W)
                if 'Conv' in name:
                    last_conv = arr
                if 'Tanh' in name or 'ReLU' in name:
                    last_tanh = arr
        return last_conv, last_tanh

    def _norm_channel(ch_2d):
        """Normalise a single (H, W) channel map to [0, 1]."""
        lo, hi = ch_2d.min(), ch_2d.max()
        if hi > lo:
            return (ch_2d - lo) / (hi - lo)
        return ch_2d

    # ── Colormaps: plain grayscale for all maps (no artificial colour) ───────
    cmap_conv = 'gray'
    cmap_tanh = 'gray'

    # ── Style constants ───────────────────────────────────────────────────────
    HDR_A      = '#1565C0'   # dark blue for model A header
    HDR_B      = '#7B1FA2'   # purple for model B header




    def _to_display(img_tensor):
        """Return the raw normalised model input as a [0,255] uint8 array.
        No inversion — shows exactly what the model receives:
        high value = ink (bright), low value = background (dark)."""
        arr = img_tensor[0].numpy().astype(np.float32)
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        return np.clip(arr * 255, 0, 255).astype(np.uint8)

    # ── Generate figures: 1 per sample per pair ───────────────────────────────
    # Style: white background, dark header bar, GT image section, comparison grid.

    # 4 columns: Conv2d ch[0], Conv2d ch[-1], Activation ch[0], Activation ch[-1]
    COL_NAMES = [
        'Conv2d\nch 1 (first)',
        'Conv2d\nch N (last)',
        'Activation\nch 1 (first)',
        'Activation\nch N (last)',
    ]

    # Figure width fixed; height computed from actual image pixel ratio
    FIG_W     = 13.0   # inches wide (wider for 4 cols)
    LABEL_W   = 1.8    # model-label column
    N_COLS    = 4      # Conv2d[0], Conv2d[-1], Act[0], Act[-1]
    CELL_W    = (FIG_W - LABEL_W - 0.4) / N_COLS

    # Style colours (image-2 inspired: white doc, dark header)
    WHITE     = '#ffffff'
    HDR_BAR   = '#2b2d3e'    # dark navy top bar
    HDR_TXT   = '#f0f0f0'
    SEC_HDR   = '#1a1d2e'    # section headers
    MUTED     = '#666666'
    DIV_LINE  = '#dddddd'

    saved = []
    for (fid, title, name_a, name_b, label_a, label_b) in ABLATION_PAIRS:
        print(f"\n  Pair: {name_a}  vs  {name_b}")

        model_a, _ = _load_model_from_name(name_a, base_dir, device)
        model_b, _ = _load_model_from_name(name_b, base_dir, device)
        if model_a is None or model_b is None:
            print("  SKIP: could not load one or both models.")
            continue

        pair_dir = os.path.join(out_dir, fid)
        os.makedirs(pair_dir, exist_ok=True)

        dataset = TesseractDataset(shared_files)
        N = len(dataset)

        for s in range(N):
            img_tensor, _, _, _ = dataset[s]
            stem = os.path.splitext(os.path.basename(shared_files[s]))[0]
            x    = img_tensor.unsqueeze(0).to(device)
            inp  = _to_display(img_tensor)   # uint8 (H, W)

            conv_a, tanh_a = _get_activations(model_a, x.clone())
            conv_b, tanh_b = _get_activations(model_b, x.clone())

            # Determine natural GT image aspect ratio for GT section height
            img_h_px, img_w_px = inp.shape
            # We display GT at 70% of figure width, scale height accordingly
            gt_disp_w = FIG_W * 0.55          # inches
            gt_disp_h = gt_disp_w * (img_h_px / img_w_px)
            gt_disp_h = max(gt_disp_h, 0.4)   # minimum height

            # Row heights
            HDR_H     = 0.45   # top header bar
            GT_SEC_H  = 0.30 + gt_disp_h + 0.20   # label + image + padding
            DIV_H     = 0.10   # divider rows
            COL_LBL_H = 0.35   # "Input / Conv2d / Activation" column labels
            # Activation maps are (H/2 × W/N) due to pooling; show proportionally
            act_h_px  = max(img_h_px // 2, 4)
            act_disp_h = CELL_W * (act_h_px / (img_w_px or 1))
            act_disp_h = np.clip(act_disp_h, 0.35, 1.6)
            MODEL_ROW_H = max(act_disp_h + 0.1, 0.5)

            fig_h = HDR_H + GT_SEC_H + DIV_H + COL_LBL_H + 2 * MODEL_ROW_H + DIV_H + 0.15

            # ── Build figure ──────────────────────────────────────────────────
            fig = plt.figure(figsize=(FIG_W, fig_h), facecolor=WHITE)
            fig.patch.set_facecolor(WHITE)

            # GridSpec rows: [hdr_bar | gt_section | divider | col_labels | modelA | modelB | bottom_pad]
            row_ratios = [HDR_H, GT_SEC_H, DIV_H, COL_LBL_H,
                          MODEL_ROW_H, MODEL_ROW_H, 0.15]
            col_ratios = [LABEL_W] + [CELL_W] * N_COLS

            gs = gridspec.GridSpec(
                7, 1 + N_COLS,
                figure=fig,
                height_ratios=row_ratios,
                width_ratios=col_ratios,
                hspace=0.0, wspace=0.04,
                left=0.02, right=0.98, top=1.0, bottom=0.0,
            )

            # ── Row 0: Header bar ─────────────────────────────────────────────
            ax_hdr = fig.add_subplot(gs[0, :])
            ax_hdr.set_facecolor(HDR_BAR)
            for sp in ax_hdr.spines.values(): sp.set_visible(False)
            ax_hdr.set_xticks([]); ax_hdr.set_yticks([])
            ax_hdr.text(0.02, 0.5,
                        f'{title}   |   Sample {s+1}: {stem[:30]}',
                        ha='left', va='center', fontsize=10, fontweight='bold',
                        color=HDR_TXT, transform=ax_hdr.transAxes)

            # ── Row 1: Ground Truth Image section ─────────────────────────────
            # Spans all columns; split into [section label (left) | image (center)]
            ax_gt_sec = fig.add_subplot(gs[1, :])
            ax_gt_sec.axis('off')
            ax_gt_sec.set_facecolor(WHITE)
            # Section label "Ground Truth Image" top-left
            ax_gt_sec.text(0.01, 0.97, 'Ground Truth Image',
                           ha='left', va='top', fontsize=11, fontweight='bold',
                           color=SEC_HDR, transform=ax_gt_sec.transAxes)

            # Embed GT image as inset axes (centred, natural ratio)
            gt_w_frac  = (gt_disp_w / FIG_W)
            gt_h_frac  = (gt_disp_h / GT_SEC_H)
            gt_x0      = (1.0 - gt_w_frac) / 2
            gt_y0      = (1.0 - gt_h_frac) / 2 - 0.05
            ax_gt = ax_gt_sec.inset_axes([gt_x0, gt_y0, gt_w_frac, gt_h_frac])
            ax_gt.imshow(inp, cmap='gray', aspect='equal', interpolation='bilinear')
            ax_gt.set_xticks([]); ax_gt.set_yticks([])
            for sp in ax_gt.spines.values():
                sp.set_edgecolor('#999999')
                sp.set_linewidth(0.8)

            # ── Row 2: Horizontal divider ─────────────────────────────────────
            ax_div1 = fig.add_subplot(gs[2, :])
            ax_div1.axis('off')
            ax_div1.axhline(0.5, color=DIV_LINE, linewidth=1.0, xmin=0.01, xmax=0.99)

            # ── Row 3: Column headers ─────────────────────────────────────────
            ax_clbl0 = fig.add_subplot(gs[3, 0])
            ax_clbl0.axis('off')
            ax_clbl0.text(0.5, 0.5, 'CNN Activation Maps',
                          ha='center', va='center', fontsize=9,
                          color=MUTED, transform=ax_clbl0.transAxes)
            for ci, cname in enumerate(COL_NAMES):
                ax_ch = fig.add_subplot(gs[3, 1 + ci])
                ax_ch.axis('off')
                ax_ch.text(0.5, 0.5, cname,
                           ha='center', va='center', fontsize=9, fontweight='bold',
                           color=SEC_HDR, transform=ax_ch.transAxes)

            # ── Rows 4-5: Model rows (Conv2d + Activation only) ───────────────
            model_rows = [
                (4, name_a, label_a, HDR_A, conv_a, tanh_a),
                (5, name_b, label_b, HDR_B, conv_b, tanh_b),
            ]
            for (grow, mname, mlabel, hdr_color, m_conv, m_tanh) in model_rows:
                # Model label cell
                ax_lbl = fig.add_subplot(gs[grow, 0])
                ax_lbl.set_facecolor(hdr_color)
                ax_lbl.set_xticks([]); ax_lbl.set_yticks([])
                for sp in ax_lbl.spines.values(): sp.set_visible(False)
                # Two-line label: model name + arch description
                ax_lbl.text(0.5, 0.60, mname,
                            ha='center', va='center', fontsize=8, fontweight='bold',
                            color='white', transform=ax_lbl.transAxes)
                ax_lbl.text(0.5, 0.25, mlabel,
                            ha='center', va='center', fontsize=7,
                            color='#ddddff', transform=ax_lbl.transAxes)

                # 4 cells: conv[ch0], conv[ch-1], tanh[ch0], tanh[ch-1]
                cell_specs = []
                for arr, cmap in [(m_conv, cmap_conv), (m_tanh, cmap_tanh)]:
                    if arr is not None:
                        cell_specs.append((_norm_channel(arr[0]),  cmap))   # first ch
                        cell_specs.append((_norm_channel(arr[-1]), cmap))   # last ch
                    else:
                        cell_specs.append((None, cmap))
                        cell_specs.append((None, cmap))

                for ci, (data, cmap) in enumerate(cell_specs):
                    ax = fig.add_subplot(gs[grow, 1 + ci])
                    ax.set_facecolor(WHITE)
                    if data is None:
                        ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                                transform=ax.transAxes, fontsize=9, color='gray')
                    else:
                        ax.imshow(data, cmap=cmap, aspect='equal',
                                  interpolation='bilinear')
                    ax.set_xticks([]); ax.set_yticks([])
                    for sp in ax.spines.values():
                        sp.set_edgecolor(DIV_LINE)
                        sp.set_linewidth(0.7)

            # ── Row 6: bottom padding (empty) ─────────────────────────────────
            ax_bot = fig.add_subplot(gs[6, :])
            ax_bot.axis('off')

            # ── Save ──────────────────────────────────────────────────────────
            fname    = f"sample_{s+1:02d}_{stem[:20]}.png"
            out_path = os.path.join(pair_dir, fname)
            fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=WHITE)
            plt.close(fig)
            print(f"    [{s+1}/{N}] ✓ {out_path}")
            saved.append(out_path)

    print(f"\n  All comparison figures saved to: {out_dir}")
    for p in saved:
        print(f"    {p}")


# ──────────────────────────────────────────────────────────────────────────────
#  WER Distribution Analysis — grouped by GT word count
#  Usage: python torch_train.py wer-distribution --model_name lao
# ──────────────────────────────────────────────────────────────────────────────

def cmd_wer_distribution(args):
    """
    Analyse WER distribution grouped by number of words in the ground truth.

    For every evaluation sample, count words in GT (split by whitespace),
    compute per-sample WER, then aggregate:
      • Histogram of sample counts per GT word-count
      • Mean WER ± std per GT word-count bucket
      • Gaussian fit overlay on the histogram
    Outputs a publication-quality PNG figure + CSV data.
    """
    print("=" * 70)
    print("  WER Distribution Analysis — by GT Word Count")
    print("=" * 70)

    import csv as csv_mod
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from collections import defaultdict

    from tesstrain_core.utils import resolve_paths
    from tesstrain_core.dataset import TesseractDataset
    from tesstrain_core.model import build_model
    from tesstrain_core.metrics import edit_distance, normalize_lao_text

    p = resolve_paths(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    tessdata_dir = getattr(args, 'tessdata_dir', '')

    if tessdata_dir:
        tess_path = os.path.join(tessdata_dir, f"{args.model_name}.traineddata")
        if not os.path.exists(tess_path):
            print(f"  ERROR: Tesseract model not found: {tess_path}")
            return
        print(f"  Loading Tesseract: {tess_path}")
        from tesstrain_core.tesseract_transfer import load_tesseract_pretrained_state_dict

        if hasattr(args, 'net_spec') and args.net_spec:
            net_spec = args.net_spec
        else:
            mname = args.model_name
            h = 64 if "I64" in mname else 48
            if "K5C32" in mname: ct = "Ct5,5,32"
            elif "K5C16" in mname: ct = "Ct5,5,16"
            elif "K3C32" in mname: ct = "Ct3,3,32"
            else: ct = "Ct3,3,16"
            net_spec = f"[1,{h},0,1 {ct} Mp3,3 Lfys{h} Lfx96 Lrx96 Lfx256 O1c###]"

        tess_state, tess_meta = load_tesseract_pretrained_state_dict(tess_path, net_spec)
        num_classes = list(tess_state.values())[-2].shape[0]
        model = build_model(net_spec, num_classes).to(device)
        model.load_state_dict(tess_state)
        model.eval()
    else:
        ckpt_path = getattr(args, 'start_model', None)
        if not ckpt_path:
            ckpt_path = os.path.join(p.output_dir, f"{args.model_name}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  ERROR: Checkpoint not found at {ckpt_path}")
            return
        print(f"  Loading PyTorch: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        net_spec = ckpt.get('net_spec', '[1,36,0,1 Ct3,3,16 Mp3,3 Lfys48 Lfx96 Lrx96 Lfx256 O1c###]')
        num_classes = list(ckpt['model_state_dict'].values())[-2].shape[0]
        model = build_model(net_spec, num_classes).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        print(f"  NET_SPEC: {net_spec}")

    # ── Load eval samples ─────────────────────────────────────────────────────
    custom_dir = getattr(args, 'custom_dir', '')

    if custom_dir:
        # Custom dataset: scan for image + .gt.txt pairs
        print(f"  Custom dir: {custom_dir}")
        test_pairs = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff'):
            for img_path in sorted(glob.glob(os.path.join(custom_dir, ext))):
                base = os.path.splitext(img_path)[0]
                gt_path = base + '.gt.txt'
                if os.path.exists(gt_path):
                    with open(gt_path, encoding='utf-8') as f:
                        gt = f.read().strip()
                    test_pairs.append((img_path, gt))
        if not test_pairs:
            print(f"  ERROR: No image + .gt.txt pairs found in {custom_dir}")
            return
        print(f"  Eval samples: {len(test_pairs)} (custom image+gt pairs)")

        # Need charset + target_h from the loaded model checkpoint
        import re as _re
        if tessdata_dir:
            # Tesseract model — charset from tess_meta
            inv_cs = tess_meta.get('inv_charset', {}) if isinstance(tess_meta, dict) else {}
            target_h = 64 if "I64" in args.model_name else 48
        else:
            inv_cs = {v: k for k, v in ckpt['charset'].items()}
            target_h = 48
            m_h = _re.match(r'\[?\d+,(\d+),', ckpt.get('net_spec', ''))
            if m_h: target_h = int(m_h.group(1))

        from tesstrain_core.recognize import recognize_line

        # ── Per-sample inference (custom images) ──────────────────────────────
        word_count_data = defaultdict(list)
        model.eval()
        with torch.no_grad():
            for i, (img_path, gt_text) in enumerate(test_pairs):
                gt_text = normalize_lao_text(gt_text)
                gt_words = gt_text.split()
                n_words = len(gt_words)
                if n_words == 0:
                    continue

                pil_img = Image.open(img_path).convert('L')
                pred_text = recognize_line(model, pil_img, target_h, device, inv_cs)
                pred_text = normalize_lao_text(pred_text)
                pred_words = pred_text.split()

                wer_val = edit_distance(pred_words, gt_words) / n_words
                word_count_data[n_words].append(wer_val)

                if (i + 1) % 200 == 0:
                    print(f"    Processed {i+1}/{len(test_pairs)}...")

    else:
        # Default: use .lstmf eval split
        files = []
        if os.path.exists(p.val_list):
            with open(p.val_list) as f:
                files = [l.strip() for l in f if l.strip()]
        if not files:
            files = sorted(glob.glob(os.path.join(p.ground_truth_dir, '**', '*.lstmf'), recursive=True))
        if not files:
            print("  ERROR: No .lstmf files found.")
            return

        dataset = TesseractDataset(files)
        inv_charset = dataset.inv_charset if hasattr(dataset, 'inv_charset') else {}
        print(f"  Eval samples: {len(dataset)} (.lstmf files)")

        # ── Per-sample inference (.lstmf) ─────────────────────────────────────
        word_count_data = defaultdict(list)
        model.eval()
        with torch.no_grad():
            for i in range(len(dataset)):
                img_tensor, _, _, _ = dataset[i]
                gt_text = normalize_lao_text(dataset.samples[i][1])
                gt_words = gt_text.split()
                n_words = len(gt_words)
                if n_words == 0:
                    continue

                x = img_tensor.unsqueeze(0).to(device)
                log_probs = model(x)
                pred_text = _ctc_greedy_decode(log_probs, inv_charset)
                pred_text = normalize_lao_text(pred_text)
                pred_words = pred_text.split()

                wer_val = edit_distance(pred_words, gt_words) / n_words
                word_count_data[n_words].append(wer_val)

                if (i + 1) % 200 == 0:
                    print(f"    Processed {i+1}/{len(dataset)}...")

    print(f"  Inference complete. {len(word_count_data)} distinct word-count groups.")

    if not word_count_data:
        print("  ERROR: No valid samples to analyse.")
        return

    # ── Compute stats ─────────────────────────────────────────────────────────
    sorted_wc = sorted(word_count_data.keys())
    stats = []
    for wc in sorted_wc:
        wers = word_count_data[wc]
        arr = np.array(wers)
        stats.append({
            'word_count': wc,
            'n_samples': len(wers),
            'mean_wer': float(arr.mean()),
            'std_wer': float(arr.std()),
            'median_wer': float(np.median(arr)),
            'min_wer': float(arr.min()),
            'max_wer': float(arr.max()),
            'perfect': int(np.sum(arr == 0.0)),
        })

    # ── Console summary ──────────────────────────────────────────────────────
    print(f"\n  {'Words':>6} {'Samples':>8} {'Mean WER':>10} {'Std':>8} "
          f"{'Median':>8} {'Min':>6} {'Max':>6} {'Perfect':>8}")
    print("  " + "─" * 72)
    for s in stats:
        print(f"  {s['word_count']:>6} {s['n_samples']:>8} "
              f"{s['mean_wer']*100:>9.2f}% {s['std_wer']*100:>7.2f}% "
              f"{s['median_wer']*100:>7.2f}% {s['min_wer']*100:>5.1f}% "
              f"{s['max_wer']*100:>5.1f}% {s['perfect']:>7}")

    # ── Export CSV ────────────────────────────────────────────────────────────
    sub_folder = 'wer_distribution_test' if custom_dir else 'wer_distribution_training'
    out_dir = os.path.join(p.output_dir, sub_folder)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'wer_distribution.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv_mod.writer(f)
        writer.writerow(['word_count', 'n_samples', 'mean_wer', 'std_wer',
                          'median_wer', 'min_wer', 'max_wer', 'perfect_count'])
        for s in stats:
            writer.writerow([s['word_count'], s['n_samples'],
                             f"{s['mean_wer']:.6f}", f"{s['std_wer']:.6f}",
                             f"{s['median_wer']:.6f}", f"{s['min_wer']:.6f}",
                             f"{s['max_wer']:.6f}", s['perfect']])
    print(f"\n  ✓ CSV saved to: {csv_path}")

    # ── Lao font ──────────────────────────────────────────────────────────────
    for _fp in [
        os.path.join(os.path.dirname(__file__), '..', 'font', 'phetsarath_ot.ttf'),
        os.path.join(os.path.dirname(__file__), 'font', 'phetsarath_ot.ttf'),
        'font/phetsarath_ot.ttf',
    ]:
        if os.path.exists(_fp):
            fm.fontManager.addfont(_fp)
            break

    # ── Plot: two-panel figure ────────────────────────────────────────────────
    # Panel 1: Sample count histogram with Gaussian fit
    # Panel 2: Mean WER bar chart per word count with error bars
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), facecolor='#ffffff')
    fig.patch.set_facecolor('#ffffff')

    word_counts = [wc for wc in sorted_wc for _ in word_count_data[wc]]
    wc_array = np.array(word_counts, dtype=float)

    # ── Panel 1: Distribution of samples per word count ───────────────────────
    bins = np.arange(min(sorted_wc) - 0.5, max(sorted_wc) + 1.5, 1.0)
    n_hist, bin_edges, patches = ax1.hist(
        wc_array, bins=bins, density=True,
        color='#80DEEA', edgecolor='#4DD0E1', linewidth=0.8, alpha=0.85,
        zorder=2,
    )

    # Gaussian fit
    if len(wc_array) > 2:
        mu = wc_array.mean()
        sigma = wc_array.std()
        if sigma > 0:
            x_fit = np.linspace(min(sorted_wc) - 1, max(sorted_wc) + 1, 200)
            y_fit = (1.0 / (sigma * np.sqrt(2 * np.pi))) * \
                    np.exp(-0.5 * ((x_fit - mu) / sigma) ** 2)
            ax1.plot(x_fit, y_fit, color='#E53935', linewidth=2.0,
                     label=f'Gaussian Fit (μ={mu:.1f}, σ={sigma:.1f})', zorder=3)
            ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)

    ax1.set_xlabel('Word Count in Ground Truth', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax1.set_title(
        f'Distribution of GT Word Count — {args.model_name}',
        fontsize=13, fontweight='bold', color='#1a1d2e', pad=12,
    )
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # ── Panel 2: Mean WER per word count (bar + error bars) ───────────────────
    x_pos = np.arange(len(sorted_wc))
    means = [word_count_data[wc] for wc in sorted_wc]
    mean_vals = np.array([np.mean(m) for m in means]) * 100
    std_vals  = np.array([np.std(m) for m in means]) * 100
    n_samples = np.array([len(m) for m in means])

    # Color gradient: green (low WER) → red (high WER)
    norm_vals = mean_vals / max(mean_vals.max(), 1e-6)
    colors = []
    for nv in norm_vals:
        r = min(1.0, 2 * nv)
        g = min(1.0, 2 * (1 - nv))
        colors.append((r, g, 0.3, 0.85))

    ax2.bar(x_pos, mean_vals, color=colors, edgecolor='#555555',
                   linewidth=0.8, zorder=2)
    ax2.errorbar(x_pos, mean_vals, yerr=std_vals, fmt='none',
                 ecolor='#333333', capsize=4, capthick=1.2, linewidth=1.2,
                 zorder=3)

    # Annotate sample count on each bar
    for xi, mv, ns in zip(x_pos, mean_vals, n_samples):
        ax2.text(xi, mv + std_vals[xi] + 1.5, f'n={ns}',
                 ha='center', va='bottom', fontsize=7.5, color='#555555')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(wc) for wc in sorted_wc], fontsize=9)
    ax2.set_xlabel('Word Count in Ground Truth', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Mean WER (%)', fontsize=11, fontweight='bold')
    ax2.set_title(
        f'Mean WER by GT Word Count — {args.model_name}',
        fontsize=13, fontweight='bold', color='#1a1d2e', pad=12,
    )
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, min(mean_vals.max() + std_vals.max() + 15, 105))

    plt.tight_layout(pad=2.0)
    fig_path = os.path.join(out_dir, 'wer_distribution.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='#ffffff')
    plt.close(fig)
    print(f"  ✓ Figure saved to: {fig_path}")

    # ── Extra: Perfect-rate bar chart ─────────────────────────────────────────
    fig2, ax3 = plt.subplots(1, 1, figsize=(10, 4), facecolor='#ffffff')
    fig2.patch.set_facecolor('#ffffff')

    perfect_rates = []
    for s in stats:
        rate = s['perfect'] / s['n_samples'] * 100 if s['n_samples'] > 0 else 0
        perfect_rates.append(rate)
    perfect_rates = np.array(perfect_rates)

    bar_colors = ['#66BB6A' if pr > 80 else '#FFA726' if pr > 50 else '#EF5350'
                  for pr in perfect_rates]
    ax3.bar(x_pos, perfect_rates, color=bar_colors, edgecolor='#555555',
            linewidth=0.8, zorder=2)
    for xi, pr, ns in zip(x_pos, perfect_rates, n_samples):
        ax3.text(xi, pr + 1.5, f'{pr:.0f}%', ha='center', va='bottom',
                 fontsize=8, fontweight='bold', color='#333333')

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([str(wc) for wc in sorted_wc], fontsize=9)
    ax3.set_xlabel('Word Count in Ground Truth', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Perfect Recognition Rate (%)', fontsize=11, fontweight='bold')
    ax3.set_title(
        f'Perfect Recognition Rate by GT Word Count — {args.model_name}',
        fontsize=13, fontweight='bold', color='#1a1d2e', pad=12,
    )
    ax3.set_ylim(0, 105)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_axisbelow(True)

    plt.tight_layout()
    fig2_path = os.path.join(out_dir, 'wer_perfect_rate.png')
    fig2.savefig(fig2_path, dpi=200, bbox_inches='tight', facecolor='#ffffff')
    plt.close(fig2)
    print(f"  ✓ Perfect-rate figure saved to: {fig2_path}")

    print(f"\n  Analysis complete. Results in {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
#  Confusion Matrix — per-character error analysis grouped by component
#  Usage: python torch_train.py confusion-matrix --model_name LaoCRNN-I48K3C16
# ──────────────────────────────────────────────────────────────────────────────

def cmd_confusion_matrix(args):
    """
    Per-character error analysis grouped by Lao component category.

    For each GT character, track: total occurrences, correct, substitutions
    (and what it was confused with), deletions, insertions.
    Group results by component (tone, consonant, upper_vowel, etc.).

    Outputs:
      • confusion_matrix_char.csv   — per-character error breakdown
      • confusion_matrix_top.csv    — top substitution pairs
      • confusion_matrix_coer.png   — CoER bar chart by component
      • confusion_matrix_char.png   — per-character error rate chart
      • confusion_matrix_heatmap.png— confusion heatmap for top-error chars
    """
    print("=" * 70)
    print("  Confusion Matrix — Per-Character Error Analysis")
    print("=" * 70)

    import csv as csv_mod
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from collections import defaultdict, Counter

    from tesstrain_core.utils import resolve_paths
    from tesstrain_core.dataset import TesseractDataset
    from tesstrain_core.model import build_model
    from tesstrain_core.metrics import (
        normalize_lao_text, align_strings,
        get_component_type, LAO_COMPONENT_MAP,
    )

    p = resolve_paths(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # ── Load model (same pattern as wer_distribution) ─────────────────────────
    tessdata_dir = getattr(args, 'tessdata_dir', '')

    if tessdata_dir:
        tess_path = os.path.join(tessdata_dir, f"{args.model_name}.traineddata")
        if not os.path.exists(tess_path):
            print(f"  ERROR: Tesseract model not found: {tess_path}")
            return
        print(f"  Loading Tesseract: {tess_path}")
        from tesstrain_core.tesseract_transfer import load_tesseract_pretrained_state_dict
        if hasattr(args, 'net_spec') and args.net_spec:
            net_spec = args.net_spec
        else:
            mname = args.model_name
            h = 64 if "I64" in mname else 48
            if "K5C32" in mname: ct = "Ct5,5,32"
            elif "K5C16" in mname: ct = "Ct5,5,16"
            elif "K3C32" in mname: ct = "Ct3,3,32"
            else: ct = "Ct3,3,16"
            net_spec = f"[1,{h},0,1 {ct} Mp3,3 Lfys{h} Lfx96 Lrx96 Lfx256 O1c###]"
        tess_state, tess_meta = load_tesseract_pretrained_state_dict(tess_path, net_spec)
        num_classes = list(tess_state.values())[-2].shape[0]
        model = build_model(net_spec, num_classes).to(device)
        model.load_state_dict(tess_state)
        model.eval()
    else:
        ckpt_path = getattr(args, 'start_model', None)
        if not ckpt_path:
            ckpt_path = os.path.join(p.output_dir, f"{args.model_name}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  ERROR: Checkpoint not found at {ckpt_path}")
            return
        print(f"  Loading PyTorch: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        net_spec = ckpt.get('net_spec', '[1,36,0,1 Ct3,3,16 Mp3,3 Lfys48 Lfx96 Lrx96 Lfx256 O1c###]')
        num_classes = list(ckpt['model_state_dict'].values())[-2].shape[0]
        model = build_model(net_spec, num_classes).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        print(f"  NET_SPEC: {net_spec}")

    # ── Collect all (gt, pred) pairs ──────────────────────────────────────────
    custom_dir = getattr(args, 'custom_dir', '')
    all_pairs = []  # list of (gt_text, pred_text)

    if custom_dir:
        print(f"  Custom dir: {custom_dir}")
        test_pairs = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff'):
            for img_path in sorted(glob.glob(os.path.join(custom_dir, ext))):
                gt_path = os.path.splitext(img_path)[0] + '.gt.txt'
                if os.path.exists(gt_path):
                    with open(gt_path, encoding='utf-8') as f:
                        test_pairs.append((img_path, f.read().strip()))
        if not test_pairs:
            print(f"  ERROR: No image + .gt.txt pairs found in {custom_dir}")
            return
        print(f"  Eval samples: {len(test_pairs)} (custom)")

        import re as _re
        if tessdata_dir:
            inv_cs = tess_meta.get('inv_charset', {}) if isinstance(tess_meta, dict) else {}
            target_h = 64 if "I64" in args.model_name else 48
        else:
            inv_cs = {v: k for k, v in ckpt['charset'].items()}
            target_h = 48
            m_h = _re.match(r'\[?\d+,(\d+),', ckpt.get('net_spec', ''))
            if m_h: target_h = int(m_h.group(1))

        from tesstrain_core.recognize import recognize_line
        model.eval()
        with torch.no_grad():
            for i, (img_path, gt) in enumerate(test_pairs):
                pil_img = Image.open(img_path).convert('L')
                pred = recognize_line(model, pil_img, target_h, device, inv_cs)
                all_pairs.append((normalize_lao_text(gt), normalize_lao_text(pred)))
                if (i + 1) % 200 == 0:
                    print(f"    Processed {i+1}/{len(test_pairs)}...")
    else:
        files = []
        if os.path.exists(p.val_list):
            with open(p.val_list) as f:
                files = [l.strip() for l in f if l.strip()]
        if not files:
            files = sorted(glob.glob(os.path.join(p.ground_truth_dir, '**', '*.lstmf'), recursive=True))
        if not files:
            print("  ERROR: No .lstmf files found.")
            return
        dataset = TesseractDataset(files)
        inv_charset = dataset.inv_charset if hasattr(dataset, 'inv_charset') else {}
        print(f"  Eval samples: {len(dataset)} (.lstmf)")
        model.eval()
        with torch.no_grad():
            for i in range(len(dataset)):
                img_tensor, _, _, _ = dataset[i]
                gt = normalize_lao_text(dataset.samples[i][1])
                x = img_tensor.unsqueeze(0).to(device)
                log_probs = model(x)
                pred = _ctc_greedy_decode(log_probs, inv_charset)
                all_pairs.append((gt, normalize_lao_text(pred)))
                if (i + 1) % 200 == 0:
                    print(f"    Processed {i+1}/{len(dataset)}...")

    print(f"  Inference complete. {len(all_pairs)} samples.")

    # ── Align and collect per-character stats ─────────────────────────────────
    # char_stats[ch] = {total, correct, sub, del, ins, confused_with: Counter}
    char_stats = defaultdict(lambda: {
        'total': 0, 'correct': 0, 'sub': 0, 'del': 0, 'ins': 0,
        'confused_with': Counter(),
    })
    sub_pairs = Counter()  # (gt_char, pred_char) → count

    for gt_text, pred_text in all_pairs:
        edits = align_strings(pred_text, gt_text)
        for op, p_ch, t_ch in edits:
            if op == 'mat':
                char_stats[t_ch]['total'] += 1
                char_stats[t_ch]['correct'] += 1
            elif op == 'sub':
                char_stats[t_ch]['total'] += 1
                char_stats[t_ch]['sub'] += 1
                char_stats[t_ch]['confused_with'][p_ch] += 1
                sub_pairs[(t_ch, p_ch)] += 1
            elif op == 'del':
                char_stats[t_ch]['total'] += 1
                char_stats[t_ch]['del'] += 1
            elif op == 'ins':
                char_stats[p_ch]['ins'] += 1

    # ── Component-level aggregation ───────────────────────────────────────────
    comp_stats = defaultdict(lambda: {'total': 0, 'errors': 0, 'chars': []})
    for ch, st in char_stats.items():
        cat = get_component_type(ch)
        comp_stats[cat]['total'] += st['total']
        comp_stats[cat]['errors'] += st['sub'] + st['del']

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n  Component Error Rate (CoER):")
    print(f"  {'Component':<15} {'Total':>7} {'Errors':>7} {'Rate':>8}")
    print("  " + "─" * 42)
    for cat in sorted(comp_stats, key=lambda c: comp_stats[c]['errors'] / max(comp_stats[c]['total'], 1), reverse=True):
        cs = comp_stats[cat]
        rate = cs['errors'] / max(cs['total'], 1) * 100
        print(f"  {cat:<15} {cs['total']:>7} {cs['errors']:>7} {rate:>7.2f}%")

    print("\n  Top-20 confused characters:")
    print(f"  {'Char':<6} {'Category':<15} {'Total':>6} {'Err':>5} {'Rate':>7} {'Top confusion':>20}")
    print("  " + "─" * 65)
    sorted_chars = sorted(
        [(ch, st) for ch, st in char_stats.items() if st['total'] >= 3],
        key=lambda x: (x[1]['sub'] + x[1]['del']) / max(x[1]['total'], 1),
        reverse=True
    )
    for ch, st in sorted_chars[:20]:
        errs = st['sub'] + st['del']
        rate = errs / max(st['total'], 1) * 100
        top_conf = st['confused_with'].most_common(1)
        conf_str = f"→ '{top_conf[0][0]}' ×{top_conf[0][1]}" if top_conf else ""
        cat = get_component_type(ch)
        print(f"  '{ch}'    {cat:<15} {st['total']:>6} {errs:>5} {rate:>6.1f}% {conf_str:>20}")

    # ── Output directory ──────────────────────────────────────────────────────
    sub_folder = 'confusion_matrix_test' if custom_dir else 'confusion_matrix_training'
    out_dir = os.path.join(p.output_dir, sub_folder)
    os.makedirs(out_dir, exist_ok=True)

    # ── CSV: per-character breakdown ──────────────────────────────────────────
    csv_char = os.path.join(out_dir, 'confusion_matrix_char.csv')
    with open(csv_char, 'w', newline='', encoding='utf-8') as f:
        w = csv_mod.writer(f)
        w.writerow(['char', 'category', 'total', 'correct', 'substitutions',
                     'deletions', 'insertions', 'error_rate', 'top_confusion'])
        for ch in sorted(char_stats, key=lambda c: char_stats[c]['total'], reverse=True):
            st = char_stats[ch]
            errs = st['sub'] + st['del']
            rate = errs / max(st['total'], 1)
            top = st['confused_with'].most_common(1)
            top_str = f"{top[0][0]}({top[0][1]})" if top else ""
            w.writerow([ch, get_component_type(ch), st['total'], st['correct'],
                        st['sub'], st['del'], st['ins'], f"{rate:.4f}", top_str])
    print(f"\n  ✓ Per-character CSV → {csv_char}")

    # ── CSV: top substitution pairs ───────────────────────────────────────────
    csv_top = os.path.join(out_dir, 'confusion_matrix_top.csv')
    with open(csv_top, 'w', newline='', encoding='utf-8') as f:
        w = csv_mod.writer(f)
        w.writerow(['gt_char', 'gt_category', 'pred_char', 'pred_category', 'count'])
        for (g, p_c), cnt in sub_pairs.most_common(100):
            w.writerow([g, get_component_type(g), p_c, get_component_type(p_c), cnt])
    print(f"  ✓ Top confusions CSV → {csv_top}")

    # ── Lao font ──────────────────────────────────────────────────────────────
    lao_font = None
    for _fp in [
        os.path.join(os.path.dirname(__file__), '..', 'font', 'phetsarath_ot.ttf'),
        os.path.join(os.path.dirname(__file__), 'font', 'phetsarath_ot.ttf'),
        'font/phetsarath_ot.ttf',
    ]:
        if os.path.exists(_fp):
            fm.fontManager.addfont(_fp)
            lao_font = fm.FontProperties(fname=_fp, size=9)
            break

    # ── Plot 1: CoER bar chart by component ───────────────────────────────────
    cats_sorted = sorted(comp_stats, key=lambda c: comp_stats[c]['errors'] / max(comp_stats[c]['total'], 1))
    cat_labels = [c.replace('_', ' ').title() for c in cats_sorted]
    cat_rates = [comp_stats[c]['errors'] / max(comp_stats[c]['total'], 1) * 100 for c in cats_sorted]
    cat_totals = [comp_stats[c]['total'] for c in cats_sorted]

    fig, ax = plt.subplots(figsize=(10, max(4, len(cats_sorted) * 0.5)), facecolor='#ffffff')
    y_pos = np.arange(len(cats_sorted))
    colors_coer = ['#EF5350' if r > 20 else '#FFA726' if r > 10 else '#66BB6A' for r in cat_rates]
    ax.barh(y_pos, cat_rates, color=colors_coer, edgecolor='#555', linewidth=0.6)
    for yi, (rate, total) in enumerate(zip(cat_rates, cat_totals)):
        ax.text(rate + 0.5, yi, f'{rate:.1f}% (n={total})', va='center', fontsize=8, color='#333')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat_labels, fontsize=10)
    ax.set_xlabel('Error Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'Component Error Rate (CoER) — {args.model_name}',
                 fontsize=13, fontweight='bold', color='#1a1d2e', pad=12)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    coer_path = os.path.join(out_dir, 'confusion_matrix_coer.png')
    fig.savefig(coer_path, dpi=200, bbox_inches='tight', facecolor='#ffffff')
    plt.close(fig)
    print(f"  ✓ CoER chart → {coer_path}")

    # ── Plot 2: Per-character error rate by component group ───────────────────
    # Show characters with total >= 5, grouped by component
    COMP_ORDER = ['tone', 'upper_vowel', 'lower_vowel', 'before_vowel',
                  'rear_vowel', 'consonant', 'special', 'lao_number',
                  'digit', 'latin', 'space', 'symbol', 'other']
    COMP_COLORS = {
        'tone': '#E53935', 'upper_vowel': '#8E24AA', 'lower_vowel': '#5E35B1',
        'before_vowel': '#1E88E5', 'rear_vowel': '#00ACC1', 'consonant': '#43A047',
        'special': '#F4511E', 'lao_number': '#FB8C00', 'digit': '#6D4C41',
        'latin': '#3949AB', 'space': '#BDBDBD',
        'symbol': '#757575', 'other': '#9E9E9E',
    }

    qualified = [(ch, st) for ch, st in char_stats.items() if st['total'] >= 5]
    qualified.sort(key=lambda x: (COMP_ORDER.index(get_component_type(x[0]))
                                  if get_component_type(x[0]) in COMP_ORDER else 99,
                                  -(x[1]['sub'] + x[1]['del']) / max(x[1]['total'], 1)))

    if qualified:
        n_chars = min(len(qualified), 60)
        sel = qualified[:n_chars]
        fig, ax = plt.subplots(figsize=(max(12, n_chars * 0.35), 6), facecolor='#ffffff')
        x_pos = np.arange(n_chars)
        bar_colors = [COMP_COLORS.get(get_component_type(ch), '#9E9E9E') for ch, _ in sel]
        rates = [(st['sub'] + st['del']) / max(st['total'], 1) * 100 for _, st in sel]
        ax.bar(x_pos, rates, color=bar_colors, edgecolor='#555', linewidth=0.5)

        labels = [ch for ch, _ in sel]
        ax.set_xticks(x_pos)
        if lao_font:
            ax.set_xticklabels(labels, fontproperties=lao_font, rotation=0, fontsize=8)
        else:
            ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.set_ylabel('Error Rate (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Per-Character Error Rate — {args.model_name}',
                     fontsize=13, fontweight='bold', color='#1a1d2e', pad=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Legend for component colors
        from matplotlib.patches import Patch
        seen = []
        legend_handles = []
        for ch, _ in sel:
            cat = get_component_type(ch)
            if cat not in seen:
                seen.append(cat)
                legend_handles.append(Patch(facecolor=COMP_COLORS.get(cat, '#9E9E9E'),
                                            label=cat.replace('_', ' ').title()))
        ax.legend(handles=legend_handles, fontsize=8, loc='upper right',
                  ncol=2, framealpha=0.9)

        plt.tight_layout()
        char_path = os.path.join(out_dir, 'confusion_matrix_char.png')
        fig.savefig(char_path, dpi=200, bbox_inches='tight', facecolor='#ffffff')
        plt.close(fig)
        print(f"  ✓ Per-char chart → {char_path}")

    # ── Plot 3: Separate confusion heatmap per component category ──────────────
    COMP_LABELS = {
        'tone': 'Tone Mark', 'consonant': 'Consonant',
        'upper_vowel': 'Upper Vowel', 'lower_vowel': 'Lower Vowel',
        'before_vowel': 'Before Vowel', 'rear_vowel': 'Rear Vowel',
        'special': 'Special', 'lao_number': 'Lao Number',
        'digit': 'Digit', 'latin': 'Latin', 'space': 'Space',
        'symbol': 'Symbol', 'other': 'Other',
    }

    for cat, cat_chars in LAO_COMPONENT_MAP.items():
        # Collect GT chars of this category that have substitutions
        cat_gt = [ch for ch in cat_chars if any(
            g == ch for (g, _), _ in sub_pairs.items()
        )]
        if not cat_gt:
            continue

        # Collect all pred chars these GT chars were confused with
        cat_pred_counter = Counter()
        for g_ch in cat_gt:
            for (g, pc), cnt in sub_pairs.items():
                if g == g_ch:
                    cat_pred_counter[pc] += cnt
        cat_pred = [pc for pc, _ in cat_pred_counter.most_common(30)]
        if not cat_pred:
            continue

        # Build matrix
        matrix = np.zeros((len(cat_gt), len(cat_pred)), dtype=float)
        for ri, g_ch in enumerate(cat_gt):
            for (g, pc), cnt in sub_pairs.items():
                if g == g_ch and pc in cat_pred:
                    matrix[ri][cat_pred.index(pc)] = cnt

        # Skip if all zeros
        if matrix.max() == 0:
            continue

        # Remove empty rows/cols
        row_mask = matrix.sum(axis=1) > 0
        col_mask = matrix.sum(axis=0) > 0
        if not row_mask.any() or not col_mask.any():
            continue
        matrix = matrix[row_mask][:, col_mask]
        cat_gt_filtered = [ch for ch, m in zip(cat_gt, row_mask) if m]
        cat_pred_filtered = [ch for ch, m in zip(cat_pred, col_mask) if m]

        n_rows = len(cat_gt_filtered)
        n_cols = len(cat_pred_filtered)
        label = COMP_LABELS.get(cat, cat.replace('_', ' ').title())

        fig, ax = plt.subplots(
            figsize=(max(5, n_cols * 0.55 + 2), max(3, n_rows * 0.55 + 1.5)),
            facecolor='#ffffff',
        )
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        plt.colorbar(im, ax=ax, label='Count', shrink=0.8)

        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))
        if lao_font:
            ax.set_xticklabels(cat_pred_filtered, fontproperties=lao_font,
                               rotation=45, ha='right')
            ax.set_yticklabels(cat_gt_filtered, fontproperties=lao_font)
        else:
            ax.set_xticklabels(cat_pred_filtered, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(cat_gt_filtered, fontsize=9)

        ax.set_xlabel('Predicted Character', fontsize=11, fontweight='bold')
        ax.set_ylabel('Ground Truth Character', fontsize=11, fontweight='bold')
        ax.set_title(f'Confusion Heatmap: {label} — {args.model_name}',
                     fontsize=13, fontweight='bold', color='#1a1d2e', pad=12)

        # Annotate cells
        for r in range(n_rows):
            for c in range(n_cols):
                v = int(matrix[r, c])
                if v > 0:
                    txt_color = 'white' if v > matrix.max() * 0.6 else '#333'
                    ax.text(c, r, str(v), ha='center', va='center',
                            fontsize=8, color=txt_color, fontweight='bold')

        plt.tight_layout()
        fname = f'confusion_heatmap_{cat}.png'
        hpath = os.path.join(out_dir, fname)
        fig.savefig(hpath, dpi=200, bbox_inches='tight', facecolor='#ffffff')
        plt.close(fig)
        print(f"  ✓ Heatmap [{label}] → {hpath}")

    print(f"\n  Analysis complete. Results in {out_dir}")



