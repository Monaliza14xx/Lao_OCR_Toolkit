import os
import re
import sys
import glob
import time
import json
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .utils import resolve_paths
from .dataset import TesseractDataset, collate_fn
from .model import build_model
from .metrics import compute_metrics, compute_mean_rms, TrainingLogger
from .command_utils import cmd_plot
from .tesseract_transfer import (
    load_tesseract_pretrained_state_dict,
    apply_compatible_pretrained_weights,
)

def _save(model, optimizer, it, ds, args, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({'iteration': it, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'charset': ds.charset, 'num_classes': ds.num_classes,
                'net_spec': args.net_spec, 'model_name': args.model_name}, path)
    print(f"  ✓ Saved: {path}")

def cmd_training(args):
    """Start GPU training with PyTorch."""
    p = resolve_paths(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if getattr(args, 'force_GPU', False) and device.type == 'cpu':
        print("\nError: --force_GPU was specified, but no GPU (CUDA) is available.")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"  Tesseract PyTorch GPU Trainer")
    print(f"{'=' * 70}")
    print(f"  Device:          {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))
    if device.type == 'cuda':
        print(f"  GPU Memory:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Model:           {args.model_name}")
    print(f"  NET_SPEC:        {args.net_spec}")
    print(f"  Batch Size:      {args.batch_size}")
    print(f"  Learning Rate:   {args.learning_rate}")
    print(f"  Max Iterations:  {args.max_iterations}")
    print(f"  Target CER:      {args.target_error_rate}")
    print(f"{'=' * 70}\n")

    if os.path.exists(p.train_list):
        with open(p.train_list) as f:
            train_files = [l.strip() for l in f if l.strip()]
        eval_files = []
        if os.path.exists(p.val_list):
            with open(p.val_list) as f:
                eval_files = [l.strip() for l in f if l.strip()]
    else:
        all_files = sorted(glob.glob(os.path.join(p.ground_truth_dir, '**', '*.lstmf'), recursive=True))
        if not all_files:
            print(f"Error: No .lstmf files. Run 'lists' first or provide --ground_truth_dir")
            sys.exit(1)
        random.seed(args.random_seed)
        random.shuffle(all_files)
        split = int(len(all_files) * args.ratio_train)
        train_files, eval_files = all_files[:split], all_files[split:]
        print(f"  Auto-split: {len(train_files)} train, {len(eval_files)} eval")

    target_h = 48
    m = re.match(r'\[?\d+,(\d+),', args.net_spec)
    if m: target_h = int(m.group(1))

    charset = None
    charset_path = os.path.join(p.output_dir, 'charset.json')
    if os.path.exists(charset_path):
        with open(charset_path) as f:
            charset = json.load(f)
        print(f"  Loaded charset from {charset_path}")

    train_ds = TesseractDataset(train_files, target_height=target_h, charset=charset)
    eval_loader = None
    if eval_files:
        eval_ds = TesseractDataset(eval_files, target_height=target_h, charset=train_ds.charset)
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, collate_fn=collate_fn,
                                  pin_memory=device.type == 'cuda')

    with open(charset_path, 'w') as f:
        json.dump(train_ds.charset, f, ensure_ascii=False, indent=2)

    max_iter = args.max_iterations
    if getattr(args, 'epochs', 0) > 0:
        max_iter = int(math.ceil(args.epochs * len(train_ds) / args.batch_size))
        print(f"  Epochs={args.epochs} → max_iterations={max_iter}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate_fn,
                               pin_memory=device.type == 'cuda', drop_last=True)

    spec = args.net_spec
    start_iter = 0
    ckpt = None
    tess_state = None
    tess_meta = {}

    if getattr(args, 'start_model', '') and getattr(args, 'start_tess_model', ''):
        print("  Warning: both --start_model and --start_tess_model were provided.")
        print("           Using --start_model and ignoring --start_tess_model.")

    if getattr(args, 'start_model', '') and os.path.exists(args.start_model):
        print(f"  Loading checkpoint: {args.start_model}")
        ckpt = torch.load(args.start_model, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            if 'net_spec' in ckpt:
                spec = ckpt['net_spec']   # use the architecture from the checkpoint
                args.net_spec = spec      # keep args in sync so _save() persists the right spec
                print(f"  Using net_spec from checkpoint: {spec}")
            start_iter = ckpt.get('iteration', 0)
    elif getattr(args, 'start_tess_model', ''):
        if not os.path.exists(args.start_tess_model):
            print(f"Error: Tesseract model not found: {args.start_tess_model}")
            sys.exit(1)
        print(f"  Loading Tesseract model: {args.start_tess_model}")
        try:
            tess_state, tess_meta = load_tesseract_pretrained_state_dict(args.start_tess_model, spec)
        except Exception as e:
            print(f"Error: Failed to parse Tesseract model: {e}")
            print("  Tip: ensure --net_spec matches the source Tesseract network architecture.")
            sys.exit(1)

        if tess_meta.get('exported_net_spec'):
            print(f"  Source net_spec metadata: {tess_meta['exported_net_spec']}")
        if tess_meta.get('iteration') is not None:
            print(f"  Source iteration: {tess_meta.get('iteration')}")

    spec = spec.replace('###', str(train_ds.num_classes))
    print(f"\n  Model: {spec}")
    model = build_model(spec, train_ds.num_classes).to(device)
    total_p = sum(p_.numel() for p_ in model.parameters())
    print(f"  Parameters: {total_p:,}\n\n{model}\n")

    if ckpt is not None:
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        try:
            model.load_state_dict(state, strict=True)
            print(f"  Loaded weights (resuming from iter {start_iter})")
        except RuntimeError as e:
            print(f"  Strict checkpoint load failed: {e}")
            print("  Falling back to compatible-layer transfer load...")
            loaded, skipped = apply_compatible_pretrained_weights(model, state)
            start_iter = 0
            print(f"  Loaded {len(loaded)} tensors, skipped {len(skipped)}")

    if tess_state is not None:
        loaded, skipped = apply_compatible_pretrained_weights(model, tess_state)
        print(f"  Tesseract transfer loaded {len(loaded)} tensors, skipped {len(skipped)}")
        if skipped:
            # Show only a small sample to keep logs readable.
            preview = ', '.join(k for k, _ in skipped[:5])
            if len(skipped) > 5:
                preview += ', ...'
            print(f"  Skipped keys: {preview}")
        if not loaded:
            print("Error: No compatible tensors were loaded from the Tesseract model.")
            print("  Verify that --net_spec matches the source model architecture.")
            sys.exit(1)


    if getattr(args, 'optimizer', 'Adam') == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=getattr(args, 'momentum', 0.9),
            weight_decay=getattr(args, 'weight_decay', 0.0)
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=getattr(args, 'weight_decay', 0.0)
        )
        
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    logger = TrainingLogger(p.log_file)

    # --- TensorBoard ---
    tb_writer = None
    if getattr(args, 'tensorboard', True):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = getattr(args, 'tensorboard_dir', '') or os.path.join(p.output_dir, 'tensorboard')
            tb_writer = SummaryWriter(log_dir=tb_dir)
            tb_writer.add_text('config', json.dumps(vars(args), indent=2), 0)
            print(f"  TensorBoard: {tb_dir}")
            print(f"  Run: tensorboard --logdir {tb_dir}")
        except ImportError:
            print("  TensorBoard not available (pip install tensorboard to enable)")

    with open(os.path.join(p.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"{'=' * 70}")
    print(f"  Training started...")
    print(f"{'=' * 70}\n")

    iteration, epoch = start_iter, 0
    t0 = time.time()
    best_cer = float('inf')
    run_loss, run_n = 0.0, 0

    model.train()
    while iteration < max_iter:
        epoch += 1
        for imgs, labels, ilens, tlens in train_loader:
            if iteration >= max_iter: break
            imgs, labels = imgs.to(device), labels.to(device)

            out = model(imgs)
            if out.dim() == 4:
                B, C, H, W = out.shape
                out = out.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)

            lp = F.log_softmax(out, dim=2).permute(1, 0, 2)
            T = lp.shape[0]
            ail = torch.full((lp.shape[1],), T, dtype=torch.long)

            loss = ctc_loss_fn(lp, labels, ail, tlens)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            run_loss += loss.item()
            run_n += 1
            iteration += 1

            if iteration % args.debug_interval == 0:
                elapsed = time.time() - t0
                avg_loss = run_loss / run_n
                sps = (iteration - start_iter) * args.batch_size / elapsed

                train_cer, train_wer, train_acc, _, train_samples = compute_metrics(
                    model, [(imgs.cpu(), labels.cpu(), ilens, tlens)],
                    train_ds, device, max_batches=1, collect_samples=3)

                eval_cer, eval_wer, eval_acc, eval_loss_val = None, None, None, None
                if eval_loader and iteration % (args.debug_interval * 5) == 0:
                    eval_cer, eval_wer, eval_acc, _, _ = compute_metrics(
                        model, eval_loader, train_ds, device)
                    # Compute eval loss
                    model.eval()
                    eval_losses = []
                    with torch.no_grad():
                        for eb, (e_imgs, e_labels, e_ilens, e_tlens) in enumerate(eval_loader):
                            if eb >= 50: break
                            e_imgs, e_labels = e_imgs.to(device), e_labels.to(device)
                            e_out = model(e_imgs)
                            if e_out.dim() == 4:
                                eB, eC, eH, eW = e_out.shape
                                e_out = e_out.permute(0, 3, 1, 2).contiguous().view(eB, eW, eC * eH)
                            e_lp = F.log_softmax(e_out, dim=2).permute(1, 0, 2)
                            eT = e_lp.shape[0]
                            e_ail = torch.full((e_lp.shape[1],), eT, dtype=torch.long)
                            e_loss = ctc_loss_fn(e_lp, e_labels, e_ail, e_tlens)
                            eval_losses.append(e_loss.item())
                    model.train()
                    eval_loss_val = sum(eval_losses) / max(len(eval_losses), 1)

                mean_rms = compute_mean_rms(model)
                gpu_mem = torch.cuda.memory_allocated() / 1e9 if device.type == 'cuda' else 0

                print(f"\n{'─' * 70}")
                print(f"  Iteration {iteration:6d}/{max_iter}  │  Epoch {epoch}  │  {sps:.0f} samples/sec")
                print(f"  Loss={loss.item():.4f}  avg={avg_loss:.4f}  │  "
                      f"CER={train_cer:.4f}  WER={train_wer:.4f}  Acc={train_acc:.2%}  │  "
                      f"Mean RMS={mean_rms:.6f}")
                if eval_cer is not None:
                    print(f"  Eval — Loss={eval_loss_val:.4f}  CER={eval_cer:.4f}  WER={eval_wer:.4f}  "
                          f"Acc={eval_acc:.2%}")
                if gpu_mem:
                    gpu_peak = torch.cuda.max_memory_allocated() / 1e9
                    print(f"  GPU={gpu_mem:.2f}GB (peak={gpu_peak:.2f}GB)")

                if train_samples:
                    print(f"  ┌─── Sample Predictions ───")
                    for si, (gt_s, pred_s) in enumerate(train_samples):
                        gt_display = gt_s[:80] + ('...' if len(gt_s) > 80 else '')
                        pred_display = pred_s[:80] + ('...' if len(pred_s) > 80 else '')
                        print(f"  │ [{si+1}] GT:   {gt_display}")
                        print(f"  │     Pred: {pred_display}")
                    print(f"  └────────────────────────")
                print(f"{'─' * 70}")

                logger.log(iteration=iteration, epoch=epoch,
                           train_loss=round(avg_loss, 6),
                           eval_loss=round(eval_loss_val, 6) if eval_loss_val is not None else '',
                           train_cer=round(train_cer, 6),
                           train_wer=round(train_wer, 6),
                           train_acc=round(train_acc, 6),
                           eval_cer=round(eval_cer, 6) if eval_cer is not None else '',
                           eval_wer=round(eval_wer, 6) if eval_wer is not None else '',
                           eval_acc=round(eval_acc, 6) if eval_acc is not None else '',
                           mean_rms=round(mean_rms, 6),
                           learning_rate=args.learning_rate,
                           gpu_mem_gb=round(gpu_mem, 3), samples_per_sec=round(sps, 1))

                # --- TensorBoard scalars ---
                if tb_writer:
                    # Group: Train
                    tb_writer.add_scalar('Train/Loss',     avg_loss,   iteration)
                    tb_writer.add_scalar('Train/CER',      train_cer,  iteration)
                    tb_writer.add_scalar('Train/WER',      train_wer,  iteration)
                    tb_writer.add_scalar('Train/Accuracy', train_acc,  iteration)
                    # Group: Eval  (only logged when eval runs)
                    if eval_loss_val is not None:
                        tb_writer.add_scalar('Eval/Loss',     eval_loss_val, iteration)
                    if eval_cer is not None:
                        tb_writer.add_scalar('Eval/CER',      eval_cer,  iteration)
                        tb_writer.add_scalar('Eval/WER',      eval_wer,  iteration)
                    if eval_acc is not None:
                        tb_writer.add_scalar('Eval/Accuracy', eval_acc,  iteration)
                    # Group: Model
                    tb_writer.add_scalar('Model/Mean_RMS',         mean_rms,           iteration)
                    # Group: System
                    tb_writer.add_scalar('System/Learning_Rate',   args.learning_rate, iteration)
                    tb_writer.add_scalar('System/GPU_Memory_GB',   gpu_mem,            iteration)
                    tb_writer.add_scalar('System/Samples_per_Sec', sps,                iteration)
                    tb_writer.flush()

                run_loss, run_n = 0.0, 0

                if eval_loader:
                    # If validation is enabled, only update best_cer and save _best.pt when validation is run
                    if eval_cer is not None:
                        if eval_cer < best_cer:
                            best_cer = eval_cer
                            _save(model, optimizer, iteration, train_ds, args, p.model_output + '_best.pt')
                else:
                    # If validation is disabled, use train_cer
                    if train_cer < best_cer:
                        best_cer = train_cer
                        _save(model, optimizer, iteration, train_ds, args, p.model_output + '_best.pt')

                cer_check = eval_cer if eval_cer is not None else train_cer
                if cer_check <= args.target_error_rate:
                    print(f"\n  ✓ Target CER {args.target_error_rate} reached!")
                    break

            if iteration % args.save_interval == 0:
                _save(model, optimizer, iteration, train_ds, args,
                      p.model_output + f'_checkpoint_{iteration}.pt')

        if best_cer <= args.target_error_rate: break

    _save(model, optimizer, iteration, train_ds, args, p.model_output + '_final.pt')
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  ✓ Complete! {iteration} iters in {elapsed/60:.1f}m | Best CER: {best_cer:.6f}")
    if device.type == 'cuda':
        print(f"  Peak GPU: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"  Log: {p.log_file}")
    print(f"{'=' * 70}")

    if tb_writer:
        tb_writer.close()
        print(f"  TensorBoard logs: {tb_writer.log_dir}")

    try: cmd_plot(args)
    except: pass
