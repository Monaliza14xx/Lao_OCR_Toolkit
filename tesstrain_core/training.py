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
from .utils import resolve_paths, set_seed, git_info
from .dataset import TesseractDataset, collate_fn, legacy_collate_fn
from .model import build_model, output_lengths
from .metrics import compute_metrics, compute_mean_rms, TrainingLogger
from .report import render_progress, COER_KEYS
from .command_utils import cmd_plot


def apply_compatible_pretrained_weights(model, pretrained_state):
    """Copy every tensor whose name and shape match into model; return (loaded, skipped).

    Used when a --start_model checkpoint doesn't load strictly (e.g. a different charset size,
    so the output layer differs): the rest of the network is still reused.
    """
    model_state = model.state_dict()
    loaded, skipped = [], []
    for key, value in pretrained_state.items():
        if key not in model_state:
            skipped.append((key, 'missing'))
            continue
        if model_state[key].shape != value.shape:
            skipped.append((key, f"shape {tuple(value.shape)} != {tuple(model_state[key].shape)}"))
            continue
        model_state[key] = value.to(dtype=model_state[key].dtype)
        loaded.append(key)
    model.load_state_dict(model_state, strict=False)
    return loaded, skipped

def _save(model, optimizer, it, ds, args, path, scheduler=None):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    ckpt = {'iteration': it, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'charset': ds.charset, 'num_classes': ds.num_classes,
            'net_spec': args.net_spec, 'model_name': args.model_name,
            'seed': getattr(args, 'seed', None)}
    if scheduler is not None:
        ckpt['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(ckpt, path)
    print(f"  ✓ Saved: {path}")

def cmd_training(args):
    """Start GPU training with PyTorch."""
    p = resolve_paths(args)
    seed = getattr(args, 'seed', None)
    if seed is not None:      # before anything random: model init, shuffling
        set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if getattr(args, 'force_GPU', False) and device.type == 'cpu':
        print("\nError: --force_GPU was specified, but no GPU (CUDA) is available.")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print("  Tesseract PyTorch GPU Trainer")
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
    print(f"  Seed:            {seed if seed is not None else 'none (not reproducible)'}")
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
            print("Error: No .lstmf files. Run 'lists' first or provide --ground_truth_dir")
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

    # P0.1: white padding + real per-line lengths. --padding legacy = black padding and the
    # full padded width for CTC / decoding, as in runs made before the fix (E1a faithful arm).
    legacy_pad = getattr(args, 'padding', 'white') == 'legacy'
    collate = legacy_collate_fn if legacy_pad else collate_fn
    if legacy_pad:
        print("  Padding:         legacy (black, full padded width for CTC)")

    train_ds = TesseractDataset(train_files, target_height=target_h, charset=charset)
    eval_loader = None
    if eval_files:
        eval_ds = TesseractDataset(eval_files, target_height=target_h, charset=train_ds.charset)
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, collate_fn=collate,
                                  pin_memory=device.type == 'cuda')

    with open(charset_path, 'w') as f:
        json.dump(train_ds.charset, f, ensure_ascii=False, indent=2)

    max_iter = args.max_iterations
    if getattr(args, 'epochs', 0) > 0:
        max_iter = int(math.ceil(args.epochs * len(train_ds) / args.batch_size))
        print(f"  Epochs={args.epochs} → max_iterations={max_iter}")

    loader_gen = torch.Generator().manual_seed(seed) if seed is not None else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=loader_gen,
                               num_workers=args.num_workers, collate_fn=collate,
                               pin_memory=device.type == 'cuda', drop_last=True)

    spec = args.net_spec
    start_iter = 0
    ckpt = None

    if getattr(args, 'start_model', '') and os.path.exists(args.start_model):
        print(f"  Loading checkpoint: {args.start_model}")
        ckpt = torch.load(args.start_model, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            if 'net_spec' in ckpt:
                spec = ckpt['net_spec']   # use the architecture from the checkpoint
                args.net_spec = spec      # keep args in sync so _save() persists the right spec
                print(f"  Using net_spec from checkpoint: {spec}")
            start_iter = ckpt.get('iteration', 0)

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
        
    scheduler = None
    if getattr(args, 'lr_scheduler', 'none') == 'plateau':
        # Stepped every evaluation (every 5 × --debug_interval iterations) on eval CER, so
        # --lr_patience counts evaluations, not iterations.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)
        print(f"  LR scheduler:    ReduceLROnPlateau on eval CER (factor {args.lr_factor}, "
              f"patience {args.lr_patience} evals = {args.lr_patience * args.debug_interval * 5} iters, "
              f"min_lr {args.min_lr})")
        if not eval_loader:
            print("  Warning: no eval list; the LR scheduler will follow the training-batch CER instead.")

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

    # Everything needed to say exactly what this run was (P0.4)
    config = dict(vars(args))
    config['run'] = {
        'git': git_info(),
        'seed': seed,
        'net_spec_resolved': spec,
        'num_classes': train_ds.num_classes,
        'num_params': total_p,
        'train_lines': len(train_ds),
        'eval_lines': len(eval_loader.dataset) if eval_loader else 0,
        'padding': 'legacy' if legacy_pad else 'white',
        'torch': torch.__version__,
        'cuda': torch.version.cuda if device.type == 'cuda' else None,
        'device': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
        'python': sys.version.split()[0],
        'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(p.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)

    print(f"{'=' * 70}")
    print("  Training started...")
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

            out = model(imgs) if legacy_pad else model(imgs, ilens)
            if out.dim() == 4:
                B, C, H, W = out.shape
                out = out.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)

            lp = F.log_softmax(out, dim=2).permute(1, 0, 2)
            T = lp.shape[0]
            ail = (torch.full((lp.shape[1],), T, dtype=torch.long) if legacy_pad
                   else output_lengths(model, ilens, T))

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

                train_extra, eval_extra = {}, {}
                train_cer, train_wer, train_acc, train_coer, train_samples = compute_metrics(
                    model, [(imgs.cpu(), labels.cpu(), ilens, tlens)],
                    train_ds, device, max_batches=1, collect_samples=3, use_real_lengths=not legacy_pad,
                    details=train_extra)

                eval_cer, eval_wer, eval_acc, eval_loss_val, eval_coer = None, None, None, None, None
                if eval_loader and iteration % (args.debug_interval * 5) == 0:
                    eval_cer, eval_wer, eval_acc, eval_coer, _ = compute_metrics(
                        model, eval_loader, train_ds, device, use_real_lengths=not legacy_pad,
                        details=eval_extra)
                    # Compute eval loss
                    model.eval()
                    eval_losses = []
                    with torch.no_grad():
                        for eb, (e_imgs, e_labels, e_ilens, e_tlens) in enumerate(eval_loader):
                            if eb >= 50: break
                            e_imgs, e_labels = e_imgs.to(device), e_labels.to(device)
                            e_out = model(e_imgs) if legacy_pad else model(e_imgs, e_ilens)
                            if e_out.dim() == 4:
                                eB, eC, eH, eW = e_out.shape
                                e_out = e_out.permute(0, 3, 1, 2).contiguous().view(eB, eW, eC * eH)
                            e_lp = F.log_softmax(e_out, dim=2).permute(1, 0, 2)
                            eT = e_lp.shape[0]
                            e_ail = (torch.full((e_lp.shape[1],), eT, dtype=torch.long) if legacy_pad
                                     else output_lengths(model, e_ilens, eT))
                            e_loss = ctc_loss_fn(e_lp, e_labels, e_ail, e_tlens)
                            eval_losses.append(e_loss.item())
                    model.train()
                    eval_loss_val = sum(eval_losses) / max(len(eval_losses), 1)

                mean_rms = compute_mean_rms(model)
                gpu_mem = torch.cuda.memory_allocated() / 1e9 if device.type == 'cuda' else 0
                gpu_peak = torch.cuda.max_memory_allocated() / 1e9 if gpu_mem else 0

                # Best checkpoint: by eval CER when validation exists, else by train CER
                score = eval_cer if eval_loader else train_cer
                is_best = score is not None and score < best_cer

                coer, coer_source = (eval_coer, 'eval') if eval_coer is not None else (train_coer, 'train')
                print(render_progress(
                    iteration=iteration, max_iter=max_iter, epoch=epoch, sps=sps,
                    elapsed=elapsed, iters_done=iteration - start_iter,
                    loss=loss.item(), avg_loss=avg_loss,
                    train={'cer': train_cer, 'wer': train_wer, 'acc': train_acc},
                    evaluation=None if eval_cer is None else
                        {'loss': eval_loss_val, 'cer': eval_cer, 'wer': eval_wer, 'acc': eval_acc,
                         'cer_strict': eval_extra.get('cer_strict'), 'cluer': eval_extra.get('cluer')},
                    coer=coer, coer_source=coer_source, is_best=is_best,
                    mean_rms=mean_rms, lr=optimizer.param_groups[0]['lr'],
                    gpu_mem=gpu_mem, gpu_peak=gpu_peak, samples=train_samples))

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
                           learning_rate=optimizer.param_groups[0]['lr'],
                           gpu_mem_gb=round(gpu_mem, 3), samples_per_sec=round(sps, 1),
                           coer_source=coer_source,
                           train_cer_strict=round(train_extra['cer_strict'], 6),
                           train_cluer=round(train_extra['cluer'], 6),
                           eval_cer_strict=round(eval_extra['cer_strict'], 6) if eval_extra else '',
                           eval_cluer=round(eval_extra['cluer'], 6) if eval_extra else '',
                           **{f'coer_{k}': round(coer[k], 6) for k in COER_KEYS if coer and k in coer})

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
                        tb_writer.add_scalar('Eval/CER_strict', eval_extra['cer_strict'], iteration)
                        tb_writer.add_scalar('Eval/CluER',    eval_extra['cluer'], iteration)
                    if eval_acc is not None:
                        tb_writer.add_scalar('Eval/Accuracy', eval_acc,  iteration)
                    # Group: CoER (per Lao character class, eval set only)
                    for k, v in (eval_coer or {}).items():
                        tb_writer.add_scalar(f'CoER/{k}', v, iteration)
                    # Group: Model
                    tb_writer.add_scalar('Model/Mean_RMS',         mean_rms,           iteration)
                    # Group: System
                    tb_writer.add_scalar('System/Learning_Rate',   optimizer.param_groups[0]['lr'], iteration)
                    tb_writer.add_scalar('System/GPU_Memory_GB',   gpu_mem,            iteration)
                    tb_writer.add_scalar('System/Samples_per_Sec', sps,                iteration)
                    tb_writer.flush()

                run_loss, run_n = 0.0, 0

                if scheduler is not None and iteration % (args.debug_interval * 5) == 0:
                    before = optimizer.param_groups[0]['lr']
                    scheduler.step(eval_cer if eval_cer is not None else train_cer)
                    after = optimizer.param_groups[0]['lr']
                    if after < before:
                        print(f"  ↓ LR reduced {before:.2e} → {after:.2e} (no CER improvement for "
                              f"{args.lr_patience} evaluations)")

                if is_best:
                    best_cer = score
                    _save(model, optimizer, iteration, train_ds, args, p.model_output + '_best.pt', scheduler)

                cer_check = eval_cer if eval_cer is not None else train_cer
                if cer_check <= args.target_error_rate:
                    print(f"\n  ✓ Target CER {args.target_error_rate} reached!")
                    break

            if iteration % args.save_interval == 0:
                _save(model, optimizer, iteration, train_ds, args,
                      p.model_output + f'_checkpoint_{iteration}.pt', scheduler)

        if best_cer <= args.target_error_rate: break

    _save(model, optimizer, iteration, train_ds, args, p.model_output + '_final.pt', scheduler)
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    best = (f"Best CER: {best_cer:.6f}" if best_cer != float('inf') else
            f"no evaluation ran (it runs every {args.debug_interval * 5} iterations), so no _best.pt")
    print(f"  ✓ Complete! {iteration} iters in {elapsed/60:.1f}m | {best}")
    if device.type == 'cuda':
        print(f"  Peak GPU: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"  Log: {p.log_file}")
    print(f"{'=' * 70}")

    if tb_writer:
        tb_writer.close()
        print(f"  TensorBoard logs: {tb_writer.log_dir}")

    try:
        cmd_plot(args)
    except Exception as e:   # plotting is a convenience; training already finished
        print(f"  ⚠ Could not draw training plots: {e}")
