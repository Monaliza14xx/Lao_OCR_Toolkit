import os
import argparse

def resolve_paths(args):
    """Compute all derived paths from CLI args, matching tesstrain Makefile logic."""
    p = argparse.Namespace(**vars(args))
    p.data_dir = getattr(args, 'data_dir', 'data')
    p.output_dir = getattr(args, 'output_dir', '') or os.path.join(p.data_dir, args.model_name)
    p.ground_truth_dir = getattr(args, 'ground_truth_dir', '') or f"{p.output_dir}-ground-truth"
    
    # Smart fallback: if the computed ground truth dir doesn't exist, see if there's a base language dir like lao-ground-truth
    if not getattr(args, 'ground_truth_dir', '') and not os.path.isdir(p.ground_truth_dir):
        found = False
        for i in range(len(args.model_name) - 1, 0, -1):
            prefix = args.model_name[:i]
            candidate = os.path.join(p.data_dir, f"{prefix}-ground-truth")
            if os.path.isdir(candidate):
                p.ground_truth_dir = candidate
                found = True
                break
        # Final fallback: case-insensitive glob for any *-ground-truth dir
        # (handles models like LaoCRNN-* whose prefix never matches lao-ground-truth)
        if not found:
            import glob as _glob
            gt_dirs = sorted(_glob.glob(os.path.join(p.data_dir, '*-ground-truth')))
            gt_dirs = [d for d in gt_dirs if os.path.isdir(d)]
            if gt_dirs:
                p.ground_truth_dir = gt_dirs[0]
    p.log_file = getattr(args, 'log_file', '') or os.path.join(p.output_dir, 'training.log')
    p.model_output = os.path.join(p.output_dir, args.model_name)
    p.unicharset_file = os.path.join(p.output_dir, 'unicharset')
    p.proto_model = os.path.join(p.output_dir, f'{args.model_name}.traineddata')
    p.wordlist_file = getattr(args, 'wordlist_file', '') or os.path.join(p.output_dir, f'{args.model_name}.wordlist')
    p.numbers_file = getattr(args, 'numbers_file', '') or os.path.join(p.output_dir, f'{args.model_name}.numbers')
    p.punc_file = getattr(args, 'punc_file', '') or os.path.join(p.output_dir, f'{args.model_name}.punc')
    p.train_list = os.path.join(p.output_dir, 'list.train')
    p.val_list   = os.path.join(p.output_dir, 'list.val')
    p.test_list  = os.path.join(p.output_dir, 'list.test')
    os.makedirs(p.output_dir, exist_ok=True)
    return p


def set_seed(seed):
    """Seed python, numpy and torch (CPU + CUDA) and ask for deterministic kernels.

    Some CUDA ops (e.g. CTC loss backward) have no deterministic version; torch then warns
    instead of failing (warn_only=True), so GPU runs are reproducible only up to those ops.
    """
    import random
    import numpy as np
    import torch
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')   # needed for deterministic cuBLAS
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def git_info(path=None):
    """{'commit': sha or None, 'dirty': bool or None} of the repository holding this code."""
    import subprocess
    cwd = path or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=cwd, capture_output=True,
                             text=True, timeout=10).stdout.strip() or None
        status = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=cwd,
                                capture_output=True, text=True, timeout=10).stdout
        return {'commit': sha, 'dirty': bool(status.strip()) if sha else None}
    except (OSError, subprocess.SubprocessError):
        return {'commit': None, 'dirty': None}
