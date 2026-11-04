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
