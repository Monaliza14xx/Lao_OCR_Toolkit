import os
import sys
import glob
import re
import torch
from torch.utils.data import DataLoader
from .utils import resolve_paths
from .dataset import TesseractDataset, collate_fn, legacy_collate_fn
from .model import build_model
from .metrics import (compute_metrics, edit_distance, align_strings, compute_component_metrics, LAO_COMPONENT_MAP,
                      strict_cer_counts, cluster_error_counts)
import csv


def _infer_spec_from_weights(state_dict, stored_spec, num_classes):
    """
    Reconstruct the net_spec from the actual weight shapes in the checkpoint.
    Used when the stored net_spec doesn't match the saved weights.
    Assumes the standard layout: Conv → MaxPool → SumLSTM → FwdLSTM → RevLSTM → FwdLSTM → Linear
    """
    try:
        sd = state_dict
        # Layer 3: SumLSTM  — weight_ih shape: [4*hidden, conv_out]
        h3 = sd['3.lstm.weight_ih_l0'].shape[0] // 4   # Lfys hidden
        # Layer 4: FwdLSTM  — weight_ih shape: [4*hidden, h3]
        h4 = sd['4.lstm.weight_ih_l0'].shape[0] // 4   # Lfx hidden
        # Layer 5: RevLSTM  — weight_ih shape: [4*hidden, h4]
        h5 = sd['5.lstm.weight_ih_l0'].shape[0] // 4   # Lrx hidden
        # Layer 6: FwdLSTM  — weight_ih shape: [4*hidden, h5]
        h6 = sd['6.lstm.weight_ih_l0'].shape[0] // 4   # Lfx hidden

        # Reconstruct the header from the stored spec (keep [B,H,W,C prefix)
        m = re.match(r'(\[?\d+,\d+,\d+,\d+\s+\S+\s+\S+\s+)', stored_spec)
        prefix = m.group(1) if m else '[1,36,0,1 Ct3,3,16 Mp3,3 '

        inferred = f"{prefix}Lfys{h3} Lfx{h4} Lrx{h5} Lfx{h6} O1c{num_classes}]"
        return inferred
    except KeyError:
        return None


def cmd_evaluation(args):
    """Evaluate trained model on eval dataset."""
    p = resolve_paths(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_pt = p.model_output + '_best.pt'
    final_pt = p.model_output + '_final.pt'
    pt_path = best_pt if os.path.exists(best_pt) else final_pt
    if not os.path.exists(pt_path):
        print(f"Error: No checkpoint at {best_pt} or {final_pt}")
        sys.exit(1)

    ckpt = torch.load(pt_path, map_location=device, weights_only=False)

    stored_spec = ckpt.get('net_spec', '')
    num_classes  = ckpt['num_classes']
    spec = stored_spec.replace('###', str(num_classes))
    model = build_model(spec, num_classes).to(device)

    try:
        model.load_state_dict(ckpt['model_state_dict'])
    except RuntimeError:
        # Stored net_spec doesn't match weights — infer the real architecture
        inferred = _infer_spec_from_weights(ckpt['model_state_dict'], stored_spec, num_classes)
        if inferred is None:
            print(f"  ERROR: Cannot infer architecture from {pt_path}. Re-run training.")
            sys.exit(1)
        print("  WARNING: Stored net_spec was incorrect.")
        print(f"    Stored:   {stored_spec}")
        print(f"    Inferred: {inferred}")
        model = build_model(inferred, num_classes).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        spec = inferred  # use inferred spec for target_h extraction below


    eval_files = []
    if os.path.exists(p.val_list):
        with open(p.val_list) as f:
            eval_files = [l.strip() for l in f if l.strip()]
    else:
        all_files = sorted(glob.glob(os.path.join(p.ground_truth_dir, '**', '*.lstmf'), recursive=True))
        split = int(len(all_files) * args.ratio_train)
        eval_files = all_files[split:]

    if not eval_files:
        print("No eval files found!")
        sys.exit(1)

    target_h = 48
    m = re.match(r'\[?\d+,(\d+),', ckpt.get('net_spec', ''))
    if m: target_h = int(m.group(1))

    legacy_pad = getattr(args, 'padding', 'white') == 'legacy'
    eval_ds = TesseractDataset(eval_files, target_height=target_h, charset=ckpt['charset'])
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=legacy_collate_fn if legacy_pad else collate_fn)

    c_samples = -1 if getattr(args, 'csv', '') else 5
    extra = {}
    cer, wer, acc, coer, samples = compute_metrics(model, eval_loader, eval_ds, device, max_batches=9999, collect_samples=c_samples,
                                                 use_real_lengths=not legacy_pad, details=extra)
    print(f"\n{'=' * 50}")
    print(f"  Model:      {pt_path}")
    print(f"  Eval files: {len(eval_files)}")
    print(f"  CER:        {cer:.6f} ({cer*100:.2f}%)   lenient (normalised, similar pairs = 0.5)")
    print(f"  CER strict: {extra['cer_strict']:.6f} ({extra['cer_strict']*100:.2f}%)   plain Levenshtein")
    print(f"  CluER:      {extra['cluer']:.6f} ({extra['cluer']*100:.2f}%)   consonant + marks = 1 unit")
    print(f"  WER:        {wer:.6f} ({wer*100:.2f}%)")
    print(f"  Accuracy:   {acc:.6f} ({acc*100:.2f}%)")
    
    if coer:
        print("\n  Component Error Rates (CoER):")
        # Sort by error rate descending to show worst parts first
        sorted_coer = sorted(coer.items(), key=lambda x: x[1], reverse=True)
        print(f"    {'Component':<15} | {'Error Rate':<10}")
        print(f"    {'-'*15} + {'-'*10}")
        for cat, val in sorted_coer:
            print(f"    {cat:<15} | {val*100:>8.2f}%")
    if samples:
        print("\n  Sample Predictions:")
        for i, (gt_s, pred_s) in enumerate(samples):
            print(f"    [{i+1}] GT:   {gt_s[:80]}")
            print(f"        Pred: {pred_s[:80]}")
    if getattr(args, 'csv', ''):
        print(f"\n  Exporting detailed CSV to {args.csv}...")
        
        # Flatten CoER for CSV columns
        all_cats = list(LAO_COMPONENT_MAP.keys()) + ['other']
        sorted_cats = sorted(all_cats)
        
        headers = ['case', 'truth', 'pred', 'cer', 'wer']
        for cat in sorted_cats:
            headers.extend([f'{cat}_errors', f'{cat}_total', f'{cat}_rate'])
        headers.extend(['cer_strict', 'cluer'])
            
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for i, (gt_s, pred_s) in enumerate(samples):
                case_cer = edit_distance(pred_s, gt_s) / max(len(gt_s), 1)
                case_wer = edit_distance(pred_s.split(), gt_s.split()) / max(len(gt_s.split()), 1)
                
                edits = align_strings(pred_s, gt_s)
                c_stats = compute_component_metrics(edits)
                
                row = [f"Sample_{i+1}", gt_s, pred_s, f"{case_cer:.4f}", f"{case_wer:.4f}"]
                for cat in sorted_cats:
                    s = c_stats.get(cat, {'errors': 0, 'total': 0})
                    rate = s['errors'] / max(s['total'], 1) if s['total'] > 0 else 0
                    row.extend([s['errors'], s['total'], f"{rate:.4f}"])
                se, sn = strict_cer_counts(pred_s, gt_s)
                ce, cn = cluster_error_counts(pred_s, gt_s)
                row.extend([f"{se / max(sn, 1):.4f}", f"{ce / max(cn, 1):.4f}"])
                writer.writerow(row)
        print(f"  ✓ Export complete: {args.csv}")

    print(f"{'=' * 50}")

