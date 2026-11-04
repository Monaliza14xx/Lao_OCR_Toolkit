import os
import re
import sys
import json
import shutil
import numpy as np
import torch
from .model import build_model
from .utils import resolve_paths

def cmd_export(args):
    """Export the trained PyTorch model to ONNX format."""
    p = resolve_paths(args)
    
    # Try the best model first, fallback to final
    best_model_path = p.model_output + '_best.pt'
    final_model_path = p.model_output + '_final.pt'
    
    if os.path.exists(best_model_path):
        model_path = best_model_path
    elif os.path.exists(final_model_path):
        model_path = final_model_path
    else:
        print(f"Error: Could not find any .pt checkpoints for '{args.model_name}' in {p.output_dir}")
        sys.exit(1)
        
    print(f"Loading checkpoint for export: {model_path}")
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    
    spec = ckpt.get('net_spec')
    num_classes = ckpt.get('num_classes')
    charset = ckpt.get('charset')
    
    if not spec or not num_classes or charset is None:
        print("Error: The checkpoint is missing required metadata (net_spec, num_classes, charset)")
        sys.exit(1)
        
    # Build model using same logic as in training
    spec_clean = spec.replace('###', str(num_classes))
    model = build_model(spec_clean, num_classes)
    
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    # Flatten LSTM parameters to eliminate _flat_weights tracing warnings
    for m in model.modules():
        if isinstance(m, torch.nn.LSTM):
            m.flatten_parameters()

    # Read the target image height from the net_spec (e.g. [1,64,0,1 ...] → 64).
    # The dummy input MUST match the height used during training, otherwise the
    # ONNX graph traces through SumLSTM/Conv/Pool with wrong spatial dimensions.
    target_h = 48  # Tesseract default
    m_spec = re.match(r'\[?\d+,(\d+),', spec)
    if m_spec:
        target_h = int(m_spec.group(1))

    dummy_input = torch.randn(1, 1, target_h, 200, dtype=torch.float32)
    
    onnx_path = os.path.join(p.output_dir, f"{args.model_name}.onnx")
    info_path = os.path.join(p.output_dir, f"{args.model_name}_model_info.json")
    
    print(f"Exporting model to ONNX (input height={target_h})...")
    # Use the legacy TorchScript-based ONNX exporter (dynamo=False).
    # The new dynamo-based exporter (PyTorch 2.x default) fails on LSTMs
    # with dynamic shapes due to PendingUnbackedSymbolNotFound errors in
    # torch.export.export(). The legacy tracer handles LSTMs correctly.
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['image'],
        output_names=['output'],
        dynamic_axes={
            'image': {0: 'batch_size', 3: 'width'},
            'output': {0: 'batch_size', 1: 'width'}
        },
        dynamo=False,
    )

    # ── Verify ONNX output matches PyTorch numerically ──
    verified = False
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        with torch.no_grad():
            pt_out = model(dummy_input)
        ort_out = sess.run(None, {'image': dummy_input.numpy()})[0]
        max_diff = np.max(np.abs(pt_out.numpy() - ort_out))
        if max_diff < 1e-4:
            print(f"  ✓ Numerical verification passed (max diff: {max_diff:.2e})")
            verified = True
        else:
            print(f"  ⚠ Numerical mismatch detected (max diff: {max_diff:.2e})")
            print(f"    ONNX output may not match PyTorch — check results carefully.")
    except ImportError:
        print("  ℹ onnxruntime not installed — skipping numerical verification")
        print("    Install with: pip install onnxruntime")
    except Exception as e:
        print(f"  ⚠ Verification failed: {e}")

    print("\nDumping model inference metadata...")
    with open(info_path, 'w') as f:
        json.dump({
            'net_spec': spec,
            'num_classes': num_classes,
            'charset': charset,
            'input_height': target_h,
            'verified': verified,
        }, f, indent=2, ensure_ascii=False)
        
    print(f"\n{'=' * 70}")
    print("  ✓ Export Successful!")
    print(f"  ONNX Model:  {onnx_path}")
    print(f"  Metadata:    {info_path}")
    print(f"  Input Shape: [batch, 1, {target_h}, width]")
    print(f"{'=' * 70}")

    # ── Copy artefacts to the shared onnx_model directory ───────────────────────
    # Resolve data_dir from args (may be a relative path)
    data_dir = getattr(args, 'data_dir', 'data') or 'data'
    onnx_dest_dir = os.path.join(data_dir, 'onnx_model', args.model_name)
    os.makedirs(onnx_dest_dir, exist_ok=True)

    dest_onnx = os.path.join(onnx_dest_dir, f"{args.model_name}.onnx")
    dest_info = os.path.join(onnx_dest_dir, f"{args.model_name}_model_info.json")
    shutil.copy2(onnx_path, dest_onnx)
    shutil.copy2(info_path, dest_info)

    print(f"\n  ✓ Copied to:  {onnx_dest_dir}/")
    print(f"      {os.path.basename(dest_onnx)}")
    print(f"      {os.path.basename(dest_info)}")
    print(f"{'=' * 70}")
