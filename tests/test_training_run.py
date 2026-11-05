"""End-to-end `torch_train.py training` on a few tiny .lstmf files: seed, config.json, LR scheduler."""
import csv
import io
import json
import os
import struct
import subprocess
import sys

import pytest
import torch
from conftest import ROOT
from PIL import Image, ImageDraw

from tesstrain_core.dataset import load_lstmf

TEXTS = ['ກາ', 'ກິ່ນ', 'ນ້ຳ', 'ເຂົ້າ', 'ປາ', 'ມື້', 'ດີ', 'ລາວ']
SPEC = '[1,24,0,1 Ct3,3,8 Mp3,3 Lfys16 Lfx24 Lrx24 O1c###]'


def _s(b):
    return struct.pack('<I', len(b)) + b


def write_lstmf(path, png, text):
    """Minimal one-page .lstmf (the layout tesstrain_core.dataset.LstmfReader reads)."""
    page = (_s(b'line.png') + struct.pack('<i', 0) + _s(png) + _s(b'lao') + _s(text.encode())
            + struct.pack('<I', 0) + struct.pack('<I', 0) + struct.pack('<b', 0))
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', 1) + struct.pack('<B', 1) + page)


@pytest.fixture(scope='module')
def gt_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp('gt')
    for i in range(16):
        text = TEXTS[i % len(TEXTS)]
        img = Image.new('L', (20 + 14 * len(text), 24), 255)
        draw = ImageDraw.Draw(img)
        for k in range(len(text)):
            draw.rectangle([6 + 14 * k, 4 + (k % 3), 14 + 14 * k, 19], fill=0)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        write_lstmf(d / f'line{i:02d}.lstmf', buf.getvalue(), text)
    return d


def train(gt_dir, out_dir, *extra):
    cmd = [sys.executable, os.path.join(ROOT, 'torch_train.py'), 'training', '--model_name', 'tiny',
           '--ground_truth_dir', str(gt_dir), '--output_dir', str(out_dir), '--net_spec', SPEC,
           '--batch_size', '4', '--num_workers', '0', '--max_iterations', '20', '--debug_interval', '2',
           '--save_interval', '1000', '--target_error_rate', '0', '--ratio_train', '0.75',
           '--no-tensorboard', *extra]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


def test_lstmf_writer_roundtrip(gt_dir):
    pages = load_lstmf(gt_dir / 'line00.lstmf')
    assert pages[0]['transcription'] == TEXTS[0] and pages[0]['image_bytes'][:4] == b'\x89PNG'


def test_same_seed_gives_the_same_model_and_config_records_the_run(gt_dir, tmp_path):
    train(gt_dir, tmp_path / 'a', '--seed', '1')
    train(gt_dir, tmp_path / 'b', '--seed', '1')
    a = torch.load(tmp_path / 'a' / 'tiny_final.pt', weights_only=False)
    b = torch.load(tmp_path / 'b' / 'tiny_final.pt', weights_only=False)
    assert a['seed'] == 1
    for k in a['model_state_dict']:
        assert torch.equal(a['model_state_dict'][k], b['model_state_dict'][k]), k

    cfg = json.loads((tmp_path / 'a' / 'config.json').read_text())
    assert cfg['seed'] == 1 and cfg['net_spec'] == SPEC and cfg['lr_scheduler'] == 'none'
    run = cfg['run']
    assert run['seed'] == 1 and run['padding'] == 'white'
    assert run['net_spec_resolved'] == SPEC.replace('###', str(run['num_classes']))
    assert run['num_params'] > 0 and run['train_lines'] == 12 and run['eval_lines'] == 4
    assert run['git']['commit'] is None or len(run['git']['commit']) == 40


def test_plateau_scheduler_lowers_the_lr(gt_dir, tmp_path):
    out = train(gt_dir, tmp_path, '--seed', '2', '--lr_scheduler', 'plateau', '--lr_patience', '0',
                '--lr_factor', '0.5', '--min_lr', '0.0001', '--learning_rate', '0.002',
                '--max_iterations', '80')
    with open(tmp_path / 'training.log') as f:
        rows = list(csv.DictReader(f))
    lrs = [float(r['learning_rate']) for r in rows]
    assert lrs[0] == 0.002 and min(lrs) < 0.002 and min(lrs) >= 0.0001
    assert 'ReduceLROnPlateau' in out
    assert all(r['eval_cer_strict'] != '' for r in rows if r['eval_cer'] != '')
    ckpt = torch.load(tmp_path / 'tiny_final.pt', weights_only=False)
    assert 'scheduler_state_dict' in ckpt
    assert json.loads((tmp_path / 'config.json').read_text())['lr_scheduler'] == 'plateau'


def test_default_run_keeps_a_constant_lr(gt_dir, tmp_path):
    train(gt_dir, tmp_path)
    with open(tmp_path / 'training.log') as f:
        lrs = {float(r['learning_rate']) for r in csv.DictReader(f)}
    assert lrs == {0.002}
    assert 'scheduler_state_dict' not in torch.load(tmp_path / 'tiny_final.pt', weights_only=False)
