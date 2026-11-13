import numpy as np
import pytest

ort = pytest.importorskip('onnxruntime')
pytest.importorskip('transformers')
pytest.importorskip('cv2')
from PIL import Image, ImageDraw

from lao_line_ocr import ctc_greedy, load_chars, pad_batch, prep_line, reading_order
from to_paddleocr import main as to_paddleocr
from train_ppocr_rec import frames_needed, main as train_main


def test_ctc_greedy_merges_repeats_and_drops_blanks():
    chars = ['', 'ກ', 'າ', ' ']
    probs = np.eye(4)[[1, 1, 0, 1, 2, 2, 0, 3, 0]] * 0.9
    assert ctc_greedy(probs, chars) == ('ກກາ ', pytest.approx(0.9))
    assert ctc_greedy(np.eye(4)[[0, 0]], chars) == ('', 0.0)


def test_frames_needed_counts_blank_between_repeats():
    assert frames_needed([1, 2, 3]) == 3
    assert frames_needed([1, 1, 2, 2]) == 6


def test_prep_line_keeps_aspect_ratio():
    x = prep_line(np.full((24, 200, 3), 255, np.uint8))
    assert x.shape == (3, 48, 400) and x.max() == pytest.approx(1.0)
    assert prep_line(np.zeros((10, 1000, 3), np.uint8), max_w=320).shape == (3, 48, 320)
    assert pad_batch([x, prep_line(np.zeros((48, 48, 3), np.uint8))]).shape == (2, 3, 48, 400)


def test_reading_order_rows_then_left_to_right():
    def box(x, y):
        return np.array([[x, y], [x + 50, y], [x + 50, y + 20], [x, y + 20]], np.float32)
    boxes = [box(300, 102), box(10, 100), box(10, 200)]
    assert reading_order(boxes) == [[1, 0], [2]]


def test_train_and_export_offline(tmp_path):
    lines = tmp_path / 'lines'
    lines.mkdir()
    for i, text in enumerate(['ກາ', 'ນ້ຳ', 'ສະບາຍ', 'ດີ ກາ'] * 3):
        img = Image.new('RGB', (30 * len(text), 32), 'white')
        ImageDraw.Draw(img).rectangle([2, 8, 10 + 5 * i, 24], fill='black')
        img.save(lines / f'l{i}.png')
        (lines / f'l{i}.gt.txt').write_text(text, encoding='utf-8')
    data, out = tmp_path / 'data', tmp_path / 'model'
    assert to_paddleocr([str(lines), '--out', str(data), '--val-ratio', '0.25']) == 0

    train_main(['train', '--data', str(data), '--out', str(out), '--init', 'none', '--det', 'none',
                '--epochs', '1', '--batch_size', '4', '--workers', '0', '--device', 'cpu'])

    assert (out / 'best' / 'model.safetensors').exists()
    chars = load_chars(out / 'rec_dict.txt')
    assert chars[0] == '' and chars[-1] == ' ' and 'ຳ' in chars
    sess = ort.InferenceSession(str(out / 'rec.onnx'), providers=['CPUExecutionProvider'])
    probs = sess.run(None, {'x': np.zeros((2, 3, 48, 160), np.float32)})[0]
    assert probs.shape == (2, 20, len(chars))
    assert np.allclose(probs.sum(-1), 1, atol=1e-4)
