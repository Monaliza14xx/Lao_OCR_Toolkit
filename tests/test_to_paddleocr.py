import os

from PIL import Image

from to_paddleocr import main


def make_pair(d, name, text):
    os.makedirs(d, exist_ok=True)
    Image.new('L', (40, 10), 255).save(os.path.join(d, name + '.png'))
    with open(os.path.join(d, name + '.gt.txt'), 'w', encoding='utf-8') as f:
        f.write(text)


def read_list(path):
    with open(path, encoding='utf-8') as f:
        return [line.rstrip('\n').split('\t') for line in f]


def test_converts_pairs_and_builds_dict(tmp_path):
    synth, real, out = tmp_path / 'synth', tmp_path / 'real', tmp_path / 'out'
    for i in range(20):
        make_pair(synth / 'sub', f'l{i}', 'ສະບາຍດີ ນ້ຳ')
    make_pair(synth, 'nam', 'ນໍ້າ')               # ໍ + tone + າ  → stored as ນ້ຳ
    make_pair(synth, 'empty', '   ')               # dropped
    Image.new('L', (40, 10)).save(synth / 'no_gt.png')  # no .gt.txt → dropped
    make_pair(real, 'r1', 'ຂອບໃຈ')

    assert main([str(synth), '--test', str(real), '--out', str(out), '--val-ratio', '0.1']) == 0

    train = read_list(out / 'train_list.txt')
    val = read_list(out / 'val_list.txt')
    test = read_list(out / 'test_list.txt')
    assert (len(train), len(val), len(test)) == (19, 2, 1)
    for img, _ in train + val + test:
        assert (out / img).exists()               # paths are relative to --out
    assert 'ນ້ຳ' in {label for _, label in train + val}
    assert 'ນໍ້າ' not in {label for _, label in train + val}

    chars = (out / 'lao_dict.txt').read_text(encoding='utf-8').split('\n')[:-1]
    assert chars == sorted(set(chars))
    assert ' ' not in chars and 'ຳ' in chars and 'ໍ' not in chars
    assert set('ຂອບໃຈ') <= set(chars)             # test characters included


def test_no_copy_writes_absolute_paths(tmp_path):
    make_pair(tmp_path / 'synth', 'a', 'ກ')
    out = tmp_path / 'out'
    assert main([str(tmp_path / 'synth'), '--out', str(out), '--val-ratio', '0', '--no-copy']) == 0
    (img, label), = read_list(out / 'train_list.txt')
    assert os.path.isabs(img) and label == 'ກ'
    assert not (out / 'images').exists()


def test_empty_input_fails(tmp_path):
    (tmp_path / 'empty').mkdir()
    assert main([str(tmp_path / 'empty'), '--out', str(tmp_path / 'out')]) == 1
