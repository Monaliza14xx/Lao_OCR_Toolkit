"""colab_setup: find data/ in a big zip, extract it to local disk, keep web results on Drive."""
import json
import os
import sys
import zipfile

import pytest
from conftest import ROOT

sys.path.insert(0, os.path.join(ROOT, 'webapp'))
import colab_setup  # noqa: E402

FILES = {
    'tesstrain-gpu/torch_train.py': b'print(1)',
    'tesstrain-gpu/tesstrain_core/data/not_this.txt': b'x',
    'tesstrain-gpu/data/lao-ground-truth/ສະບາຍດີ_1.lstmf': b'L' * 100,
    'tesstrain-gpu/data/lao-ground-truth/ສະບາຍດີ_1.gt.txt': 'ສະບາຍດີ'.encode(),
    'tesstrain-gpu/data/LaoCRNN-I64K5C16/LaoCRNN-I64K5C16_best.pt': b'P' * 50,
}


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / 'tesstrain-gpu-test.zip'
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in FILES.items():
            z.writestr(name, data)
    return str(path)


def test_finds_top_level_data_prefix(archive):
    assert colab_setup.find_data_prefix(archive) == 'tesstrain-gpu/data/'


@pytest.mark.parametrize('use_unzip', [True, False])
def test_extracts_only_data(archive, tmp_path, monkeypatch, use_unzip):
    if not use_unzip:
        monkeypatch.setattr(colab_setup.shutil, 'which', lambda _: None)
    data = colab_setup.extract_data(archive, str(tmp_path / 'work'))
    assert data == tmp_path / 'work' / 'data'
    assert (data / 'lao-ground-truth' / 'ສະບາຍດີ_1.gt.txt').read_text() == 'ສະບາຍດີ'
    assert (data / 'LaoCRNN-I64K5C16' / 'LaoCRNN-I64K5C16_best.pt').stat().st_size == 50
    assert not (tmp_path / 'work' / 'torch_train.py').exists()
    assert not (tmp_path / 'work' / '.unzip').exists()


def test_extracts_selected_datasets_and_skips_repeat(archive, tmp_path, capsys):
    work = str(tmp_path / 'work')
    data = colab_setup.extract_data(archive, work, ['lao-ground-truth'])
    assert sorted(os.listdir(data)) == ['lao-ground-truth']
    colab_setup.extract_data(archive, work, ['lao-ground-truth'])
    assert 'Already extracted' in capsys.readouterr().out
    # a different selection extracts again
    data = colab_setup.extract_data(archive, work)
    assert sorted(os.listdir(data)) == ['LaoCRNN-I64K5C16', 'lao-ground-truth']


def test_refuses_when_disk_too_small(archive, tmp_path):
    with pytest.raises(RuntimeError, match='Not enough local disk'):
        colab_setup.extract_data(archive, str(tmp_path / 'work'), min_free_gb=1e9)


def test_sync_and_restore_round_trip(tmp_path):
    data, drive = tmp_path / 'data', tmp_path / 'drive'
    job = data / 'web_jobs' / '20261104aaa'
    job.mkdir(parents=True)
    (job / 'job.json').write_text(json.dumps({'status': 'running', 'config': {'model_name': 'lao-web'}}))
    (job / 'log.csv').write_text('iteration\n100\n')
    (data / 'lao-web').mkdir()
    (data / 'lao-web' / 'lao-web_best.pt').write_bytes(b'1' * 10)
    (data / 'lao-ground-truth').mkdir()                      # dataset from the zip: not synced
    (data / 'lao-ground-truth' / 'a.lstmf').write_bytes(b'x')

    assert colab_setup.sync_to_drive(str(data), str(drive)) == 3
    assert colab_setup.sync_to_drive(str(data), str(drive)) == 0      # unchanged files are skipped
    assert (drive / 'models' / 'lao-web' / 'lao-web_best.pt').exists()
    assert not (drive / 'models' / 'lao-ground-truth').exists()

    (data / 'lao-web' / 'lao-web_best.pt').write_bytes(b'2' * 12)
    assert colab_setup.sync_to_drive(str(data), str(drive)) == 1

    fresh = tmp_path / 'next-session' / 'data'
    assert colab_setup.restore_from_drive(str(drive), str(fresh)) == 3
    assert (fresh / 'lao-web' / 'lao-web_best.pt').read_bytes() == b'2' * 12
    assert (fresh / 'web_jobs' / '20261104aaa' / 'log.csv').exists()
