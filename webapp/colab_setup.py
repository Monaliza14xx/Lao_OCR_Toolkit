"""Helpers for colab_server.ipynb: unpack data from a big Drive zip, keep results on Drive.

Training reads many small files, so the data is extracted to Colab's local disk (fast).
Results (web job logs and the model folders those jobs write) are copied back to Drive,
and restored from there at the start of the next session.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path


def find_data_prefix(zip_path: str) -> str:
    """Return the archive prefix of the top-most `data/` folder, e.g. 'tesstrain-gpu/data/'."""
    best = None
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            parts = name.split('/')
            if 'data' in parts[:-1]:
                prefix = '/'.join(parts[:parts.index('data') + 1]) + '/'
                if best is None or prefix.count('/') < best.count('/'):
                    best = prefix
    if best is None:
        raise ValueError(f'no data/ folder inside {zip_path}')
    return best


def plan_extract(zip_path: str, datasets: list[str] | None = None) -> tuple[str, list[str], int]:
    """Members to extract (under data/, optionally only some top-level folders) and their total size."""
    prefix = find_data_prefix(zip_path)
    wanted = {d.strip('/') for d in datasets or []}
    names, total = [], 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if not info.filename.startswith(prefix) or info.filename == prefix:
                continue
            top = info.filename[len(prefix):].split('/')[0]
            if wanted and top not in wanted:
                continue
            names.append(info.filename)
            total += info.file_size
    return prefix, names, total


def extract_data(zip_path: str, dest: str, datasets: list[str] | None = None, min_free_gb: float = 5) -> Path:
    """Extract the archive's data/ folder to dest/data. Skips the work if it was already done."""
    dest_p = Path(dest)
    data_dir = dest_p / 'data'
    marker = dest_p / '.extracted.json'
    key = {'zip': os.path.abspath(zip_path), 'size': os.path.getsize(zip_path), 'datasets': sorted(datasets or [])}
    if marker.exists() and json.loads(marker.read_text()) == key and data_dir.is_dir():
        print(f'Already extracted: {data_dir}')
        return data_dir

    prefix, names, total = plan_extract(zip_path, datasets)
    if not names:
        raise ValueError(f'nothing to extract under {prefix} (datasets={datasets})')
    dest_p.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest_p).free
    print(f'{len(names):,} files, {total / 1e9:.1f} GB to extract from {prefix} ; {free / 1e9:.1f} GB free')
    if total + min_free_gb * 1e9 > free:
        raise RuntimeError('Not enough local disk. Set DATASETS to extract only the folders you need.')

    tmp = dest_p / '.unzip'
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    patterns = [prefix + '*'] if not datasets else [f'{prefix}{d.strip("/")}/*' for d in datasets]
    if shutil.which('unzip'):
        subprocess.run(['unzip', '-q', '-o', zip_path, *patterns, '-d', str(tmp)], check=True)
    else:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp, members=names)
    if data_dir.exists():
        shutil.rmtree(data_dir)
    (tmp / prefix.rstrip('/')).rename(data_dir)
    shutil.rmtree(tmp, ignore_errors=True)
    marker.write_text(json.dumps(key))
    print(f'Extracted to {data_dir}')
    return data_dir


def _copy_tree(src: Path, dst: Path) -> int:
    """Copy files that are missing or differ (size or mtime) in dst. Returns the number copied."""
    copied = 0
    for root, _, files in os.walk(src):
        rel = Path(root).relative_to(src)
        for name in files:
            s, d = Path(root) / name, dst / rel / name
            st = s.stat()
            if d.exists():
                dt = d.stat()
                if dt.st_size == st.st_size and int(dt.st_mtime) == int(st.st_mtime):
                    continue
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
            copied += 1
    return copied


def web_models(data_dir: Path) -> set[str]:
    """Model folders written by web jobs."""
    names = set()
    for f in (data_dir / 'web_jobs').glob('*/job.json'):
        try:
            names.add(json.loads(f.read_text())['config']['model_name'])
        except (OSError, ValueError, KeyError):
            continue
    return names


def sync_to_drive(data_dir: str, drive_out: str) -> int:
    """Copy web_jobs/ and the model folders of web jobs to drive_out. Safe to call repeatedly."""
    data, out = Path(data_dir), Path(drive_out)
    n = 0
    if (data / 'web_jobs').is_dir():
        n += _copy_tree(data / 'web_jobs', out / 'web_jobs')
    for m in web_models(data):
        if (data / m).is_dir():
            n += _copy_tree(data / m, out / 'models' / m)
    return n


def restore_from_drive(drive_out: str, data_dir: str) -> int:
    """Bring back job history and web-trained models from earlier sessions."""
    data, out = Path(data_dir), Path(drive_out)
    n = 0
    if (out / 'web_jobs').is_dir():
        n += _copy_tree(out / 'web_jobs', data / 'web_jobs')
    if (out / 'models').is_dir():
        for m in (out / 'models').iterdir():
            if m.is_dir():
                n += _copy_tree(m, data / m.name)
    return n
