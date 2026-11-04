"""Lao OCR training server. Runs next to the GPU (e.g. on Colab) and is reached through a tunnel.

Run from the repository root:
    LAOOCR_TOKEN=... python -m uvicorn webapp.server.app:create_app --factory --port 8000

Environment:
    LAOOCR_TOKEN         bearer token every request must carry (generated and printed if unset)
    LAOOCR_DATA          data directory (default: <repo>/data); all paths are confined to it
    LAOOCR_TRAIN_SCRIPT  training entry point (default: <repo>/torch_train.py)
"""
from __future__ import annotations

import asyncio
import os
import secrets
import shutil
import threading
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .jobs import ACTIVE, JobConfig, JobError, JobManager

REPO = Path(__file__).resolve().parents[2]
VERSION = '0.1.0'
TAIL_BYTES = 64 * 1024


def _gpu_info() -> dict:
    try:
        import torch
    except ImportError:
        return {'available': False, 'reason': 'torch not installed'}
    if not torch.cuda.is_available():
        return {'available': False, 'reason': 'CUDA not available'}
    free, total = torch.cuda.mem_get_info()
    return {'available': True, 'name': torch.cuda.get_device_name(0),
            'memory_total_gb': round(total / 1e9, 2), 'memory_free_gb': round(free / 1e9, 2),
            'cuda': torch.version.cuda, 'torch': torch.__version__}


def _count_files(root: Path, limit: int = 1_000_000) -> dict:
    counts = {'lstmf': 0, 'gt_txt': 0, 'png': 0}
    seen = 0
    for _, _, files in os.walk(root):
        for name in files:
            seen += 1
            if name.endswith('.lstmf'):
                counts['lstmf'] += 1
            elif name.endswith('.gt.txt'):
                counts['gt_txt'] += 1
            elif name.endswith(('.png', '.tif')):
                counts['png'] += 1
        if seen > limit:
            break
    return counts


def create_app(data_dir: Optional[str] = None, token: Optional[str] = None,
               train_script: Optional[str] = None) -> FastAPI:
    data = Path(data_dir or os.environ.get('LAOOCR_DATA') or REPO / 'data')
    data.mkdir(parents=True, exist_ok=True)
    token = token or os.environ.get('LAOOCR_TOKEN')
    if not token:
        token = secrets.token_urlsafe(24)
        print(f'LAOOCR_TOKEN not set; generated token: {token}', flush=True)
    script = Path(train_script or os.environ.get('LAOOCR_TRAIN_SCRIPT') or REPO / 'torch_train.py')
    jobs = JobManager(data, REPO, script)
    # Importing torch takes several seconds; do it now so the first /health answers quickly.
    threading.Thread(target=_gpu_info, daemon=True).start()

    def auth(authorization: str = Header('')) -> None:
        scheme, _, value = authorization.partition(' ')
        if scheme.lower() != 'bearer' or not secrets.compare_digest(value.encode(), token.encode()):
            raise HTTPException(401, 'missing or wrong bearer token')

    app = FastAPI(title='Lao OCR training server', version=VERSION, dependencies=[Depends(auth)])
    app.state.jobs = jobs

    @app.exception_handler(JobError)
    async def _job_error(_: Request, e: JobError):
        return JSONResponse({'detail': e.detail}, status_code=e.status)

    @app.get('/health')
    def health():
        usage = shutil.disk_usage(jobs.data_dir)
        running = jobs.running()
        return {'ok': True, 'version': VERSION, 'gpu': _gpu_info(),
                'data_dir': str(jobs.data_dir), 'disk_free_gb': round(usage.free / 1e9, 1),
                'running_job': running['id'] if running else None}

    @app.get('/datasets')
    def datasets():
        out = []
        for d in sorted(jobs.data_dir.iterdir()):
            if not d.is_dir() or d.name.startswith('.') or d.name == 'web_jobs':
                continue
            counts = _count_files(d)
            if counts['lstmf'] or counts['gt_txt']:
                out.append({'name': d.name, 'path': d.name, **counts})
        return out

    @app.get('/models')
    def models():
        out = []
        for f in sorted(jobs.data_dir.glob('*/*.pt'), key=lambda f: f.stat().st_mtime, reverse=True):
            st = f.stat()
            out.append({'name': f.stem, 'model': f.parent.name, 'path': str(f.relative_to(jobs.data_dir)),
                        'size_mb': round(st.st_size / 1e6, 2), 'modified': int(st.st_mtime)})
        return out

    @app.get('/jobs')
    def list_jobs():
        return jobs.list()

    @app.post('/jobs', status_code=201)
    def start_job(cfg: JobConfig):
        return jobs.start(cfg)

    @app.get('/jobs/{job_id}')
    def get_job(job_id: str):
        return jobs.get(job_id)

    @app.post('/jobs/{job_id}/stop')
    def stop_job(job_id: str):
        return jobs.stop(job_id)

    @app.get('/jobs/{job_id}/metrics')
    def job_metrics(job_id: str, since: int = Query(-1)):
        return jobs.metrics(job_id, since)

    @app.get('/jobs/{job_id}/logs')
    async def job_logs(job_id: str, request: Request, offset: Optional[int] = Query(None, ge=0),
                       last_event_id: Optional[str] = Header(None)):
        """Server-sent events: one `data:` per output line, `id:` is the byte offset to resume from.

        Browsers reconnect with a Last-Event-ID header, so a dropped tunnel resumes without duplicates.
        """
        path = jobs.stdout_path(job_id)
        if offset is None and last_event_id and last_event_id.isdigit():
            offset = int(last_event_id)

        async def stream():
            size = path.stat().st_size if path.exists() else 0
            pos = offset if offset is not None else max(0, size - TAIL_BYTES)
            buf = b''
            idle = 0.0
            yield 'retry: 2000\n\n'
            while True:
                if await request.is_disconnected():
                    return
                chunk = b''
                if path.exists():
                    with open(path, 'rb') as f:
                        f.seek(pos)
                        chunk = f.read(256 * 1024)
                if chunk:
                    pos += len(chunk)
                    buf += chunk.replace(b'\r', b'\n')
                    *lines, buf = buf.split(b'\n')
                    for line in lines:
                        yield f'id: {pos - len(buf)}\ndata: {line.decode("utf-8", "replace")}\n\n'
                    idle = 0.0
                    continue
                status = jobs.get(job_id)['status']
                if status not in ACTIVE:
                    if buf:
                        yield f'id: {pos}\ndata: {buf.decode("utf-8", "replace")}\n\n'
                    yield f'event: end\ndata: {status}\n\n'
                    return
                await asyncio.sleep(0.5)
                idle += 0.5
                if idle >= 15:
                    yield ': ping\n\n'
                    idle = 0.0

        return StreamingResponse(stream(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    return app
