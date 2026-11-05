"""Training job runner: one `torch_train.py training` subprocess at a time.

Each job lives in DATA_DIR/web_jobs/<id>/ with:
  job.json    metadata (status, config, argv, times)
  stdout.log  merged stdout/stderr of the process
  log.csv     the TrainingLogger CSV (one per job, so reruns never overwrite history)
"""
from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

ACTIVE = {'running', 'stopping'}


class JobConfig(BaseModel):
    """Whitelisted `torch_train.py training` options. Nothing else reaches the command line."""
    model_name: str = Field(pattern=r'^[A-Za-z0-9_.-]{1,64}$')
    net_spec: str = Field('[1,48,0,1 Ct3,3,16 Mp3,3 Lfys64 Lfx96 Lrx96 Lfx384 O1c###]',
                          pattern=r'^\[[0-9A-Za-z,#. ()\-]+\]$', max_length=400)
    ground_truth_dir: str = ''
    start_model: str = ''
    optimizer: Literal['Adam', 'SGD'] = 'Adam'
    learning_rate: float = Field(0.002, gt=0, le=1)
    momentum: float = Field(0.9, ge=0, le=1)
    weight_decay: float = Field(0.0, ge=0, le=1)
    batch_size: int = Field(64, ge=1, le=4096)
    max_iterations: int = Field(10000, ge=1, le=10_000_000)
    epochs: int = Field(0, ge=0, le=100_000)
    target_error_rate: float = Field(0.01, ge=0, le=1)
    debug_interval: int = Field(100, ge=1, le=1_000_000)
    save_interval: int = Field(2000, ge=1, le=10_000_000)
    random_seed: int = Field(0, ge=0)
    ratio_train: float = Field(0.9, gt=0, lt=1)
    num_workers: int = Field(2, ge=0, le=32)
    force_gpu: bool = False


class JobError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _num(v: str):
    if v == '' or v is None:
        return None
    try:
        f = float(v)
    except ValueError:
        return v
    return int(f) if f.is_integer() and 'e' not in v.lower() and '.' not in v else f


class JobManager:
    def __init__(self, data_dir: Path, repo_dir: Path, train_script: Path):
        self.data_dir = data_dir.resolve()
        self.repo_dir = repo_dir
        self.train_script = train_script
        self.jobs_dir = self.data_dir / 'web_jobs'
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._procs: dict[str, subprocess.Popen] = {}
        self._recover()

    # ---------- paths ----------
    def inside_data(self, path: str) -> Path:
        """Resolve a user path relative to DATA_DIR and refuse anything outside it."""
        p = Path(path)
        p = (p if p.is_absolute() else self.data_dir / p).resolve()
        if p != self.data_dir and self.data_dir not in p.parents:
            raise JobError(400, f'path is outside the data directory: {path}')
        return p

    def _dir(self, job_id: str) -> Path:
        if not job_id.isalnum():
            raise JobError(404, 'job not found')
        d = self.jobs_dir / job_id
        if not (d / 'job.json').exists():
            raise JobError(404, 'job not found')
        return d

    # ---------- records ----------
    def _write(self, job: dict) -> None:
        d = self.jobs_dir / job['id']
        tmp = d / 'job.json.tmp'
        tmp.write_text(json.dumps(job, indent=2, ensure_ascii=False))
        tmp.replace(d / 'job.json')

    def get(self, job_id: str) -> dict:
        return json.loads((self._dir(job_id) / 'job.json').read_text())

    def list(self) -> list[dict]:
        jobs = []
        for f in self.jobs_dir.glob('*/job.json'):
            try:
                jobs.append(json.loads(f.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(jobs, key=lambda j: j['created_at'], reverse=True)

    def running(self) -> Optional[dict]:
        return next((j for j in self.list() if j['status'] in ACTIVE), None)

    def _recover(self) -> None:
        """After a server restart, jobs whose process is gone are 'lost'; live ones are polled."""
        for job in self.list():
            if job['status'] not in ACTIVE:
                continue
            if _pid_alive(job.get('pid')):
                threading.Thread(target=self._poll_orphan, args=(job['id'], job['pid']), daemon=True).start()
            else:
                job.update(status='lost', finished_at=_now())
                self._write(job)

    # ---------- lifecycle ----------
    def build_argv(self, cfg: JobConfig, log_csv: Path) -> list[str]:
        argv = [sys.executable, '-u', str(self.train_script), 'training',
                '--model_name', cfg.model_name,
                '--data_dir', str(self.data_dir),
                '--log_file', str(log_csv),
                '--net_spec', cfg.net_spec,
                '--optimizer', cfg.optimizer,
                '--learning_rate', str(cfg.learning_rate),
                '--momentum', str(cfg.momentum),
                '--weight_decay', str(cfg.weight_decay),
                '--batch_size', str(cfg.batch_size),
                '--max_iterations', str(cfg.max_iterations),
                '--epochs', str(cfg.epochs),
                '--target_error_rate', str(cfg.target_error_rate),
                '--debug_interval', str(cfg.debug_interval),
                '--save_interval', str(cfg.save_interval),
                '--random_seed', str(cfg.random_seed),
                '--ratio_train', str(cfg.ratio_train),
                '--num_workers', str(cfg.num_workers)]
        for flag, value in (('--ground_truth_dir', cfg.ground_truth_dir),
                            ('--start_model', cfg.start_model)):
            if value:
                argv += [flag, str(self.inside_data(value))]
        if cfg.force_gpu:
            argv.append('--force_GPU')
        return argv

    def start(self, cfg: JobConfig) -> dict:
        with self._lock:
            busy = self.running()
            if busy:
                raise JobError(409, f"job {busy['id']} is already running; stop it first")
            job_id = datetime.now().strftime('%Y%m%d%H%M%S') + uuid.uuid4().hex[:6]
            d = self.jobs_dir / job_id
            d.mkdir()
            argv = self.build_argv(cfg, d / 'log.csv')
            job = {'id': job_id, 'status': 'running', 'config': cfg.model_dump(), 'argv': argv,
                   'output_dir': str(self.data_dir / cfg.model_name),
                   'created_at': _now(), 'started_at': _now(), 'finished_at': None,
                   'return_code': None, 'pid': None}
            out = open(d / 'stdout.log', 'wb')
            try:
                proc = subprocess.Popen(argv, cwd=self.repo_dir, stdout=out, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, start_new_session=True,
                                        env={**os.environ, 'PYTHONUNBUFFERED': '1'})
            except OSError as e:
                out.close()
                job.update(status='failed', finished_at=_now())
                self._write(job)
                raise JobError(500, f'could not start training: {e}') from e
            out.close()
            job['pid'] = proc.pid
            self._procs[job_id] = proc
            self._write(job)
        threading.Thread(target=self._wait, args=(job_id, proc), daemon=True).start()
        return job

    def _finish(self, job_id: str, return_code: Optional[int]) -> None:
        with self._lock:
            job = self.get(job_id)
            if job['status'] == 'stopping':
                job['status'] = 'stopped'
            elif return_code is None:
                job['status'] = 'exited'
            else:
                job['status'] = 'succeeded' if return_code == 0 else 'failed'
            job.update(return_code=return_code, finished_at=_now())
            self._write(job)
            self._procs.pop(job_id, None)

    def _wait(self, job_id: str, proc: subprocess.Popen) -> None:
        self._finish(job_id, proc.wait())

    def _poll_orphan(self, job_id: str, pid: int) -> None:
        while _pid_alive(pid):
            time.sleep(2)
        self._finish(job_id, None)

    def stop(self, job_id: str, grace: float = 10.0) -> dict:
        with self._lock:
            job = self.get(job_id)
            if job['status'] not in ACTIVE:
                raise JobError(409, f"job is {job['status']}, not running")
            job['status'] = 'stopping'
            self._write(job)
            pid = job['pid']
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return self.get(job_id)
        threading.Thread(target=self._kill_later, args=(pid, grace), daemon=True).start()
        return self.get(job_id)

    @staticmethod
    def _kill_later(pid: int, grace: float) -> None:
        deadline = time.time() + grace
        while time.time() < deadline:
            if not _pid_alive(pid):
                return
            time.sleep(0.2)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    # ---------- outputs ----------
    def stdout_path(self, job_id: str) -> Path:
        return self._dir(job_id) / 'stdout.log'

    def metrics(self, job_id: str, since: int = -1) -> dict:
        f = self._dir(job_id) / 'log.csv'
        if not f.exists():
            return {'columns': [], 'rows': []}
        with open(f, newline='') as fh:
            reader = csv.DictReader(fh)
            rows = []
            for row in reader:
                r = {k: (v if k in ('timestamp', 'coer_source') else _num(v)) for k, v in row.items() if k}
                if isinstance(r.get('iteration'), int) and r['iteration'] > since:
                    rows.append(r)
            return {'columns': reader.fieldnames or [], 'rows': rows}
