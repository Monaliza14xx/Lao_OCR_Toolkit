"""Web server: auth, argument whitelisting, path confinement, job lifecycle, metrics, log stream."""
import os
import time

import pytest
from conftest import ROOT

pytest.importorskip('fastapi')
pytest.importorskip('httpx')
from fastapi.testclient import TestClient  # noqa: E402

from webapp.server.app import create_app  # noqa: E402

FAKE = os.path.join(ROOT, 'webapp', 'server', 'fake_trainer.py')
TOKEN = 'test-token'
AUTH = {'Authorization': f'Bearer {TOKEN}'}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('FAKE_TRAIN_STEP_SEC', '0.02')
    app = create_app(data_dir=str(tmp_path), token=TOKEN, train_script=FAKE)
    with TestClient(app) as c:
        yield c


def wait_status(client, job_id, statuses, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f'/jobs/{job_id}', headers=AUTH).json()
        if job['status'] in statuses:
            return job
        time.sleep(0.05)
    raise AssertionError(f'job stayed {job["status"]}')


def test_requires_token(client):
    assert client.get('/health').status_code == 401
    assert client.get('/health', headers={'Authorization': 'Bearer nope'}).status_code == 401
    r = client.get('/health', headers=AUTH)
    assert r.status_code == 200 and r.json()['ok'] is True


def test_rejects_bad_config(client):
    assert client.post('/jobs', json={'model_name': '../x'}, headers=AUTH).status_code == 422
    assert client.post('/jobs', json={'model_name': 'a', 'net_spec': '[1; rm -rf /]'},
                       headers=AUTH).status_code == 422
    r = client.post('/jobs', json={'model_name': 'a', 'start_model': '/etc/passwd'}, headers=AUTH)
    assert r.status_code == 400 and 'outside' in r.json()['detail']


def test_tesseract_transfer_not_exposed(client):
    r = client.post('/jobs', json={'model_name': 'lao', 'max_iterations': 20, 'debug_interval': 10,
                                   'start_tess_model': 'lao.traineddata'}, headers=AUTH)
    assert r.status_code == 201
    assert '--start_tess_model' not in r.json()['argv']
    wait_status(client, r.json()['id'], {'succeeded', 'failed'})


def test_job_runs_to_completion(client):
    r = client.post('/jobs', json={'model_name': 'lao', 'max_iterations': 500, 'debug_interval': 50},
                    headers=AUTH)
    assert r.status_code == 201
    job = r.json()
    assert '--log_file' in job['argv'] and job['argv'][3] == 'training'
    job = wait_status(client, job['id'], {'succeeded', 'failed'})
    assert job['status'] == 'succeeded' and job['return_code'] == 0

    m = client.get(f'/jobs/{job["id"]}/metrics', headers=AUTH).json()
    assert [r['iteration'] for r in m['rows']] == list(range(50, 501, 50))
    assert isinstance(m['rows'][0]['train_loss'], float) and m['rows'][0]['eval_cer'] is None
    later = client.get(f'/jobs/{job["id"]}/metrics?since=400', headers=AUTH).json()['rows']
    assert [r['iteration'] for r in later] == [450, 500]

    with client.stream('GET', f'/jobs/{job["id"]}/logs?offset=0', headers=AUTH) as s:
        body = ''.join(s.iter_text())
    assert 'data:   [   500]' in body and 'event: end\ndata: succeeded' in body


def test_one_job_at_a_time_and_stop(client):
    cfg = {'model_name': 'lao', 'max_iterations': 100000, 'debug_interval': 10}
    first = client.post('/jobs', json=cfg, headers=AUTH).json()
    assert client.post('/jobs', json=cfg, headers=AUTH).status_code == 409
    assert client.get('/health', headers=AUTH).json()['running_job'] == first['id']

    assert client.post(f'/jobs/{first["id"]}/stop', headers=AUTH).json()['status'] == 'stopping'
    assert wait_status(client, first['id'], {'stopped'})['status'] == 'stopped'
    assert client.post('/jobs', json=cfg, headers=AUTH).status_code == 201


def test_failed_job(client, monkeypatch):
    monkeypatch.setenv('FAKE_TRAIN_FAIL', '1')
    job = client.post('/jobs', json={'model_name': 'lao', 'max_iterations': 200, 'debug_interval': 20},
                      headers=AUTH).json()
    job = wait_status(client, job['id'], {'succeeded', 'failed'})
    assert job['status'] == 'failed' and job['return_code'] == 1


def test_lost_after_restart(tmp_path, client):
    job = client.post('/jobs', json={'model_name': 'lao', 'max_iterations': 100000, 'debug_interval': 10},
                      headers=AUTH).json()
    client.post(f'/jobs/{job["id"]}/stop', headers=AUTH)
    wait_status(client, job['id'], {'stopped'})
    # Simulate a server that died while the job was running and its process is gone.
    mgr = client.app.state.jobs
    rec = mgr.get(job['id'])
    rec.update(status='running', pid=2 ** 22 + 12345)
    mgr._write(rec)
    app2 = create_app(data_dir=str(tmp_path), token=TOKEN, train_script=FAKE)
    assert app2.state.jobs.get(job['id'])['status'] == 'lost'


def test_datasets_and_models(tmp_path, client):
    gt = tmp_path / 'lao-ground-truth'
    gt.mkdir()
    (gt / 'a.gt.txt').write_text('ສະບາຍດີ')
    (gt / 'a.lstmf').write_bytes(b'')
    (tmp_path / 'lao').mkdir()
    (tmp_path / 'lao' / 'lao_best.pt').write_bytes(b'0' * 10)
    ds = client.get('/datasets', headers=AUTH).json()
    assert ds == [{'name': 'lao-ground-truth', 'path': 'lao-ground-truth', 'lstmf': 1, 'gt_txt': 1, 'png': 0}]
    models = client.get('/models', headers=AUTH).json()
    assert models[0]['path'] == 'lao/lao_best.pt'
