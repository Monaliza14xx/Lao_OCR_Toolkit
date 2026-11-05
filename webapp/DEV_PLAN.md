# Web UI for Lao OCR Training — Development Plan

**Status:** v0.2 (2026-11-04). M0–M3 built and tested locally with `server/fake_trainer.py`.
The Colab notebook and a real GPU run have not been tested yet.

## 1. Goal

Control GPU training of the Lao OCR models from a browser on the local machine, while all heavy work runs on a Google Colab GPU.

- Start / stop / monitor training jobs with live loss and CER charts.
- Run OCR (recognize) on uploaded images with a chosen model.
- Evaluate models and download checkpoints / ONNX exports.

**Non-goals (v1):** multi-user accounts, a job queue across several GPUs, in-browser data labelling, replacing the CLI. The CLI stays the source of truth.

## 2. Architecture

```
Local machine                              Colab VM (GPU)
┌────────────────────────┐   HTTPS    ┌─────────────────────────────┐
│ Nuxt 4 (ssr:false)     │  tunnel    │ FastAPI  (webapp/server)    │
│  app/   pages, charts  │──────────► │  ├ job runner (subprocess)  │
│  server/ Nitro proxy   │  Bearer    │  ├ log/CSV tailer → SSE     │
│   holds URL + token    │  token     │  ├ recognize (model cache)  │
└────────────────────────┘            │  └ file/model endpoints     │
                                      │ torch_train.py (unchanged)  │
                                      │ Google Drive: data, models  │
                                      └─────────────────────────────┘
```

Key decisions

| Decision | Choice | Why |
|---|---|---|
| Frontend | Nuxt 4, Nuxt UI v3, `vue-chartjs` | Chosen by user; `app/` layout, `shared/` types |
| Browser → Colab | Via Nitro proxy (`server/api/colab/[...path].ts`) | No CORS; token + tunnel URL never reach the browser; URL changes every session |
| Tunnel | `cloudflared` quick tunnel (ngrok as fallback) | Free, no account, stable |
| Backend | FastAPI + uvicorn | Async, SSE, auto OpenAPI → generate TS types |
| Training | `subprocess.Popen(torch_train.py training …)` | Reuses CLI untouched; process isolation; killable |
| Metrics | Tail `data/<model>/log.csv` (`TrainingLogger.HEADER`) | Already has loss, CER, WER, CoER, lr, GPU mem, samples/s |
| Auth | Random bearer token generated in the notebook | Tunnel URL is public; server can run code on the GPU |
| Persistence | Everything under a Drive folder; job metadata in `jobs.json` | Colab VM is disposable |

## 3. API (Colab server)

| Method + path | Purpose | Backed by |
|---|---|---|
| `GET /health` | GPU name, free memory, version, running job | `torch.cuda` |
| `POST /jobs` | Start training from a validated config; returns `job_id` | `torch_train.py training` |
| `GET /jobs`, `GET /jobs/{id}` | List / detail (status, args, pid, times) | `jobs.json` |
| `POST /jobs/{id}/stop` | SIGTERM, then SIGKILL after timeout | `Popen` |
| `GET /jobs/{id}/logs` | SSE stream of stdout | process pipe → file tail |
| `GET /jobs/{id}/metrics?since=iter` | Incremental CSV rows as JSON | `log.csv` |
| `GET /models` | `.pt` checkpoints with net_spec, charset size, iteration | checkpoint metadata |
| `POST /recognize` | Image + model + psm + beam_width → text, per-line boxes | `recognize.py` (model cached in memory) |
| `POST /evaluate` | Run `evaluation`, return CER/WER/CoER | `evaluate.py` |
| `GET /models/{name}/download?format=pt\|onnx` | Download / export | `export.py` |
| `POST /datasets` / `GET /datasets` | Register Drive paths; zip upload for small sets | filesystem |

Config schema (`shared/types` in Nuxt and Pydantic model on server, kept in sync via OpenAPI): `model_name, net_spec, optimizer, learning_rate, batch_size, epochs | max_iterations, start_model, ground_truth_dir, save_interval, debug_interval, random_seed`.

Rules: whitelist argument names (never pass a raw command line), resolve all paths inside the Drive root, **one training job at a time** (single GPU), reject if a job is running.

## 4. Frontend (Nuxt 4)

```
webapp/ui/
  app/
    app.vue
    pages/  index.vue (dashboard) · train.vue · jobs/[id].vue · recognize.vue · models.vue · settings.vue
    components/  ConfigForm · MetricChart · LogViewer · CoerBars · LaoText
    composables/ useColab · useJobStream (EventSource) · useMetrics
  server/api/
    settings.post.ts · settings.get.ts       # URL + token, kept in server memory / .env
    colab/[...path].ts                       # proxy, pipes SSE with sendStream
  shared/  types.ts                          # JobConfig, MetricRow, Model
  nuxt.config.ts                             # ssr:false, modules: @nuxt/ui
```

Pages
1. **Settings** — paste tunnel URL + token, "Test connection" shows GPU name.
2. **Train** — config form with presets (scratch / fine-tune / thesis E1; Tesseract transfer is deliberately left out), VGSL string field with quick validation, dataset picker, Start.
3. **Job detail** — live chart (train/eval loss, CER, WER), CoER bars per Lao component, samples/s and GPU memory, streaming log, Stop button.
4. **Recognize** — drag-and-drop image, model + psm + beam width, shows text with Lao font (Phetsarath / Noto Sans Lao), copy button.
5. **Models** — table of checkpoints, evaluate, export ONNX, download.

## 5. Colab notebook (`colab_server.ipynb`)

1. Check GPU, mount Drive, clone repo, `pip install -r requirements.txt fastapi uvicorn sse-starlette python-multipart`.
2. Install `cloudflared`.
3. Generate token, start uvicorn in the background.
4. Start the tunnel, print `URL` + `TOKEN`, with a "paste into Settings" hint.
5. Keep-alive cell (heartbeat) and a final "stop everything" cell.

## 6. Milestones

| # | Milestone | Done when | Est. |
|---|---|---|---|
| M0 | Spike: FastAPI + tunnel on Colab, `curl /health` from laptop | GPU name returned over tunnel with token | 0.5 d |
| M1 | Job runner: start/stop/status, logs SSE, metrics endpoint; server tests with a fake training script | Real `training` run starts, streams, stops cleanly on Colab | 1.5 d |
| M2 | Nuxt skeleton: settings, proxy (incl. SSE), health indicator | Local UI shows Colab GPU name | 1 d |
| M3 | Train page + job detail with live charts | Launch a short job from UI, watch loss/CER update, stop it | 2 d |
| M4 | Recognize: model cache, upload, result view | Image → Lao text in under ~1 s after model load | 1 d |
| M5 | Models page: list, evaluate, ONNX export, download | Round trip works for one model | 1 d |
| M6 | Hardening: session-loss recovery, resume, error states, README + notebook polish | Kill the Colab runtime mid-job → UI shows "lost", resume works | 1.5 d |

Total ≈ 8–9 working days of effort; M0–M3 is the useful MVP.

## 7. Risks and open items found in the code

1. **Resume is not real yet.** `TrainingLogger.__init__` opens `log.csv` with mode `'w'`, so restarting a job erases the history. `_save` stores `iteration` and optimizer state, but the loader's use of them with `--start_model` needs checking. Needed for M6: append mode (or a per-run log file) and a `--resume` flag that restores iteration, optimizer and LR scheduler.
2. **Batch padding bug** (README Known issue #1): predictions change with batch size for right-to-left / attention layers. Recognize endpoint uses batch size 1 until fixed; show a warning for those specs.
3. **Colab limits:** idle timeout ~90 min, hard cap ~12 h on free tier, tunnel URL changes on every restart. Mitigation: all state on Drive, "reconnect" flow in Settings, `save_interval` default lowered for web runs.
4. **Security:** public tunnel + code execution. Token required on every route, arguments whitelisted, paths confined to the Drive root, no shell strings.
5. **Large data transfer:** don't upload gigabytes through the browser. Datasets live on Drive; synthetic generation (`gen_data_from_text.py`) can be a later job type.
6. **`log.csv` is written only every `debug_interval` steps** and the eval columns only on eval steps; the chart must handle empty cells.
7. **Colab terms of service:** personal use via tunnel is generally tolerated; if it becomes a problem, the same server runs unchanged on RunPod / Lambda / a lab GPU box.

## 8. Testing

- Server: pytest with a fake `torch_train.py` that emits a CSV on a timer (start/stop/stream/metrics, argument validation, path confinement, single-job rule). Runs in the existing CI without a GPU.
- Frontend: Vitest for composables, Playwright smoke test against a mock server.
- Manual: one 200-step real job on Colab, one recognize call, one ONNX export.

## 9. Open questions

1. Datasets: is everything already on Drive, or do you want upload from the UI?
2. Should the UI also trigger synthetic data generation and document segmentation, or only training / OCR?
3. Is the UI for your own use only (single user, token auth is enough)?
4. Keep `webapp/` in this repo (recommended, shares `tesstrain_core`) or a separate repo?
