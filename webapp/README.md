# Web UI manual: train on a Colab GPU from your browser

The web UI lets you start, watch and stop Lao OCR training from a browser on your own computer.
The training itself runs on a Google Colab GPU.

```
Your computer                                   Google Colab (GPU)
┌──────────────────────────┐                    ┌──────────────────────────────┐
│ Browser → localhost:3000 │                    │ FastAPI server (port 8000)   │
│ Nuxt UI + Nitro proxy    │ ── HTTPS tunnel ─► │  runs torch_train.py training│
│ (keeps URL + token)      │   (Cloudflare)     │  reads/writes Google Drive   │
└──────────────────────────┘                    └──────────────────────────────┘
```

| Part | Runs on | Job |
|---|---|---|
| `colab_server.ipynb` | Colab | Mounts Drive, extracts `data/` from your zip, starts the server and the tunnel, prints **URL** + **TOKEN** |
| `colab_setup.py` | Colab | Used by the notebook: extracts the zip, copies results to Drive and back |
| `server/` | Colab | Starts and stops `torch_train.py training`, streams its output, serves `log.csv` to the charts |
| `ui/` | your computer | Dashboard, Train form, live loss / CER / CoER charts, log viewer |

Contents: [Requirements](#1-requirements) · [One-time setup](#2-one-time-setup) · [Every session](#3-every-session) ·
[Train and monitor](#4-train-and-monitor) · [Stop](#5-stop-and-shut-down) · [Where files go](#6-where-files-go) ·
[Troubleshooting](#7-troubleshooting) · [Develop without a GPU](#8-develop-without-a-gpu) · [Security](#9-security)

---

## 1. Requirements

| Where | What |
|---|---|
| Google | A Google account with Drive. Colab free tier (T4) works; Colab Pro gives longer sessions |
| Your computer | [Node.js](https://nodejs.org) **20 or newer** (`node -v`) and npm |
| Code | This repository pushed to GitHub, **including the `webapp/` folder**: the notebook clones it from there |
| Data | A zip on Drive that contains a `data/` folder with the ground truth (`.lstmf`, `.gt.txt`, `.png`) and models |

No Python is needed on your computer. The UI never touches the GPU.

---

## 2. One-time setup

### 2.1 Data on Google Drive

The notebook reads the data straight from your zip on Drive. By default:

```
MyDrive/Niigata_University/Master/2026/Compared/Update_50k_iters/tesstrain-gpu-20260714.zip
└── tesstrain-gpu/                 (any top folder name works)
    ├── torch_train.py, ...        ← code in the zip is NOT used (it is older and has no webapp/)
    └── data/                      ← this folder is extracted
        ├── lao-ground-truth/      ← datasets: *.lstmf (+ *.gt.txt, *.png)
        └── <model_name>/          ← models: *.pt, charset.json
```

- The notebook finds the top-most `data/` folder in the zip by itself.
- The zip is extracted to Colab's **local disk** (`/content/work/data`), not to Drive.
  Training reads thousands of small files, and the local disk is much faster than Drive.
- The zip stays unchanged on Drive. You don't need to unzip anything yourself.
- Any subfolder of `data/` that contains `.lstmf` or `.gt.txt` files appears in the **Ground truth** list
  on the Train page, and every `.pt` file in `data/*/` appears in the **Checkpoint** list.

**Disk space.** Free Colab has roughly 70–100 GB of local disk. The zip is 20 GB, and the notebook checks
before extracting that there is room. If there isn't, set `DATASETS` in the notebook's **Settings** cell to only
the folders you need, for example `DATASETS = ['lao-ground-truth', 'LaoCRNN-I64K5C16']`.

**Google account.** In Colab, mount the Google account that owns the zip (the one whose *My Drive* contains
`Niigata_University/`). If the zip is somewhere else, change `ZIP_PATH` in the **Settings** cell. If the
path is wrong, the notebook prints the files it does find in that folder.

### Results are kept on Drive

The local disk is erased when the Colab session ends, so the notebook copies results back to Drive
**every 2 minutes**, and once more when you run **Stop everything**:

```
MyDrive/Niigata_University/Master/2026/Compared/webui_runs/   (RESULTS_DIR)
├── .webui_token               ← the UI token, same every session
├── web_jobs/<job_id>/         ← job record, log.csv, stdout.log
└── models/<model_name>/       ← checkpoints of models trained from the web UI
```

At the start of the next session, the notebook copies these back into `data/`. So your job history,
and checkpoints you can fine-tune from, survive across sessions. Only models trained from the web UI are
copied back to Drive. The original models from the zip are not copied again, since they're already in the zip.

### 2.2 Push the code

The notebook clones `https://github.com/Monaliza14xx/Lao_OCR_Toolkit.git`, branch `main`.
Push your current code so `webapp/` is on GitHub. For another fork or branch, edit `REPO_URL` / `BRANCH` in the notebook.

### 2.3 Install the UI on your computer

```bash
cd webapp/ui
```

```bash
npm install
```

You only do this once (and again after pulling changes to `package.json`).

---

## 3. Every session

### 3.1 Start the server on Colab

1. Open `webapp/colab_server.ipynb` in Colab
   (File → Open notebook → GitHub, or upload it from your computer).
2. **Runtime → Change runtime type → T4 GPU** (or any GPU) → Save.
3. Run the cells from the top, in order:

| Cell | What it does | Notes |
|---|---|---|
| `nvidia-smi` | Shows the GPU and free local disk | If it fails, the runtime has no GPU: go back to step 2 |
| **1. Settings** | `ZIP_PATH`, `DATASETS`, `RESULTS_DIR`, repo URL and branch | Edit only if your files are elsewhere |
| **2. Drive, code** | Mounts Drive, checks the zip exists, clones or updates the repo in `/content/Lao_OCR_Toolkit` | Approve the Drive pop-up with the account that owns the zip |
| Dependencies | Installs Python packages, `unzip` and `cloudflared` | 1–2 minutes |
| **3. Extract data** | Extracts `data/` from the zip to `/content/work/data`, then restores earlier results from `RESULTS_DIR` | **10–20 minutes** for 20 GB, once per session; re-running skips it |
| **4. Start the server** | Starts the server, the tunnel and the 2-minute Drive sync, prints the URL and token | See below |
| **5. Keep alive** | Prints GPU, job and last-sync status every minute | Leave it running while you train |
| **6. Stop everything** | Final copy to Drive, then stops the server and tunnel | Run it before you close Colab |

The server cell ends with something like:

```
============================================================
Paste these into the web UI → Settings
URL:   https://quiet-river-example.trycloudflare.com
TOKEN: 3pF...k2Q
============================================================
```

- The **URL** is new every session.
- The **TOKEN** is saved in `RESULTS_DIR/.webui_token` on Drive and stays the same across sessions.

### 3.2 Start the UI on your computer

```bash
cd webapp/ui && npm run dev
```

Open http://localhost:3000.

### 3.3 Connect

1. Go to **Settings**.
2. Paste the **Server URL**.
3. Paste the **Token** (only the first time; afterwards leave it empty and the saved one is kept).
4. Click **Save and connect**.

The badge in the top right turns green and shows the GPU name (for example *Tesla T4*).
The settings are stored on your computer in `webapp/ui/.data/`, not in the browser.

**Next session:** start the notebook again, then in Settings paste only the new URL and click **Save and connect**.

---

## 4. Train and monitor

### 4.1 Start a job

Open **Train**.

1. Optional: click a preset to fill the form.

   | Preset | Use it for |
   |---|---|
   | **From scratch** | Default network, Adam, learning rate 0.002 |
   | **Fine-tune** | Continue from a `.pt` checkpoint with a lower learning rate (0.0005) |
   | **Thesis E1** | Replicates the thesis setting: `Ct5,5,32`, height 64, batch 64, 50,000 iterations |

2. **Model and data**
   - **Model name**: checkpoints go to `data/<model name>/`. Re-using a name overwrites that model's checkpoints.
   - **Ground truth**: pick a dataset folder, or *Auto* (`data/<model name>-ground-truth`).
   - **Network (VGSL)**: the architecture string, e.g. `[1,48,0,1 Ct3,3,16 Mp3,3 Lfys64 Lfx96 Lrx96 Lfx384 O1c###]`.
   - **Start from**: *Scratch*, or *Checkpoint (.pt)* to fine-tune; the list shows every `.pt` under `data/*/`.

3. **Optimisation and stopping**: optimizer, learning rate, batch size, target CER (stops early below it),
   iterations or epochs, how often to log, how often to save a checkpoint.
   On Colab, keep **Checkpoint every N steps** low (e.g. 1000) so a disconnect loses little.

4. **Advanced** (optional): weight decay, train/eval ratio, random seed, DataLoader workers,
   *Require GPU* (on by default: the job fails instead of silently training on CPU).

5. Click **Start training**. The page switches to the job.

Only **one job runs at a time** (one GPU). *New training job* is disabled while a job is running.

The form remembers your last settings in this browser.

### 4.2 Watch a job

The job page updates every few seconds:

| Panel | Shows |
|---|---|
| Header | Status (`running`, `stopping`, `stopped`, `succeeded`, `failed`, `lost`), start time, duration, exit code |
| Stat cards | Iteration and progress, latest train CER, best eval CER (and at which iteration), samples/s, learning rate, GPU memory |
| **Loss** | Train and eval loss per iteration (log scale once it spans a decade) |
| **Error rate** | Train CER, eval CER, eval WER |
| **CoER** | Component error rate per Lao character group at the latest eval step; tone, upper and lower vowels (the stacked marks) in green |
| **Output** | Live console output of `torch_train.py`; *Show command* shows the exact command line |

Eval values appear only on eval steps, so the eval lines have fewer points than the train lines.

The **Dashboard** lists all jobs, newest first; click a row to open it.
You can close the browser tab while training: the job keeps running on Colab. Reopen the page to continue watching.

---

## 5. Stop and shut down

- **Stop a job:** job page → **Stop** → confirm. The server sends SIGTERM, then kills the process after 10 s.
  Status becomes `stopped`. Checkpoints already saved are kept.
- **Stop the UI:** `Ctrl+C` in the terminal running `npm run dev`.
- **Stop the server:** in Colab, interrupt the keep-alive cell and run **6. Stop everything**.
  It copies the latest results to Drive first. If you disconnect the runtime without it, you lose at most
  the last 2 minutes of results.

---

## 6. Where files go

During a session, all paths are inside `DATA_DIR` on the Colab local disk (`/content/work/data/`).
Web jobs and their model folders are copied to `RESULTS_DIR` on Drive (`web_jobs/` and `models/<model_name>/`):

| Path | Contents |
|---|---|
| `<model_name>/<model_name>_best.pt` | Best checkpoint (by eval CER) |
| `<model_name>/<model_name>_checkpoint_<N>.pt` | Periodic checkpoints |
| `<model_name>/<model_name>_final.pt` | Last weights when the run ends normally |
| `<model_name>/charset.json` | Character set used by the model |
| `web_jobs/<job_id>/job.json` | Job record: status, settings, command, times |
| `web_jobs/<job_id>/log.csv` | Metrics log (same columns as the CLI `training.log`), one per job |
| `web_jobs/<job_id>/stdout.log` | Full console output |

Each job writes its own `log.csv`, so running the same model name again does not erase earlier logs
(the checkpoints in `<model_name>/` are overwritten, though).

The CLI tools work on these files as usual, for example
`python torch_train.py plot --model_name lao --log_file data/web_jobs/<job_id>/log.csv`.

---

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Badge says **offline**, "Colab server not configured" | Settings not saved yet: paste URL and token, *Save and connect* |
| **offline**, 502 / fetch failed | The Colab runtime stopped or the URL is from an old session. Re-run the server cell and paste the new URL |
| **401 missing or wrong bearer token** | Token mismatch: copy the TOKEN from the notebook again and paste it in Settings |
| Badge shows **connected, no GPU** | Colab runtime has no GPU. Change runtime type to GPU and re-run all cells. With *Require GPU* on, jobs fail immediately |
| Ground truth list is empty | The folders under `data/` have no `.lstmf` / `.gt.txt`, or `DATASETS` left them out. Check the *Data dir* on the Settings page and the `data/ contains:` line of the extract cell |
| `FileNotFoundError` for the zip | Wrong `ZIP_PATH`, or Drive is mounted with another Google account. The cell lists the files it does see in that folder |
| `Not enough local disk` | Set `DATASETS` to only the folders you need and re-run the extract cell |
| Extract cell is slow | Normal: 20 GB is read from Drive once per session. Re-running the cell in the same session skips it |
| Checkpoints missing in the next session | The runtime was deleted less than 2 minutes after they were written, or **Stop everything** was not run. Look in `RESULTS_DIR/models/` |
| Job **failed** right away | Read the **Output** panel. Common: `No .lstmf files` (wrong dataset), CUDA out of memory (lower batch size), wrong VGSL string |
| "job … is already running" | One job at a time. Stop it first, or wait |
| Job shows **lost** | The server restarted (Colab disconnected) while the job ran. Use the last `.pt` in `data/<model_name>/` with **Fine-tune** to continue |
| Server cell: `URL: not found` | `cloudflared` did not get a tunnel. Look at `/content/tunnel.log`, then re-run the cell |
| Server cell raises an error | The server crashed on start; the error printed is `/content/server.log` |
| Charts stay at "Waiting for the first log rows" | Normal for the first *Log every N steps* iterations; first rows appear after N steps |
| Colab disconnects after a while | Free Colab stops idle sessions (~90 min) and caps sessions (~12 h). Keep the tab open, keep the keep-alive cell running, save checkpoints often |

---

## 8. Develop without a GPU

Run the server on your computer with `server/fake_trainer.py`, which writes a realistic `log.csv` without
training anything. From the repository root (Python with `pip install -r webapp/server/requirements.txt`):

```bash
LAOOCR_TOKEN=dev-token LAOOCR_TRAIN_SCRIPT=webapp/server/fake_trainer.py python -m uvicorn webapp.server.app:create_app --factory --port 8000
```

Then `npm run dev` in `webapp/ui` and use `http://127.0.0.1:8000` with token `dev-token` in Settings.
Use `LAOOCR_DATA=/some/folder` to choose the data directory (default `data/`).

Server tests:

```bash
pytest tests/test_webapp_server.py
```

Server environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `LAOOCR_TOKEN` | generated and printed | Bearer token required on every request |
| `LAOOCR_DATA` | `<repo>/data` | Data directory; all paths must be inside it |
| `LAOOCR_TRAIN_SCRIPT` | `<repo>/torch_train.py` | Training entry point |

UI environment variables (optional, Settings page overrides them): `NUXT_COLAB_URL`, `NUXT_COLAB_TOKEN`.

---

## 9. Security

The tunnel URL is public on the internet and the server runs training code on your GPU, so:

- Every request needs the bearer token. Don't share it or commit it.
  To get a new one, delete `RESULTS_DIR/.webui_token` and re-run the server cell.
- Only known training options are accepted (no shell commands), and every path must be inside the data directory.
- The browser never sees the URL or token: the local Nuxt server adds them when it forwards requests.

Not included in the web UI yet: recognize, evaluation, ONNX export, model management, resuming a run.
Use the CLI for these. See [DEV_PLAN.md](DEV_PLAN.md) for the roadmap.
