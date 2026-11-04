# 🚀 Tesstrain-GPU

**Tesseract OCR Training Pipeline — GPU-Accelerated with PyTorch**

Train custom Tesseract OCR models 10–50× faster using GPU acceleration. Drop-in replacement for the original [tesstrain](https://github.com/tesseract-ocr/tesstrain) Makefile workflow.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **GPU Training** | 10–50× faster than Tesseract's `lstmtraining` |
| **PyTorch LSTM** | Standard PyTorch nn.LSTM with CTC loss |
| **Optimizers** | Adam and SGD + Momentum supported |
| **Beam Search + DAWG** | Dictionary-constrained decoding with word/number/punctuation DAWGs |
| **Language Model** | N-gram LM (char trigrams + word bigrams) for beam search |
| **Page Segmentation** | PSM 3, 6, 7, 13 for full-page and single-line OCR |
| **Unicode Recoder** | NFC/NFD normalization, ligature decomposition |
| **ONNX Export** | Cross-platform deployment with ONNX Runtime |
| **Synthetic Data** | Generate training images with GPU-accelerated augmentations + calibrated presets |
| **Real-World Calibration** | Analyze real scanned documents → auto-derive augmentation params to close the domain gap |
| **Character Analytics** | Analyze character frequencies, Unicode codepoints, and categories |
| **ICU Segmentation** | Accurate WER for non-space-delimited languages (Lao, Thai, CJK) |

---

## 📑 Table of Contents
- [✨ Features](#-features)
- [📦 Installation](#-installation)
- [🚀 Quick Start (5 minutes)](#-quick-start-5-minutes)
- [📝 Synthetic Data Generation](#-synthetic-data-generation)
  - [Augmentation Presets](#augmentation-presets---preset)
  - [Real-World Calibration Workflow](#real-world-calibration-workflow)
- [📖 Detailed Usage](#-detailed-usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Recognition (Inference)](#recognition-inference)
  - [Export](#export)
- [📁 Project Structure](#-project-structure)
- [🏗️ Pipeline Architecture](#️-pipeline-architecture)
- [🔧 Advanced Features](#-advanced-features)
- [⚡ Performance](#-performance)
- [🆚 Comparison with Tesseract](#-comparison-with-tesseract)
- [📊 Evaluation & Analysis Tools](#-evaluation--analysis-tools)
  - [tesseract_tools.py — Training Plots, Model Comparison & Timing](#tesseract_toolspy)
  - [tesstrain_core/compare.py — PyTorch Model Comparison](#tesstrain_corecomparepy)
  - [compare_cross.py — Original vs Fine-tuned Cross-Comparison](#compare_crosspy)
- [🐛 Troubleshooting](#-troubleshooting)
- [🛠️ Utilities](#️-utilities)

---

## 📦 Installation

### 1. Clone

```bash
git clone https://github.com/your-username/tesstrain-gpu.git
cd tesstrain-gpu
```

### 2. Install dependencies

```bash
# Core requirements
pip install -r requirements.txt

# For GPU training (choose your CUDA version):

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only (no GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Optional dependencies

```bash
# ONNX export & inference
pip install onnx onnxruntime       # CPU
pip install onnx onnxruntime-gpu   # GPU

# Training charts and live metrics
pip install matplotlib tensorboard

# ICU word segmentation (for Lao/Thai/CJK WER)
# Requires: apt install libicu-dev (Ubuntu) or brew install icu4c (macOS)
pip install PyICU
```

### 4. Verify installation

```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Google Colab (zero-install)

```python
# Everything is pre-installed on Colab!
!git clone https://github.com/your-username/tesstrain-gpu.git
%cd tesstrain-gpu

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")  # Tesla T4

# Start training
!python3 torch_train.py training --model_name lao --max_iterations 5000
```

---

## 🚀 Quick Start (5 minutes)

### Step 1: Prepare ground truth

There are three main ways to get training data:

#### Way 1: Real documents (Highest Accuracy)
Manually crop lines from real scanned documents and transcribe them.
Place your images (`.tif`/`.png`) and transcriptions (`.gt.txt`) in `data/<model_name>-ground-truth/`:
```
data/lao-ground-truth/
  ├── line_001.tif       # Grayscale line image
  ├── line_001.gt.txt    # Text file containing exactly: "ສະບາຍດີ"
```

#### Way 2: Synthetic generation from your own text (Fastest)
Render images directly from a plain text corpus using the included script. This is the fastest way to bootstrap a new language or fine-tune an existing model on your specific domain's vocabulary.
```bash
python3 gen_data_from_text.py \
    --text corpus.txt \
    --output_dir data/lao-ground-truth \
    --font "Noto Sans Lao" \
    --count 10000 \
    --augment
```
*(See the [Synthetic Data Generation](#synthetic-data-generation-gen_data_from_textpy) section for advanced options like multi-font, noise, and blur.)*

#### Way 3: Migrate from original Tesseract training
If you already have a `data/<model_name>-ground-truth/` folder containing `.tif` and `.gt.txt` pairs from using the original `tesstrain` Makefile, you can reuse it entirely without any changes!

### Step 2: Run the full pipeline

```bash
# 1. Build character set
python3 torch_train.py unicharset --model_name lao

# 2. Create file lists  (80% train / 10% val / 10% test)
python3 torch_train.py lists --model_name lao --ratio 80 10 10

# 3. Train on GPU
python3 torch_train.py training --model_name lao \
    --max_iterations 10000 --target_error_rate 0.01

# 4. Export for deployment
python3 torch_train.py export --model_name lao
```

That's it! Your model is ready at `data/lao/lao.onnx`.

---

## 📝 Synthetic Data Generation

You can generate training samples by rendering text using `gen_data_from_text.py`. Supports **two processing modes**:

| Mode | Flag | Best for | How it works |
|------|------|----------|-------------|
| **CPU** | `--device cpu` | Small datasets, many cores | Multiprocessing with `--workers` |
| **GPU** | `--device cuda` | Large datasets (50k+) | Batched augmentations on GPU with `--batch_size` |

**Example 1: GPU mode with `--preset` (recommended)**
```bash
python3 gen_data_from_text.py \
    --corpus /path/to/lao_corpus.txt \
    --output_dir data/lao-synth-ground-truth \
    --font font1.ttf font2.ttf font3.ttf \
    --count 50000 \
    --preset real-match \
    --font_size_min 28 --font_size_max 72 \
    --device cuda --batch_size 256 \
    --ext jpg --to_lstmf
```

**Example 2: GPU mode (manual augmentation params)**
```bash
python3 gen_data_from_text.py \
    --corpus /path/to/lao_corpus.txt \
    --output_dir data/lao-synth-ground-truth \
    --font font1.ttf font2.ttf font3.ttf \
    --count 140000 \
    --noise 0.3 --blur 0.3 --augment 0.5 \
    --font_size_min 28 --font_size_max 48 \
    --device cuda --batch_size 256 \
    --ext png --to_lstmf
```

**Example 3: CPU mode (alternative)**
```bash
python3 gen_data_from_text.py \
    --corpus /path/to/lao_corpus.txt \
    --output_dir data/lao-synth-ground-truth \
    --font /path/to/NotoSansLao.ttf \
    --count 10000 \
    --device cpu --workers 8 \
    --to_lstmf
```

**Example 4: Convert existing images to `.lstmf`**
```bash
python3 gen_data_from_text.py --convert_only --ext png \
    --output_dir data/lao140k-ground-truth
```

#### How it works:
1. **Text Rendering** (CPU): PIL renders text lines with random fonts/sizes.
2. **Augmentation** (CPU or GPU): Applies noise, blur, rotation, perspective, contrast, erosion/dilation, paper texture.
3. **Output**: Saves `.tif`/`.png`/`.jpg` + `.gt.txt` + optional `.lstmf`.

#### Output Example:
- `line_0001.png`: A 48px high image containing the text.
- `line_0001.gt.txt`: A text file containing exactly what is in the image.

---

### Augmentation Presets (`--preset`)

The `--preset` flag applies a named bundle of augmentation parameters calibrated for different use cases. Using a preset **overrides** the manual `--noise`/`--blur`/`--augment` flags and also enables the advanced real-world augmentation functions not available in manual mode.

| Preset | Description | Use When |
|--------|-------------|----------|
| `clean` | No augmentation — pure synthetic render | Ablation / baseline comparison |
| `scan-mild` | Light scan degradation | Good-quality scanned documents |
| `scan-heavy` | Heavy degradation: high noise, blur, JPEG | Aged or low-quality scans |
| `real-match` | **Calibrated from real document analysis** | When you have run `analyze_real_data.py` on your actual target documents |

```bash
# Baseline (no augmentation)
python3 gen_data_from_text.py --corpus corpus.txt --output_dir data/out \
    --font font.ttf --count 10000 --preset clean

# Calibrated to real Lao administrative documents (Phase 1 measurements)
python3 gen_data_from_text.py --corpus corpus.txt --output_dir data/out \
    --font font.ttf --count 50000 --preset real-match --device cuda --batch_size 256
```

#### Preset Parameter Reference

Each preset controls the same set of parameters. Below is the full breakdown showing exactly what each preset sets:

| Parameter | `clean` | `scan-mild` | `scan-heavy` | `real-match` | What it controls |
|-----------|---------|-------------|--------------|--------------|------------------|
| `noise_prob` | 0.0 | 0.50 | 0.85 | **0.70** | Probability a sample receives noise |
| `noise_intensity` | 0.0 | 0.025 | 0.070 | **0.046** | Noise σ as fraction of 255 (σ = value × 255) |
| `blur_prob` | 0.0 | 0.20 | 0.60 | **0.20** | Probability a sample gets blurred |
| `blur_radius` | 0.0 | 0.50 | 1.50 | **0.0** | Gaussian blur kernel radius |
| `augment_prob` | 0.0 | 0.40 | 0.70 | **0.60** | Probability of geometric transforms (rotation, scale, perspective) |
| `brightness_max` | 0.0 | 10.0 | 30.0 | **7.7** | Max brightness shift in pixel units (±) |
| `contrast_max` | 0.0 | 0.10 | 0.35 | **0.138** | Max contrast scale factor (±) |
| `morph_prob` | 0.0 | 0.15 | 0.40 | **0.30** | Erosion/dilation probability |
| `rotation_max_angle` | 0° | 2.0° | 5.0° | **3.96°** | Max random rotation angle (±) |
| `jpeg_quality_range` | — | 80–92 | 50–75 | **65–90** | JPEG re-encode quality range (lower = more artifacts) |
| `paper_texture_prob` | 0.0 | 0.40 | 0.70 | **0.50** | Probability of applying paper grain texture |
| `bg_mean` | 255 | 250 | 240 | **252.8** | Target background pixel mean (real paper vs pure white) |
| `bg_std` | 0.0 | 4.0 | 12.0 | **6.4** | Background pixel std (controls grain intensity) |
| `uneven_illumination_prob` | 0.0 | 0.20 | 0.50 | **0.30** | Probability of scanner light-falloff gradient |
| `colored_ink_prob` | 0.0 | 0.01 | 0.03 | **0.02** | Probability of converting ink to a non-black color |

> **`real-match` design rationale:** Every value is derived from `analyze_real_data.py` measurements on 50 real Lao administrative document lines:
> - `noise_intensity = 0.046` ← measured noise σ = 11.75 / 255
> - `blur_radius = 0.0` ← all 50 real images classified as **sharp** (Laplacian variance > 500)
> - `brightness_max = 7.7`, `contrast_max = 0.138` ← calibrated from Michelson contrast gap (real: 0.773 vs synthetic: 0.85)
> - `bg_mean = 252.8`, `bg_std = 6.4` ← measured from real paper background regions
> - `rotation_max_angle = 3.96°` ← 1.5× the observed skew range (±1.97°) for slight over-coverage
> - `colored_ink_prob = 0.02` ← 1/50 real lines had red ink

#### Augmentation Execution Order

Each preset runs augmentations in this fixed order (later steps depend on earlier ones):

```
1. Base augmentation  →  noise + blur + brightness/contrast + morph (CPU or GPU)
2. Paper texture      →  add_real_paper_texture()     (shifts background toward real paper mean)
3. Illumination       →  add_uneven_illumination()    (gradient overlay simulating scanner falloff)
4. Colored ink        →  add_colored_ink()            (replaces black ink with red/blue/dark-green)
5. JPEG artifacts     →  add_jpeg_artifacts()         (re-encode at random quality; applied 50% of the time)
6. Scan noise         →  add_scan_noise_profile()     (correlated sensor noise on top of JPEG)
7. Rotation           →  random_rotation()            (applied last so fill colour matches background)
```

#### Real-World Augmentation Functions

| Function | Preset(s) | Key Parameters | Simulates |
|----------|-----------|----------------|-----------|
| `add_jpeg_artifacts` | `scan-mild`, `scan-heavy`, `real-match` | `quality_range=(65,90)` | Scan JPEG compression: blocking, ringing, chroma artefacts |
| `add_uneven_illumination` | `scan-mild`, `scan-heavy`, `real-match` | `max_gradient=0.12` (12% brightness swing) | Scanner light falloff; edges darker/lighter than centre |
| `add_colored_ink` | `scan-mild`, `scan-heavy`, `real-match` | colors: `red`, `dark-red`, `blue` | Colored stamp ink, official document header color lines |
| `add_real_paper_texture` | `scan-mild`, `scan-heavy`, `real-match` | `bg_mean=252.8`, `bg_std=6.4` | Real paper grain (multi-scale Perlin) vs flat white (255) |
| `add_scan_noise_profile` | `scan-mild`, `scan-heavy`, `real-match` | `noise_sigma=σ`, 70% Gaussian + 30% correlated | Scanner sensor noise (row/column drift + pixel noise) |

#### Manual Augmentation Options

> Use these when you want fine-grained control without a preset.

| Option | Default | Description | Effect on Model |
|--------|---------|-------------|-----------------|
| `--noise` | 0.0 | Gaussian noise probability (0.0–1.0) | Robustness to scanned/faxed docs |
| `--noise_intensity` | 0.04 | Noise σ as fraction of 255 | Controls noise strength |
| `--blur` | 0.0 | Gaussian blur probability (0.0–1.0) | Out-of-focus camera robustness |
| `--blur_radius` | 0.9 | Blur kernel radius | Controls blur strength |
| `--augment` | 0.0 | Geometric augmentation probability | Skew, scale, perspective robustness |
| `--brightness_max` | 20.0 | Max brightness shift (pixel units ±) | Exposure variation |
| `--contrast_max` | 0.2 | Max contrast scale factor (±) | Contrast variation |
| `--morph_prob` | 0.15 | Erosion/dilation probability | Over/under-inked print simulation |
| `--font_size_min` | 28 | Minimum font size in pt | — |
| `--font_size_max` | 48 | Maximum font size in pt | Size diversity |

> **Tip:** Use `--font_size_min 28 --font_size_max 72` to cover header (large), body (medium), and label (small) line types found in real documents.

---

### Real-World Calibration Workflow

If you have a set of real scanned document line images (`.jpg` + `.gt.txt` pairs), you can automatically measure the domain gap and calibrate `gen_data_from_text.py` to close it.

#### Step 1 — Profile your real documents

```bash
cd /path/to/your-real-documents
python3 analyze_real_data.py --data_dir . --output_dir ./analysis_output
```

Outputs:
- `analysis_output/analysis_report.md` — full metrics table (contrast, noise σ, background, skew, char frequency, line-type distribution)
- `analysis_output/augmentation_params.json` — calibrated parameters ready to copy into a preset
- `analysis_output/full_analysis.json` — machine-readable full analysis
- `analysis_output/*.png` — distribution plots (dimensions, quality metrics, character frequency)

#### Step 2 — Enrich your training corpus

```bash
python3 build_augmented_corpus.py \
    --real_gt /path/to/your-real-documents \
    --base_corpus data/lao-corpus.txt \
    --output data/lao-corpus-enriched.txt \
    --n_oversample 5
```

- Merges real-world GT text into your base corpus
- Detects under-represented characters → oversamples lines containing them
- Outputs a `.meta.txt` file with per-character coverage report

#### Step 3 — Generate calibrated training data

```bash
python3 gen_data_from_text.py \
    --corpus data/lao-corpus-enriched.txt \
    --output_dir data/lao-augmented-v2 \
    --font tesstrain_core/font/Phetsarath-Regular.ttf \
           tesstrain_core/font/Phetsarath-Bold.ttf \
           tesstrain_core/font/phetsarath_ot.ttf \
    --font_size_min 28 --font_size_max 72 \
    --count 50000 \
    --preset real-match \
    --device cuda --batch_size 256 --ext jpg --to_lstmf
```

#### Step 4 — Validate the domain gap is closed

```bash
python3 validate_augmentation.py \
    --real_dir /path/to/your-real-documents \
    --synth_dir data/lao-augmented-v2 \
    --analysis_params /path/to/analysis_output/augmentation_params.json \
    --n_synth_sample 500
```

Outputs `data/lao-augmented-v2/validation/`:
- `validation_report.md` — KL divergence scores (target: all < 0.5) with pass/fail grade
- `comparison_gallery/` — side-by-side real vs. synthetic image pairs
- `kl_divergence.json` — machine-readable scores for CI integration

> **Recommended threshold:** KL divergence < 0.15 (🟢 Good) for background, contrast, and noise. Values > 0.5 indicate a significant remaining domain gap.


---

## 📖 Detailed Usage

### File Lists

```bash
# 3-way split: 80% train / 10% val / 10% test
python3 torch_train.py lists --model_name lao --ratio 80 10 10

# 2-way split: 90% train / 10% val
python3 torch_train.py lists --model_name lao --ratio 90 10

# Legacy: use a float fraction (default 0.90 train)
python3 torch_train.py lists --model_name lao --ratio_train 0.90
```

Output files created in `data/<model_name>/`:

| File | Purpose |
|------|---------|
| `list.train` | Paths to training `.lstmf` files |
| `list.val` | Paths to validation `.lstmf` files |
| `list.test` | Paths to test `.lstmf` files *(only with 3-way split)* |

### Training

```bash
python3 torch_train.py training --model_name lao \
    --net_spec "[1,48,0,1 Ct3,3,16 Mp3,3 Lfys64 Lfx96 Lrx96 Lfx512 O1c###]" \
    --optimizer Adam \
    --learning_rate 0.002 \
    --batch_size 64 \
    --max_iterations 10000 \
    --target_error_rate 0.01 \
    --debug_interval 100
```

#### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | *required* | Name for the model (e.g., `lao`, `eng`) |
| `--net_spec` | VGSL string | Network architecture in VGSL format |
| `--optimizer` | `Adam` | Optimizer choice (`Adam` or `SGD`) |
| `--learning_rate` | `0.002` | Learning rate |
| `--momentum` | `0.9` | Momentum value (if `--optimizer SGD`) |
| `--weight_decay` | `0.0` | Weight decay / L2 penalty |
| `--batch_size` | `64` | Training batch size |
| `--max_iterations` | `10000` | Maximum training steps |
| `--epochs` | — | Train for N epochs (alternative to iterations) |
| `--target_error_rate` | `0.01` | Stop early when CER reaches this value |
| `--debug_interval` | `100` | Print metrics every N iterations |
| `--start_model` | — | Path to checkpoint for fine-tuning |
| `--start_tess_model` | — | Path to Tesseract `.traineddata` or `.lstm` for transfer-learning warm-start |
| `--data_dir` | `data` | Root data directory |
| `--tensorboard` | `True` | Log live metrics to TensorBoard |
| `--tensorboard_dir` | `data/.../tensorboard` | Custom TensorBoard log directory |

#### Live Tracking & Visualizations

**TensorBoard (Live)**
Training metrics (Loss, CER, WER, Learning Rate) are automatically logged to a TensorBoard directory (`data/<model>/tensorboard/`) in real time.
```bash
# In a new terminal, start the TensorBoard server:
tensorboard --logdir data/lao/tensorboard
```
Open your browser to `http://localhost:6006` to watch your model learn.

**Training Plots (Static)**
At the end of training, a `<model>_plot.png` chart is automatically generated in your model directory showing Loss, CER, WER, execution speeds, and character frequencies over time. You can also manually generate this at any time:
```bash
python3 torch_train.py plot --model_name lao
```

#### Training output

```
──────────────────────────────────────────────────────────────────────
  Iteration    500/10000  │  Epoch 2  │  312 samples/sec
  Loss=2.1456  avg=2.3102  │  CER=0.4521  WER=0.7834  │  Mean RMS=0.043521
  GPU: 2.1 GB / 14.7 GB  (peak: 3.8 GB)  │  ETA: 12:34
──────────────────────────────────────────────────────────────────────
  GT:   ສາທາລະນະລັດ ປະຊາທິປະໄຕ ປະຊາຊົນລາວ
  Pred: ສາທາລະນະລັດ ປະຊາທີປະໄຕ ປະຊາຊົນລາວ
──────────────────────────────────────────────────────────────────────
```

### Evaluation

```bash
python3 torch_train.py evaluation --model_name lao --batch_size 32
```

> **Note:** The architecture is automatically read from the checkpoint, so evaluation always works even if the checkpoint was trained with a custom `--net_spec`.

### Recognition (Inference)

```bash
# Single line (greedy decode)
python3 torch_train.py recognize --model_name lao --image line.png

# Full page with beam search + dictionary
python3 torch_train.py recognize --model_name lao --image page.png \
    --psm 3 --beam_width 10 --wordlist wordlist.txt

# With language model
python3 torch_train.py recognize --model_name lao --image page.png \
    --psm 3 --beam_width 10 \
    --lm_corpus data/lao-ground-truth --lm_weight 0.5
```

#### Page Segmentation Modes (PSM)

| PSM | Mode | Description |
|-----|------|-------------|
| `3` | Auto | Detect text blocks and lines automatically |
| `6` | Block | Assume single uniform text block |
| `7` | Line | Treat as a single text line |
| `13` | Raw | Single character or raw line |

### Feature Map Inspection

To visually debug what the neural network is "seeing" inside its convolutional and LSTM layers:

```bash
python3 torch_train.py inspection --model_name lao --num_samples 10
```

This exports activation maps for 10 random images from your `list.val` set to `data/lao/inspection/`. For each layer (Conv2d, MaxPool2d, etc.), it saves the outputs as grayscale PNGs so you can inspect how characters and diacritics are being processed.

### Compare Models

You can overlay the training logs of multiple models to directly compare their Validation Loss and CER curves. Additionally, if you provide a `--custom_img_dir`, it will test all models on the `.gt.txt` and image pairs found in that custom dataset.

```bash
python3 torch_train.py compare --models lao lao_v2 lao_big \
    --custom_img_dir data/my-custom-test-set
```

### Export

```bash
python3 torch_train.py export --model_name lao

# Outputs:
#   data/lao/lao.onnx          → onnxruntime inference
#   data/lao/lao_model_info.json
```

**Note on Export Format:** PyTorch models are directly exported to the ONNX format, which provides a fast, standardized way to run inference in production on CPU or GPU.

#### Using exported models

```python
# ONNX Runtime (Python)
import numpy as np
from PIL import Image
import onnxruntime as ort
import json

# 1. Load image and resize to target height (e.g., 48px)
img = Image.open('image.png').convert('L')
w, h = img.size
new_w = max(1, int(w * (48 / h)))
img = img.resize((new_w, 48), Image.BILINEAR)

# 2. Normalize to [-1.0, 1.0] and add batch/channel dims -> [1, 1, 48, W]
arr = np.array(img, dtype=np.float32)
arr = (arr - 127.5) / 127.5
preprocessed_image = arr[np.newaxis, np.newaxis, :, :]

# 3. Run ONNX Inference
session = ort.InferenceSession('data/lao/lao.onnx')
output = session.run(None, {'image': preprocessed_image})[0]

# 4. CTC Greedy Decode
with open('data/lao/lao_model_info.json') as f:
    charset = json.load(f)['charset']
inv_charset = {v: k for k, v in charset.items()}

pred_ids = np.argmax(output[0], axis=1) # Shape: (W, NumClasses)
text = []
for i in range(len(pred_ids)):
    if pred_ids[i] != 0 and (i == 0 or pred_ids[i] != pred_ids[i-1]):
        text.append(inv_charset.get(pred_ids[i], '?'))

print("Predicted:", "".join(text).replace('<space>', ' '))
```

## 🛠️ Utilities

```bash
# Analyze character frequencies from ground truth
python3 torch_train.py charset-stats --model_name lao
# or specify directory directly:
python3 torch_train.py charset-stats --ground_truth_dir data/lao-ground-truth
```

Output:
```
  #    Char     Unicode       Count        %  Category              Name
  1       າ      U+0EB2           4   11.11%  Letter, other         LAO VOWEL SIGN AA
  2       ສ      U+0EAA           2    5.56%  Letter, other         LAO LETTER SO SUNG
  ...  

  ── Category Summary ──────────────────────────────
  Other letters              16  ( 44.4%)
  Lowercase letters           8  ( 22.2%)
  ...
  ✓ Saved: data/lao/charset_stats.csv
```

```bash
# Generate training chart
python3 torch_train.py plot --model_name lao

# Clean all generated files
python3 torch_train.py clean --model_name lao
```

---

## 📁 Project Structure

```
tesstrain-gpu/
├── torch_train.py              # CLI: 11 subcommands (train, evaluate, recognize, …)
├── gen_data_from_text.py       # Synthetic data generator (CPU/GPU) + 4 augmentation presets
├── analyze_real_data.py        # Phase 1: Profile real scanned documents → calibrated aug params
├── build_augmented_corpus.py   # Phase 3: Enrich corpus from real GT + oversample rare chars
├── validate_augmentation.py    # Phase 3: KL divergence + gallery validation of augmented data
├── tesseract_tools.py          # Training plots · model comparison · timing report
├── compare_cross.py            # Cross-compare original Tesseract vs fine-tuned (CSV)
├── tesstrain_core/
│   └── compare.py              # PyTorch model comparison + diagnostic plots
├── requirements.txt
├── LICENSE
└── README.md

# Generated after training:
data/
├── <model_name>/
│   ├── <model>_best.pt             # Best checkpoint (lowest CER)
│   ├── <model>_final.pt            # Final checkpoint
│   ├── <model>.traineddata         # Tesseract binary model
│   ├── <model>_final.traineddata   # Compiled Tesseract model (for comparison)
│   ├── <model>.onnx                # ONNX model
│   ├── <model>_model_info.json     # Model metadata
│   ├── training.log                # Training log (with timestamps if segmented)
│   ├── training_plot.png           # Error-rate chart
│   ├── charset.json                # Character → index mapping
│   ├── config.json                 # Training configuration
│   └── unicharset                  # Tesseract unicharset
├── lao-augmented-v2/               # Generated by gen_data_from_text.py --preset real-match
│   └── validation/                 # Generated by validate_augmentation.py
│       ├── validation_report.md
│       ├── kl_divergence.json
│       └── comparison_gallery/
├── lao-ground-truth-test/          # Test images + .gt.txt pairs
├── output/                         # PyTorch comparison outputs
│   ├── plot_global_bar.png
│   ├── plot_coer_bar.png
│   ├── plot_demo_grid.png
│   ├── plot_worst_coer_grid.png
│   ├── plot_metric_explanation_<model>.png  # one per model
│   └── report*.csv
├── tesseract_comparison/           # tesseract_tools.py compare outputs
└── cross_comparison/               # compare_cross.py outputs

# Generated by analyze_real_data.py (stored alongside real documents):
<real-documents-dir>/analysis_output/
├── analysis_report.md          # Full domain gap report
├── augmentation_params.json    # Calibrated params for --preset real-match
├── full_analysis.json          # Machine-readable per-image stats
├── dimensions.png              # Height/width distribution plots
├── quality_metrics.png         # Noise/contrast/sharpness distribution
└── char_frequency.png          # Top-30 character frequency bar chart

plots/                              # tesseract_tools.py training plots
```

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Training Pipeline                                                  │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │  .gt.txt  │    │  .lstmf  │    │ PyTorch  │    │   Export     │  │
│  │  .tif     │──▸ │  lists   │──▸ │ Training │──▸ │.traineddata  │  │
│  │  images   │    │          │    │  (GPU)   │    │.onnx         │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────────┘  │
│   unicharset       lists          training        traineddata      │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  Inference Pipeline                                                 │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │  Input   │    │   Page   │    │  LSTM    │    │  Beam Search │  │
│  │  Image   │──▸ │  Segment │──▸ │ Forward  │──▸ │  + DAWG + LM │  │
│  │          │    │  (PSM)   │    │  Pass    │    │  → Text      │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────────┘  │
│                    recognize                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Advanced Features

### Beam Search + DAWG Decoding

```bash
python3 torch_train.py recognize --model_name lao --image page.png \
    --beam_width 10 --wordlist wordlist.txt
```

| DAWG | What it does |
|------|-------------|
| **Word DAWG** | Boosts dictionary words, penalizes unknown |
| **Number DAWG** | Recognizes integers, decimals, dates, currency, phone, time |
| **Punctuation DAWG** | Validates placement, prevents repetition |

### Language Model (N-gram)

```bash
python3 torch_train.py recognize --model_name lao --image page.png \
    --beam_width 10 --lm_corpus corpus.txt --lm_weight 0.5
```

- Character trigrams + word bigrams with backoff smoothing
- Auto-trains from `.gt.txt` directory or plain text file

### Fine-Tuning

```bash
# Continue from best checkpoint (use a lower learning rate)
python3 torch_train.py training --model_name lao \
    --start_model data/lao/lao_best.pt \
    --epochs 50 --learning_rate 0.0005
```

> **Tip:** The network architecture (`--net_spec`) is **automatically loaded from the checkpoint** — you don't need to specify it again. The architecture from the checkpoint always takes precedence.

### Transfer Learning: Tesseract CPU -> GPU

You can warm-start PyTorch GPU training from an existing Tesseract CPU model (`.traineddata` or extracted `.lstm`).

```bash
python3 torch_train.py training --model_name lao \
    --start_tess_model /path/to/lao.traineddata \
    --net_spec "[1,48,0,1 Ct3,3,16 Mp3,3 Lfys64 Lfx96 Lrx96 Lfx384 O1c###]" \
    --epochs 50 --learning_rate 0.0005
```

Notes:
- `--net_spec` should match the source Tesseract architecture.
- The loader applies all compatible tensors and skips mismatched ones (commonly the final output layer when charset sizes differ).
- This makes cross-domain adaptation practical: reuse backbone layers from CPU models, then continue fine-tuning on GPU data.

### Network Architecture (VGSL)

```
Default: [1,48,0,1 Ct3,3,16 Mp3,3 Lfys64 Lfx96 Lrx96 Lfx512 O1c###]

Layer breakdown:
  [1,48,0,1]  Input: batch=1, height=48, width=variable, depth=1 (grayscale)
  Ct3,3,16    Conv: 3×3 kernel, 16 filters, tanh activation
  Mp3,3       MaxPool: 3×3
  Lfys64      LSTM: forward-y (vertical), 64 units, with sigmoid gate
  Lfx96       LSTM: forward-x (horizontal), 96 units
  Lrx96       LSTM: reverse-x (right-to-left), 96 units
  Lfx512      LSTM: forward-x, 512 units
  O1c###      Output: 1D softmax, ### = num_classes (auto-filled)
```

#### VGSL Token Reference

| Token | Description | Stage | Example |
|-------|-------------|-------|---------|
| `[B,H,W,C]` | Input dimensions | — | `[1,48,0,1]` |
| `C[t\|s\|r]Y,X,F` | Conv2d + activation (t=tanh, s=sigmoid, r/default=relu) | CNN | `Ct3,3,16` |
| `MpY,X` | MaxPool2d with stride = kernel | CNN | `Mp3,3` |
| `L[f\|r\|b\|m][x\|y][s]N` | LSTM (f=fwd, r=rev, b=bidi, m=multi) | Seq | `Lfx96`, `Lfys64` |
| `Bn` | Batch Normalization (BatchNorm2d in CNN, BatchNorm1d in seq) | Any | `Bn` |
| `D<rate>` | Dropout (rate is integer 0–99, only active during training) | Any | `D20` (= 20%) |
| `Fc<N>` | Dense / fully-connected: Linear(N) + ReLU | Seq | `Fc256` |
| `O<d>c<N>` | Output: Linear layer, N = class count (`###` = auto) | Seq | `O1c###` |

> **Note:** `Bn`, `D`, and `Fc` auto-detect whether the network is still in the 2-D CNN
> stage or has collapsed to 1-D sequences, and use the appropriate PyTorch module
> (`BatchNorm2d` vs `BatchNorm1d`, `Dropout2d` vs `Dropout`, etc.).

#### Example Architectures

```bash
# Default (Tesseract-compatible)
--net_spec "[1,48,0,1 Ct3,3,16 Mp3,3 Lfys64 Lfx96 Lrx96 Lfx512 O1c###]"

# With BN + Dropout (regularized)
--net_spec "[1,48,0,1 Ct3,3,16 Bn D20 Mp3,3 Lfys64 Lfx96 D30 Lrx96 Lfx512 O1c###]"

# With Dense layer (bottleneck between LSTMs)
--net_spec "[1,48,0,1 Ct3,3,16 Bn Mp3,3 Lfys64 Lfx96 Fc256 D40 Lrx96 Lfx512 O1c###]"
```

> **⚠️ Tesseract compatibility:** The `Bn`, `D`, and `Fc` tokens are PyTorch-only
> extensions. Models using them can be trained and exported to ONNX, but **cannot** be
> converted back to a native Tesseract `.traineddata` file.

---

## ⚡ Performance

| Metric | Tesseract `lstmtraining` | tesstrain-gpu |
|--------|--------------------------|---------------|
| **Training speed** | ~20 samples/sec (CPU) | ~500 samples/sec (GPU) |
| **10K iterations** | ~2 hours | ~5 minutes |
| **Memory** | 2–4 GB RAM | 2–6 GB VRAM |
| **Speedup** | 1× | **10–50×** |

### GPU Memory Estimates (Tesla T4, 15GB)

| Batch Size | ~Memory | ~Throughput |
|------------|---------|-------------|
| 32 | 2 GB | 200 samples/s |
| 64 | 3 GB | 400 samples/s |
| 128 | 5 GB | 600 samples/s |
| 256 | 8 GB | 750 samples/s |

---

## 🆚 Comparison with Tesseract

| Feature | Tesseract | tesstrain-gpu |
|---------|-----------|---------------|
| Training backend | C++ CPU | **PyTorch GPU** |
| Optimizer | SGD + momentum | **Adam** / **SGD** |
| Model export | `.traineddata` | `.onnx` |
| Beam search | ✅ | ✅ + DAWG + LM |
| Page segmentation | 14 modes | 4 modes (3, 6, 7, 13) |
| Language model | Word n-gram | **Char + word n-gram** |
| Unicode recoder | ✅ | ✅ NFC/NFD + ligatures |
| Synthetic data | `text2image` (C++) | `gen_data_from_text.py` |

---

## 🐛 Troubleshooting

### CUDA out of memory
```bash
# Reduce batch size
python3 torch_train.py training --model_name lao --batch_size 32
```

### No GPU detected
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show GPU name
```

### PyICU install fails
```bash
# Ubuntu/Debian
sudo apt install libicu-dev pkg-config
pip install PyICU

# macOS
brew install icu4c
export ICU_VERSION=$(brew list --versions icu4c | head -1 | awk '{print $2}')
export PATH="/usr/local/opt/icu4c/bin:$PATH"
export PKG_CONFIG_PATH="/usr/local/opt/icu4c/lib/pkgconfig"
pip install PyICU
```

### ONNX export fails
```bash
# Install onnx with matching version
pip install onnx>=1.14.0 onnxruntime>=1.15.0
```

---

## 📄 Command Reference

| Command | Description | Key Arguments |
|---------|-------------|---------------|
| `unicharset` | Build charset from `.gt.txt` | `--model_name` |
| `lists` | Create train/eval/valid file lists | `--model_name`, `--ratio 80 10 10` |
| `proto-model` | Build proto `.traineddata` | `--model_name` |
| `training` | GPU training | `--optimizer`, `--epochs`, `--batch_size`, `--learning_rate`, `--start_model`, `--start_tess_model` |
| `export` | Export PyTorch model to ONNX | `--model_name` |
| `evaluation` | Eval CER / WER / Accuracy | `--model_name`, `--batch_size` |
| `recognize` | Run OCR inference | `--image`, `--psm`, `--beam_width`, `--wordlist` |
| `plot` | Training dashboard (7 plots) | `--model_name` |
| `inspection` | Export feature map images | `--model_name`, `--num_samples` |
| `compare` | Compare logs and evaluate on custom set | `--models`, `--custom_img_dir` |
| `charset-stats` | Analyze character frequencies | `--model_name` |
| `clean` | Remove generated files | `--model_name` |

---

## 📊 Evaluation & Analysis Tools

Three dedicated scripts handle post-training evaluation and model comparison.
They are independent of `torch_train.py` and can be run at any time after
training is complete.

---

### `tesseract_tools.py`

**Combined tool: training-log plots · Tesseract model comparison · training-time report**

```bash
# Generate training / eval plots from all models under data/
python tesseract_tools.py plots

# Compare compiled Tesseract models on your test image set
python tesseract_tools.py compare

# Training-time report for all models (wall-clock, iter/s, BCER per segment)
python tesseract_tools.py timereport

# Training-time report for one specific model
python tesseract_tools.py timereport --model_name lao48cnn32

# Run everything
python tesseract_tools.py all
```

#### Part 1 — Training Log Plots (`plots`)

Parses every `data/<model>/training.log` and `data/<model>/eval/*.eval.log`
and produces publication-ready PNGs in `plots/`:

| Plot file | Contents |
|-----------|----------|
| `plot_loss_combined.png` | Mean-RMS (≈ loss) for all models overlaid |
| `plot_acc_combined.png` | Character accuracy (100 − BCER) overlaid |
| `plot_acc_premium.png` | Smoothed accuracy with ± std band + best eval ★ marker |
| `plot_train_loss.png` | Mean-RMS per model (subplots) |
| `plot_train_acc.png` | Training accuracy per model (subplots) |
| `plot_eval_loss.png` | BCER eval (sparse checkpoints) per model |
| `plot_eval_acc.png` | Eval accuracy per model |
| `plot_wer_train.png` | Word Error Rate (training) all models combined |
| `plot_wer_eval.png` | Word Error Rate (eval checkpoints) all models combined |

#### Part 2 — Model Comparison (`compare`)

Runs every compiled `.traineddata` model in `data/` against `data/lao-ground-truth-test/`
and outputs to `data/tesseract_comparison/`:

| Output file | Contents |
|-------------|----------|
| `plot_global_bar.png` | CER / WER / Avg CoER / Accuracy grouped bar chart |
| `plot_coer_bar.png` | Per-category CoER (Tone, Vowel, Consonant…) bar chart |
| `plot_demo_grid.png` | Random sample image + alignment grid per model |
| `plot_metric_explanation_<model>.png` | Character-level CoER + CER/WER alignment diagram |
| `report.csv` | Global CER / WER / Accuracy / CoER |
| `report_characters.csv` | Per-character error breakdown |
| `report_mistakes.csv` | Top substitution/deletion/insertion errors |

#### Part 3 — Training Time Report (`timereport`)

Parses `[YYYY-MM-DD HH:MM:SS]` timestamps written by the segmented training
script and produces:

| Output | Contents |
|--------|----------|
| Console table | Segment boundaries, wall-clock time, iter/s, best/last BCER |
| `report_training_time_<model>.csv` | Machine-readable version of the table |
| `plot_training_time_<model>.png` | Dual-axis bar (wall time) + line (best BCER) chart |

#### Config block (top of file)

```python
IMAGE_FOLDER        = "data/lao-ground-truth-test"
CUSTOM_TESSDATA_DIR = "data"          # where *_final.traineddata files live
CUSTOM_MODEL_NAMES  = ["lao48cnn16_final", "lao48cnn32_final", ...]
TESS_CONFIG         = "--psm 7 --oem 1"
```

---

### `tesstrain_core/compare.py`

**PyTorch model comparison pipeline** — runs inside the `tesstrain_core`
package and is invoked via `torch_train.py compare`.

```bash
python torch_train.py compare \
    --models lao48cnn16 lao48cnn32 lao48cnn16t16 lao48cnn16t32 \
    --custom_img_dir data/lao-ground-truth-test
```

Outputs to `data/comparison/` (or `--out_dir`):

| Output file | Contents |
|-------------|----------|
| `plot_global_bar.png` | CER / WER / Avg CoER / Accuracy grouped bar chart |
| `plot_coer_bar.png` | Per-category CoER bar chart |
| `plot_demo_grid.png` | Random sample + alignment grid per model |
| `plot_worst_coer_grid.png` | Worst-performing samples with images highlighted |
| `plot_metric_explanation_<model>.png` | Per-model diagnostic diagram (same 2 GT samples across all models) |
| `report.csv` | Global metrics |
| `report_mistakes.csv` | Top error patterns |

> **Shared GT samples**: `plot_metric_explanation_*.png` always shows the
> **same two ground-truth strings** across every model so you can compare
> prediction errors side by side.

---

### `compare_cross.py`

**Cross-compare original Tesseract models vs your fine-tuned models — CSV only.**

Useful for a clean, quantitative baseline comparison for papers or reports.

```bash
# Default paths (edit CONFIG block at the top)
python compare_cross.py

# Override paths
python compare_cross.py \
  --image_folder data/lao-ground-truth-test \
  --out_dir data/cross_comparison \
  --orig_tessdata /usr/share/tessdata
```

#### Config block (top of file)

```python
# Original Tesseract models (looked up in system tessdata or ORIGINAL_TESSDATA_DIR)
ORIGINAL_MODELS = ["lao"]   # add "lao_best", "eng" etc. as needed

# Fine-tuned models (must exist as data/<name>.traineddata)
CUSTOM_MODELS   = ["lao48cnn16_final", "lao48cnn32_final",
                   "lao48cnn16t16_final", "lao48cnn16t32_final"]

IMAGE_FOLDER        = "data/lao-ground-truth-test"
CUSTOM_TESSDATA_DIR = "data"
ORIGINAL_TESSDATA_DIR = None    # None = system default
```

#### Output CSVs in `data/cross_comparison/`

| File | Contents |
|------|----------|
| `cross_report.csv` | CER / WER / Accuracy / Avg CoER per model with **Model Type** column (`Original Tesseract` / `Fine-tuned`) |
| `cross_report_coer.csv` | Per-category CoER (Tone / Vowel / Consonant…) for all models |
| `cross_report_characters.csv` | Per-character GT / Correct / Errors / Error Rate |
| `cross_report_mistakes.csv` | All substitution / deletion / insertion errors sorted by frequency |

A delta summary is printed to the console:

```
  Best original  : lao  (CER 18.42%)
  Best fine-tuned: lao48cnn32_final  (CER 2.31%)
  Δ CER  = +16.11%   (positive = fine-tuned is better)
  Δ WER  = +24.87%
  Δ Acc  = +15.63%
```

#### Requirements

```bash
pip install pytesseract pillow matplotlib numpy
# Tesseract binary must be installed and in PATH
```

---

## 📜 License

Apache 2.0 — Same as [Tesseract](https://github.com/tesseract-ocr/tesseract) and [tesstrain](https://github.com/tesseract-ocr/tesstrain).
