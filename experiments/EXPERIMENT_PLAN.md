# Experiment Plan: Which part of CNN + BiLSTM + CTC limits Lao OCR?

**Status:** draft v0.1 (2026-11-04). Nothing here has been trained yet. The only thing verified is that the VGSL
specs marked ✓ build and run a forward pass with `tesstrain_core.model.build_model`.

## 0. Research question

> In a single-stage CNN → (Bi)LSTM → CTC recogniser, which module is the bottleneck for **vertically stacked
> Lao components** (tone, upper vowel, lower vowel), and where is the sweet spot between accuracy and cost?

The limitation we study is *morphological interception*: a consonant, an upper vowel, a tone mark and a lower
vowel share the same horizontal span, but CTC has to emit them as a left-to-right sequence of labels.
That gives three competing explanations, one per module:

| | Module | Hypothesis: "this is the bottleneck" | If true, you will see |
|---|---|---|---|
| **H-CNN** | CNN + height collapse | Features that cannot keep tier identity (collapsing H to 1 row mixes the tiers) | Marks are *never* detected, even when the GT alignment is given. Fixed by taller receptive field, no-collapse / attention collapse, more height resolution |
| **H-LSTM** | BiLSTM | The sequence model cannot recover tier information, or hides visual errors behind a learned language prior | Linear probe on CNN features is good, final output is bad. Random-string CER is much worse than text CER |
| **H-CTC** | CTC alignment + label unit | Many labels per column, an arbitrary emission order, too few frames per cluster | Marks are detected (high posterior somewhere) but are deleted, swapped or misplaced. Fixed by more frames (x-stride), cluster / tier-wise labels |

Decision rule (stated now, before any result): a module is **critical** if varying it moves stack-component
CoER (tone, upper, lower) by more than 2× the seed standard deviation *and* explains the largest share of
variance in the Phase 3 factorial (η²).
The **sweet spot** is the knee of the Pareto front over (CoER_stack, params, CPU ms/line).

## 1. What the current code already tells us (read before designing)

These come from reading `tesstrain_core/model.py`, `README.md` and `metrics.md`. They change the design.

1. **The default "BiLSTM" is not bidirectional.** `Lfx96 Lrx96 Lfx384` is forward → reverse → forward, three stacked
   *uni*-directional layers. A true BiLSTM is `Lbx`. So "is the BiLSTM critical" is not yet a clean question: Phase 2 group B separates
   direction from depth from width.
2. **Black padding leaks into right-to-left and attention layers.** README "Known issues" #1: with batch padding of −1
   (black), 296 / 400 predictions changed between batch size 1 and 64. Any ablation involving `Lrx`, `Lbx`,
   `Atys`, `Matys`, `Hybys`, `SelfAttnX`, `ConfX` is confounded until this is fixed (or evaluated at `--batch_size 1`).
   **This blocks Phase 2.**
3. **Default CER is lenient.** `metrics.py` counts ດ/ຕ, ນ/ມ, ໃ/ໄ … as half an error and strips trailing punctuation. Ablations must report the strict score too.
4. **`SelfAttnX` has no positional encoding** (`HorizontalSelfAttention` is plain `nn.MultiheadAttention` + LayerNorm). A stack of it is order-blind except for what the CNN leaks. Compare `ConfX` (has a depthwise conv) or add sinusoidal PE first; do not conclude "attention is worse than LSTM" from `SelfAttnX`.
5. **Time resolution is fixed by the pooling.** `Mp3,3` gives T ≈ W/3 (✓ W=512 → T=170). Changing the pooling changes the CTC problem, not just the CNN. This is why x-stride sits in the CTC group (C1), not the CNN group.
6. **Existing evidence is thin for causal claims.** Single seed, 50 real lines (3,093 chars), `THESIS_REPORT.md` results tables are still empty, and the "SweetSpot" in `paper/lao_ocr_sweet_spot_calculations.md` (`Ct5,5,32 Mp3,3 Hybys64 …`, 938,256 params) is not yet tested against its own ablations with multiple seeds.
7. **Parameter counts differ a lot between variants** (✓ measured, 80 classes): baseline 933,776; no seq-model 31,120; `Lbx96 Lbx384` 1,987,472; `SelfAttnX4×2` 64,656. Compare variants at equal params *and* at equal spec, otherwise "bigger wins" is the only finding.

## 1b. Experiment 1 (E1): replicate the master's thesis setting

**Goal:** the first run of the PhD uses the *same environment as `thesis_Obi.pdf`*: same data, same iterations, same hyperparameters.
It proves the pipeline reproduces the thesis before anything is changed, and it turns the thesis' single-seed numbers into a seed distribution (the new B0).

### What the thesis fixes (read from `methodology.tex`, `experiment1–4.tex`, `benchmark_report.csv`)

| Item | Thesis value | Source |
|---|---|---|
| Train data | 100,000 synthetic lines, 1,486,130 chars, 79 classes, 4 fonts (Phetsarath Regular / OT / Bold, Saysettha OT), 20–44 pt | Methodology, Table "Dataset Statistics" |
| Augmentation (train) | noise p=0.20, intensity 0.06; blur p=0.20, radius 0.9; random aug 40%; max brightness 10.0; max contrast 0.6; shape distortion 2% | same |
| Synthetic test | 2,000 lines, 59,415 chars, 14–44 pt, noise intensity 0.10, brightness 20.0, contrast 0.8, distortion 4% | same |
| Real test | 50 scanned-form lines, 3,093 chars, 69 classes, zero-shot (no fine-tuning) | Methodology |
| Input | grayscale, fixed height **I = 64** (also 48 in Exp. 1), variable width | Methodology |
| Network | `Ct{K},{K},{C}` → `Mp3,3` → `Lfys64` → `Lfx96` `Lrx96` → one more LSTM → `O1c###`; **K = 5**, C ∈ {16, 32, 64} | Methodology, Exp. 4 |
| Optimizer | Adam (β₁ 0.9, β₂ 0.999), LR **0.002**, **ReduceLROnPlateau** | Exp. 1–4 setup tables |
| Batch / iterations | batch **64**, **50,000** iterations | Methodology |
| Loss / decode | `CTCLoss(blank=0)`, greedy for evaluation | `training.py` |
| Hardware | one NVIDIA T4 (16 GB) | Methodology |
| Anchors to reproduce | **I64K5C16** (headline, 91.49% synthetic acc., 3.64% CER) and **I64K5C32** (best real-world CER, 5.46%). Optional: I64K3C16, I48K5C32 | `benchmark_report.csv` |

Use these two as the E1 configurations. `B0` in the rest of this plan becomes `I64K5C16` (the thesis' final setting) with `I64K5C32` as the second anchor.

### Things the thesis does not pin down, and that I could not find on this machine

These must be settled before E1, otherwise "same as the thesis" is not true. Please confirm from your Colab history, the Drive archive `Compared/Update_50k_iters/tesstrain-gpu-20260714.zip`, or the run `config.json` files:

1. **LSTM stack.** ✅ **Resolved 2026-11-05:** the master's model is `[1,64,0,1 Ct5,5,16 Mp3,3 Lfys64 Lfx96 Lrx96 Lfx384 O1c###]` (last LSTM = **384**; 929,264 params at 80 classes). Original note: Thesis text says "three BiLSTM layers, 96 / 192 / 384". The block diagram says `Lfx96 Lrx96 Lfx512`. The repo default is `Lfx96 Lrx96 Lfx384`. These are different models (the sweet-spot note's 929,264 params assume 384). The `config.json` saved in each run folder records the real `--net_spec`.
2. **ReduceLROnPlateau.** ✅ *Code added 2026-11-05:* `--lr_scheduler plateau` (on eval CER, defaults factor 0.5 / patience 5 evaluations / min_lr 1e-5); the thesis' own values are still unknown. Original note: The thesis names it, but `tesstrain_core/training.py` in this repo has **no scheduler** (fixed LR; `git log -S` finds no commit that ever had it). Its factor, patience, monitored metric and minimum LR are unknown. The version of the code that trained the thesis models is probably the one in the Drive zips.
3. **Early stopping.** `--target_error_rate` defaults to 0.01: training stops when *eval* CER ≤ 1%. In `benchmark_report.csv` the PyTorch runs stopped at very different iterations (I64K9C32 at 9,500, I64K5C16 at 20,500, I64K5C32 at 30,500, others the full 50,000), although the thesis tables say 50,000 for all. Decide whether "same iteration" means the rule (stop at ≤ 1% eval CER, max 50,000) or exactly 50,000 for everyone.
4. **Eval split.** Early stopping and "best" checkpointing use a held-out *eval* list (`BestEvalBCER` ≈ 0.5–1% in the CSV), which is drawn from the training distribution. Its size and seed are not in the thesis.
5. **The 100k-line training set and the 2k test set are not on this machine.** `data/lao-ground-truth-test/` in the repo is empty, and `data/real-lines/` holds a different document (69 files). The files exist in your Drive (`lao-ground-truth-test-2k.zip`, and the large `tesstrain-gpu-2026071x.zip` archives). The 50 real thesis lines are not located yet.
   Do **not** regenerate with the current `gen_data_from_text.py`: it was rewritten after the thesis (HarfBuzz shaping, fixed line box, per-sample seeds, font coverage filtering), so it would give a different dataset.
6. **Seeds.** Not recorded. The thesis numbers are therefore one unknown draw. ✅ *From now on:* `--seed` and `config.json` record them.

### E1 protocol

| Step | Action | Pass criterion |
|---|---|---|
| E1.0 | Recover the thesis code snapshot, data (train 100k, eval list, test 2k, real 50) and each model's `config.json`. Store file hashes. Pin the environment (torch, CUDA, Python versions) | One manifest file; spec string, LR schedule, early-stop rule all written down from the logs, not from memory |
| E1.1 | **Faithful arm (E1a):** run I64K5C16 and I64K5C32 with the thesis code *as it was*: same data, 50,000 iterations (or the same stop rule), Adam 0.002 + the same scheduler, batch 64, same augmentation. Seeds {1, 2, 3}. No padding or metric fixes | Seed mean within the thesis single-run value ± 2σ for CER, Acc, per-tier CoER (synthetic and real). If the thesis value is outside, investigate before going on |
| E1.2 | **Sensitivity arm (E1b):** same as E1a, but with white padding + real lengths (P0.1) and strict CER (P0.2). Everything else identical | Reports how much of the thesis result depends on the padding leak and the lenient metric. This is a result, not a failure |
| E1.3 | **Stop-rule arm (E1c, only if item 3 shows a mismatch):** fixed 50,000 iterations with no early stop, best-eval checkpoint kept | Shows whether runs that stopped at 9,500–30,500 iterations were under-trained, and whether the K / C ranking of the thesis survives equal training |
| E1.4 | Freeze: these checkpoints, logs and σ table are the **baseline for every later experiment** | Phase 1 diagnostics (D1–D6) run on these checkpoints |

Order matters: E1a first (faithful), then E1b, so each change is isolated. Only after E1 is accepted do any architecture changes start.

### What is allowed to differ in later experiments

Only the one factor under test. Dataset, iterations, optimizer, LR schedule, batch size, augmentation, evaluation sets, seeds and stop rule stay exactly as E1a (or E1b once the padding fix is adopted, stated explicitly in every table).
If a later experiment needs a different setting (for example more frames per label in C1 needs more LSTM time steps, not a different LR), say so in the table.

## 1c. Experiment 2 (E2): replace the CNN with ResNet18 or LCNetV4

**Question:** with the thesis recipe fixed (E1), does a stronger CNN backbone, ResNet18 or PP-LCNetV4, reduce stacked-component errors (tone, upper vowel, lower vowel) compared with the thesis CNN (`Ct5,5,C Mp3,3`)?
**Rule:** only the CNN changes. Everything after it (`Lfys64 Lfx96 Lrx96 Lfx{384|512} O1c###`), data, iterations, optimizer, scheduler, batch, seeds and stop rule stay as in E1. Anchor = `I64K5C16` (and `I64K5C32` if you want both).

### What I checked (numbers measured here with random weights, input 64 × 272, 80 classes)

LCNetV4 is the PP-OCRv6 backbone, a MetaFormer-style block (3×3 depthwise token mixer + 1×1 channel mixer) with re-parameterised depthwise convs; recognition mode uses (2,1) strides at stages 3 and 4 ([paper](https://arxiv.org/html/2606.13108), [PaddleOCR docs](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)).
A PyTorch implementation is already in your Python environment (`transformers` 5.19: `transformers.models.pp_lcnet_v4`). ResNet18 is not (no `torchvision`), but it is ~30 lines.

| Backbone (CNN only) | Params | MMACs / line | Output (C, H′, T) at W=272 | T / W | CNN-only CPU ms/line* |
|---|---|---|---|---|---|
| **B0** `Ct5,5,16 Mp3,3` | **416** | 7 | 16, 21, 90 | 0.331 | 0.8 |
| **B0** `Ct5,5,32 Mp3,3` | **832** | 14 | 32, 21, 90 | 0.331 | 1.8 |
| ResNet18, width 16, T = W/3 | 699,888 | 361 | 128, 4, 91 | 0.335 | 8.0 |
| ResNet18, width 32, T = W/3 | 2,794,464 | 1,444 | 256, 4, 91 | 0.335 | 12.9 |
| ResNet18, width 64 (standard), T = W/3 | 11,167,680 | 5,775 | 512, 4, 91 | 0.335 | 68.2 |
| LCNetV4 "small" tier (HF default), T = W/3 | 2,683,464 | 1,466 | 384, 4, 91 | 0.335 | 36.0 |
| LCNetV4 "tiny" tier (approximate, see below) | 519,964 | 180 | 160, 4, 68 | 0.25 | 8.6 |

\*Measured on this Mac, batch 1, 10 repeats. Only relative values mean anything; RQ3 (< 25 ms/line on an office CPU) must be measured on the target machine with the full model.

Four facts that decide the design:

1. **The thesis CNN is tiny: 416 or 832 parameters.** Of the 929,264 total (C16), nearly all are LSTM. Any real backbone therefore changes the model from "LSTM with a one-layer front end" to "backbone with an LSTM on top": 0.7M to 11M extra parameters and 50 to 800× the CNN compute. "Better backbone" and "more capacity" cannot be separated unless capacity is controlled (arms C1 and C2 below).
2. **Standard strides break CTC.** Stock ResNet18 (total stride 32) gives T = W/32. The average training line is about 271 px wide at H=64 with 14.9 labels (1,486,130 chars / 100,000 lines; 229 × 54 average size), so T ≈ 8.5 < 14.9 labels: CTC has no valid alignment. The repo uses `CTCLoss(zero_infinity=True)`, so such samples would silently contribute **zero loss** instead of raising an error. Frames per label: x-stride 3 → 6.1, 4 → 4.6, 8 → 2.3, 16 → 1.1, 32 → 0.57. Both backbones must keep the width stride low (OCR practice: stride (2,1) in later stages).
3. **Time resolution is a CTC factor (C1 above), so keep it equal.** LCNetV4 recognition mode and the usual ResNet-OCR layouts give T = W/4 (or W/2), not the thesis' W/3. To change only the CNN, the primary arms put the whole x-reduction in the stem so that **T = W/3** exactly (ResNet18 stem stride (2,3); LCNetV4 `stem_strides=((2,3),1,1,(2,1),1)`, both ✓ measured T/W = 0.335). The native W/4 layouts run as a sensitivity arm. Do not use LCNetV4's own `avg_pool2d((3,2))` from the PP-OCRv6 tiny recogniser: it cuts T to W/8 (2.3 frames per label).
4. **The height that reaches `Lfys64` changes from 21 rows to 4.** The backbones end at H′ = 4 (64 / 16). That is a second change besides "backbone". Control B0-H4 below isolates it.

### Arms (all trained from scratch, grayscale input, thesis recipe, seeds {1, 2, 3})

| ID | CNN | Role |
|---|---|---|
| **B0** | thesis `Ct5,5,16 Mp3,3` (from E1) | reference |
| B0-H4 | `Ct5,5,16 Mp3,3 Mpa5,1,5,1,0,0` ✓ (same 416 CNN params, H′ = 4) | does fewer rows alone change CoER? |
| C1 | plain deep CNN, ≈ 0.69M CNN params: `Ct5,5,32 Bn Mp3,3 Cr3,3,96 Bn Cr3,3,128 Bn Mpa3,1,3,1,0,0 Cr3,3,192 Bn Cr3,3,192 Bn Mpa2,1,2,1,0,0` ✓ (total 1,667,824) | capacity control with no residual / no metaformer design |
| R-16 | ResNet18, width 16, T = W/3 (0.70M) | ResNet at matched CNN capacity to C1 |
| R-32 | ResNet18, width 32 (2.79M) | |
| R-64 | ResNet18, width 64 (11.2M, the standard model) | upper capacity bound; reports the cost |
| L-S | LCNetV4 small tier, T = W/3 (2.68M) | |
| L-T | LCNetV4 tiny tier (≈ 0.5M) | **needs verification**: I built it from the paper's table (1×48, 1×48, 3×96, 4×160, small stem); the exact block tuples (squeeze-excitation positions, stem channels) must be checked against the Hub config of `PP-OCRv6_tiny_rec`. The HF default config that I measured is the paper's *small* tier, not tiny |
| Native-T | R-16 and L-S with their native layouts (T = W/2 and W/4) | sensitivity to time resolution |

≈ 9 CNN variants × 3 seeds ≈ **27 runs** (plus B0 from E1). If compute is short, the minimum set is B0, B0-H4, C1, R-16, L-S, L-T (18 runs).
Optional, labelled as a different experiment ("transfer"): initialise LCNetV4 from the PP-OCRv6 weights on the Hub. It adds pretraining on other scripts, so it must never share a table row with the from-scratch results.

### Predictions (written before running)

- **P-E2a:** backbones improve tone and upper-vowel CoER on synthetic data (larger vertical receptive field, more depth), but lower-vowel CoER improves less, because the collapse (`Lfys64`) and CTC still receive one mixed vector per column (H-CNN vs H-CTC; read together with D2 / D3).
- **P-E2b:** the gain at matched parameters (C1, R-16, L-T) is much smaller than at R-64 / L-S, i.e. mostly capacity.
- **P-E2c:** large backbones widen the synthetic-to-real gap. Your Exp. 4 already shows the pattern for filters ("Filter Depth Paradox": C = 64 gives a deletion spike on real scans), and the training set has only 4 fonts. Expect R-64 to win on test-A and not on test-R.
- **P-E2d:** B0-H4 alone changes little. If it changes a lot, the earlier "backbone gain" is partly a row-count effect.

Decision rule: a backbone is "better" only if it beats **both** B0 and its capacity control (C1) on stack-component CoER by more than 2σ on test-A, test-B (held-out fonts) *and* test-R, and stays inside the CPU budget.

### Extra risks specific to E2

- **BatchNorm and padding.** ResNet18 and LCNetV4 use BatchNorm; the thesis CNN does not. The repo's padding fix (P0.1, default `--padding white` with per-line lengths) makes *eval-mode* predictions independent of the batch (tested for `Res18` and `Lcn4`). In *training*, BN still averages over padded columns (they are zero after masking), so batch composition changes the batch statistics a little. Run E2 with the fix (E1b arm) as well as the faithful `--padding legacy` one, and use batch size 1 when comparing a BN model with B0.
- **Re-parameterisation.** LCNetV4 trains with a 3-branch depthwise conv that fuses to one at inference. Train in the training form, fuse before timing and ONNX export; check that the fused model gives identical logits (max abs diff < 1e-4).
- **LR / scheduler.** Same Adam 0.002 + scheduler as the thesis. A single learning rate is tuned for a model with no BatchNorm, so give each backbone one small LR check ({0.001, 0.002, 0.004}, 1 seed, short run) before the 3-seed runs, and report it. Otherwise a loss could just be "wrong LR".
- **Early stop.** The ≤ 1% eval-CER rule favours models that reach the easy eval set fast and may stop a slower-converging backbone early. Report E2 with the fixed-50,000 arm (E1c) as well.
- **Implementation (done, 2026-11-13).** `build_model` has a `Res18,<width>[,<sx>]` token (`ResNet18Backbone`, [model.py](../tesstrain_core/model.py)), and the LCNetV4 arms are written with the repo's existing `Lcn4` stage, so there is **no `transformers` dependency**. All E2 arms are in `experiments/step0_specs.py` (group `backbone`; native-T arms in `backbone-native`). Print them with `python3 experiments/step0_specs.py --only backbone`, or the 3-seed commands with `--commands --only backbone`. Tests: `tests/test_model.py` (shapes, T equals `Mp3,3`, padding independence, 11,167,680 params for `Res18,64`, backward), `tests/test_step0_specs.py`, `tests/test_training_run.py` (short real training with a `Res18` and an `Lcn4` spec, checkpoint reload). Full suite: 145 passed, 3 skipped.
- **Measured differences from the table above.** Totals include the thesis LSTM head (`Lfys64 Lfx96 Lrx96 Lfx384`, whose input width grows with the backbone channels): E2-R16 1,657,408; E2-R32 3,784,752; E2-R64 12,223,504; E2-LT 1,719,028; E2-LS 4,874,072 (the paper's small tier is 2.68M for the backbone alone; the repo's `Lcn4` counts the 3-branch training form and its SE is per stage, not alternating). The `Lcn4` stems pad by 1, so those arms give 171 frames at 512 px instead of 170 (B0, ResNet18: 170).
- **`Lcn4` is LCNetV4-*style*, not a bit-exact port.** The stem is two plain stride convs instead of the paper's branch stem, and SE sits on whole stages. Say "LCNetV4-style (our re-implementation)" in the paper; the Hub weights cannot be loaded into it.

## 2. Phase 0: protocol (blocking, ~1 week)

| ID | Task | Done when |
|---|---|---|
| P0.1 | Fix padding: pad with white (+1) and pass real lengths to CTC (already planned in README) | Same prediction at batch size 1 and 64 (≥ 99% identical) |
| P0.2 | Report **strict** CER (plain Levenshtein) next to the lenient one; add **cluster error rate** (CluER: a consonant + its marks is one unit) | Both in `metrics.py` output |
| P0.3 | Freeze splits: train (4 fonts) / val / **test-A** (same fonts, new text) / **test-B** (held-out fonts) / **test-R** (50 real lines, later LaoDocBench) | Split lists committed with a hash |
| P0.4 | One training recipe for everything = **the thesis recipe recovered in E1.0** (Adam 0.002 + scheduler, batch 64, 50,000 iterations, thesis augmentation and data; *not* the `real-match` preset), seed list `{1,2,3}` (5 for finalists) | One config file; every run logs git hash + spec + seed |
| P0.5 | Baseline **B0** = thesis `I64K5C16` (and `I64K5C32`), 5 seeds → measures seed noise σ for every metric. With `Lfx384` the C32 variant is `[1,64,0,1 Ct5,5,32 Mp3,3 Lfys64 Lfx96 Lrx96 Lfx384 O1c###]` ✓ (933,776 params); confirm 384 vs 512 first (E1.0 item 1) | σ table for CER, CluER, CoER per tier |
| P0.6 | Log T/L ratio (frames per label) and blank rate for every run | Columns in results CSV |

- ✅ **P0.1 done (2026-11-05, branch `step0-toolkit`):** white padding + real per-line lengths for CTC, decoding, `Lrx`/`Lbx`, attention and conv borders; batch size 1 vs 64 agree on 100% of lines in `tests/test_padding.py`; `--padding legacy` keeps the old behaviour for E1a.
- ✅ **P0.2 done (2026-11-05):** strict CER (plain Levenshtein) and CluER (consonant + tone / upper / lower / ຼ marks = 1 unit) in `evaluation`, `training.log`, TensorBoard and the console report; lenient CER unchanged.
- ✅ **P0.4 tooling done (2026-11-05), recipe still open:** `--seed` (torch / numpy / random / shuffle / deterministic cuDNN), and `config.json` records git commit, seed, resolved spec and all args; `--lr_scheduler plateau` (`--lr_factor`, `--lr_patience`, `--min_lr`) exists, but the thesis values must still come from E1.0. `experiments/step0_specs.py` lists the Step 0 specs and prints the commands per seed.
- ⬜ P0.3, P0.5, P0.6 not started (P0.6: T/L ratio and blank rate are not logged yet).

Everything below is relative to B0 and reports mean ± std over seeds, never a single run.

## 3. Phase 1: diagnostics on B0 (cheap, no new architecture)

These locate the bottleneck *before* spending GPU time on ablations. They reuse the 5 B0 checkpoints.

| ID | Experiment | How | What it separates |
|---|---|---|---|
| D1 | **Error anatomy** | Split CoER per tier into S / D / I (data already in `metrics.py`) | Deletions dominate → emission / CTC. Substitutions (ຸ↔ູ) dominate → visual resolution |
| D2 | **Forced-alignment posterior** | CTC-forced-align the GT (`torchaudio.functional.forced_align`); per tier, record mean posterior of the GT label at its aligned frame, and whether the model's argmax path has it at all | Mark never has posterior → H-CNN. Has posterior but is lost or swapped at decode → H-CTC |
| D3 | **Layer-wise linear probe** | Freeze B0. Train a linear classifier on forced-aligned frames from (a) CNN output after collapse, (b) after LSTM 1, (c) 2, (d) 3. Report per-tier frame accuracy | Where tier information is created or destroyed. Direct answer to "which module is critical" |
| D4 | **Random-string test** | Render lines of random valid Lao clusters (no language structure) with `gen_data_from_text.py`; compare CER with natural text | Large gap = LSTM acts as a language model and hides visual errors (H-LSTM) |
| D5 | **Order sensitivity** | Count how often marks are right but in the wrong order within a cluster (tone vs upper vowel) | Pure ordering errors are a CTC label-convention problem, not a vision problem |
| D6 | **Oracle upper bound** (optional) | Render tiers as separate input channels (base / upper / tone / lower) from the synthetic renderer and train B0 on that | Ceiling if tiers were perfectly separated. If this is still bad, the limit is not "interception" |

Expected deliverable: one figure, "tier information per layer" (D3), plus a table of D1/D2 per tier.
If D3 shows the CNN output already separates the tiers, Phase 2 shifts effort to groups B and C; if not, to group A.

## 4. Phase 2: single-factor ablations (one module at a time)

Rule: change **one** thing, keep everything else at B0, 3 seeds. ✓ = spec built and ran a forward pass; ○ = needs a spec check or code.
Each variant is run at equal spec and, where params differ by more than 25%, again param-matched (marked *pm*).

### Group A: CNN and height collapse (H-CNN)

| ID | Factor | Levels | Spec / change | Why |
|---|---|---|---|---|
| A1 | Kernel K | 3, **5**, 7, 9 | `Ct{K},{K},32` ○ | Receptive field in y. Redo the master's result under the fixed protocol with seeds |
| A2 | Depth | **1 conv**, 2 conv, 3 conv + `Bn` | `Ct5,5,32 Mp2,2 Ct5,5,64 Mp3,3` ✓ (993,232), 3-conv ○ | Same receptive field by depth rather than by one big kernel |
| A3 | Width C | 16, **32**, 64 | `Ct5,5,{C}` ○ | Capacity |
| A4 | Input height H | 32, 48, **64**, 96 | `1,{H},0,1` ○ | Resolution available to separate tiers |
| **A5** | **Collapse operator** | `Lfys` (**B0**), flatten (no collapse), `Atys`, `Matys`, `Hybys`, mean-pool, max-pool, +CoordConv y-channel | flatten ✓ `…Mp3,3 Lfx96…` (1,142,160); `Atys64` ✓ (915,088); `Matys64` ✓ (917,136); `Hybys64` ✓ (938,256); mean / max / CoordConv ○ (new tokens) | **Core test of H-CNN.** Flatten keeps the row position of every feature; every other operator mixes rows. Mean-pool is the "collapse with no learned selection" control |
| A6 | Y reduction | none, /2, **/3**, /4 | `Mpa3,3,{sy},3,…` ○ | How much vertical detail the collapse receives |

### Group B: sequence model (H-LSTM)

| ID | Factor | Levels | Spec / change | Why |
|---|---|---|---|---|
| B1 | Sequence model removed | none (+ `Fc` layers to match params) | `Lfys64 O1c###` ✓ (31,120, so also *pm* with `Fc`) | Does context matter at all once the CNN has a wide x receptive field? |
| B2 | Direction | `Lfx` only ✓ (747,920); `Lfx Lrx Lfx` (**B0**); true BiLSTM `Lbx96 Lbx384` ✓ (1,987,472, *pm*: `Lbx48 Lbx192` ○) | | Separates "bidirectional" from "deeper" |
| B3 | Hidden size | 64, 128, 256, **384**, 512 on the last layer | ○ | Capacity curve, finds the knee |
| B4 | Depth | 1, 2, **3** | ○ | Same |
| B5 | Architecture | LSTM vs `ConfX4 ConfX4` ✓ (124,816, *pm* needed) vs `SelfAttnX` + positional encoding ○ | | Is the recurrent layer special, or any context mixer works? Do not use `SelfAttnX` without PE (see §1.4) |

### Group C: CTC alignment and label unit (H-CTC)

| ID | Factor | Levels | Spec / change | Why |
|---|---|---|---|---|
| **C1** | **Time resolution** | x-stride 1, 2, **3**, 4, 6 → T/L = measure | `Mpa3,3,3,1,0,1` ✓ (T=512), `Mpa3,3,3,2,0,0` ✓ (T=255) | More frames per column. Tests whether CTC is starved. Keep the LSTM sizes fixed |
| **C2** | **Label unit** | (a) code points (**B0**), (b) canonical mark order, (c) grapheme clusters (consonant + marks as one class), (d) tier-wise multi-head CTC (base / upper / lower heads, shared encoder) | (a) as is; (b) reorder tone/upper vowel in labels ○; (c) new charset ○; (d) new training code ○ | **Core test of H-CTC.** (d) is the "2D multi-tier" idea from the PhD proposal, tested here at the CTC level |
| C3 | Auxiliary supervision | none, intermediate CTC on the collapse output, per-tier auxiliary CTC | ○ | Does extra supervision at the CNN output fix what the head cannot? |
| C4 | Decoding | greedy vs prefix beam (after fixing the double-letter bug, README #3) | ○ | Probably a small effect; confirm and move on |
| C5 | Is CTC itself the limit? (optional) | CTC vs a small attention decoder on the same encoder | ○ | Contrast baseline, links to the DAN direction |

## 5. Phase 3: factorial and variance decomposition

One-at-a-time ablations miss interactions (e.g. a better collapse only helps with more frames). Pick **2 levels per factor**
from Phase 2 (the best and the worst-but-sane), then run the full grid:

| Factor | Level − | Level + |
|---|---|---|
| F1 collapse | `Lfys` | best of A5 |
| F2 x-stride | 3 (B0) | best of C1 |
| F3 sequence model | `Lfx Lrx Lfx` | best of B2 / B5 |
| F4 label unit | code points | best of C2 |

2⁴ = 16 cells × 3 seeds = 48 runs. Responses: CoER tone, upper, lower, CluER, CER.
Fit a main-effects + two-way-interaction model and report **η² per factor** and per interaction.
The headline result is a sentence of the form "X% of the variance in lower-vowel CoER is explained by the collapse operator, Y% by label unit, Z% by sequence model"
and answers "which part is critical" with numbers.

## 6. Phase 4: confirmation

- Take the Pareto-optimal configurations (expected 2–3) and B0. Train with **5 seeds**.
- Evaluate on test-A (same fonts), **test-B (held-out fonts)**, test-R (real scans).
  The master's results show the gap: 91.49% synthetic vs 80.28% best on real, so an ablation that wins only on test-A is not a finding.
- **Data-vs-architecture control:** train B0 and the winner at 10k / 25k / 50k / 100k lines. If the gap closes at 100k, the limit is data, not architecture.
- Cost axis: params, FLOPs, CPU ms/line (target < 25 ms from RQ3), INT8 CER drop.
- Sweet spot = knee of the Pareto front over (CoER_stack, CPU ms/line); report the full front, not one point.

## 7. Metrics and statistics

| Metric | Use |
|---|---|
| CER strict and lenient | Headline; always both |
| CoER per tier (tone, upper, lower, before, rear, consonant) with S/D/I split | Main diagnostic; the PhD contribution |
| **CluER** (new, P0.2) | CER hides stack errors because a missed tone mark is 1 of ~15 characters. Counting a cluster as one unit makes them visible |
| Line accuracy | Strictest |
| T/L ratio, blank rate, D2 posterior | Explanatory, not outcome |
| Params, FLOPs, CPU ms/line | Cost axis |

Statistics: mean ± std over seeds; paired bootstrap over test lines (10,000 resamples, 95% CI) for any A-vs-B claim;
Holm correction across the claims in one table; report effect size next to p. With 3,093 real characters, CoER of rare classes (lower vowel, special) has wide CIs: say so, and wait for LaoDocBench (800–1,000 lines) before claiming real-scan results.

## 8. Run budget and order

| Phase | Configs × seeds | Runs |
|---|---|---|
| 0: B0 | 1 × 5 | 5 |
| 1: diagnostics | reuse B0 | 0 (+ optional D6: 1 × 3) |
| 2: ablations | A ≈ 17, B ≈ 10, C ≈ 9 → ~36 × 3 | ~108 |
| 3: factorial | 16 × 3 | 48 |
| 4: confirmation | ~3 × 5 + data-size runs | ~25 |
| **Total** | | **~190 runs** |

Runtime per run is not known to me; multiply by your T4 time for 20k iterations. To cut cost, screen Phase 2 with 1 seed at 50% of the iteration budget
and re-run only the variants within ±3σ of B0 or better with 3 full seeds. This is safe only because Phase 0 measures σ first.

**Minimum viable set (one conference paper, ~60 runs):** P0, D1–D4, A5, B1 + B2, C1 + C2(d), and a half-fraction of Phase 3 (2⁴⁻¹ = 8 cells × 3 seeds).
This is enough to answer "which module is critical" with η² and to place the sweet spot.

## 9. Threats to validity

- **Synthetic-to-real gap** (91.49 vs 80.28): rank the architectures on test-B and test-R as well, and report if the ranking flips.
- **Font bias:** 4 fonts, all looped. Test-B with held-out fonts is mandatory; add more fonts before the thesis version.
- **Shared recipe is tuned for B0.** Give each finalist a small LR sweep ({1e-3, 2e-3, 4e-3}) so the winner is not simply "closest to the recipe".
- **Padding and metric changes (P0.1, P0.2) invalidate old numbers.** Re-run B0; do not compare against the 91.49% from before the fix.
- **Parameter count vs capacity:** report both equal-spec and param-matched.
- **Test-set reuse:** choose configurations on val only; touch test-B / test-R once per finalist.

## 10. Suggested first three steps

0. **E1 (§1b): recover the thesis code, data and configs, then replicate I64K5C16 and I64K5C32 with 3 seeds.** This comes before everything else.
1. P0.1 + P0.2 (padding fix, strict CER, CluER) as the E1b arm, then the 5-seed B0 (P0.5). Without these no later number is trustworthy.
2. D1 → D2 → D3 on the B0 checkpoints. One week of mostly analysis code, no new training, and it tells you which of groups A / B / C deserves the budget.
3. A5 and C2(d), the two ablations that attack the vertical-stack problem most directly, at 3 seeds.
