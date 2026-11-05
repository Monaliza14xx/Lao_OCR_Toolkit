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
