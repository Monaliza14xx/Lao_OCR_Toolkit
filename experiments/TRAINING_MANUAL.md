# Training manual

How to train, check and compare the models of [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) with
`torch_train.py`. Every flag below was read from `torch_train.py` / `tesstrain_core/training.py`
(not from memory); things I could not confirm are marked **unverified**.

## 1. Before you start

```bash
git clone https://github.com/Monaliza14xx/Lao_OCR_Toolkit.git && cd Lao_OCR_Toolkit   # Colab: use a GPU runtime
pip install -r requirements.txt
python3 -c "import torch; print(torch.cuda.is_available())"        # must print True for --force_GPU
```

**Data.** Training reads `.lstmf` files (image + text), not loose PNG/JPG:

```bash
# images + .gt.txt pairs  →  .lstmf, once
python3 gen_data_from_text.py --convert_only --ext jpg --output_dir data/lao-ground-truth
```

For the thesis replication use the **original** 100k-line set (see EXPERIMENT_PLAN §1b, item 5). Do not regenerate it with the current generator.

**Train / eval split.** Two ways:

| Way | How | Use for |
|---|---|---|
| Automatic | no lists: the files are shuffled with `--random_seed` (default 0) and split by `--ratio_train` (default 0.90) | quick tries |
| **Fixed lists** | `python3 torch_train.py lists --model_name X --ground_truth_dir data/lao-ground-truth --ratio 90 10` → `data/X/list.train`, `list.val`; training uses them if `list.train` exists | **all experiments**: every run must see the same eval split |

The "eval" split is the one used for early stopping and the `_best.pt` checkpoint. Your test sets (2,000 synthetic lines, 50 real lines) are separate and are only scored after training (§5).

## 2. Anatomy of a training command

Your example, annotated:

```bash
!time python3 torch_train.py training --model_name LaoCRNN-I64K5C32-D20 \    # run name; also the checkpoint file prefix
    --net_spec "[1,64,0,1 Ct5,5,32 Mp3,3 Lfys64 Lfx96 D20 Lrx96 Lfx384 O1c###]" \   # the network (§3)
    --ground_truth_dir data/lao-ground-truth \          # folder with the .lstmf files
    --batch_size 64 --max_iterations 50000 \            # thesis: 64 and 50,000
    --learning_rate 0.002 \                             # thesis: Adam, 0.002
    --target_error_rate 0.001 --debug_interval 500 \    # stop when eval CER <= 0.1% ; log every 500 steps
    --wordlist_file data/lao_dict.txt \                 # see note below: no effect on this training
    --output_dir data/LaoCRNN-I64K5C32-D20 --force_GPU  # where files go ; fail instead of running on CPU
```

Notes on what that command really does:

- `!time` is a Colab/Jupyter shell escape; in a terminal use `time python3 …`.
- **`--target_error_rate 0.001` is 10× stricter than the default (0.01)**, which is what the thesis runs used. The run stops as soon as the *eval* CER is ≤ this value. Use `0` to disable early stopping (fixed 50,000 iterations, plan arm E1c).
- **Evaluation runs only every 5 × `--debug_interval` iterations** (here every 2,500). With `--debug_interval 500` the early-stop check and `_best.pt` are therefore updated only at 2,500, 5,000, … A run shorter than that has no `_best.pt`.
- `--wordlist_file` (also `--numbers_file`, `--punc_file`): I found it used only in `tesstrain_core/command_utils.py` (Tesseract tooling), not in `training.py`. **Unverified:** that the PyTorch training ignores it completely. Do not expect it to change the result.
- `D20` is Dropout 20% (the thesis model has none). A different network, so it needs its own name and is not B0.
- `###` in `O1c###` is replaced by the number of classes found in the data (79 characters + blank = 80 for the thesis data).

## 3. The network string (`--net_spec`)

`[1,64,0,1 …]` = batch 1 (placeholder), **height 64**, width variable, 1 channel. Then the layers:

| Token | Meaning |
|---|---|
| `Ct5,5,16` | conv 5×5, 16 channels, tanh |
| `Mp3,3` | max-pool 3×3, stride 3 (T = W/3) |
| `Lfys64` | LSTM down each column, keeps the last state: collapses the height |
| `Lfx96` / `Lrx96` / `Lbx96` | LSTM along the line: forward / reverse / bidirectional |
| `O1c###` | linear output + CTC |
| `Res18,16[,sx]` | ResNet18 backbone, width 16, T = W/sx (default 3) |
| `Lcn4[s],C,D[,sy,sx]` | LCNetV4-style stage |

Thesis model **B0** (I64K5C16): `[1,64,0,1 Ct5,5,16 Mp3,3 Lfys64 Lfx96 Lrx96 Lfx384 O1c###]` (929,264 params).
Check the LSTM width (384 vs 512) before claiming "same as the thesis" (EXPERIMENT_PLAN §1b, item 1).
Full token table: README, "Network Architecture (VGSL)".

## 4. Recipes

All recipes use the **same data, lists, batch size, iterations and learning rate** and change only what the row says.

| Goal | Add to the command |
|---|---|
| **E1a** faithful rerun of your old Colab runs | exactly the old flags (`--batch_size 64 --max_iterations 50000 --learning_rate 0.002 --target_error_rate 0.001 --debug_interval 500`, **no** `--lr_scheduler`, i.e. constant LR) plus `--padding legacy` (the old code padded with black) and `--seed 1` (repeatability only) |
| **E1b** same, padding fixed | the same flags without `--padding legacy` (`--padding white` is the default) |
| Thesis text says ReduceLROnPlateau | only if the old `training.log` shows a falling `learning_rate` column: then add `--lr_scheduler plateau` (settings unknown) |
| **E1c** equal training for all | add `--target_error_rate 0` |
| **E3** cluster CTC labels | `--label_unit cluster` (T1), `--label_unit cluster_am` (T2) or `--label_unit open --min_cluster_count 20` (T3); B0 = no flag. Use a new `--output_dir`: a folder that already has a `charset.json` of another unit is refused |
| Repeat with another seed | `--seed 2`, `--seed 3` (also change `--model_name` / `--output_dir`) |

Minimal reproducible command (B0, seed 1, white padding):

```bash
time python3 torch_train.py training --model_name B0-s1 \
    --net_spec "[1,64,0,1 Ct5,5,16 Mp3,3 Lfys64 Lfx96 Lrx96 Lfx384 O1c###]" \
    --ground_truth_dir data/lao-ground-truth \
    --batch_size 64 --max_iterations 50000 --learning_rate 0.002 \
    --lr_scheduler plateau --seed 1 \
    --debug_interval 500 --target_error_rate 0.01 \
    --output_dir data/runs/B0-s1 --force_GPU
```

**All Step 0 / E2 configurations with 3 seeds each** (specs and the shared recipe are generated, so none can drift):

```bash
python3 experiments/step0_specs.py                       # table: name, params, rows, frames
python3 experiments/step0_specs.py --commands --only B0 backbone --seeds 1 2 3 \
    --output_root data/runs \
    --extra "--ground_truth_dir data/lao-ground-truth --debug_interval 500 --force_GPU" > run_all.sh
bash run_all.sh                                          # or run the lines one by one
```

The runs are named `step0-<config>-s<seed>` (for example `step0-E2-R16-s2`); use those names in §6. `B0` + `backbone` = 8 configurations × 3 seeds = 24 commands.
`--only` takes names (`B0`, `E2-R16`) or groups (`squash`, `rnn`, `cnn`, `xstride`, `backbone`, `backbone-native`).
The generated recipe is `--batch_size 64 --learning_rate 0.002 --max_iterations 50000 --lr_scheduler plateau`; override it with `--train_args`.

**Colab tips.** Write `--output_dir` to Google Drive (or copy `data/runs` there at the end) because the runtime is wiped; a run that is interrupted can be continued from a saved checkpoint (`--start_model data/runs/X/X_checkpoint_<N>.pt`, saved every `--save_interval` = 2,000 steps), but the optimizer state and the seed stream are **not** guaranteed to match an uninterrupted run, so rerun from scratch for the final numbers (**unverified** in detail).

## 5. What a run leaves behind

In `--output_dir`:

| File | Content |
|---|---|
| `<name>_best.pt` | checkpoint with the lowest eval CER (only if an evaluation ran) |
| `<name>_final.pt` | last step |
| `<name>_checkpoint_<N>.pt` | every `--save_interval` steps |
| `config.json` | all arguments, resolved spec, seed, git commit, parameter count, padding mode |
| `charset.json` | the character set (index ↔ character) |
| `training.log` | CSV: loss, train/eval CER (strict and lenient), learning rate, speed per logged step |
| `tensorboard/` | `tensorboard --logdir data/runs/X/tensorboard` |

Checkpoints store `net_spec`, so every later command rebuilds the network from the checkpoint alone.

## 6. After training: score on the test sets

```bash
# eval split (same as early stopping)
python3 torch_train.py evaluation --model_name B0-s1 --output_dir data/runs/B0-s1 \
    --ground_truth_dir data/lao-ground-truth

# the thesis test sets: 2,000 synthetic lines and the 50 real lines (folders of image + .gt.txt)
python3 torch_train.py compare --models B0-s1 E2-R16-s1 \
    --custom_img_dir data/lao-ground-truth-test \
    --custom_realworld_img_dir data/real-lines-thesis \
    --output_dir data/comparison
```

`compare` expects the models under `--data_dir` (default `data/<model_name>/`), so either keep that layout or pass `--data_dir data/runs` (**unverified** that `compare` honours `--data_dir`; check its first output lines).
`data/real-lines/` in this repo is a different document, not the 50 thesis lines.
For batch-independent scoring of BatchNorm models (ResNet18, `Lcn4`) use the default `--padding white`.

## 7. Rules that keep the comparison fair

1. One recipe for the whole table (§4); change only the factor under test.
2. Same `list.train` / `list.val` and the same test folders for every run.
3. At least 3 seeds (`--seed 1 2 3`); report mean ± std, not the best run.
4. Never pick a configuration on the test sets; use the eval split.
5. Keep `config.json` and `training.log` of every run. They are the record of what was run.
6. Report whether a run stopped early (`Target CER … reached` in the log) and at which iteration.

## 8. Problems

| Symptom | Likely cause / fix |
|---|---|
| `--force_GPU was specified, but no GPU` | Colab: Runtime → Change runtime type → GPU |
| `No .lstmf files` | wrong `--ground_truth_dir`, or the images were not converted (§1) |
| Loss is `nan` / `inf`, or never drops | CTC cannot align when frames < labels (T < L): a backbone with a large width stride. Keep T ≈ W/3. The loss is built with `zero_infinity=True`, so impossible samples add **zero** loss silently; check `output_lengths` with `python3 experiments/step0_specs.py` (column `T@512`) |
| Out of memory | lower `--batch_size` (changes the recipe: record it), or use a narrower backbone (`Res18,16` instead of `Res18,64`) |
| Stops at a few thousand iterations | the early-stop rule fired (`--target_error_rate`); use `0` for a fixed budget |
| No `_best.pt` | fewer than 5 × `--debug_interval` iterations ran, so no evaluation happened |
| Different results with the same seed | `--seed` makes runs repeatable only up to the CUDA ops that have no deterministic version (the CTC loss backward is one; torch warns instead of failing), and only on the same hardware and library versions. Two runs with one seed can differ slightly on GPU; run all seeds of a table on the same setup |
