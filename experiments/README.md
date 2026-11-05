# Experiments

Scripts used for the research behind this toolkit: Tesseract-vs-PyTorch benchmarks,
cross-model comparisons, corpus enrichment, augmentation validation and paper figures.
They are kept for reproducibility. Several of them expect the model names and folder
layout used in those experiments (for example `LaoCRNN-I48K3C16`, `tesseract_data/`),
so they are less general than the toolkit itself.

Run them from the repository root, for example:

```bash
python3 experiments/benchmark.py --no-plots
python3 experiments/compare_cross.py --help
```

| Script | Purpose |
|--------|---------|
| `benchmark.py` | Full Tesseract-vs-PyTorch benchmark suite (`./run.sh benchmark`) |
| `run_benchmark.py` | Checks dependencies, then runs `compare_pt_vs_tess.py` |
| `compare_pt_vs_tess.py` | Tesseract (`.traineddata`) vs PyTorch (`.pt`) on the same images |
| `compare_cross.py` | Original Tesseract models vs fine-tuned ones (CSV reports) |
| `build_augmented_corpus.py` | Enrich a corpus with real ground truth and oversample rare characters |
| `validate_augmentation.py` | KL divergence + image gallery: augmented vs real data |
| `plot_training.py` | Plots from Tesseract LSTM training logs |
| `plot_confusion_matrix.py` | Per-class character confusion matrices from prediction CSVs |
| `generate_projection_profile.py` | Projection-profile figure of a rendered Lao word |
| `step0_specs.py` | Step 0 part-swap net_specs ([`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)): table of params / rows / frames, and `torch_train.py` commands per seed |
