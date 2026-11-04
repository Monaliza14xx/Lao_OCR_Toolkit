#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh  –  Tesseract training tools launcher
#
# Usage (from any directory):
#   ./run.sh plots                                        – Tesseract training-log plots   → plots/
#   ./run.sh plots --no-plots                             – skip plot rendering
#   ./run.sh compare                                      – Tesseract model comparison      → data/tesseract_comparison/
#   ./run.sh compare --no-plots                           – skip comparison plots
#   ./run.sh metric_explanation                           – CoER + CER/WER explanation      → data/tesseract_comparison/
#   ./run.sh metric_explanation --custom-gt=path/to/f.gt.txt  – same, with custom GT text
#   ./run.sh metric_explanation --num-samples=5 --min_length=30  – filter by length
#   ./run.sh inspection                                   – Tesseract sample inspection grids → data/tesseract_comparison/inspection/
#   ./run.sh all                                          – plots + compare + TensorBoard
#   ./run.sh all --no-plots                               – skip all plots
#   ./run.sh tensorboard                                  – Export Tesseract logs           → runs/
#   ./run.sh benchmark                                    – Tesseract vs PyTorch full suite → benchmark_output/
#   ./run.sh benchmark --no-plots                         – skip benchmark plots
#   ./run.sh benchmark --no-plots --models M1 M2 M3       – restrict to specific models
# ─────────────────────────────────────────────────────────────────────────────

# Always work relative to the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CMD="${1:-plots}"

print_sep() { echo "════════════════════════════════════════"; }

case "$CMD" in

  plots)
    print_sep
    echo "  Generating training-log plots …"
    print_sep
    python3 tesseract_tools.py plots \
        --data_dir tesseract_data \
        --out_dir  plots \
        "${@:2}"
    echo ""
    echo "✓ Plots saved to:  $SCRIPT_DIR/plots/"
    ;;

  compare)
    print_sep
    echo "  Running model comparison …"
    print_sep
    python3 tesseract_tools.py compare \
        --compare_dir tesseract_data/tesseract_comparison \
        --realworld_dir data/lao-real-documents \
        "${@:2}"
    echo ""
    echo "✓ Results saved to:  $SCRIPT_DIR/tesseract_data/tesseract_comparison/"
    ;;

  all)
    print_sep
    echo "  Running PLOTS + COMPARE + TENSORBOARD (Tesseract) …"
    print_sep
    python3 tesseract_tools.py all \
        --data_dir    tesseract_data \
        --out_dir     plots \
        --compare_dir tesseract_data/tesseract_comparison \
        --realworld_dir data/lao-real-documents \
        "${@:2}"
    echo ""
    echo "  Running PyTorch Model Comparison Analytics …"
    print_sep
    # Extract --no-plots and --models MODEL... from extra args to forward to torch_train.py compare
    _torch_args=()
    _in_models=0
    for _arg in "${@:2}"; do
        if [[ "$_arg" == "--no-plots" ]]; then
            _torch_args+=("--no-plots")
            _in_models=0
        elif [[ "$_arg" == "--models" ]]; then
            _torch_args+=("--models")
            _in_models=1
        elif [[ $_in_models -eq 1 && "$_arg" != --* ]]; then
            _torch_args+=("$_arg")
        else
            _in_models=0
        fi
    done
    python3 torch_train.py compare \
        --custom_img_dir           data/lao-ground-truth-test \
        --custom_realworld_img_dir data/lao-real-documents \
        --output_dir               data/comparison \
        "${_torch_args[@]}"
    echo ""
    echo "  Exporting Tesseract logs to TensorBoard …"
    python3 log_to_tensorboard.py --data_dir tesseract_data
    echo ""
    echo "✓ Plots           → $SCRIPT_DIR/plots/"
    echo "✓ Tess Comparison → $SCRIPT_DIR/tesseract_data/tesseract_comparison/"
    echo "✓ PT  Comparison  → $SCRIPT_DIR/data/comparison/"
    echo "✓ TensorBoard     → $SCRIPT_DIR/runs/"
    ;;

  tensorboard)
    print_sep
    echo "  Exporting logs to TensorBoard …"
    print_sep
    python3 log_to_tensorboard.py
    echo ""
    echo "✓ Done.  Start TensorBoard with:"
    echo "    tensorboard --logdir=$SCRIPT_DIR/runs"
    echo "  Then open:  http://localhost:6006/"
    ;;

  benchmark)
    print_sep
    echo "  Running full benchmark suite …"
    print_sep
    export TF_CPP_MIN_LOG_LEVEL=3
    python3 benchmark.py "${@:2}"
    echo ""
    echo "✓ Done.  Check the benchmark_output folder."
    ;;

  metric_explanation)
    print_sep
    echo "  Generating metric explanation diagrams …"
    print_sep
    # Forward any extra args (e.g. --custom-gt=...) straight to the Python CLI
    python3 tesseract_tools.py metric_explanation \
        --compare_dir data/tesseract_comparison \
        --models \
            LaoCRNN-I48K3C16 LaoCRNN-I48K3C32 \
            LaoCRNN-I48K5C16 LaoCRNN-I48K5C32 \
            I64K2C16 I64K2C32 \
            LaoCRNN-I64K3C16 LaoCRNN-I64K3C32 \
            I64K4C16 I64K4C32 \
            LaoCRNN-I64K5C16 LaoCRNN-I64K5C32 LaoCRNN-I64K5C64 \
            LaoCRNN-I64K7C16 LaoCRNN-I64K7C32 \
            LaoCRNN-I64K9C16 LaoCRNN-I64K9C32 \
        "${@:2}"
    echo ""
    echo "✓ Diagrams saved to:  $SCRIPT_DIR/data/tesseract_comparison/"
    ;;

  inspection)
    MODELS=(
        "LaoCRNN-I48K3C16" "LaoCRNN-I48K3C32"
        "LaoCRNN-I48K5C16" "LaoCRNN-I48K5C32"
        "I64K2C16"         "I64K2C32"
        "LaoCRNN-I64K3C16" "LaoCRNN-I64K3C32"
        "I64K4C16"         "I64K4C32"
        "LaoCRNN-I64K5C16" "LaoCRNN-I64K5C32" "LaoCRNN-I64K5C64"
        "LaoCRNN-I64K7C16" "LaoCRNN-I64K7C32"
        "LaoCRNN-I64K9C16" "LaoCRNN-I64K9C32"
    )

    print_sep
    echo "  Generating inspection diagrams (PyTorch) …"
    print_sep
    for model in "${MODELS[@]}"; do
        echo "► Inspecting PyTorch CNNs: $model"
        python3 torch_train.py inspection --model_name "$model" "${@:2}"
        echo ""
    done

    echo "✓ PyTorch CNN layers saved to: $SCRIPT_DIR/data/<model>/inspection/"
    ;;

  *)
    echo "Usage: $0 {plots|compare|metric_explanation|all|tensorboard|benchmark} [options]"
    echo ""
    echo "  plots              – Tesseract training-log plots           (→ plots/)"
    echo "  compare            – Tesseract CER/WER/CoER comparison      (→ data/tesseract_comparison/)"
    echo "  metric_explanation – CoER + CER/WER alignment diagrams      (→ data/tesseract_comparison/)"
    echo "                       Options:"
    echo "                         --custom-gt=<file.gt.txt>  use a specific GT text file"
    echo "  inspection         – Tesseract sample inspection grids      (→ data/tesseract_comparison/inspection/)"
    echo "  all                – plots + compare + TensorBoard (Tess)   (→ plots/, runs/)"
    echo "  tensorboard        – export Tesseract logs to TensorBoard   (→ runs/)"
    echo "  benchmark          – Tess vs PT full suite                  (→ benchmark_output/)"
    exit 1
    ;;
esac
