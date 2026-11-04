#!/usr/bin/env python3
import argparse

from tesstrain_core import (
    cmd_unicharset,
    cmd_lists,
    cmd_proto_model,
    cmd_training,
    cmd_evaluation,
    cmd_recognize,
    cmd_plot,
    cmd_clean,
    cmd_charset_stats
)
from tesstrain_core.inspection import cmd_inspection, cmd_compare_inspection, cmd_wer_distribution, cmd_confusion_matrix
from tesstrain_core.compare import cmd_compare
from tesstrain_core.projection import cmd_projection
from tesstrain_core.export import cmd_export


def add_common_args(parser):
    g = parser.add_argument_group('common options')
    g.add_argument('--model_name',       default='foo',  metavar='NAME',
                   help='Model name used for file/folder naming  (default: foo)')
    g.add_argument('--data_dir',         default='data', metavar='DIR',
                   help='Root data directory                     (default: data)')
    g.add_argument('--output_dir',       default='',     metavar='DIR',
                   help='Override output dir                     (default: data/MODEL_NAME)')
    g.add_argument('--ground_truth_dir', default='',     metavar='DIR',
                   help='Dir containing .lstmf / .gt.txt files  (default: data/MODEL_NAME/ground_truth)')
    g.add_argument('--tessdata',         default='',     metavar='DIR',
                   help='Tessdata directory (for proto-model)    (default: $TESSDATA_PREFIX)')
    g.add_argument('--random_seed', type=int,   default=0,    metavar='N',
                   help='RNG seed for reproducibility            (default: 0)')
    g.add_argument('--ratio_train', type=float, default=0.90, metavar='R',
                   help='Fraction of files used for training     (default: 0.90)')
    g.add_argument('--batch_size',  type=int,   default=64,   metavar='N',
                   help='Mini-batch size                         (default: 64)')
    g.add_argument('--num_workers', type=int,   default=2,    metavar='N',
                   help='DataLoader worker processes             (default: 2)')
    g.add_argument('--log_file',    default='', metavar='FILE',
                   help='CSV log output path                     (default: data/MODEL_NAME/log.csv)')

def main():
    parser = argparse.ArgumentParser(
        prog='torch_train.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            'Tesseract PyTorch GPU Training Pipeline\n'
            '----------------------------------------\n'
            'Train, evaluate, and export Tesseract LSTM models on GPU.\n'
            'Choose one of the subcommands below to run a pipeline step.\n'
        ),
        epilog="""\
--------------------------------------------------------
 Typical workflow
--------------------------------------------------------
  1. unicharset    Build character set from ground truth
  2. lists         Split files into train / eval lists
  3. proto-model   Package a starter .traineddata
  4. training      Train the model on GPU
  5. evaluation    Measure CER / WER on eval set
  6. export        Export PyTorch -> ONNX
--------------------------------------------------------
 Other commands
--------------------------------------------------------
  recognize        OCR a single image file
  plot             Generate a training error-rate chart
  charset-stats    Show character frequency statistics
  clean            Remove all generated files
--------------------------------------------------------

Quick start example:
  python torch_train.py training \\
      --model_name lao --batch_size 32 --epochs 50

Get help for a subcommand:
  python torch_train.py training -h
""",
    )


    sub = parser.add_subparsers(dest='command')

    # unicharset
    p = sub.add_parser('unicharset', help='Build character set from .gt.txt ground truth files')
    add_common_args(p)

    # lists
    p = sub.add_parser('lists', help='Split .lstmf files into train / eval / valid lists')
    add_common_args(p)
    p.add_argument('--ratio', type=int, nargs='+', metavar='N',
                   help='Split ratio as integers. '
                        '2 values: train val (e.g. 90 10). '
                        '3 values: train val test (e.g. 80 10 10). '
                        'Values are treated as percentages and normalized to 100. '
                        'If omitted, uses --ratio_train (default 0.90).')

    # proto-model
    p = sub.add_parser('proto-model', help='Package a starter .traineddata (needed before training)')
    add_common_args(p)

    # training
    p = sub.add_parser('training', help='Train the LSTM model on GPU',
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       description='Train the Tesseract LSTM model on GPU.\n'
                                   'Use --epochs for epoch-based stopping, or --max_iterations for step-based.',
                       epilog="""\
--------------------------------------------------------
 Quick start: Train from scratch
--------------------------------------------------------
  python torch_train.py training \\
      --model_name lao \\
      --batch_size 32 \\
      --epochs 100 \\
      --learning_rate 0.002

--------------------------------------------------------
 Quick start: Fine-tune from an existing checkpoint
--------------------------------------------------------
  python torch_train.py training \\
      --model_name lao \\
      --start_model data/lao/lao_best.pt \\
      --batch_size 32 \\
      --epochs 50 \\
      --learning_rate 0.0005

--------------------------------------------------------
 Quick start: Transfer from Tesseract CPU model
--------------------------------------------------------
  python torch_train.py training \\
      --model_name lao \\
      --start_tess_model /path/to/lao.traineddata \\
      --net_spec "[1,48,0,1 Ct3,3,16 Mp3,3 Lfys64 Lfx96 Lrx96 Lfx384 O1c###]" \\
      --batch_size 32 \\
      --epochs 50 \\
      --learning_rate 0.0005

  Tips:
    - Use a lower --learning_rate when fine-tuning (e.g. 0.0005)
    - --start_model accepts a .pt checkpoint saved during a previous run
    - --start_tess_model accepts a Tesseract .traineddata or .lstm file
    - For --start_tess_model, make sure --net_spec matches the source model
    - Best checkpoint is saved as data/MODEL_NAME/MODEL_NAME_best.pt
""")
    add_common_args(p)
    g = p.add_argument_group('model')
    g.add_argument('--start_model', default='', metavar='PATH',
                   help='Checkpoint (.pt) to fine-tune from        (default: train from scratch)')
    g.add_argument('--start_tess_model', default='', metavar='PATH',
                   help='Tesseract model (.traineddata/.lstm) for transfer learning warm-start')
    g.add_argument('--net_spec',
                   default='[1,48,0,1 Ct3,3,16 Mp3,3 Lfys64 Lfx96 Lrx96 Lfx384 O1c###]',
                   metavar='SPEC',
                   help='VGSL network architecture string')
    g = p.add_argument_group('optimizer')
    g.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'], metavar='{Adam,SGD}',
                   help='Optimizer algorithm                        (default: Adam)')
    g.add_argument('--learning_rate', type=float, default=0.002, metavar='LR',
                   help='Initial learning rate                      (default: 0.002)')
    g.add_argument('--momentum', type=float, default=0.9, metavar='M',
                   help='Momentum - only used with SGD              (default: 0.9)')
    g.add_argument('--weight_decay', type=float, default=0.0, metavar='WD',
                   help='L2 weight decay penalty                    (default: 0.0)')
    g.add_argument('--force_GPU', action='store_true',
                   help='Require GPU for training; fail if not available')
    g = p.add_argument_group('stopping criteria')
    g.add_argument('--max_iterations', type=int, default=10000, metavar='N',
                   help='Stop after N gradient steps                (default: 10000)')
    g.add_argument('--epochs', type=int, default=0, metavar='N',
                   help='Stop after N full epochs (overrides --max_iterations, 0=off)')
    g.add_argument('--target_error_rate', type=float, default=0.01, metavar='CER',
                   help='Stop early when eval CER drops below this  (default: 0.01)')
    g = p.add_argument_group('logging & checkpoints')
    g.add_argument('--debug_interval', type=int, default=100, metavar='N',
                   help='Print training stats every N steps         (default: 100)')
    g.add_argument('--save_interval', type=int, default=2000, metavar='N',
                   help='Save a checkpoint every N steps            (default: 2000)')
    g.add_argument('--tensorboard', action='store_true', default=True,
                   help='Enable TensorBoard logging                 (default: on)')
    g.add_argument('--no-tensorboard', dest='tensorboard', action='store_false',
                   help='Disable TensorBoard logging')
    g.add_argument('--tensorboard_dir', default='', metavar='DIR',
                   help='TensorBoard log dir  (default: data/MODEL_NAME/tensorboard)')
    g = p.add_argument_group('language model hints (optional)')
    g.add_argument('--wordlist_file', default='', metavar='FILE', help='Word list file')
    g.add_argument('--numbers_file', default='', metavar='FILE', help='Numbers list file')
    g.add_argument('--punc_file',    default='', metavar='FILE', help='Punctuation list file')


    p = sub.add_parser('evaluation', help='Measure CER / Accuracy on the eval set')
    add_common_args(p)
    p.add_argument('--beam_width', type=int, default=0, metavar='N',
                   help='CTC decode beam width  (0=greedy, 5-10=beam search, default: 0)')
    p.add_argument('--csv', default='', metavar='FILE',
                   help='Export detailed evaluation metrics to CSV file')

    # recognize
    p = sub.add_parser('recognize', help='Run OCR on a single image file')
    add_common_args(p)
    p.add_argument('--image', required=True, metavar='FILE',
                   help='Path to the image to recognize')
    p.add_argument('--psm', type=int, default=7, metavar='N',
                   help='Page segmentation mode: 3=auto  6=block  7=single-line  13=raw  (default: 7)')
    p.add_argument('--beam_width', type=int, default=0, metavar='N',
                   help='CTC decode beam width  (0=greedy, 5-10=beam search, default: 0)')
    p.add_argument('--wordlist', default='', metavar='FILE',
                   help='Word list file for DAWG-constrained decoding')
    p.add_argument('--lm_corpus', default='', metavar='PATH',
                   help='Text corpus / .gt.txt dir for n-gram language model')
    p.add_argument('--lm_weight', type=float, default=0.5, metavar='W',
                   help='Language model weight  0.0–1.0             (default: 0.5)')

    # plot
    p = sub.add_parser('plot', help='Generate a training error-rate chart from the CSV log')
    add_common_args(p)

    p = sub.add_parser('charset-stats', help='Show character frequency statistics from .gt.txt files')
    add_common_args(p)


    # inspection
    p = sub.add_parser('inspection', help='Export feature maps (activations) of random evaluation images')
    add_common_args(p)
    p.add_argument('--num-samples', '--num_samples', dest='num_samples', type=int, default=10, help='Number of images to inspect (default: 10)')
    p.add_argument('--min_length', '--min-length', dest='min_length', type=int, default=3, help='Minimum length of GT text to be selected (default: 3)')
    p.add_argument('--seed', type=int, default=42, help='RNG seed for sample selection — same seed = same samples across all models (default: 42)')
    p.add_argument('--tessdata_dir', default='', help='Directory with .traineddata models (if inspecting a Tesseract model)')

    # compare
    p = sub.add_parser('compare', help='Compare training logs and optionally evaluate models on a custom image set')
    p.add_argument('--models', nargs='*', default=[], metavar='MODEL', help='List of models to compare (e.g., lao lao_v2)')
    p.add_argument('--custom_img_dir', default='', metavar='DIR', help='Directory with .gt.txt and images for custom evaluation')
    p.add_argument('--custom_realworld_img_dir', default='', metavar='DIR', help='Directory with .gt.txt and images for custom real-world evaluation')
    p.add_argument('--output_dir', default='data/comparison', metavar='DIR', help='Directory to save comparison plot (default: data/comparison)')
    p.add_argument('--report', default='', metavar='FILE', help='Export detailed model comparison report to Markdown')
    p.add_argument('--tess_models_dir', default='', metavar='DIR', help='Directory with .traineddata models to include in comparison')
    p.add_argument('--no-plots', dest='no_plots', action='store_true', help='Skip generating matplotlib comparison plots')

    # projection
    p = sub.add_parser('projection', help='Generate vertical projection profile alignment grids for model predictions on custom datasets')
    p.add_argument('--models', nargs='*', default=[], metavar='MODEL', help='List of models to evaluate')
    p.add_argument('--num-samples', '--num_samples', dest='num_samples', type=int, default=5, help='Number of samples to visualize per dataset')
    p.add_argument('--min_length', '--min-length', dest='min_length', type=int, default=40, help='Minimum length of GT text to select')
    p.add_argument('--custom_img_dir', default='', metavar='DIR', help='Directory with .gt.txt and images for custom evaluation')
    p.add_argument('--custom_realworld_img_dir', default='', metavar='DIR', help='Directory with .gt.txt and images for custom real-world evaluation')
    p.add_argument('--output_dir', default='data/projection', metavar='DIR', help='Directory to save projection plots')

    # compare-inspection
    p = sub.add_parser('compare-inspection', help='Cross-model visual inspection comparing activation maps and predictions')
    add_common_args(p)
    p.add_argument('--num_samples', type=int, default=5, help='Number of shared samples to visualize per pair (default: 5)')
    p.add_argument('--min_length', '--min-length', dest='min_length', type=int, default=3, help='Minimum GT length to be selected (default: 3)')
    p.add_argument('--seed', type=int, default=42, help='RNG seed — same seed = same samples across all pairs (default: 42)')
    p.add_argument('--base_data_dir', default='data', metavar='DIR', help='Base data directory containing model folders (default: data)')
    p.add_argument('--val_list', default='', metavar='FILE', help='Path to the validation split file (e.g. data/lao/lists.val)')

    # wer-distribution
    p = sub.add_parser('wer-distribution', help='Analyse WER distribution grouped by GT word count')
    add_common_args(p)
    p.add_argument('--tessdata_dir', default='', help='Directory with .traineddata models (if analysing a Tesseract model)')
    p.add_argument('--custom_dir', default='', metavar='DIR',
                   help='Custom dataset directory with images + .gt.txt files (overrides default eval split)')

    # confusion-matrix
    p = sub.add_parser('confusion-matrix', help='Per-character error analysis grouped by Lao component category')
    add_common_args(p)
    p.add_argument('--tessdata_dir', default='', help='Directory with .traineddata models (if analysing a Tesseract model)')
    p.add_argument('--custom_dir', default='', metavar='DIR',
                   help='Custom dataset directory with images + .gt.txt files (overrides default eval split)')

    # clean
    p = sub.add_parser('clean', help='Remove all generated files for the model')
    add_common_args(p)

    # export
    p = sub.add_parser('export', help='Export PyTorch model to ONNX format',
                       conflict_handler='resolve')
    add_common_args(p)
    # Override --model_name to accept one or more model names for batch export
    p.add_argument('--model_name', nargs='+', required=True, metavar='NAME',
                   help='One or more model names to export  '
                        '(e.g. LaoCRNN-I64K5C16 LaoCRNN-I64K3C16 I64K2C16)')

    args = parser.parse_args()

    cmds = {
        'unicharset': cmd_unicharset,
        'lists': cmd_lists,
        'proto-model': cmd_proto_model,
        'training': cmd_training,
        'evaluation': cmd_evaluation,
        'recognize': cmd_recognize,
        'plot': cmd_plot,
        'charset-stats': cmd_charset_stats,
        'inspection': cmd_inspection,
        'compare-inspection': cmd_compare_inspection,
        'wer-distribution': cmd_wer_distribution,
        'confusion-matrix': cmd_confusion_matrix,
        'compare': cmd_compare,
        'projection': cmd_projection,
        'export': cmd_export,
        'clean': cmd_clean,
    }

    if args.command in cmds:
        if args.command == 'export':
            # Batch export: iterate over every model name supplied
            for name in args.model_name:
                args_copy = argparse.Namespace(**vars(args))
                args_copy.model_name = name
                cmds['export'](args_copy)
        else:
            cmds[args.command](args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
