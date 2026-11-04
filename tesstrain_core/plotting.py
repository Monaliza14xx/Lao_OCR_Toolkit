import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ──────────────────────────────────────────────────────────────────────────────
# PAPER-READY THEME CONFIG
#
# Design goals:
#   • White canvas with light grey gridlines (clean paper style)
#   • Semi-transparent raw data + bold smoothed overlay
#   • Clean sans-serif typography suitable for IEEE papers
#   • High contrast for B&W printing
#   • 300 DPI output for publication
# ──────────────────────────────────────────────────────────────────────────────

# ── Canvas ────────────────────────────────────────────────────────────────────
TB_BG        = '#FFFFFF'     # Figure background (white for paper)
TB_CANVAS    = '#FFFFFF'     # Axes background (white for paper)
TB_GRID      = '#E0E0E0'     # Light grey gridlines on white canvas
TB_SPINE     = '#CCCCCC'     # Subtle spine colour
TB_TEXT      = '#333333'     # Main text
TB_TICK      = '#666666'     # Tick labels

# ── Semantic line colours ─────────────────────────────────────────────────────
C_TRAIN      = '#90CAF9'     # Light blue  (raw train)
C_EVAL       = '#FF8A65'     # Light orange (raw eval)
C_SMOOTH     = '#1565C0'     # Bold blue   (smoothed train)
C_ESMOOTH    = '#C62828'     # Bold red    (smoothed eval)
C_ACC_T      = '#2E7D32'     # Green       (train accuracy)
C_ACC_E      = '#E65100'     # Orange      (eval accuracy)
C_LR         = '#7B1FA2'     # Purple      (learning rate)
C_GPU        = '#00796B'     # Teal        (GPU metrics)

# ── TensorBoard-like multi-run palette ────────────────────────────────────────
TB_PALETTE = [
    '#FF6F00',   # amber-800
    '#1565C0',   # blue-800
    '#2E7D32',   # green-800
    '#C62828',   # red-800
    '#6A1B9A',   # purple-800
    '#00838F',   # cyan-800
    '#EF6C00',   # orange-800
    '#AD1457',   # pink-800
    '#283593',   # indigo-800
    '#558B2F',   # light-green-800
]

# ── Typography ────────────────────────────────────────────────────────────────
FONT_FAMILY  = 'sans-serif'
FONT_TITLE   = 13
FONT_LABEL   = 11
FONT_TICK    = 10
FONT_LEGEND  = 9

# ── Output ────────────────────────────────────────────────────────────────────
DPI_PAPER    = 300            # Publication DPI


# ──────────────────────────────────────────────────────────────────────────────
# CORE UTILS
# ──────────────────────────────────────────────────────────────────────────────

def smooth_ema(data, weight=0.6):
    """Exponential Moving Average smoothing."""
    if len(data) < 2:
        return data
    smoothed = []
    # Standard format for this plotter: [(iteration, value), ...]
    last = data[0][1]
    for x, y in data:
        s = last * weight + y * (1 - weight)
        smoothed.append((x, s))
        last = s
    return smoothed


def apply_premium_theme(fig, axes_list):
    """Reset to the original style of matplotlib."""
    import matplotlib.pyplot as plt
    plt.rcdefaults()


def add_stats_box(ax, text, loc='upper right'):
    """Add a monospace stats summary box."""
    props = dict(boxstyle='round,pad=0.5', facecolor=TB_BG,
                 edgecolor=TB_SPINE, alpha=0.9)
    anchors = {
        'upper right': (0.98, 0.95), 'upper left': (0.02, 0.95),
        'lower right': (0.98, 0.05), 'lower left': (0.02, 0.05),
    }
    x, y = anchors.get(loc, (0.98, 0.95))
    ha = 'right' if 'right' in loc else 'left'
    va = 'top' if 'upper' in loc else 'bottom'
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8,
            verticalalignment=va, horizontalalignment=ha,
            bbox=props, family='monospace', color=TB_TEXT, zorder=10)


def add_best_annotation(ax, data, color, mode='min', label_prefix='Best'):
    """Add a subtle annotation pointing to the best value in series."""
    if not data: return
    if mode == 'min':
        bx, by = min(data, key=lambda p: p[1])
    else:
        bx, by = max(data, key=lambda p: p[1])

    label = f'{label_prefix}: {by:.2f} @ {bx}'
    ax.annotate(label, xy=(bx, by),
                xytext=(15, 15), textcoords='offset points',
                fontsize=8, color=color, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor=TB_BG,
                          edgecolor=color, alpha=0.85), zorder=15)


def fill_between_smooth(ax, data, smoothed, color, alpha=0.08):
    """Fill variance area between raw and smoothed lines."""
    if len(data) < 2: return
    xs = [x for x, _ in data]
    ys_raw = [y for _, y in data]
    ys_smooth = [y for _, y in smoothed]
    if len(xs) == len(ys_smooth):
        ax.fill_between(xs, ys_smooth, ys_raw, color=color, alpha=alpha, zorder=1)


def tb_plot_series(ax, data, color, label,
                   smoothing=0.6, raw_alpha=0.25, raw_lw=0.8,
                   smooth_lw=2.2, fill=True):
    """
    Plot a single series in TensorBoard style:
      • Faint raw data behind
      • Bold smoothed line on top
      • Optional fill between raw & smoothed
    """
    if not data:
        return
    sm = smooth_ema(data, weight=smoothing)
    ax.plot(*zip(*data), color=color, alpha=raw_alpha, lw=raw_lw)
    ax.plot(*zip(*sm), color=color, lw=smooth_lw, label=label)
    if fill:
        fill_between_smooth(ax, data, sm, color, alpha=0.06)


def tb_save(fig, path, dpi=DPI_PAPER):
    """Save figure with paper-quality settings."""
    fig.savefig(path, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor(),
                edgecolor='none')
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# DASHBOARD CLASS
# ──────────────────────────────────────────────────────────────────────────────

class DashboardPlotter:
    def __init__(self, model_name, output_dir):
        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _get_metric(self, entries, key, iters_key='iteration'):
        """Extract (iter, val) pairs for a specific metric."""
        return [(e[iters_key], e[key]) for e in entries
                if e.get(key) is not None and e.get(key) != '']

    def generate_dashboard(self, entries, epoch_boundaries=None):
        """Generate the unified 6-panel dashboard."""
        fig, axes = plt.subplots(3, 2, figsize=(18, 16))
        fig.suptitle(f'{self.model_name} — Training Dashboard',
                     fontsize=20, fontweight='bold', color=TB_TEXT, y=0.98)
        apply_premium_theme(fig, axes.flatten())

        # Extract data
        tcer       = self._get_metric(entries, 'train_cer')
        ecer       = self._get_metric(entries, 'eval_cer')
        tacc       = self._get_metric(entries, 'train_acc')
        eacc       = self._get_metric(entries, 'eval_acc')
        twer       = self._get_metric(entries, 'train_wer')
        ewer       = self._get_metric(entries, 'eval_wer')

        # --- Column 1: Train ---
        # 1. train_acc (row 0, col 0)
        ax = axes[0, 0]
        if tacc:
            tb_plot_series(ax, tacc, C_ACC_T, 'Train Acc')
        ax.set_title('Training Character Accuracy', fontweight='bold')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right')

        # 2. train_loss_cer (row 1, col 0)
        ax = axes[1, 0]
        if tcer:
            tb_plot_series(ax, tcer, C_SMOOTH, 'Train CER')
        ax.set_title('Training Character Error Rate (CER)', fontweight='bold')
        ax.set_ylabel('CER')
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper right')

        # 3. train_loss_wer (row 2, col 0)
        ax = axes[2, 0]
        if twer:
            tb_plot_series(ax, twer, C_SMOOTH, 'Train WER')
        ax.set_title('Training Word Error Rate (WER)', fontweight='bold')
        ax.set_ylabel('WER')
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper right')
        ax.set_xlabel('Iteration')

        # --- Column 2: Eval ---
        # 4. eval_acc (row 0, col 1)
        ax = axes[0, 1]
        if eacc:
            tb_plot_series(ax, eacc, C_ACC_E, 'Eval Acc', smoothing=0.5)
        ax.set_title('Evaluation Character Accuracy', fontweight='bold')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right')

        # 5. eval_loss_cer (row 1, col 1)
        ax = axes[1, 1]
        if ecer:
            tb_plot_series(ax, ecer, C_ESMOOTH, 'Eval CER', smoothing=0.5)
        ax.set_title('Evaluation Character Error Rate (CER)', fontweight='bold')
        ax.set_ylabel('CER')
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper right')

        # 6. eval_loss_wer (row 2, col 1)
        ax = axes[2, 1]
        if ewer:
            tb_plot_series(ax, ewer, C_ESMOOTH, 'Eval WER', smoothing=0.5)
        ax.set_title('Evaluation Word Error Rate (WER)', fontweight='bold')
        ax.set_ylabel('WER')
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper right')
        ax.set_xlabel('Iteration')

        # Handle epoch boundaries
        if epoch_boundaries:
            for ax in axes.flatten():
                for it, _ep in epoch_boundaries:
                    ax.axvline(x=it, color='#aaaaaa', linestyle=':', alpha=0.4, lw=0.8)

        # Cleanup
        for a in fig.axes:
            a.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = os.path.join(self.output_dir, 'training_plot.png')
        tb_save(fig, out_path)
        return out_path

    def generate_premium_charts(self, entries, epoch_boundaries=None):
        """Generate high-quality individual plots for specific metrics."""
        saved = []

        # Helper for stats
        def _stats(data, label=''):
            vals = [v for _, v in data]
            if not vals: return ''
            return f'{label}\n  Min:  {min(vals):.4f}\n  Max:  {max(vals):.4f}\n  Last: {vals[-1]:.4f}\n  Mean: {sum(vals)/len(vals):.4f}'

        # Standard chart generator
        def _make_chart(title, ylabel, filename, series_configs, log_y=False, ylim=None):
            fig, ax = plt.subplots(figsize=(12, 6))
            apply_premium_theme(fig, ax)
            ax.set_title(f'{self.model_name} — {title}', fontweight='bold', pad=15)
            ax.set_xlabel('Iteration')
            ax.set_ylabel(ylabel)
            if log_y: ax.set_yscale('log')
            if ylim: ax.set_ylim(ylim)

            stats_text = []
            for data, label, color, is_eval in series_configs:
                if not data: continue
                smoothing = 0.5 if is_eval else 0.6
                tb_plot_series(ax, data, color, label, smoothing=smoothing)
                stats_text.append(_stats(data, f'── {label} ──'))
                mode = 'max' if 'Acc' in title else 'min'
                add_best_annotation(ax, data, color, mode=mode)

            if stats_text:
                add_stats_box(ax, '\n'.join(stats_text))

            if epoch_boundaries:
                for it, ep in epoch_boundaries:
                    ax.axvline(x=it, color='#aaaaaa', linestyle=':', alpha=0.5, lw=0.8)
                    ax.text(it, ax.get_ylim()[1], f'E{ep}', fontsize=7,
                            color='#777777', ha='center', va='bottom')

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(loc='best')
            plt.tight_layout()
            out = os.path.join(self.output_dir, filename)
            tb_save(fig, out)
            return out

        # 1. CER
        tcer = self._get_metric(entries, 'train_cer')
        ecer = self._get_metric(entries, 'eval_cer')
        if tcer or ecer:
            saved.append(_make_chart('Character Error Rate (CER)', 'CER', 'plot_cer.png',
                                    [(tcer, 'Train CER', C_SMOOTH, False),
                                     (ecer, 'Eval CER', C_ESMOOTH, True)], ylim=(0, None)))

        # 2. Loss
        tloss = self._get_metric(entries, 'train_loss')
        eloss = self._get_metric(entries, 'eval_loss')
        if tloss or eloss:
            saved.append(_make_chart('Loss (CTC)', 'Loss', 'plot_loss_combined.png',
                                    [(tloss, 'Train Loss', C_SMOOTH, False),
                                     (eloss, 'Eval Loss', C_ESMOOTH, True)], log_y=True))

        # 3. Accuracy
        tacc = self._get_metric(entries, 'train_acc')
        eacc = self._get_metric(entries, 'eval_acc')
        if tacc or eacc:
            saved.append(_make_chart('Accuracy (Exact-Match)', 'Accuracy', 'plot_acc_combined.png',
                                    [(tacc, 'Train Acc', C_ACC_T, False),
                                     (eacc, 'Eval Acc', C_ACC_E, True)], ylim=(0, 1.05)))

        return saved


def plot_multi_model(title, ylabel, filename, model_series, output_dir, log_y=False, ylim=None):
    """
    Generate a plot overlaying multiple models.
    model_series: list of (name, data_list, is_eval)
    data_list: list of (iteration, value)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    apply_premium_theme(fig, ax)
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(ylabel)
    if log_y: ax.set_yscale('log')
    if ylim: ax.set_ylim(ylim)

    for i, (name, data, is_eval) in enumerate(model_series):
        if not data: continue
        color = TB_PALETTE[i % len(TB_PALETTE)]
        smoothing = 0.5 if is_eval else 0.6
        sm = smooth_ema(data, weight=smoothing)
        ax.plot(*zip(*data), color=color, alpha=0.15, lw=0.6)
        ax.plot(*zip(*sm), color=color, lw=2.2, label=name)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='best')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, filename)
    tb_save(fig, out)
    return out
