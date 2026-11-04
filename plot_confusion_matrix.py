import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import defaultdict

# ── Project imports ──────────────────────────────────────────────────────────
from tesseract_tools import align_strings, LAO_COMPONENT_MAP

LAO_FONT_PATH = "font/phetsarath_ot.ttf"

def generate_confusion_matrix(csv_path, out_dir, category, models_filter=None):
    if category not in LAO_COMPONENT_MAP:
        print(f"Error: Category '{category}' not found in LAO_COMPONENT_MAP.")
        return

    target_chars = LAO_COMPONENT_MAP[category]

    print(f"[{category}] Loading predictions from {csv_path}...")
    
    confusion_data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row.get("model", "")
                if models_filter and model not in models_filter:
                    continue
                    
                gt_str = row.get("gt", "")
                pred_str = row.get("prediction", "")
                
                edits = align_strings(pred_str, gt_str)
                
                for op, p_c, t_c in edits:
                    relevant = False
                    if t_c and t_c in target_chars:
                        relevant = True
                    elif op == "ins" and p_c in target_chars:
                        relevant = True
                        
                    if relevant:
                        y_true = t_c if t_c else "<INS>"
                        y_pred = p_c if p_c else "<DEL>"
                        confusion_data[model][y_true][y_pred] += 1
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    if not confusion_data:
        print(f"[{category}] No matching data found.")
        return

    prop = None
    if os.path.exists(LAO_FONT_PATH):
        prop = fm.FontProperties(fname=LAO_FONT_PATH, size=10)

    for model, matrix_dict in confusion_data.items():
        # Create model directory
        model_out_dir = os.path.join(out_dir, model.replace("/", "_").replace("\\", "_"))
        os.makedirs(model_out_dir, exist_ok=True)
        
        all_true = set(matrix_dict.keys())
        all_pred = set()
        matrix_dict_bucketed = defaultdict(lambda: defaultdict(int))
        for r, preds in matrix_dict.items():
            for c, count in preds.items():
                if c in target_chars or c == "<DEL>" or c == "<INS>":
                    matrix_dict_bucketed[r][c] += count
                else:
                    matrix_dict_bucketed[r]["<OTHER>"] += count
                    
        all_true = set(matrix_dict_bucketed.keys())
        all_pred = set()
        for t_c, preds in matrix_dict_bucketed.items():
            all_pred.update(preds.keys())
            
        row_labels = target_chars.copy()
        col_labels = target_chars.copy()
            
        active_rows = []
        for r in row_labels:
            if sum(matrix_dict_bucketed.get(r, {}).values()) > 0:
                active_rows.append(r)
                
        if not active_rows:
            continue
            
        active_cols = []
        for c in col_labels:
            col_total = sum(matrix_dict_bucketed.get(r, {}).get(c, 0) for r in active_rows)
            if col_total > 0:
                active_cols.append(c)
                
        matrix_raw = np.zeros((len(active_rows), len(active_cols)), dtype=int)
        matrix_pct = np.zeros((len(active_rows), len(active_cols)), dtype=float)
        
        for i, r in enumerate(active_rows):
            row_total = sum(matrix_dict_bucketed.get(r, {}).values())
            for j, c in enumerate(active_cols):
                val = matrix_dict_bucketed.get(r, {}).get(c, 0)
                matrix_raw[i, j] = val
                matrix_pct[i, j] = (val / row_total * 100.0) if row_total > 0 else 0.0
                
        fig_width = max(8, len(active_cols) * 0.5)
        fig_height = max(6, len(active_rows) * 0.5)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Color based on percentage (0 to 100)
        cax = ax.imshow(matrix_pct, cmap="Blues", aspect="auto", vmin=0, vmax=100)
        
        ax.set_xticks(np.arange(len(active_cols)))
        ax.set_yticks(np.arange(len(active_rows)))
        
        ax.set_xticklabels(active_cols)
        ax.set_yticklabels(active_rows)
        
        if prop:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(prop)
                
        for i in range(len(active_rows)):
            for j in range(len(active_cols)):
                val_raw = matrix_raw[i, j]
                val_pct = matrix_pct[i, j]
                if val_raw > 0:
                    text_color = "white" if val_pct > 50 else "black"
                    ax.text(j, i, f"{val_pct:.0f}%", ha="center", va="center", color=text_color, fontsize=9)
                    
        ax.set_title(f"Confusion Matrix ({category}) - {model}", fontsize=14)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual (Ground Truth)", fontsize=12)
        
        plt.tight_layout()
        out_file = os.path.join(model_out_dir, f"{category}.png")
        plt.savefig(out_file, dpi=150)
        plt.close(fig)
        print(f"  ✓ Saved {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate Confusion Matrices from predictions CSV")
    parser.add_argument("--csv_path", required=True, help="Path to predictions_report.csv")
    parser.add_argument("--out_dir", default="plots/confusion_matrices", help="Output directory")
    parser.add_argument("--category", default="all", help="Category name (e.g. consonant, tone, or 'all' for all categories)")
    parser.add_argument("--models", nargs="+", help="Specific models to plot", default=[])
    args = parser.parse_args()
    
    cats_to_run = list(LAO_COMPONENT_MAP.keys()) if args.category == "all" else [args.category]
    
    for cat in cats_to_run:
        generate_confusion_matrix(args.csv_path, args.out_dir, cat, args.models)

if __name__ == "__main__":
    main()
