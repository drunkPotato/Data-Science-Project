#!/usr/bin/env python3
"""
Generate publication-ready visualizations and tables from experiment results.
Outputs high-quality figures and formatted tables for scientific papers.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality matplotlib defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

def load_experiment_results(runs_dir):
    """Load all experiment results from runs directory"""
    runs_dir = Path(runs_dir)

    if not runs_dir.exists():
        print(f"Error: Runs directory not found: {runs_dir}")
        return []

    experiments = []

    for exp_dir in sorted(runs_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        exp_data = {"name": exp_dir.name, "dir": exp_dir}

        # Load metrics
        best_metrics = exp_dir / "best_metrics.json"
        if best_metrics.exists():
            with open(best_metrics) as f:
                exp_data["val_metrics"] = json.load(f)

        test_metrics = exp_dir / "test_metrics.json"
        if test_metrics.exists():
            with open(test_metrics) as f:
                exp_data["test_metrics"] = json.load(f)

        # Load config
        config_file = exp_dir / "config.yaml"
        if config_file.exists():
            import yaml
            with open(config_file) as f:
                exp_data["config"] = yaml.safe_load(f)

        # Load training history
        history_file = exp_dir / "history.json"
        if history_file.exists():
            with open(history_file) as f:
                exp_data["history"] = json.load(f)

        # Load confusion matrices
        val_cm = exp_dir / "best_confusion_matrix.npy"
        if val_cm.exists():
            exp_data["val_confusion_matrix"] = np.load(val_cm)

        test_cm = exp_dir / "test_confusion_matrix.npy"
        if test_cm.exists():
            exp_data["test_confusion_matrix"] = np.load(test_cm)

        if "val_metrics" in exp_data or "test_metrics" in exp_data:
            experiments.append(exp_data)

    return experiments

def create_overall_comparison_table(experiments, output_dir):
    """Create formatted table comparing all experiments"""
    rows = []
    for exp in experiments:
        row = {"Experiment": exp["name"]}

        if "config" in exp:
            cfg = exp["config"]
            row["Epochs"] = cfg["training"]["epochs"]
            row["LR"] = f"{cfg['training']['lr']:.0e}"
            row["Batch Size"] = cfg["training"]["batch_size"]
            row["Loss"] = cfg["loss"]["type"].upper()
            row["Mixup"] = cfg["augmentation"]["mixup_alpha"]

        if "val_metrics" in exp:
            row["Val Acc"] = f"{exp['val_metrics']['val_acc']:.4f}"
            row["Val F1"] = f"{exp['val_metrics']['val_macro_f1']:.4f}"

        if "test_metrics" in exp:
            row["Test Acc"] = f"{exp['test_metrics']['test_acc']:.4f}"
            row["Test F1"] = f"{exp['test_metrics']['test_macro_f1']:.4f}"

        rows.append(row)

    df = pd.DataFrame(rows)

    # Save as CSV and Excel
    df.to_csv(output_dir / "table_overall_comparison.csv", index=False)
    df.to_excel(output_dir / "table_overall_comparison.xlsx", index=False)

    print(f"Created: table_overall_comparison.csv/.xlsx")
    return df

def create_per_emotion_table(experiments, output_dir, split="test"):
    """Create per-emotion performance table"""
    metrics_key = f"{split}_metrics"

    # Get all emotions
    emotions = None
    for exp in experiments:
        if metrics_key in exp and "per_class" in exp[metrics_key]:
            emotions = list(exp[metrics_key]["per_class"].keys())
            break

    if not emotions:
        print(f"No per-class metrics found for {split} set")
        return None

    # Create table for each metric
    for metric in ["precision", "recall", "f1"]:
        rows = []
        for emotion in emotions:
            row = {"Emotion": emotion.title()}
            for exp in experiments:
                if metrics_key in exp and "per_class" in exp[metrics_key]:
                    value = exp[metrics_key]["per_class"].get(emotion, {}).get(metric, 0)
                    row[exp["name"]] = f"{value:.4f}"
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_dir / f"table_{split}_{metric}_per_emotion.csv", index=False)
        df.to_excel(output_dir / f"table_{split}_{metric}_per_emotion.xlsx", index=False)

        print(f"Created: table_{split}_{metric}_per_emotion.csv/.xlsx")

def plot_overall_comparison(experiments, output_dir):
    """Bar chart comparing overall accuracy and F1"""
    data = []
    for exp in experiments:
        if "test_metrics" in exp:
            data.append({
                'Experiment': exp['name'],
                'Accuracy': exp['test_metrics']['test_acc'] * 100,
                'Macro F1': exp['test_metrics']['test_macro_f1'] * 100
            })

    if not data:
        print("No test metrics found for overall comparison")
        return

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width/2, df['Accuracy'], width, label='Accuracy', alpha=0.8)
    ax.bar(x + width/2, df['Macro F1'], width, label='Macro F1', alpha=0.8)

    ax.set_xlabel('Experiment')
    ax.set_ylabel('Score (%)')
    ax.set_title('Overall Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Experiment'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(output_dir / "fig_overall_comparison.png", bbox_inches='tight')
    plt.savefig(output_dir / "fig_overall_comparison.pdf", bbox_inches='tight')
    plt.close()

    print("Created: fig_overall_comparison.png/.pdf")

def plot_per_emotion_heatmap(experiments, output_dir, metric="f1", split="test"):
    """Heatmap showing per-emotion performance across experiments"""
    metrics_key = f"{split}_metrics"

    # Get emotions
    emotions = None
    for exp in experiments:
        if metrics_key in exp and "per_class" in exp[metrics_key]:
            emotions = sorted(list(exp[metrics_key]["per_class"].keys()))
            break

    if not emotions:
        return

    # Build matrix
    data = []
    exp_names = []
    for exp in experiments:
        if metrics_key in exp and "per_class" in exp[metrics_key]:
            row = [exp[metrics_key]["per_class"].get(emotion, {}).get(metric, 0)
                   for emotion in emotions]
            data.append(row)
            exp_names.append(exp["name"])

    if not data:
        return

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(10, len(exp_names) * 0.4 + 2))

    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                xticklabels=[e.title() for e in emotions],
                yticklabels=exp_names,
                cbar_kws={'label': metric.upper()},
                ax=ax)

    ax.set_title(f'Per-Emotion {metric.upper()} Scores ({split.title()} Set)')
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Experiment')

    plt.tight_layout()
    plt.savefig(output_dir / f"fig_{split}_{metric}_heatmap.png", bbox_inches='tight')
    plt.savefig(output_dir / f"fig_{split}_{metric}_heatmap.pdf", bbox_inches='tight')
    plt.close()

    print(f"Created: fig_{split}_{metric}_heatmap.png/.pdf")

def plot_confusion_matrix(exp, output_dir, split="test"):
    """Plot confusion matrix for a single experiment"""
    cm_key = f"{split}_confusion_matrix"
    metrics_key = f"{split}_metrics"

    if cm_key not in exp or metrics_key not in exp:
        return

    cm = exp[cm_key]
    emotions = sorted(list(exp[metrics_key]["per_class"].keys()))

    fig, ax = plt.subplots(figsize=(8, 7))

    # Normalize by row (true labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=[e.title() for e in emotions],
                yticklabels=[e.title() for e in emotions],
                cbar_kws={'label': 'Normalized Rate'},
                ax=ax)

    ax.set_title(f"Confusion Matrix: {exp['name']} ({split.title()} Set)")
    ax.set_ylabel('True Emotion')
    ax.set_xlabel('Predicted Emotion')

    plt.tight_layout()

    safe_name = exp['name'].replace(' ', '_').replace('/', '_')
    plt.savefig(output_dir / f"fig_confusion_{safe_name}_{split}.png", bbox_inches='tight')
    plt.savefig(output_dir / f"fig_confusion_{safe_name}_{split}.pdf", bbox_inches='tight')
    plt.close()

    print(f"Created: fig_confusion_{safe_name}_{split}.png/.pdf")

def plot_learning_curves(exp, output_dir):
    """Plot training and validation curves"""
    if "history" not in exp:
        return

    history = exp["history"]

    if not history:
        return

    epochs = range(1, len(history) + 1)

    # Check what metrics are available
    has_train_loss = 'train_loss' in history[0]
    has_val_loss = 'val_loss' in history[0]
    has_train_acc = 'train_acc' in history[0]
    has_val_acc = 'val_acc' in history[0]
    has_val_f1 = 'val_macro_f1' in history[0]

    # Decide how many subplots we need
    num_plots = 0
    if has_train_loss or has_val_loss:
        num_plots += 1
    if has_train_acc or has_val_acc or has_val_f1:
        num_plots += 1

    if num_plots == 0:
        return

    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 4))
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Loss curves
    if has_train_loss or has_val_loss:
        if has_train_loss:
            train_loss = [h['train_loss'] for h in history]
            axes[plot_idx].plot(epochs, train_loss, label='Training Loss', marker='o', markersize=3)

        if has_val_loss:
            val_loss = [h['val_loss'] for h in history]
            axes[plot_idx].plot(epochs, val_loss, label='Validation Loss', marker='s', markersize=3)

        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].set_title('Loss Curves')
        axes[plot_idx].legend()
        axes[plot_idx].grid(alpha=0.3)
        plot_idx += 1

    # Accuracy/F1 curves
    if has_train_acc or has_val_acc or has_val_f1:
        if has_train_acc:
            train_acc = [h['train_acc'] * 100 for h in history]
            axes[plot_idx].plot(epochs, train_acc, label='Training Accuracy', marker='o', markersize=3)

        if has_val_acc:
            val_acc = [h['val_acc'] * 100 for h in history]
            axes[plot_idx].plot(epochs, val_acc, label='Validation Accuracy', marker='s', markersize=3)

        if has_val_f1:
            val_f1 = [h['val_macro_f1'] * 100 for h in history]
            axes[plot_idx].plot(epochs, val_f1, label='Validation F1', marker='^', markersize=3)

        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Score (%)')
        axes[plot_idx].set_title('Accuracy and F1 Curves')
        axes[plot_idx].legend()
        axes[plot_idx].grid(alpha=0.3)

    plt.tight_layout()

    safe_name = exp['name'].replace(' ', '_').replace('/', '_')
    plt.savefig(output_dir / f"fig_learning_curves_{safe_name}.png", bbox_inches='tight')
    plt.savefig(output_dir / f"fig_learning_curves_{safe_name}.pdf", bbox_inches='tight')
    plt.close()

    print(f"Created: fig_learning_curves_{safe_name}.png/.pdf")

def plot_hyperparameter_impact(experiments, output_dir):
    """Plot impact of different hyperparameters"""
    # Group experiments by hyperparameter type

    # Learning rate experiments
    lr_exps = [e for e in experiments if 'lr_' in e['name']]
    if len(lr_exps) >= 2:
        plot_hyperparameter_group(lr_exps, "Learning Rate", output_dir, "lr")

    # Batch size experiments
    batch_exps = [e for e in experiments if 'batch_' in e['name']]
    if len(batch_exps) >= 2:
        plot_hyperparameter_group(batch_exps, "Batch Size", output_dir, "batch")

    # Epoch experiments
    epoch_exps = [e for e in experiments if 'epochs_' in e['name']]
    if len(epoch_exps) >= 2:
        plot_hyperparameter_group(epoch_exps, "Training Epochs", output_dir, "epochs")

    # Augmentation experiments
    aug_exps = [e for e in experiments if 'aug_' in e['name']]
    if len(aug_exps) >= 2:
        plot_hyperparameter_group(aug_exps, "Augmentation Strategy", output_dir, "augmentation")

def plot_hyperparameter_group(exps, title, output_dir, filename_prefix):
    """Helper to plot a group of hyperparameter experiments"""
    data = []
    for exp in exps:
        if "test_metrics" in exp:
            data.append({
                'name': exp['name'],
                'accuracy': exp['test_metrics']['test_acc'] * 100,
                'f1': exp['test_metrics']['test_macro_f1'] * 100
            })

    if not data:
        return

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width/2, df['accuracy'], width, label='Accuracy', alpha=0.8)
    ax.bar(x + width/2, df['f1'], width, label='Macro F1', alpha=0.8)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Score (%)')
    ax.set_title(f'{title} Impact on Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(df['name'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(output_dir / f"fig_{filename_prefix}_impact.png", bbox_inches='tight')
    plt.savefig(output_dir / f"fig_{filename_prefix}_impact.pdf", bbox_inches='tight')
    plt.close()

    print(f"Created: fig_{filename_prefix}_impact.png/.pdf")

def generate_latex_table(df, caption, label, output_dir, filename):
    """Generate LaTeX table code"""
    latex_code = df.to_latex(index=False,
                             caption=caption,
                             label=label,
                             float_format="%.4f")

    with open(output_dir / filename, 'w') as f:
        f.write(latex_code)

    print(f"Created: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready visualizations")
    parser.add_argument("--runs_dir", type=str,
                       default="experiments/runs",
                       help="Directory containing experiment runs")
    parser.add_argument("--output_dir", type=str,
                       default="experiments/results_figures",
                       help="Directory to save output figures and tables")
    parser.add_argument("--best_only", action="store_true",
                       help="Only generate visualizations for top 3 experiments")

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    runs_dir = base_dir / args.runs_dir
    output_dir = base_dir / args.output_dir

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading experiments from {runs_dir}...")
    experiments = load_experiment_results(runs_dir)

    if not experiments:
        print(f"No experiments found in {runs_dir}")
        return

    print(f"Found {len(experiments)} experiments\n")

    # Sort by test F1 score
    experiments_with_test = [e for e in experiments if "test_metrics" in e]
    experiments_with_test.sort(
        key=lambda x: x["test_metrics"]["test_macro_f1"],
        reverse=True
    )

    if args.best_only and len(experiments_with_test) > 3:
        print("Generating visualizations for top 3 experiments only\n")
        best_experiments = experiments_with_test[:3]
    else:
        best_experiments = experiments_with_test

    print("="*60)
    print("GENERATING PUBLICATION-READY VISUALIZATIONS")
    print("="*60 + "\n")

    # Generate tables
    print("1. Creating comparison tables...")
    create_overall_comparison_table(experiments_with_test, output_dir)
    create_per_emotion_table(experiments_with_test, output_dir, split="test")
    create_per_emotion_table(experiments_with_test, output_dir, split="val")

    print("\n2. Creating overall comparison plots...")
    plot_overall_comparison(experiments_with_test, output_dir)

    print("\n3. Creating per-emotion heatmaps...")
    plot_per_emotion_heatmap(experiments_with_test, output_dir, metric="f1", split="test")
    plot_per_emotion_heatmap(experiments_with_test, output_dir, metric="precision", split="test")
    plot_per_emotion_heatmap(experiments_with_test, output_dir, metric="recall", split="test")

    print("\n4. Creating hyperparameter impact plots...")
    plot_hyperparameter_impact(experiments_with_test, output_dir)

    print("\n5. Creating confusion matrices...")
    for exp in best_experiments:
        plot_confusion_matrix(exp, output_dir, split="test")

    print("\n6. Creating learning curves...")
    for exp in best_experiments:
        plot_learning_curves(exp, output_dir)

    print("\n" + "="*60)
    print(f"All visualizations saved to: {output_dir}")
    print("="*60)
    print("\nFiles created:")
    print("  - CSV/Excel tables (for Word documents)")
    print("  - PNG figures (for presentations)")
    print("  - PDF figures (for LaTeX documents)")
    print("\nYou can now insert these into your paper!")

if __name__ == "__main__":
    main()
