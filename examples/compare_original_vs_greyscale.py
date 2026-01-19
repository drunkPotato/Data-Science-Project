#!/usr/bin/env python3
"""
Compare model performance between original and greyscale versions of Dataset 3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Publication-quality settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'


def load_and_calculate_metrics(csv_path, version_name):
    """Load results and calculate metrics"""
    df = pd.read_csv(csv_path)
    df = df[df['status'] == 'success'].copy()

    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
    models = df['model'].unique()

    metrics = {}
    for model in models:
        model_df = df[df['model'] == model].copy()
        y_true = model_df['true_emotion'].values
        y_pred = model_df['dominant_emotion'].values

        accuracy = accuracy_score(y_true, y_pred)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=emotions, average=None, zero_division=0
        )
        macro_f1 = np.mean(f1)

        metrics[model] = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'version': version_name
        }

    return metrics


def plot_comparison(original_metrics, greyscale_metrics, output_dir):
    """Create comparison visualization"""
    models = list(original_metrics.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy comparison
    x = np.arange(len(models))
    width = 0.35

    original_acc = [original_metrics[m]['accuracy'] * 100 for m in models]
    greyscale_acc = [greyscale_metrics[m]['accuracy'] * 100 for m in models]

    bars1 = ax1.bar(x - width/2, original_acc, width, label='Original', alpha=0.8, color='#4ECDC4')
    bars2 = ax1.bar(x + width/2, greyscale_acc, width, label='Greyscale', alpha=0.8, color='#FF6B6B')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8)

    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy Comparison: Original vs Greyscale')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 105])

    # F1-Score comparison
    original_f1 = [original_metrics[m]['macro_f1'] * 100 for m in models]
    greyscale_f1 = [greyscale_metrics[m]['macro_f1'] * 100 for m in models]

    bars3 = ax2.bar(x - width/2, original_f1, width, label='Original', alpha=0.8, color='#4ECDC4')
    bars4 = ax2.bar(x + width/2, greyscale_f1, width, label='Greyscale', alpha=0.8, color='#FF6B6B')

    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8)

    ax2.set_xlabel('Model')
    ax2.set_ylabel('Macro F1-Score (%)')
    ax2.set_title('F1-Score Comparison: Original vs Greyscale')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_original_vs_greyscale.png", bbox_inches='tight')
    plt.savefig(output_dir / "comparison_original_vs_greyscale.pdf", bbox_inches='tight')
    plt.close()

    print("Created: comparison_original_vs_greyscale.png/.pdf")


def generate_comparison_table(original_metrics, greyscale_metrics, output_dir):
    """Generate comparison table"""
    rows = []
    for model in original_metrics.keys():
        orig = original_metrics[model]
        grey = greyscale_metrics[model]

        rows.append({
            'Model': model,
            'Original_Accuracy': orig['accuracy'],
            'Greyscale_Accuracy': grey['accuracy'],
            'Accuracy_Diff': grey['accuracy'] - orig['accuracy'],
            'Original_F1': orig['macro_f1'],
            'Greyscale_F1': grey['macro_f1'],
            'F1_Diff': grey['macro_f1'] - orig['macro_f1']
        })

    df = pd.DataFrame(rows)
    output_path = output_dir / "comparison_metrics.csv"
    df.to_csv(output_path, index=False)
    print(f"Created: comparison_metrics.csv")

    return df


def print_comparison_report(original_metrics, greyscale_metrics):
    """Print comparison report"""
    print("\n" + "="*80)
    print("ORIGINAL vs GREYSCALE COMPARISON")
    print("="*80)

    for model in original_metrics.keys():
        orig = original_metrics[model]
        grey = greyscale_metrics[model]

        acc_diff = (grey['accuracy'] - orig['accuracy']) * 100
        f1_diff = (grey['macro_f1'] - orig['macro_f1']) * 100

        print(f"\n{model}:")
        print(f"  Original  - Accuracy: {orig['accuracy']*100:.2f}%  F1: {orig['macro_f1']*100:.2f}%")
        print(f"  Greyscale - Accuracy: {grey['accuracy']*100:.2f}%  F1: {grey['macro_f1']*100:.2f}%")
        print(f"  Difference: Accuracy: {acc_diff:+.2f}%  F1: {f1_diff:+.2f}%")

        if abs(acc_diff) < 1:
            print("  -> Minimal impact from greyscaling")
        elif acc_diff > 0:
            print("  -> Greyscale version performs BETTER")
        else:
            print("  -> Original version performs BETTER")

    print("\n" + "="*80 + "\n")


def main():
    # File paths
    original_file = Path(__file__).parent.parent / "data" / "processed" / "dataset3_model_comparison.csv"
    greyscale_file = Path(__file__).parent.parent / "data" / "processed" / "dataset3_greyscale_model_comparison.csv"
    output_dir = Path(__file__).parent.parent / "results" / "dataset3_comparison"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("COMPARING ORIGINAL vs GREYSCALE DATASET 3 RESULTS")
    print("="*80)

    # Check files exist
    if not original_file.exists():
        print(f"\nERROR: Original results not found: {original_file}")
        print("Run: python examples/evaluate_models_dataset3.py")
        return

    if not greyscale_file.exists():
        print(f"\nERROR: Greyscale results not found: {greyscale_file}")
        print("Run: python examples/evaluate_models_dataset3_greyscale.py")
        return

    # Load metrics
    print("\nLoading results...")
    original_metrics = load_and_calculate_metrics(original_file, "Original")
    greyscale_metrics = load_and_calculate_metrics(greyscale_file, "Greyscale")

    # Print comparison
    print_comparison_report(original_metrics, greyscale_metrics)

    # Generate visualizations
    print("Generating comparison visualizations...")
    plot_comparison(original_metrics, greyscale_metrics, output_dir)

    # Generate comparison table
    generate_comparison_table(original_metrics, greyscale_metrics, output_dir)

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print(f"All outputs saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
