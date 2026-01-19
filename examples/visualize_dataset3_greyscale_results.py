#!/usr/bin/env python3
"""
Generate comprehensive visualizations for model evaluation on GREYSCALE Emotion Dataset 3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

# Publication-quality settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def load_results(csv_path):
    """Load model evaluation results"""
    df = pd.read_csv(csv_path)

    # Filter successful predictions only
    df = df[df['status'] == 'success'].copy()

    print(f"Loaded {len(df)} successful predictions")
    print(f"Models: {df['model'].unique()}")

    return df


def calculate_metrics(df, model_name):
    """Calculate comprehensive metrics for a model"""
    model_df = df[df['model'] == model_name].copy()

    if len(model_df) == 0:
        print(f"WARNING: No data for model {model_name}")
        return None

    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
    y_true = model_df['true_emotion'].values
    y_pred = model_df['dominant_emotion'].values

    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=emotions, average=None, zero_division=0
    )

    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'per_class': {
            emotions[i]: {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': int(support[i])
            } for i in range(len(emotions))
        },
        'y_true': y_true,
        'y_pred': y_pred
    }


def plot_overall_comparison(metrics_dict, output_dir):
    """Bar chart comparing overall performance metrics"""
    fig, ax = plt.subplots(figsize=(12, 6))

    models = list(metrics_dict.keys())
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    # Prepare data
    data = {
        'Accuracy': [metrics_dict[m]['accuracy'] * 100 for m in models],
        'Precision': [metrics_dict[m]['macro_precision'] * 100 for m in models],
        'Recall': [metrics_dict[m]['macro_recall'] * 100 for m in models],
        'F1-Score': [metrics_dict[m]['macro_f1'] * 100 for m in models]
    }

    x = np.arange(len(models))
    width = 0.2
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for i, metric in enumerate(metrics_names):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, data[metric], width, label=metric,
                     alpha=0.8, color=colors[i])

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Score (%)')
    ax.set_title('Overall Model Performance on GREYSCALE Dataset 3 Test Set')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(output_dir / "dataset3_greyscale_overall_comparison.png", bbox_inches='tight')
    plt.savefig(output_dir / "dataset3_greyscale_overall_comparison.pdf", bbox_inches='tight')
    plt.close()

    print("Created: dataset3_greyscale_overall_comparison.png/.pdf")


def plot_per_emotion_f1(metrics_dict, output_dir):
    """Grouped bar chart for per-emotion F1 scores"""
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
    models = list(metrics_dict.keys())

    # Prepare data
    data = {model: [metrics_dict[model]['per_class'][e]['f1'] * 100 for e in emotions]
            for model in models}

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(emotions))
    width = 0.25
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for i, model in enumerate(models):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data[model], width, label=model,
                     alpha=0.8, color=colors[i % len(colors)])

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 3:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}',
                       ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Emotion')
    ax.set_ylabel('F1-Score (%)')
    ax.set_title('Per-Emotion F1-Score Comparison - GREYSCALE Dataset 3 Test Set')
    ax.set_xticks(x)
    ax.set_xticklabels([e.title() for e in emotions])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(output_dir / "dataset3_greyscale_per_emotion_f1.png", bbox_inches='tight')
    plt.savefig(output_dir / "dataset3_greyscale_per_emotion_f1.pdf", bbox_inches='tight')
    plt.close()

    print("Created: dataset3_greyscale_per_emotion_f1.png/.pdf")


def plot_confusion_matrices(metrics_dict, output_dir):
    """Create confusion matrix for each model"""
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
    models = list(metrics_dict.keys())

    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5))

    if len(models) == 1:
        axes = [axes]

    for idx, model in enumerate(models):
        y_true = metrics_dict[model]['y_true']
        y_pred = metrics_dict[model]['y_pred']

        cm = confusion_matrix(y_true, y_pred, labels=emotions)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Plot
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=[e[:3].upper() for e in emotions],
                   yticklabels=[e[:3].upper() for e in emotions],
                   ax=axes[idx], cbar_kws={'label': 'Percentage (%)'})

        axes[idx].set_title(f'{model}\nAccuracy: {metrics_dict[model]["accuracy"]*100:.1f}%')
        axes[idx].set_ylabel('True Emotion')
        axes[idx].set_xlabel('Predicted Emotion')

    plt.suptitle('Confusion Matrices - GREYSCALE Dataset 3', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "dataset3_greyscale_confusion_matrices.png", bbox_inches='tight')
    plt.savefig(output_dir / "dataset3_greyscale_confusion_matrices.pdf", bbox_inches='tight')
    plt.close()

    print("Created: dataset3_greyscale_confusion_matrices.png/.pdf")


def plot_precision_recall_comparison(metrics_dict, output_dir):
    """Compare precision and recall across models"""
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
    models = list(metrics_dict.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Precision comparison
    data_precision = {model: [metrics_dict[model]['per_class'][e]['precision'] * 100 for e in emotions]
                     for model in models}

    x = np.arange(len(emotions))
    width = 0.25
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for i, model in enumerate(models):
        offset = (i - 1) * width
        ax1.bar(x + offset, data_precision[model], width, label=model,
               alpha=0.8, color=colors[i % len(colors)])

    ax1.set_xlabel('Emotion')
    ax1.set_ylabel('Precision (%)')
    ax1.set_title('Per-Emotion Precision - GREYSCALE Dataset 3')
    ax1.set_xticks(x)
    ax1.set_xticklabels([e.title() for e in emotions], rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 105])

    # Recall comparison
    data_recall = {model: [metrics_dict[model]['per_class'][e]['recall'] * 100 for e in emotions]
                  for model in models}

    for i, model in enumerate(models):
        offset = (i - 1) * width
        ax2.bar(x + offset, data_recall[model], width, label=model,
               alpha=0.8, color=colors[i % len(colors)])

    ax2.set_xlabel('Emotion')
    ax2.set_ylabel('Recall (%)')
    ax2.set_title('Per-Emotion Recall - GREYSCALE Dataset 3')
    ax2.set_xticks(x)
    ax2.set_xticklabels([e.title() for e in emotions], rotation=15, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(output_dir / "dataset3_greyscale_precision_recall.png", bbox_inches='tight')
    plt.savefig(output_dir / "dataset3_greyscale_precision_recall.pdf", bbox_inches='tight')
    plt.close()

    print("Created: dataset3_greyscale_precision_recall.png/.pdf")


def generate_metrics_table(metrics_dict, output_dir):
    """Generate detailed metrics table as CSV"""
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

    rows = []
    for model in metrics_dict.keys():
        for emotion in emotions:
            pc = metrics_dict[model]['per_class'][emotion]
            rows.append({
                'Model': model,
                'Emotion': emotion,
                'Precision': pc['precision'],
                'Recall': pc['recall'],
                'F1-Score': pc['f1'],
                'Support': pc['support']
            })

        # Add overall metrics
        rows.append({
            'Model': model,
            'Emotion': 'MACRO AVG',
            'Precision': metrics_dict[model]['macro_precision'],
            'Recall': metrics_dict[model]['macro_recall'],
            'F1-Score': metrics_dict[model]['macro_f1'],
            'Support': sum(metrics_dict[model]['per_class'][e]['support'] for e in emotions)
        })

    df = pd.DataFrame(rows)
    output_path = output_dir / "dataset3_greyscale_detailed_metrics.csv"
    df.to_csv(output_path, index=False)
    print(f"Created: dataset3_greyscale_detailed_metrics.csv")

    return df


def print_summary_report(metrics_dict):
    """Print comprehensive summary report to console"""
    print("\n" + "="*80)
    print("GREYSCALE EMOTION DATASET 3 - MODEL EVALUATION SUMMARY")
    print("="*80)

    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

    for model in metrics_dict.keys():
        print(f"\n{model}:")
        print(f"  Overall Accuracy: {metrics_dict[model]['accuracy']*100:.2f}%")
        print(f"  Macro Precision:  {metrics_dict[model]['macro_precision']*100:.2f}%")
        print(f"  Macro Recall:     {metrics_dict[model]['macro_recall']*100:.2f}%")
        print(f"  Macro F1-Score:   {metrics_dict[model]['macro_f1']*100:.2f}%")

        print("\n  Per-Emotion Performance:")
        print(f"    {'Emotion':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("    " + "-"*56)

        for emotion in emotions:
            pc = metrics_dict[model]['per_class'][emotion]
            print(f"    {emotion.title():<12} {pc['precision']*100:>9.1f}% "
                  f"{pc['recall']*100:>9.1f}% {pc['f1']*100:>9.1f}% {pc['support']:>10}")

    print("\n" + "="*80)

    # Find best model
    best_accuracy = max(metrics_dict.items(), key=lambda x: x[1]['accuracy'])
    best_f1 = max(metrics_dict.items(), key=lambda x: x[1]['macro_f1'])

    print("\nBest Performance on Greyscale Dataset:")
    print(f"  Highest Accuracy: {best_accuracy[0]} ({best_accuracy[1]['accuracy']*100:.2f}%)")
    print(f"  Highest F1-Score: {best_f1[0]} ({best_f1[1]['macro_f1']*100:.2f}%)")
    print("="*80 + "\n")


def main():
    # Configuration
    results_file = Path(__file__).parent.parent / "data" / "processed" / "dataset3_greyscale_model_comparison.csv"
    output_dir = Path(__file__).parent.parent / "results" / "dataset3_greyscale_evaluation"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("VISUALIZING GREYSCALE EMOTION DATASET 3 MODEL EVALUATION RESULTS")
    print("="*80)

    # Check if results file exists
    if not results_file.exists():
        print(f"\nERROR: Results file not found: {results_file}")
        print("Please run the greyscale evaluation script first:")
        print("  python examples/evaluate_models_dataset3_greyscale.py")
        return

    # Load results
    print(f"\nLoading results from: {results_file}")
    df = load_results(results_file)

    # Calculate metrics for each model
    models = df['model'].unique()
    print(f"\nCalculating metrics for models: {', '.join(models)}")

    metrics_dict = {}
    for model in models:
        metrics = calculate_metrics(df, model)
        if metrics:
            metrics_dict[model] = metrics

    if not metrics_dict:
        print("ERROR: No valid metrics calculated")
        return

    # Print summary report
    print_summary_report(metrics_dict)

    # Generate visualizations
    print("\nGenerating visualizations...")
    print(f"Output directory: {output_dir}")
    print()

    plot_overall_comparison(metrics_dict, output_dir)
    plot_per_emotion_f1(metrics_dict, output_dir)
    plot_confusion_matrices(metrics_dict, output_dir)
    plot_precision_recall_comparison(metrics_dict, output_dir)

    # Generate detailed metrics table
    print()
    generate_metrics_table(metrics_dict, output_dir)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print(f"All outputs saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
