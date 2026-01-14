#!/usr/bin/env python3
"""
Generate detailed per-emotion visualizations for individual models (DeepFace and FER)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

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

def extract_true_emotion(path):
    """Extract true emotion from file path"""
    path_parts = path.replace('\\', '/').split('/')
    for part in path_parts:
        if part in ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']:
            return part
    return 'unknown'

def load_and_process_data(csv_path):
    """Load comparison CSV and process"""
    df = pd.read_csv(csv_path)

    # Add true emotion
    df['true_emotion'] = df['image_path'].apply(extract_true_emotion)

    # Filter successful predictions only
    df = df[df['status'] == 'success']

    return df

def calculate_per_emotion_metrics(df, model_name):
    """Calculate detailed metrics for each emotion"""
    model_df = df[df['model'] == model_name].copy()

    if len(model_df) == 0:
        return None

    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
    y_true = model_df['true_emotion'].values
    y_pred = model_df['dominant_emotion'].values

    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=emotions, average=None, zero_division=0
    )

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = np.mean(f1)

    results = {
        'emotions': emotions,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'support': support,
        'accuracy': accuracy * 100,
        'macro_f1': macro_f1 * 100
    }

    return results

def plot_individual_model_metrics(metrics, model_name, output_dir):
    """Create detailed per-emotion bar chart for a single model"""
    emotions = [e.title() for e in metrics['emotions']]

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(emotions))
    width = 0.25

    bars1 = ax.bar(x - width, metrics['precision'], width, label='Precision',
                   alpha=0.9, color='#FF6B6B', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, metrics['recall'], width, label='Recall',
                   alpha=0.9, color='#4ECDC4', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, metrics['f1'], width, label='F1-Score',
                   alpha=0.9, color='#45B7D1', edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 2:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=8)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    ax.set_xlabel('Emotion', fontweight='bold')
    ax.set_ylabel('Score (%)', fontweight='bold')
    ax.set_title(f'{model_name}: Per-Emotion Performance Metrics', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=0)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])

    # Add overall metrics text box
    textstr = f'Overall Accuracy: {metrics["accuracy"]:.2f}%\nMacro F1-Score: {metrics["macro_f1"]:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    safe_name = model_name.replace('-', '_').replace(' ', '_')
    plt.savefig(output_dir / f"fig_per_emotion_{safe_name}.png", bbox_inches='tight')
    plt.savefig(output_dir / f"fig_per_emotion_{safe_name}.pdf", bbox_inches='tight')
    plt.close()

    print(f"Created: fig_per_emotion_{safe_name}.png/.pdf")

def create_individual_metrics_table(metrics, model_name, output_dir):
    """Create detailed table for a single model"""
    data = []
    for i, emotion in enumerate(metrics['emotions']):
        data.append({
            'Emotion': emotion.title(),
            'Precision': f"{metrics['precision'][i]:.2f}%",
            'Recall': f"{metrics['recall'][i]:.2f}%",
            'F1-Score': f"{metrics['f1'][i]:.2f}%",
            'Support': int(metrics['support'][i])
        })

    # Add overall row
    data.append({
        'Emotion': 'Overall',
        'Precision': '-',
        'Recall': '-',
        'F1-Score': f"{metrics['macro_f1']:.2f}%",
        'Support': int(sum(metrics['support']))
    })

    df = pd.DataFrame(data)

    safe_name = model_name.replace('-', '_').replace(' ', '_')
    df.to_csv(output_dir / f"table_per_emotion_{safe_name}.csv", index=False)
    df.to_excel(output_dir / f"table_per_emotion_{safe_name}.xlsx", index=False)

    print(f"Created: table_per_emotion_{safe_name}.csv/.xlsx")

    return df

def plot_stacked_comparison(metrics_dict, output_dir):
    """Create side-by-side comparison of both models' per-emotion metrics"""
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
    models = list(metrics_dict.keys())

    fig, axes = plt.subplots(1, len(models), figsize=(18, 7), sharey=True)

    if len(models) == 1:
        axes = [axes]

    colors = {'Precision': '#FF6B6B', 'Recall': '#4ECDC4', 'F1-Score': '#45B7D1'}

    for idx, model in enumerate(models):
        ax = axes[idx]
        metrics = metrics_dict[model]
        emotion_labels = [e.title() for e in metrics['emotions']]

        x = np.arange(len(emotions))
        width = 0.25

        bars1 = ax.bar(x - width, metrics['precision'], width, label='Precision',
                      alpha=0.9, color=colors['Precision'], edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x, metrics['recall'], width, label='Recall',
                      alpha=0.9, color=colors['Recall'], edgecolor='black', linewidth=0.5)
        bars3 = ax.bar(x + width, metrics['f1'], width, label='F1-Score',
                      alpha=0.9, color=colors['F1-Score'], edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Emotion', fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Score (%)', fontweight='bold')
        ax.set_title(f'{model}', fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(emotion_labels, rotation=45, ha='right')
        ax.legend(loc='upper right', framealpha=0.95, fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, 105])

        # Add accuracy text
        textstr = f'Acc: {metrics["accuracy"]:.1f}%\nF1: {metrics["macro_f1"]:.1f}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props)

    plt.suptitle('Side-by-Side Model Comparison: Per-Emotion Metrics',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()

    plt.savefig(output_dir / "fig_side_by_side_comparison.png", bbox_inches='tight')
    plt.savefig(output_dir / "fig_side_by_side_comparison.pdf", bbox_inches='tight')
    plt.close()

    print("Created: fig_side_by_side_comparison.png/.pdf")

def print_model_summary(metrics, model_name):
    """Print detailed summary for a model"""
    print(f"\n{'='*70}")
    print(f"{model_name} - DETAILED PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  Macro F1-Score: {metrics['macro_f1']:.2f}%")
    print(f"\nPer-Emotion Breakdown:")
    print(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)

    for i, emotion in enumerate(metrics['emotions']):
        print(f"{emotion.title():<12} "
              f"{metrics['precision'][i]:>10.2f}%  "
              f"{metrics['recall'][i]:>10.2f}%  "
              f"{metrics['f1'][i]:>10.2f}%  "
              f"{int(metrics['support'][i]):>10}")

def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / "data/processed/emotion_results_comparison.csv"
    output_dir = base_dir / "experiments/results_figures"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading comparison data...")
    df = load_and_process_data(csv_path)

    models = df['model'].unique()
    print(f"Found models: {', '.join(models)}\n")

    # Calculate and visualize metrics for each model
    metrics_dict = {}

    for model in models:
        print(f"\nProcessing {model}...")
        metrics = calculate_per_emotion_metrics(df, model)

        if metrics:
            metrics_dict[model] = metrics

            # Generate individual visualizations
            plot_individual_model_metrics(metrics, model, output_dir)
            create_individual_metrics_table(metrics, model, output_dir)
            print_model_summary(metrics, model)

    # Generate side-by-side comparison
    if len(metrics_dict) > 1:
        print("\nCreating side-by-side comparison...")
        plot_stacked_comparison(metrics_dict, output_dir)

    print("\n" + "="*70)
    print(f"All figures saved to: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
