#!/usr/bin/env python3
"""
Generate publication-ready visualizations comparing DeepFace, FER, and Custom-ResNet18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

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

def calculate_metrics(df, model_name):
    """Calculate accuracy, precision, recall, F1 for a model"""
    model_df = df[df['model'] == model_name].copy()

    if len(model_df) == 0:
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

    macro_f1 = np.mean(f1)

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'per_class': {
            emotions[i]: {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            } for i in range(len(emotions))
        }
    }

def plot_overall_comparison(metrics_dict, output_dir):
    """Bar chart comparing overall performance"""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(metrics_dict.keys())
    accuracies = [metrics_dict[m]['accuracy'] * 100 for m in models]
    f1_scores = [metrics_dict[m]['macro_f1'] * 100 for m in models]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='#4ECDC4')
    bars2 = ax.bar(x + width/2, f1_scores, width, label='Macro F1', alpha=0.8, color='#FF6B6B')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model')
    ax.set_ylabel('Score (%)')
    ax.set_title('Overall Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(output_dir / "fig_model_comparison_overall.png", bbox_inches='tight')
    plt.savefig(output_dir / "fig_model_comparison_overall.pdf", bbox_inches='tight')
    plt.close()

    print("Created: fig_model_comparison_overall.png/.pdf")

def plot_per_emotion_comparison(metrics_dict, output_dir):
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
            if height > 5:  # Only label if visible
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}',
                       ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Emotion')
    ax.set_ylabel('F1-Score (%)')
    ax.set_title('Per-Emotion F1-Score Comparison Across Models')
    ax.set_xticks(x)
    ax.set_xticklabels([e.title() for e in emotions])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(output_dir / "fig_model_comparison_per_emotion.png", bbox_inches='tight')
    plt.savefig(output_dir / "fig_model_comparison_per_emotion.pdf", bbox_inches='tight')
    plt.close()

    print("Created: fig_model_comparison_per_emotion.png/.pdf")

def plot_confusion_matrices(df, output_dir):
    """Create confusion matrix for each model"""
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
    models = df['model'].unique()

    for model_name in models:
        model_df = df[df['model'] == model_name]

        y_true = model_df['true_emotion'].values
        y_pred = model_df['dominant_emotion'].values

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=emotions)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(9, 8))

        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                   xticklabels=[e.title() for e in emotions],
                   yticklabels=[e.title() for e in emotions],
                   cbar_kws={'label': 'Normalized Rate'},
                   ax=ax)

        ax.set_title(f"Confusion Matrix: {model_name}")
        ax.set_ylabel('True Emotion')
        ax.set_xlabel('Predicted Emotion')

        plt.tight_layout()

        safe_name = model_name.replace('-', '_').replace(' ', '_')
        plt.savefig(output_dir / f"fig_confusion_{safe_name}.png", bbox_inches='tight')
        plt.savefig(output_dir / f"fig_confusion_{safe_name}.pdf", bbox_inches='tight')
        plt.close()

        print(f"Created: fig_confusion_{safe_name}.png/.pdf")

def plot_heatmap_comparison(metrics_dict, output_dir):
    """Heatmap showing all models' per-emotion performance"""
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
    models = list(metrics_dict.keys())

    # Build matrix: rows=models, cols=emotions
    data = []
    for model in models:
        row = [metrics_dict[model]['per_class'][e]['f1'] * 100 for e in emotions]
        data.append(row)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100,
               xticklabels=[e.title() for e in emotions],
               yticklabels=models,
               cbar_kws={'label': 'F1-Score (%)'},
               ax=ax)

    ax.set_title('Model Performance Heatmap (F1-Score %)')
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Model')

    plt.tight_layout()
    plt.savefig(output_dir / "fig_model_heatmap.png", bbox_inches='tight')
    plt.savefig(output_dir / "fig_model_heatmap.pdf", bbox_inches='tight')
    plt.close()

    print("Created: fig_model_heatmap.png/.pdf")

def create_comparison_table(metrics_dict, output_dir):
    """Create Excel table with all metrics"""
    models = list(metrics_dict.keys())
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

    # Overall metrics table
    overall_data = []
    for model in models:
        overall_data.append({
            'Model': model,
            'Accuracy': f"{metrics_dict[model]['accuracy']*100:.2f}%",
            'Macro F1': f"{metrics_dict[model]['macro_f1']*100:.2f}%"
        })

    df_overall = pd.DataFrame(overall_data)
    df_overall.to_csv(output_dir / "table_model_comparison_overall.csv", index=False)
    df_overall.to_excel(output_dir / "table_model_comparison_overall.xlsx", index=False)

    # Per-emotion table
    emotion_data = []
    for emotion in emotions:
        row = {'Emotion': emotion.title()}
        for model in models:
            f1 = metrics_dict[model]['per_class'][emotion]['f1'] * 100
            row[model] = f"{f1:.2f}%"
        emotion_data.append(row)

    df_emotion = pd.DataFrame(emotion_data)
    df_emotion.to_csv(output_dir / "table_model_comparison_per_emotion.csv", index=False)
    df_emotion.to_excel(output_dir / "table_model_comparison_per_emotion.xlsx", index=False)

    print("Created: table_model_comparison_overall.csv/.xlsx")
    print("Created: table_model_comparison_per_emotion.csv/.xlsx")

def print_summary(metrics_dict):
    """Print summary to console"""
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70 + "\n")

    for model, metrics in metrics_dict.items():
        print(f"{model}:")
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  Macro F1: {metrics['macro_f1']*100:.2f}%")
        print()

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

    # Calculate metrics for each model
    print("Calculating metrics...")
    metrics_dict = {}
    for model in models:
        metrics = calculate_metrics(df, model)
        if metrics:
            metrics_dict[model] = metrics

    print_summary(metrics_dict)

    print("\nGenerating visualizations...")
    print("-" * 70)

    # Generate all figures
    plot_overall_comparison(metrics_dict, output_dir)
    plot_per_emotion_comparison(metrics_dict, output_dir)
    plot_heatmap_comparison(metrics_dict, output_dir)
    plot_confusion_matrices(df, output_dir)
    create_comparison_table(metrics_dict, output_dir)

    print("\n" + "="*70)
    print(f"All figures saved to: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
