#!/usr/bin/env python3
"""
Analyze and visualize model agreement:
When all three models agree on an emotion, how likely is it to be correct?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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

def analyze_model_agreement(df):
    """Analyze when all models agree and their correctness"""
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

    # Pivot data to have one row per image with columns for each model
    pivot_df = df.pivot_table(
        index='image_path',
        columns='model',
        values='dominant_emotion',
        aggfunc='first'
    ).reset_index()

    # Add true emotion
    pivot_df['true_emotion'] = pivot_df['image_path'].apply(extract_true_emotion)

    # Get model columns
    model_cols = [col for col in pivot_df.columns if col not in ['image_path', 'true_emotion']]

    # Check if all models agree
    pivot_df['all_agree'] = pivot_df.apply(
        lambda row: len(set([row[col] for col in model_cols if pd.notna(row[col])])) == 1,
        axis=1
    )

    # Get the agreed prediction
    pivot_df['agreed_emotion'] = pivot_df.apply(
        lambda row: row[model_cols[0]] if row['all_agree'] else None,
        axis=1
    )

    # Check if agreement is correct
    pivot_df['agreement_correct'] = pivot_df.apply(
        lambda row: row['agreed_emotion'] == row['true_emotion'] if row['all_agree'] else None,
        axis=1
    )

    # Calculate statistics per emotion
    results = {}

    for emotion in emotions:
        # Filter for when all models agree on this emotion
        agreed_on_emotion = pivot_df[
            (pivot_df['all_agree']) &
            (pivot_df['agreed_emotion'] == emotion)
        ]

        if len(agreed_on_emotion) > 0:
            correct = agreed_on_emotion['agreement_correct'].sum()
            total = len(agreed_on_emotion)
            accuracy = (correct / total) * 100

            results[emotion] = {
                'total_agreements': total,
                'correct_agreements': correct,
                'accuracy_when_agree': accuracy
            }
        else:
            results[emotion] = {
                'total_agreements': 0,
                'correct_agreements': 0,
                'accuracy_when_agree': 0.0
            }

    # Overall statistics
    all_agreements = pivot_df[pivot_df['all_agree']]
    overall_stats = {
        'total_images': len(pivot_df),
        'total_agreements': len(all_agreements),
        'agreement_rate': (len(all_agreements) / len(pivot_df)) * 100,
        'correct_when_agree': all_agreements['agreement_correct'].sum(),
        'accuracy_when_agree': (all_agreements['agreement_correct'].sum() / len(all_agreements)) * 100 if len(all_agreements) > 0 else 0
    }

    return results, overall_stats, pivot_df

def plot_agreement_accuracy(results, overall_stats, output_dir):
    """Plot accuracy when all models agree on each emotion"""
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
    emotion_labels = [e.title() for e in emotions]

    accuracies = [results[e]['accuracy_when_agree'] for e in emotions]
    counts = [results[e]['total_agreements'] for e in emotions]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                     gridspec_kw={'height_ratios': [2, 1]})

    # Top plot: Accuracy when all models agree
    colors = ['#2ecc71' if acc >= 80 else '#f39c12' if acc >= 60 else '#e74c3c'
              for acc in accuracies]

    bars = ax1.bar(emotion_labels, accuracies, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1)

    # Add value labels and count annotations
    for i, (bar, acc, count) in enumerate(zip(bars, accuracies, counts)):
        height = bar.get_height()

        # Accuracy percentage
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Agreement count
        ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                f'n={count}',
                ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax1.set_ylabel('Accuracy When All Models Agree (%)', fontweight='bold')
    ax1.set_title('Model Agreement Analysis: Correctness by Emotion\nWhen DeepFace, FER, and Custom-ResNet18 All Agree',
                  fontweight='bold', pad=20, fontsize=13)
    ax1.set_ylim([0, 105])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=overall_stats['accuracy_when_agree'], color='red',
                linestyle='--', linewidth=2, label=f"Overall: {overall_stats['accuracy_when_agree']:.1f}%")
    ax1.legend(loc='upper right')

    # Add color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.8, label='High (≥80%)'),
        Patch(facecolor='#f39c12', alpha=0.8, label='Medium (60-80%)'),
        Patch(facecolor='#e74c3c', alpha=0.8, label='Low (<60%)')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', title='Accuracy Level')

    # Bottom plot: Number of agreements per emotion
    bars2 = ax2.bar(emotion_labels, counts, color='#3498db', alpha=0.8,
                    edgecolor='black', linewidth=1)

    for bar, count in zip(bars2, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Emotion', fontweight='bold')
    ax2.set_ylabel('Number of Agreements', fontweight='bold')
    ax2.set_title('Frequency of Unanimous Model Agreement', fontweight='bold', pad=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add overall stats text box
    textstr = '\n'.join([
        'Overall Statistics:',
        f'Total Images: {overall_stats["total_images"]}',
        f'Total Agreements: {overall_stats["total_agreements"]} ({overall_stats["agreement_rate"]:.1f}%)',
        f'Correct When Agree: {overall_stats["correct_when_agree"]}',
        f'Accuracy When Agree: {overall_stats["accuracy_when_agree"]:.1f}%'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax1.text(0.98, 0.97, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    plt.savefig(output_dir / "fig_model_agreement_accuracy.png", bbox_inches='tight')
    plt.savefig(output_dir / "fig_model_agreement_accuracy.pdf", bbox_inches='tight')
    plt.close()

    print("Created: fig_model_agreement_accuracy.png/.pdf")

def plot_agreement_heatmap(pivot_df, output_dir):
    """Create heatmap showing agreement patterns"""
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

    # Create confusion matrix: true emotion vs agreed emotion (when all agree)
    agreed_df = pivot_df[pivot_df['all_agree']].copy()

    # Create matrix
    matrix = np.zeros((len(emotions), len(emotions)))

    for i, true_emotion in enumerate(emotions):
        for j, agreed_emotion in enumerate(emotions):
            count = len(agreed_df[
                (agreed_df['true_emotion'] == true_emotion) &
                (agreed_df['agreed_emotion'] == agreed_emotion)
            ])
            matrix[i, j] = count

    # Normalize by row (true emotion)
    matrix_norm = np.zeros_like(matrix)
    for i in range(len(emotions)):
        row_sum = matrix[i, :].sum()
        if row_sum > 0:
            matrix_norm[i, :] = (matrix[i, :] / row_sum) * 100

    fig, ax = plt.subplots(figsize=(10, 9))

    # Create annotations with both counts and percentages
    annot = np.empty_like(matrix, dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                annot[i, j] = f'{int(matrix[i, j])}\n({matrix_norm[i, j]:.1f}%)'
            else:
                annot[i, j] = ''

    sns.heatmap(matrix_norm, annot=annot, fmt='', cmap='YlOrRd',
               xticklabels=[e.title() for e in emotions],
               yticklabels=[e.title() for e in emotions],
               cbar_kws={'label': 'Percentage (%)'},
               ax=ax, linewidths=0.5, linecolor='gray')

    ax.set_title('Agreement Confusion Matrix\n(When All Models Agree, What Do They Predict vs. True Label)',
                 fontweight='bold', pad=20)
    ax.set_ylabel('True Emotion', fontweight='bold')
    ax.set_xlabel('Unanimously Agreed Emotion', fontweight='bold')

    plt.tight_layout()

    plt.savefig(output_dir / "fig_agreement_confusion_matrix.png", bbox_inches='tight')
    plt.savefig(output_dir / "fig_agreement_confusion_matrix.pdf", bbox_inches='tight')
    plt.close()

    print("Created: fig_agreement_confusion_matrix.png/.pdf")

def create_agreement_table(results, overall_stats, output_dir):
    """Create detailed table of agreement statistics"""
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

    data = []
    for emotion in emotions:
        data.append({
            'Emotion': emotion.title(),
            'Total Agreements': results[emotion]['total_agreements'],
            'Correct Agreements': results[emotion]['correct_agreements'],
            'Accuracy When Agree': f"{results[emotion]['accuracy_when_agree']:.2f}%"
        })

    # Add overall row
    data.append({
        'Emotion': 'Overall',
        'Total Agreements': overall_stats['total_agreements'],
        'Correct Agreements': overall_stats['correct_when_agree'],
        'Accuracy When Agree': f"{overall_stats['accuracy_when_agree']:.2f}%"
    })

    df = pd.DataFrame(data)

    df.to_csv(output_dir / "table_model_agreement.csv", index=False)
    df.to_excel(output_dir / "table_model_agreement.xlsx", index=False)

    print("Created: table_model_agreement.csv/.xlsx")

    return df

def print_summary(results, overall_stats):
    """Print detailed summary"""
    print("\n" + "="*70)
    print("MODEL AGREEMENT ANALYSIS")
    print("="*70)
    print(f"\nOverall Statistics:")
    print(f"  Total Images Analyzed: {overall_stats['total_images']}")
    print(f"  Images Where All Models Agree: {overall_stats['total_agreements']} ({overall_stats['agreement_rate']:.1f}%)")
    print(f"  Correct When All Agree: {overall_stats['correct_when_agree']}")
    print(f"  Accuracy When All Agree: {overall_stats['accuracy_when_agree']:.2f}%")

    print(f"\n{'Emotion':<12} {'Agreements':<12} {'Correct':<12} {'Accuracy':<12}")
    print("-" * 70)

    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
    for emotion in emotions:
        print(f"{emotion.title():<12} "
              f"{results[emotion]['total_agreements']:<12} "
              f"{results[emotion]['correct_agreements']:<12} "
              f"{results[emotion]['accuracy_when_agree']:>10.2f}%")

    print("\nKey Insights:")
    print("-" * 70)

    # Find best and worst emotions
    valid_emotions = [(e, results[e]) for e in emotions if results[e]['total_agreements'] > 10]
    if valid_emotions:
        best_emotion = max(valid_emotions, key=lambda x: x[1]['accuracy_when_agree'])
        worst_emotion = min(valid_emotions, key=lambda x: x[1]['accuracy_when_agree'])

        print(f"  Highest accuracy when agree: {best_emotion[0].title()} "
              f"({best_emotion[1]['accuracy_when_agree']:.1f}%)")
        print(f"  Lowest accuracy when agree: {worst_emotion[0].title()} "
              f"({worst_emotion[1]['accuracy_when_agree']:.1f}%)")

    most_agreements = max(emotions, key=lambda e: results[e]['total_agreements'])
    print(f"  Most frequent agreement: {most_agreements.title()} "
          f"({results[most_agreements]['total_agreements']} times)")

    print("\n  When all three models agree, they are correct "
          f"{overall_stats['accuracy_when_agree']:.1f}% of the time.")

def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / "data/processed/emotion_results_comparison.csv"
    output_dir = base_dir / "experiments/results_figures"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading comparison data...")
    df = load_and_process_data(csv_path)

    print("Analyzing model agreement...")
    results, overall_stats, pivot_df = analyze_model_agreement(df)

    print_summary(results, overall_stats)

    print("\nGenerating visualizations...")
    print("-" * 70)

    plot_agreement_accuracy(results, overall_stats, output_dir)
    plot_agreement_heatmap(pivot_df, output_dir)
    create_agreement_table(results, overall_stats, output_dir)

    print("\n" + "="*70)
    print(f"All figures saved to: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
