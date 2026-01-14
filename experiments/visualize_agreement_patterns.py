#!/usr/bin/env python3
"""
Additional visualizations for inter-model agreement analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import combinations

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
    df['true_emotion'] = df['image_path'].apply(extract_true_emotion)
    df = df[df['status'] == 'success']
    return df

def analyze_agreement_patterns(df):
    """Analyze different agreement patterns"""
    # Pivot data
    pivot_df = df.pivot_table(
        index='image_path',
        columns='model',
        values='dominant_emotion',
        aggfunc='first'
    ).reset_index()

    pivot_df['true_emotion'] = pivot_df['image_path'].apply(extract_true_emotion)

    model_cols = [col for col in pivot_df.columns if col not in ['image_path', 'true_emotion']]

    # Count agreement patterns
    def count_agreements(row):
        predictions = [row[col] for col in model_cols if pd.notna(row[col])]
        unique_preds = set(predictions)
        return len(predictions) - len(unique_preds) + 1  # Number agreeing

    pivot_df['num_agreeing'] = pivot_df.apply(count_agreements, axis=1)

    # Check correctness for each agreement level
    pivot_df['is_correct'] = pivot_df.apply(
        lambda row: any(row[col] == row['true_emotion'] for col in model_cols if pd.notna(row[col])),
        axis=1
    )

    # All three agree
    pivot_df['all_agree'] = pivot_df.apply(
        lambda row: len(set([row[col] for col in model_cols if pd.notna(row[col])])) == 1,
        axis=1
    )

    # Exactly two agree
    pivot_df['two_agree'] = pivot_df.apply(
        lambda row: len([row[col] for col in model_cols if pd.notna(row[col])]) == 3 and
                   len(set([row[col] for col in model_cols if pd.notna(row[col])])) == 2,
        axis=1
    )

    # All disagree
    pivot_df['all_disagree'] = pivot_df.apply(
        lambda row: len(set([row[col] for col in model_cols if pd.notna(row[col])])) == 3,
        axis=1
    )

    return pivot_df, model_cols

def plot_agreement_levels(pivot_df, output_dir):
    """Plot distribution of agreement levels"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Agreement pattern distribution
    patterns = {
        'All 3 Agree': pivot_df['all_agree'].sum(),
        'Exactly 2 Agree': pivot_df['two_agree'].sum(),
        'All Disagree': pivot_df['all_disagree'].sum()
    }

    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    bars = ax1.bar(patterns.keys(), patterns.values(), color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    # Add percentages
    total = sum(patterns.values())
    for bar, (label, count) in zip(bars, patterns.items()):
        height = bar.get_height()
        percentage = (count / total) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylabel('Number of Images', fontweight='bold')
    ax1.set_title('Distribution of Model Agreement Patterns', fontweight='bold', pad=15)
    ax1.set_ylim([0, max(patterns.values()) * 1.15])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Right: Accuracy by agreement level
    accuracy_by_pattern = {
        'All 3 Agree': (pivot_df[pivot_df['all_agree']]['is_correct'].sum() /
                       pivot_df['all_agree'].sum() * 100) if pivot_df['all_agree'].sum() > 0 else 0,
        'Exactly 2 Agree': (pivot_df[pivot_df['two_agree']]['is_correct'].sum() /
                           pivot_df['two_agree'].sum() * 100) if pivot_df['two_agree'].sum() > 0 else 0,
        'All Disagree': (pivot_df[pivot_df['all_disagree']]['is_correct'].sum() /
                        pivot_df['all_disagree'].sum() * 100) if pivot_df['all_disagree'].sum() > 0 else 0
    }

    bars2 = ax2.bar(accuracy_by_pattern.keys(), accuracy_by_pattern.values(),
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar, (label, acc) in zip(bars2, accuracy_by_pattern.items()):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.set_title('Prediction Accuracy by Agreement Level', fontweight='bold', pad=15)
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Chance Level')

    plt.tight_layout()
    plt.savefig(output_dir / "fig_agreement_levels.png", bbox_inches='tight')
    plt.savefig(output_dir / "fig_agreement_levels.pdf", bbox_inches='tight')
    plt.close()

    print("Created: fig_agreement_levels.png/.pdf")
    return patterns, accuracy_by_pattern

def plot_pairwise_agreement(pivot_df, model_cols, output_dir):
    """Plot pairwise agreement between models"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    model_pairs = list(combinations(model_cols, 2))

    for idx, (model1, model2) in enumerate(model_pairs):
        ax = axes[idx]

        # Filter valid rows
        valid_df = pivot_df[pivot_df[model1].notna() & pivot_df[model2].notna()].copy()

        # Agreement rate
        agreement = (valid_df[model1] == valid_df[model2]).sum()
        total = len(valid_df)
        agreement_rate = (agreement / total * 100) if total > 0 else 0

        # Create confusion-like matrix
        emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
        matrix = np.zeros((len(emotions), len(emotions)))

        for i, emo1 in enumerate(emotions):
            for j, emo2 in enumerate(emotions):
                count = ((valid_df[model1] == emo1) & (valid_df[model2] == emo2)).sum()
                matrix[i, j] = count

        # Normalize by row
        matrix_norm = matrix / matrix.sum(axis=1, keepdims=True)
        matrix_norm = np.nan_to_num(matrix_norm)

        # Plot
        im = ax.imshow(matrix_norm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

        # Diagonal emphasis
        for i in range(len(emotions)):
            rect = plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                                edgecolor='blue', linewidth=2.5)
            ax.add_patch(rect)

        ax.set_xticks(range(len(emotions)))
        ax.set_yticks(range(len(emotions)))
        ax.set_xticklabels([e.title() for e in emotions], rotation=45, ha='right')
        ax.set_yticklabels([e.title() for e in emotions])

        ax.set_xlabel(model2.replace('-', '\n'), fontweight='bold', fontsize=10)
        ax.set_ylabel(model1.replace('-', '\n'), fontweight='bold', fontsize=10)
        ax.set_title(f'Agreement: {agreement_rate:.1f}%\n({agreement}/{total} images)',
                    fontweight='bold', pad=10)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Frequency', rotation=270, labelpad=15, fontsize=9)

    plt.suptitle('Pairwise Model Agreement Patterns', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_pairwise_agreement.png", bbox_inches='tight')
    plt.savefig(output_dir / "fig_pairwise_agreement.pdf", bbox_inches='tight')
    plt.close()

    print("Created: fig_pairwise_agreement.png/.pdf")

def plot_agreement_by_emotion(pivot_df, output_dir):
    """Plot agreement rates and accuracy per emotion"""
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

    agreement_rates = []
    accuracies_when_agree = []
    total_samples = []

    for emotion in emotions:
        emo_df = pivot_df[pivot_df['true_emotion'] == emotion]
        total_samples.append(len(emo_df))

        if len(emo_df) > 0:
            agree_count = emo_df['all_agree'].sum()
            agreement_rate = (agree_count / len(emo_df)) * 100
            agreement_rates.append(agreement_rate)

            if agree_count > 0:
                correct_when_agree = emo_df[emo_df['all_agree']]['is_correct'].sum()
                acc = (correct_when_agree / agree_count) * 100
                accuracies_when_agree.append(acc)
            else:
                accuracies_when_agree.append(0)
        else:
            agreement_rates.append(0)
            accuracies_when_agree.append(0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    emotion_labels = [e.title() for e in emotions]
    x = np.arange(len(emotions))

    # Top: Agreement rates
    colors1 = ['#3498db' if rate > 45 else '#e67e22' for rate in agreement_rates]
    bars1 = ax1.bar(x, agreement_rates, color=colors1, alpha=0.8,
                    edgecolor='black', linewidth=1)

    for i, (bar, rate, total) in enumerate(zip(bars1, agreement_rates, total_samples)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                f'n={total}',
                ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax1.set_ylabel('Agreement Rate (%)', fontweight='bold')
    ax1.set_title('How Often All 3 Models Agree (by True Emotion)',
                  fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(emotion_labels)
    ax1.set_ylim([0, 65])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=46.1, color='red', linestyle='--', linewidth=2,
                label='Overall Agreement Rate (46.1%)')
    ax1.legend(loc='upper right')

    # Bottom: Accuracy when all agree
    colors2 = ['#2ecc71' if acc >= 80 else '#f39c12' if acc >= 60 else '#e74c3c'
              for acc in accuracies_when_agree]
    bars2 = ax2.bar(x, accuracies_when_agree, color=colors2, alpha=0.8,
                    edgecolor='black', linewidth=1)

    for bar, acc in zip(bars2, accuracies_when_agree):
        height = bar.get_height()
        if height > 5:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Emotion', fontweight='bold')
    ax2.set_ylabel('Accuracy When All Agree (%)', fontweight='bold')
    ax2.set_title('Prediction Correctness When All 3 Models Agree',
                  fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(emotion_labels)
    ax2.set_ylim([0, 105])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=84.7, color='red', linestyle='--', linewidth=2,
                label='Overall Accuracy When Agree (84.7%)')
    ax2.legend(loc='upper right')

    # Add color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.8, label='High (≥80%)'),
        Patch(facecolor='#f39c12', alpha=0.8, label='Medium (60-80%)'),
        Patch(facecolor='#e74c3c', alpha=0.8, label='Low (<60%)')
    ]
    ax2.legend(handles=legend_elements, loc='upper left', title='Accuracy Level')

    plt.tight_layout()
    plt.savefig(output_dir / "fig_agreement_by_emotion.png", bbox_inches='tight')
    plt.savefig(output_dir / "fig_agreement_by_emotion.pdf", bbox_inches='tight')
    plt.close()

    print("Created: fig_agreement_by_emotion.png/.pdf")

def create_agreement_statistics_table(patterns, accuracy_by_pattern, pivot_df, output_dir):
    """Create summary statistics table"""
    data = []

    for pattern, count in patterns.items():
        total = len(pivot_df)
        percentage = (count / total) * 100
        accuracy = accuracy_by_pattern[pattern]

        data.append({
            'Agreement Pattern': pattern,
            'Number of Images': count,
            'Percentage of Total': f'{percentage:.2f}%',
            'Prediction Accuracy': f'{accuracy:.2f}%'
        })

    df = pd.DataFrame(data)

    df.to_csv(output_dir / "table_agreement_statistics.csv", index=False)
    df.to_excel(output_dir / "table_agreement_statistics.xlsx", index=False)

    print("Created: table_agreement_statistics.csv/.xlsx")

def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / "data/processed/emotion_results_comparison.csv"
    output_dir = base_dir / "experiments/results_figures"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading comparison data...")
    df = load_and_process_data(csv_path)

    print("Analyzing agreement patterns...")
    pivot_df, model_cols = analyze_agreement_patterns(df)

    print("\nGenerating visualizations...")
    print("-" * 70)

    patterns, accuracy_by_pattern = plot_agreement_levels(pivot_df, output_dir)
    plot_pairwise_agreement(pivot_df, model_cols, output_dir)
    plot_agreement_by_emotion(pivot_df, output_dir)
    create_agreement_statistics_table(patterns, accuracy_by_pattern, pivot_df, output_dir)

    print("\n" + "="*70)
    print(f"All figures saved to: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
