#!/usr/bin/env python3
"""
Ensemble approach: Use Custom-ResNet18 as base, but override with DeepFace+FER agreement
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
    """Load and pivot results for ensemble analysis"""
    df = pd.read_csv(csv_path)
    df = df[df['status'] == 'success'].copy()

    # Pivot so each image has all model predictions in one row
    pivot_df = df.pivot_table(
        index=['image_path', 'true_emotion', 'label'],
        columns='model',
        values='dominant_emotion',
        aggfunc='first'
    ).reset_index()

    # Drop rows where any model failed
    pivot_df = pivot_df.dropna(subset=['DeepFace-Emotion', 'FER', 'Custom-ResNet18'])

    print(f"Loaded {len(pivot_df)} images with predictions from all 3 models")

    return pivot_df


def apply_ensemble_logic(df):
    """
    Apply ensemble logic:
    - Base: Custom-ResNet18
    - Override: If DeepFace AND FER agree on something different, use their prediction
    """
    ensemble_predictions = []
    override_cases = []

    for idx, row in df.iterrows():
        custom_pred = row['Custom-ResNet18']
        deepface_pred = row['DeepFace-Emotion']
        fer_pred = row['FER']
        true_emotion = row['true_emotion']

        # Check if DeepFace and FER agree
        deepface_fer_agree = (deepface_pred == fer_pred)

        # If DeepFace and FER agree on something different than Custom, override
        if deepface_fer_agree and (deepface_pred != custom_pred):
            final_pred = deepface_pred
            override_cases.append({
                'image_path': row['image_path'],
                'true_emotion': true_emotion,
                'custom_pred': custom_pred,
                'override_pred': deepface_pred,
                'was_custom_correct': custom_pred == true_emotion,
                'is_override_correct': deepface_pred == true_emotion
            })
        else:
            # Use Custom-ResNet18 prediction
            final_pred = custom_pred

        ensemble_predictions.append(final_pred)

    df['ensemble_prediction'] = ensemble_predictions

    return df, override_cases


def calculate_metrics(y_true, y_pred, model_name):
    """Calculate comprehensive metrics"""
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=emotions, average=None, zero_division=0
    )

    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    return {
        'model': model_name,
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


def analyze_override_impact(override_cases):
    """Analyze when overrides helped vs hurt"""
    if not override_cases:
        return None

    override_df = pd.DataFrame(override_cases)

    total_overrides = len(override_df)
    custom_was_correct = override_df['was_custom_correct'].sum()
    override_is_correct = override_df['is_override_correct'].sum()

    # Cases where override helped (Custom was wrong, override is right)
    helped = len(override_df[
        (~override_df['was_custom_correct']) & (override_df['is_override_correct'])
    ])

    # Cases where override hurt (Custom was right, override is wrong)
    hurt = len(override_df[
        (override_df['was_custom_correct']) & (~override_df['is_override_correct'])
    ])

    # Both wrong or both right
    both_wrong = len(override_df[
        (~override_df['was_custom_correct']) & (~override_df['is_override_correct'])
    ])
    both_right = len(override_df[
        (override_df['was_custom_correct']) & (override_df['is_override_correct'])
    ])

    return {
        'total_overrides': total_overrides,
        'helped': helped,
        'hurt': hurt,
        'both_wrong': both_wrong,
        'both_right': both_right,
        'net_gain': helped - hurt,
        'override_df': override_df
    }


def plot_accuracy_comparison(metrics_list, output_dir):
    """Compare accuracy across all models + ensemble"""
    fig, ax = plt.subplots(figsize=(12, 7))

    models = [m['model'] for m in metrics_list]
    accuracies = [m['accuracy'] * 100 for m in metrics_list]
    f1_scores = [m['macro_f1'] * 100 for m in metrics_list]

    x = np.arange(len(models))
    width = 0.35

    # Use different colors for ensemble
    colors_acc = ['#4ECDC4'] * (len(models) - 1) + ['#FF6B6B']
    colors_f1 = ['#45B7D1'] * (len(models) - 1) + ['#FFA07A']

    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy',
                   color=colors_acc, alpha=0.8)
    bars2 = ax.bar(x + width/2, f1_scores, width, label='Macro F1',
                   color=colors_f1, alpha=0.8)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model')
    ax.set_ylabel('Score (%)')
    ax.set_title('Performance Comparison: Individual Models vs. Ensemble\n' +
                'Ensemble: Custom-ResNet18 + Override when DeepFace & FER Agree')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])

    # Highlight the ensemble bar
    ax.axvline(x=len(models)-1.5, color='red', linestyle='--', alpha=0.3, linewidth=2)

    plt.tight_layout()
    plt.savefig(output_dir / "ensemble_accuracy_comparison.png", bbox_inches='tight')
    plt.savefig(output_dir / "ensemble_accuracy_comparison.pdf", bbox_inches='tight')
    plt.close()

    print("Created: ensemble_accuracy_comparison.png/.pdf")


def plot_per_emotion_comparison(metrics_list, output_dir):
    """Compare per-emotion F1 scores"""
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

    # Prepare data - focus on Custom vs Ensemble
    custom_metrics = next(m for m in metrics_list if m['model'] == 'Custom-ResNet18')
    ensemble_metrics = next(m for m in metrics_list if m['model'] == 'Ensemble')

    custom_f1 = [custom_metrics['per_class'][e]['f1'] * 100 for e in emotions]
    ensemble_f1 = [ensemble_metrics['per_class'][e]['f1'] * 100 for e in emotions]

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(emotions))
    width = 0.35

    bars1 = ax.bar(x - width/2, custom_f1, width, label='Custom-ResNet18',
                   alpha=0.8, color='#4ECDC4')
    bars2 = ax.bar(x + width/2, ensemble_f1, width, label='Ensemble',
                   alpha=0.8, color='#FF6B6B')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 3:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}',
                       ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Emotion')
    ax.set_ylabel('F1-Score (%)')
    ax.set_title('Per-Emotion F1-Score: Custom-ResNet18 vs. Ensemble')
    ax.set_xticks(x)
    ax.set_xticklabels([e.title() for e in emotions])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(output_dir / "ensemble_per_emotion_f1.png", bbox_inches='tight')
    plt.savefig(output_dir / "ensemble_per_emotion_f1.pdf", bbox_inches='tight')
    plt.close()

    print("Created: ensemble_per_emotion_f1.png/.pdf")


def plot_confusion_matrix(metrics, output_dir):
    """Plot confusion matrix for ensemble"""
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

    y_true = metrics['y_true']
    y_pred = metrics['y_pred']

    cm = confusion_matrix(y_true, y_pred, labels=emotions)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
               xticklabels=[e[:3].upper() for e in emotions],
               yticklabels=[e[:3].upper() for e in emotions],
               ax=ax, cbar_kws={'label': 'Percentage (%)'})

    ax.set_title(f'Ensemble Confusion Matrix\nAccuracy: {metrics["accuracy"]*100:.1f}%',
                fontsize=14)
    ax.set_ylabel('True Emotion')
    ax.set_xlabel('Predicted Emotion')

    plt.tight_layout()
    plt.savefig(output_dir / "ensemble_confusion_matrix.png", bbox_inches='tight')
    plt.savefig(output_dir / "ensemble_confusion_matrix.pdf", bbox_inches='tight')
    plt.close()

    print("Created: ensemble_confusion_matrix.png/.pdf")


def plot_override_impact(override_analysis, output_dir):
    """Visualize impact of overrides"""
    if not override_analysis:
        print("No overrides occurred")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart of override outcomes
    labels = ['Helped\n(Custom wrong → Override right)',
             'Hurt\n(Custom right → Override wrong)',
             'Both Wrong',
             'Both Right']
    sizes = [override_analysis['helped'],
            override_analysis['hurt'],
            override_analysis['both_wrong'],
            override_analysis['both_right']]
    colors = ['#2ECC71', '#E74C3C', '#95A5A6', '#3498DB']
    explode = (0.1, 0.1, 0, 0)

    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
    ax1.set_title(f'Override Impact Analysis\n(Total Overrides: {override_analysis["total_overrides"]})')

    # Bar chart showing net impact
    categories = ['Helped', 'Hurt', 'Net Gain']
    values = [override_analysis['helped'],
             override_analysis['hurt'],
             override_analysis['net_gain']]
    colors_bar = ['#2ECC71', '#E74C3C', '#3498DB' if override_analysis['net_gain'] >= 0 else '#E74C3C']

    bars = ax2.bar(categories, values, color=colors_bar, alpha=0.8)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        label_y = height if height >= 0 else height - 5
        ax2.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{int(height)}',
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=11)

    ax2.set_ylabel('Number of Cases')
    ax2.set_title('Override Impact Summary')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "ensemble_override_impact.png", bbox_inches='tight')
    plt.savefig(output_dir / "ensemble_override_impact.pdf", bbox_inches='tight')
    plt.close()

    print("Created: ensemble_override_impact.png/.pdf")


def analyze_disagreement_cases(df):
    """Analyze cases where all 3 models disagree"""
    disagreement_cases = []

    for idx, row in df.iterrows():
        custom = row['Custom-ResNet18']
        deepface = row['DeepFace-Emotion']
        fer = row['FER']
        true_emotion = row['true_emotion']

        # Check if all three predictions are different
        if custom != deepface and custom != fer and deepface != fer:
            disagreement_cases.append({
                'image_path': row['image_path'],
                'true_emotion': true_emotion,
                'custom_pred': custom,
                'deepface_pred': deepface,
                'fer_pred': fer,
                'custom_correct': custom == true_emotion,
                'deepface_correct': deepface == true_emotion,
                'fer_correct': fer == true_emotion
            })

    if not disagreement_cases:
        return None

    disagreement_df = pd.DataFrame(disagreement_cases)

    total = len(disagreement_df)
    custom_correct = disagreement_df['custom_correct'].sum()
    deepface_correct = disagreement_df['deepface_correct'].sum()
    fer_correct = disagreement_df['fer_correct'].sum()
    none_correct = len(disagreement_df[
        (~disagreement_df['custom_correct']) &
        (~disagreement_df['deepface_correct']) &
        (~disagreement_df['fer_correct'])
    ])

    return {
        'total_disagreements': total,
        'custom_correct': custom_correct,
        'deepface_correct': deepface_correct,
        'fer_correct': fer_correct,
        'none_correct': none_correct,
        'custom_accuracy': custom_correct / total if total > 0 else 0,
        'deepface_accuracy': deepface_correct / total if total > 0 else 0,
        'fer_accuracy': fer_correct / total if total > 0 else 0,
        'disagreement_df': disagreement_df
    }


def plot_disagreement_performance(disagreement_analysis, overall_metrics, output_dir):
    """Visualize performance on disagreement cases vs overall"""
    if not disagreement_analysis:
        print("No disagreement cases found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Accuracy on disagreement cases
    models = ['Custom-ResNet18', 'DeepFace-Emotion', 'FER']
    disagreement_acc = [
        disagreement_analysis['custom_accuracy'] * 100,
        disagreement_analysis['deepface_accuracy'] * 100,
        disagreement_analysis['fer_accuracy'] * 100
    ]

    bars = ax1.bar(models, disagreement_acc, color=['#4ECDC4', '#FF6B6B', '#45B7D1'], alpha=0.8)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)

    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title(f'Performance on Difficult Cases (All Models Disagree)\n' +
                  f'Total Cases: {disagreement_analysis["total_disagreements"]}')
    ax1.set_xticklabels(models, rotation=15, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 100])

    # Right plot: Comparison with overall accuracy
    model_names = ['Custom-ResNet18', 'DeepFace', 'FER']
    overall_acc = [
        next(m for m in overall_metrics if m['model'] == 'Custom-ResNet18')['accuracy'] * 100,
        next(m for m in overall_metrics if m['model'] == 'DeepFace-Emotion')['accuracy'] * 100,
        next(m for m in overall_metrics if m['model'] == 'FER')['accuracy'] * 100
    ]

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax2.bar(x - width/2, overall_acc, width, label='Overall Accuracy',
                    alpha=0.8, color='#2ECC71')
    bars2 = ax2.bar(x + width/2, disagreement_acc, width, label='Disagreement Cases',
                    alpha=0.8, color='#E74C3C')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=8)

    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Overall Accuracy vs. Accuracy on Difficult Cases')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=15, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(output_dir / "ensemble_disagreement_performance.png", bbox_inches='tight')
    plt.savefig(output_dir / "ensemble_disagreement_performance.pdf", bbox_inches='tight')
    plt.close()

    print("Created: ensemble_disagreement_performance.png/.pdf")


def plot_agreement_patterns(df, output_dir):
    """Visualize different agreement patterns"""
    # Calculate agreement patterns
    patterns = {
        'All 3 Agree': 0,
        'Custom + DeepFace': 0,
        'Custom + FER': 0,
        'DeepFace + FER (Override)': 0,
        'All Disagree': 0
    }

    for idx, row in df.iterrows():
        custom = row['Custom-ResNet18']
        deepface = row['DeepFace-Emotion']
        fer = row['FER']

        if custom == deepface == fer:
            patterns['All 3 Agree'] += 1
        elif custom == deepface and custom != fer:
            patterns['Custom + DeepFace'] += 1
        elif custom == fer and custom != deepface:
            patterns['Custom + FER'] += 1
        elif deepface == fer and deepface != custom:
            patterns['DeepFace + FER (Override)'] += 1
        else:
            patterns['All Disagree'] += 1

    fig, ax = plt.subplots(figsize=(10, 6))

    pattern_names = list(patterns.keys())
    pattern_counts = list(patterns.values())
    colors = ['#2ECC71', '#3498DB', '#9B59B6', '#FF6B6B', '#95A5A6']

    bars = ax.bar(pattern_names, pattern_counts, color=colors, alpha=0.8)

    # Add value labels and percentages
    total = sum(pattern_counts)
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}\n({percentage:.1f}%)',
               ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Number of Images')
    ax.set_title('Model Agreement Patterns')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "ensemble_agreement_patterns.png", bbox_inches='tight')
    plt.savefig(output_dir / "ensemble_agreement_patterns.pdf", bbox_inches='tight')
    plt.close()

    print("Created: ensemble_agreement_patterns.png/.pdf")

    return patterns


def generate_detailed_report(metrics_list, override_analysis, patterns, disagreement_analysis, output_dir):
    """Generate comprehensive text report"""
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("ENSEMBLE EVALUATION REPORT")
    report_lines.append("Strategy: Custom-ResNet18 + Override when DeepFace & FER Agree")
    report_lines.append("="*80)
    report_lines.append("")

    # Overall metrics comparison
    report_lines.append("OVERALL PERFORMANCE COMPARISON:")
    report_lines.append("-"*80)
    report_lines.append(f"{'Model':<25} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    report_lines.append("-"*80)

    for m in metrics_list:
        report_lines.append(
            f"{m['model']:<25} "
            f"{m['accuracy']*100:>11.2f}% "
            f"{m['macro_precision']*100:>11.2f}% "
            f"{m['macro_recall']*100:>11.2f}% "
            f"{m['macro_f1']*100:>11.2f}%"
        )

    report_lines.append("")

    # Find improvements
    custom_metrics = next(m for m in metrics_list if m['model'] == 'Custom-ResNet18')
    ensemble_metrics = next(m for m in metrics_list if m['model'] == 'Ensemble')

    acc_improvement = (ensemble_metrics['accuracy'] - custom_metrics['accuracy']) * 100
    f1_improvement = (ensemble_metrics['macro_f1'] - custom_metrics['macro_f1']) * 100

    report_lines.append("ENSEMBLE IMPROVEMENT OVER CUSTOM-RESNET18:")
    report_lines.append("-"*80)
    report_lines.append(f"Accuracy improvement: {acc_improvement:+.2f}%")
    report_lines.append(f"F1-Score improvement: {f1_improvement:+.2f}%")
    report_lines.append("")

    # Override analysis
    if override_analysis:
        report_lines.append("OVERRIDE ANALYSIS:")
        report_lines.append("-"*80)
        report_lines.append(f"Total predictions: {len(metrics_list[0]['y_true'])}")
        report_lines.append(f"Times DeepFace & FER agreed (and differed from Custom): {override_analysis['total_overrides']}")
        report_lines.append(f"  - Overrides that helped (Custom wrong → Override right): {override_analysis['helped']}")
        report_lines.append(f"  - Overrides that hurt (Custom right → Override wrong): {override_analysis['hurt']}")
        report_lines.append(f"  - Both were wrong: {override_analysis['both_wrong']}")
        report_lines.append(f"  - Both were right: {override_analysis['both_right']}")
        report_lines.append(f"Net gain from overrides: {override_analysis['net_gain']:+d} correct predictions")
        report_lines.append("")

    # Agreement patterns
    report_lines.append("AGREEMENT PATTERNS:")
    report_lines.append("-"*80)
    total = sum(patterns.values())
    for pattern, count in patterns.items():
        percentage = (count / total) * 100
        report_lines.append(f"{pattern:<30}: {count:>5} ({percentage:>5.1f}%)")
    report_lines.append("")

    # Disagreement analysis
    if disagreement_analysis:
        report_lines.append("PERFORMANCE ON DIFFICULT CASES (ALL MODELS DISAGREE):")
        report_lines.append("-"*80)
        report_lines.append(f"Total disagreement cases: {disagreement_analysis['total_disagreements']} " +
                          f"({disagreement_analysis['total_disagreements']/len(metrics_list[0]['y_true'])*100:.1f}% of all predictions)")
        report_lines.append(f"None of the models were correct: {disagreement_analysis['none_correct']}")
        report_lines.append("")
        report_lines.append("Individual model performance on disagreement cases:")
        report_lines.append(f"  Custom-ResNet18: {disagreement_analysis['custom_correct']}/{disagreement_analysis['total_disagreements']} " +
                          f"correct ({disagreement_analysis['custom_accuracy']*100:.1f}%)")
        report_lines.append(f"  DeepFace:        {disagreement_analysis['deepface_correct']}/{disagreement_analysis['total_disagreements']} " +
                          f"correct ({disagreement_analysis['deepface_accuracy']*100:.1f}%)")
        report_lines.append(f"  FER:             {disagreement_analysis['fer_correct']}/{disagreement_analysis['total_disagreements']} " +
                          f"correct ({disagreement_analysis['fer_accuracy']*100:.1f}%)")
        report_lines.append("")

        # Compare to overall accuracy
        custom_overall = next(m for m in metrics_list if m['model'] == 'Custom-ResNet18')['accuracy'] * 100
        deepface_overall = next(m for m in metrics_list if m['model'] == 'DeepFace-Emotion')['accuracy'] * 100
        fer_overall = next(m for m in metrics_list if m['model'] == 'FER')['accuracy'] * 100

        report_lines.append("Accuracy drop on difficult cases compared to overall:")
        report_lines.append(f"  Custom-ResNet18: {custom_overall:.1f}% -> {disagreement_analysis['custom_accuracy']*100:.1f}% " +
                          f"({disagreement_analysis['custom_accuracy']*100 - custom_overall:+.1f}%)")
        report_lines.append(f"  DeepFace:        {deepface_overall:.1f}% -> {disagreement_analysis['deepface_accuracy']*100:.1f}% " +
                          f"({disagreement_analysis['deepface_accuracy']*100 - deepface_overall:+.1f}%)")
        report_lines.append(f"  FER:             {fer_overall:.1f}% -> {disagreement_analysis['fer_accuracy']*100:.1f}% " +
                          f"({disagreement_analysis['fer_accuracy']*100 - fer_overall:+.1f}%)")
        report_lines.append("")

    report_lines.append("="*80)

    # Save report
    report_path = output_dir / "ensemble_evaluation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"Created: ensemble_evaluation_report.txt")

    # Print to console
    print('\n' + '\n'.join(report_lines))


def main():
    # Configuration
    results_file = Path(__file__).parent.parent / "data" / "processed" / "dataset3_model_comparison.csv"
    output_dir = Path(__file__).parent.parent / "results" / "ensemble_evaluation"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("ENSEMBLE APPROACH: CUSTOM-RESNET18 + DEEPFACE/FER OVERRIDE")
    print("="*80)

    # Check if results exist
    if not results_file.exists():
        print(f"\nERROR: Results file not found: {results_file}")
        print("Please run: python examples/evaluate_models_dataset3.py")
        return

    # Load results
    print(f"\nLoading results from: {results_file}")
    df = load_results(results_file)

    # Apply ensemble logic
    print("\nApplying ensemble logic...")
    df, override_cases = apply_ensemble_logic(df)

    print(f"Total predictions: {len(df)}")
    print(f"Override cases (DeepFace & FER agreed, differed from Custom): {len(override_cases)}")

    # Calculate metrics for all models + ensemble
    print("\nCalculating metrics...")
    metrics_list = []

    # Individual models
    for model_name in ['DeepFace-Emotion', 'FER', 'Custom-ResNet18']:
        metrics = calculate_metrics(
            df['true_emotion'].values,
            df[model_name].values,
            model_name
        )
        metrics_list.append(metrics)

    # Ensemble
    ensemble_metrics = calculate_metrics(
        df['true_emotion'].values,
        df['ensemble_prediction'].values,
        'Ensemble'
    )
    metrics_list.append(ensemble_metrics)

    # Analyze override impact
    print("\nAnalyzing override impact...")
    override_analysis = analyze_override_impact(override_cases)

    # Analyze agreement patterns
    print("\nAnalyzing agreement patterns...")
    patterns = plot_agreement_patterns(df, output_dir)

    # Analyze disagreement cases
    print("\nAnalyzing disagreement cases (difficult samples)...")
    disagreement_analysis = analyze_disagreement_cases(df)
    if disagreement_analysis:
        print(f"Found {disagreement_analysis['total_disagreements']} cases where all models disagree")

    # Generate visualizations
    print("\nGenerating visualizations...")
    print(f"Output directory: {output_dir}\n")

    plot_accuracy_comparison(metrics_list, output_dir)
    plot_per_emotion_comparison(metrics_list, output_dir)
    plot_confusion_matrix(ensemble_metrics, output_dir)
    plot_override_impact(override_analysis, output_dir)
    if disagreement_analysis:
        plot_disagreement_performance(disagreement_analysis, metrics_list, output_dir)

    # Generate detailed report
    print()
    generate_detailed_report(metrics_list, override_analysis, patterns, disagreement_analysis, output_dir)

    print("\n" + "="*80)
    print("ENSEMBLE EVALUATION COMPLETE")
    print(f"All outputs saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
