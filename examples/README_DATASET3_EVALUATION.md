# Emotion Dataset 3 Model Evaluation

This guide explains how to evaluate DeepFace, FER, and Custom-ResNet18 models on Emotion Dataset 3's test set and visualize the results.

## Overview

Emotion Dataset 3 contains 3,068 test images across 7 emotion classes:
- angry (329 images)
- disgusted (74 images)
- fearful (160 images)
- happy (1,185 images)
- sad (478 images)
- surprised (162 images)
- neutral (680 images)

## Step 1: Run Model Evaluation

Evaluate all three models on the test set:

```bash
python examples/evaluate_models_dataset3.py
```

This script will:
- Load the test set from `data/raw/emotion dataset 3/test_labels.csv`
- Process all 3,068 test images with DeepFace, FER, and Custom-ResNet18
- Save results to `data/processed/dataset3_model_comparison.csv`
- Display a summary report showing:
  - Success rates for each model
  - Predicted emotion distributions
  - Processing statistics

Expected runtime: 10-30 minutes depending on your hardware

## Step 2: Generate Visualizations

Create comprehensive visualizations of model performance:

```bash
python examples/visualize_dataset3_results.py
```

This script will:
- Load the evaluation results
- Calculate detailed metrics (accuracy, precision, recall, F1-score)
- Generate publication-quality visualizations
- Save all outputs to `results/dataset3_evaluation/`

### Generated Visualizations

1. **dataset3_overall_comparison.png**
   - Bar chart comparing accuracy, precision, recall, and F1-score across models

2. **dataset3_per_emotion_f1.png**
   - Grouped bar chart showing F1-scores for each emotion class

3. **dataset3_confusion_matrices.png**
   - Confusion matrices for all three models showing prediction patterns

4. **dataset3_precision_recall.png**
   - Side-by-side comparison of precision and recall per emotion

5. **dataset3_class_distribution.png**
   - Bar chart showing the distribution of emotions in the test set

6. **dataset3_detailed_metrics.csv**
   - Comprehensive table with all metrics for further analysis

All visualizations are saved in both PNG (for viewing) and PDF (for publications) formats.

## Output Files

### Results CSV
`data/processed/dataset3_model_comparison.csv` contains:
- image_path: Path to the test image
- true_emotion: Ground truth emotion label
- label: Numeric label (1-7)
- model: Model name (DeepFace-Emotion, FER, or Custom-ResNet18)
- dominant_emotion: Predicted emotion
- {emotion}_score: Confidence scores for each emotion class
- status: success or error

### Visualization Directory
`results/dataset3_evaluation/` contains all generated plots and metrics tables.

## Metrics Explained

- **Accuracy**: Percentage of correct predictions across all classes
- **Precision**: Of all predictions for an emotion, how many were correct
- **Recall**: Of all actual instances of an emotion, how many were detected
- **F1-Score**: Harmonic mean of precision and recall (balanced metric)
- **Macro Average**: Unweighted average across all emotion classes

## Notes

- Images in Dataset 3 are 48x48 grayscale, already aligned faces
- The evaluation script sets `extract_face=False` since images are pre-cropped
- Class imbalance: 'happy' is overrepresented (38.6%), 'disgusted' is underrepresented (2.4%)
- This may affect model performance on minority classes

## Requirements

Ensure you have the required packages installed:
```bash
pip install -r requirements.txt
```

Key dependencies:
- opencv-python
- deepface
- fer
- torch, torchvision
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
