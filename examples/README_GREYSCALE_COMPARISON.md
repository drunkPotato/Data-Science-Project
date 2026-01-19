# Greyscale vs Original Dataset 3 Comparison

This guide explains how to compare model performance between the original and greyscale versions of Emotion Dataset 3.

## Overview

This comparison tests whether converting Dataset 3 images to greyscale affects the performance of DeepFace, FER, and Custom-ResNet18 models. Dataset 3 images are already 48x48 pixels, so this tests if color information (even in low resolution) provides additional value for emotion detection.

## Complete Workflow

### Step 1: Create Greyscale Copy of Dataset 3

```bash
python examples/create_greyscale_dataset3.py
```

This script will:
- Copy the entire test set from `data/raw/emotion dataset 3/`
- Convert all images to greyscale (or copy if already greyscale)
- Save to `data/raw/emotion dataset 3 greyscale/`
- Preserve the folder structure and CSV labels

Expected output:
- Greyscale dataset at: `data/raw/emotion dataset 3 greyscale/DATASET/test/`
- Labels CSV at: `data/raw/emotion dataset 3 greyscale/test_labels.csv`

### Step 2: Evaluate Models on Original Dataset

```bash
python examples/evaluate_models_dataset3.py
```

This evaluates all three models on the original (potentially color) dataset and saves results to:
- `data/processed/dataset3_model_comparison.csv`

### Step 3: Visualize Original Results

```bash
python examples/visualize_dataset3_results.py
```

Generates visualizations in:
- `results/dataset3_evaluation/`

### Step 4: Evaluate Models on Greyscale Dataset

```bash
python examples/evaluate_models_dataset3_greyscale.py
```

This evaluates all three models on the greyscale dataset and saves results to:
- `data/processed/dataset3_greyscale_model_comparison.csv`

### Step 5: Visualize Greyscale Results

```bash
python examples/visualize_dataset3_greyscale_results.py
```

Generates visualizations in:
- `results/dataset3_greyscale_evaluation/`

### Step 6: Compare Original vs Greyscale

```bash
python examples/compare_original_vs_greyscale.py
```

This creates a direct comparison showing:
- Side-by-side accuracy and F1-score comparisons
- Performance differences for each model
- Whether greyscaling helps, hurts, or has no effect

Results saved to:
- `results/dataset3_comparison/`

## Quick Commands (Full Pipeline)

Run all steps in sequence:

```bash
# Create greyscale version
python examples/create_greyscale_dataset3.py

# Evaluate original
python examples/evaluate_models_dataset3.py
python examples/visualize_dataset3_results.py

# Evaluate greyscale
python examples/evaluate_models_dataset3_greyscale.py
python examples/visualize_dataset3_greyscale_results.py

# Compare both
python examples/compare_original_vs_greyscale.py
```

## Expected Findings

The comparison will reveal:

1. **If greyscaling improves performance**: The greyscale preprocessing may help models focus on structural features rather than color artifacts
2. **If greyscaling reduces performance**: Color information (even minimal) may provide useful emotion cues
3. **If there's no significant difference**: Dataset 3 images may already be effectively greyscale or models are color-agnostic

## Output Files

### Dataset Copies
- `data/raw/emotion dataset 3/` - Original dataset (untouched)
- `data/raw/emotion dataset 3 greyscale/` - Greyscale copy

### Evaluation Results
- `data/processed/dataset3_model_comparison.csv` - Original results
- `data/processed/dataset3_greyscale_model_comparison.csv` - Greyscale results

### Visualizations
- `results/dataset3_evaluation/` - Original visualizations
- `results/dataset3_greyscale_evaluation/` - Greyscale visualizations
- `results/dataset3_comparison/` - Comparison visualizations

## Notes

- The original Dataset 3 is never modified, only copied
- Images are converted using OpenCV's `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`
- If images are already greyscale, they are simply copied
- All three models (DeepFace, FER, Custom-ResNet18) are evaluated on both versions
- The comparison script requires both evaluations to be completed first

## Requirements

Ensure all dependencies are installed:
```bash
pip install opencv-python pandas numpy matplotlib seaborn scikit-learn
```

Plus model-specific requirements:
- DeepFace: `pip install deepface`
- FER: `pip install fer`
- Custom-ResNet18: PyTorch (`pip install torch torchvision`)
