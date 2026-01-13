# Experiment Tracking System

A comprehensive system for tracking emotion detection model experiments with different hyperparameters.

## Overview

This system helps you:
- Run controlled experiments with different hyperparameters
- Track all results with per-emotion metrics (precision, recall, F1)
- Compare experiments side-by-side
- Export results for analysis

## Directory Structure

```
experiments/
├── configs/          # YAML experiment configurations
├── runs/            # Output from each experiment
├── run_experiment.py    # Run a single experiment
├── run_batch.py        # Run multiple experiments
└── compare_results.py  # Compare experiment results
```

## Quick Start

### 1. Run a Single Experiment

```bash
python experiments/run_experiment.py \
    experiments/configs/baseline.yaml \
    --data_root /path/to/your/data
```

### 2. Run Multiple Experiments

```bash
python experiments/run_batch.py \
    --data_root /path/to/your/data
```

This will run all experiments in `experiments/configs/` sequentially.

### 3. Compare Results

```bash
python experiments/compare_results.py
```

Or with CSV export:

```bash
python experiments/compare_results.py --export results.csv
```

## Available Experiment Configs

### Baseline
- `baseline.yaml`: Standard configuration (30 epochs, lr=3e-4, batch=128)

### Learning Rate Experiments
- `lr_low.yaml`: Lower LR (1e-4)
- `lr_high.yaml`: Higher LR (1e-3)

### Epoch Experiments
- `epochs_10.yaml`: Quick 10 epoch baseline
- `epochs_50.yaml`: Extended 50 epochs

### Batch Size Experiments
- `batch_64.yaml`: Smaller batch (64)
- `batch_256.yaml`: Larger batch (256)

### Loss Function Experiments
- `baseline.yaml`: Cross-entropy loss
- `loss_focal.yaml`: Focal loss for imbalance

### Augmentation Experiments
- `aug_none.yaml`: No augmentation
- `baseline.yaml`: Moderate augmentation
- `aug_heavy.yaml`: Heavy augmentation

### Class Balancing Experiments
- `balance_no_weights.yaml`: No balancing
- `baseline.yaml`: Class weights only
- `balance_weighted_sampler.yaml`: Weighted sampler

## Creating Custom Experiments

Create a new YAML file in `experiments/configs/`:

```yaml
name: my_experiment
description: Testing something new

model:
  arch: resnet18
  dropout: 0.3
  pretrained: true

training:
  epochs: 30
  batch_size: 128
  eval_batch_size: 256
  lr: 0.0003
  weight_decay: 0.05
  optimizer: adamw
  freeze_epochs: 0
  onecycle_pct_start: 0.3

loss:
  type: ce  # or 'focal'
  label_smoothing: 0.1
  focal_gamma: 2.0
  use_class_weights: true

augmentation:
  mixup_alpha: 0.2
  random_erasing: 0.0

data:
  img_size: 224
  weighted_sampler: false

other:
  patience: 7
  seed: 42
  workers: 8
```

## Output Structure

Each experiment creates:

```
experiments/runs/[experiment_name]/
├── config.yaml              # Copy of config used
├── best.pt                  # Best model checkpoint
├── best_metrics.json        # Validation metrics (overall + per-class)
├── best_confusion_matrix.npy # Confusion matrix
├── history.json             # Training history
├── test_metrics.json        # Test set metrics (if test set exists)
└── test_confusion_matrix.npy
```

## Metrics Tracked

For each experiment, the system tracks:

**Overall Metrics:**
- Accuracy
- Macro F1-score

**Per-Emotion Metrics:**
- Precision (for each: angry, disgusted, fearful, happy, sad, surprised, neutral)
- Recall
- F1-score

## Advanced Usage

### Filter Experiments by Prefix

Run only learning rate experiments:
```bash
python experiments/run_batch.py \
    --data_root /path/to/data \
    --filter lr_
```

### Compare Test Set Results

```bash
python experiments/compare_results.py --split test
```

### Export for External Analysis

```bash
python experiments/compare_results.py --export results.csv
```

The CSV contains all hyperparameters and per-emotion metrics for easy analysis in Excel or pandas.

## Tips

1. Start with `baseline.yaml` to establish a reference point
2. Run `epochs_10.yaml` first to quickly test if your data pipeline works
3. Use `--force` flag to overwrite existing experiments if needed
4. Check `experiment_log.jsonl` for a complete history of all runs
5. The system automatically saves the best model based on validation F1-score
6. Early stopping patience is configurable per experiment

## Troubleshooting

**Experiment already exists:**
```bash
python experiments/run_experiment.py config.yaml --data_root /path --force
```

**Out of memory:**
- Try smaller batch size configs (e.g., `batch_64.yaml`)
- Reduce `workers` in the config

**Want to see what will run:**
```bash
python experiments/run_batch.py --data_root /path
# Press Ctrl+C at the confirmation prompt to cancel
```
