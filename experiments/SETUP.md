# Experiment System Setup Guide

## Prerequisites

1. Python environment with all dependencies installed
2. Your emotion dataset organized in the expected format:
   ```
   data_root/
   ├── train/
   │   ├── angry/
   │   ├── disgusted/
   │   ├── fearful/
   │   ├── happy/
   │   ├── sad/
   │   ├── surprised/
   │   └── neutral/
   ├── val/
   │   └── [same structure]
   └── test/ (optional)
       └── [same structure]
   ```

## Installation

Install the experiment system dependencies:

```bash
pip install PyYAML tabulate
```

Or if you have a fresh environment:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Test with a single quick experiment

```bash
python experiments/run_experiment.py \
    experiments/configs/epochs_10.yaml \
    --data_root /path/to/your/data
```

This will run a quick 10-epoch experiment to verify everything works.

### 2. Run the baseline

```bash
python experiments/run_experiment.py \
    experiments/configs/baseline.yaml \
    --data_root /path/to/your/data
```

### 3. View results

```bash
python experiments/compare_results.py
```

### 4. Run all experiments

```bash
python experiments/run_batch.py --data_root /path/to/your/data
```

This will sequentially run all 12 experiments. You'll be prompted to confirm before starting.

## Output Location

All results are saved to:
```
experiments/runs/<experiment_name>/
```

Each experiment directory contains:
- `config.yaml` - Configuration used
- `best.pt` - Best model checkpoint
- `best_metrics.json` - Validation metrics with per-emotion breakdown
- `history.json` - Training history
- `best_confusion_matrix.npy` - Confusion matrix
- `test_metrics.json` - Test metrics (if test set exists)

## Tips for Running Experiments

1. **Start Small**: Run `epochs_10.yaml` first to verify your setup (takes ~5-10 min)

2. **Run Overnight**: All 12 experiments will take several hours. Use batch mode:
   ```bash
   nohup python experiments/run_batch.py --data_root /path/to/data > experiment.log 2>&1 &
   ```

3. **Run By Category**: Test one hyperparameter at a time:
   ```bash
   python experiments/run_batch.py --data_root /path/to/data --filter lr_
   python experiments/run_batch.py --data_root /path/to/data --filter epochs_
   ```

4. **GPU Considerations**:
   - Default batch size is 128 (requires ~8GB GPU memory)
   - If OOM errors, use `batch_64.yaml` configs
   - Can force CPU with `--cpu` flag (will be very slow)

5. **Monitoring Progress**:
   - Each experiment shows live progress with tqdm bars
   - Check `experiment_log.jsonl` for high-level status
   - Training history saved after each epoch

## Analyzing Results

### View Summary Table

```bash
python experiments/compare_results.py
```

Shows:
- Overall performance comparison
- Per-emotion metrics for each experiment
- Best experiment per emotion
- Configuration differences

### Export to CSV

```bash
python experiments/compare_results.py --export results.csv
```

Open in Excel or use pandas for deeper analysis:
```python
import pandas as pd
df = pd.read_csv('results.csv')

# Compare learning rates
df[df['experiment'].str.contains('lr_')][['experiment', 'val_macro_f1']]

# Best per-class F1 scores
emotion_cols = [c for c in df.columns if 'val_' in c and '_f1' in c]
df[['experiment'] + emotion_cols].sort_values('val_macro_f1', ascending=False)
```

### Compare Test vs Validation

```bash
python experiments/compare_results.py --split test
```

## Troubleshooting

**Error: "Config file not found"**
- Make sure you're running from the project root
- Use full path or relative path from project root

**Error: "Data root not found"**
- Check your data path
- Ensure train/ and val/ subdirectories exist

**Experiment exists**
- Use `--force` to overwrite
- Or rename the experiment in the config

**Out of memory**
- Reduce batch size in config
- Reduce number of workers
- Use CPU mode (slow): edit config or training script

**Different number of classes**
- The code expects 7 emotion classes
- If you have different classes, it will use whatever folders exist
- Per-class metrics will adapt automatically

## Understanding Results

Key metrics to compare:

1. **Overall Performance**: val_acc, val_macro_f1
2. **Per-Emotion F1**: Which emotions improve/degrade?
3. **Training Efficiency**: Which configs converge faster?
4. **Generalization**: val vs test performance gap

Common patterns:
- **Learning rate too high**: Unstable training, oscillating loss
- **Learning rate too low**: Slow convergence, may not plateau in 30 epochs
- **Overfitting**: High val performance drops on test set
- **Class imbalance**: Low F1 on minority classes (disgust, fear)

## Next Steps After Analysis

1. **Identify best single hyperparameter changes**
   - Which LR works best?
   - Does focal loss help minority classes?
   - Is augmentation helping or hurting?

2. **Combine winning strategies**
   - Create new configs combining best settings
   - Example: best_lr + focal_loss + heavy_aug

3. **Deep dive on problem emotions**
   - If "disgust" consistently underperforms, try:
     - Weighted sampler
     - Focal loss with higher gamma
     - Data augmentation focused on that class

4. **Iterate**
   - Create new configs based on findings
   - Run focused experiments
   - Compare results
