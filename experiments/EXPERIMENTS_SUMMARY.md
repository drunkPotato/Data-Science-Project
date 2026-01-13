# Experiment Configurations Summary

## All Available Experiments (11 total)

### 1. Baseline
**File:** `baseline.yaml`
**Purpose:** Reference configuration for comparison
**Key params:** 30 epochs, LR=3e-4, batch=128, CE loss, moderate augmentation

---

## Learning Rate Experiments (2)

### 2. Low Learning Rate
**File:** `lr_low.yaml`
**Purpose:** Test if slower learning improves final performance
**Changes from baseline:** LR = 1e-4 (vs 3e-4)

### 3. High Learning Rate
**File:** `lr_high.yaml`
**Purpose:** Test if faster learning still converges well
**Changes from baseline:** LR = 1e-3 (vs 3e-4)

---

## Epoch Experiments (2)

### 4. Quick Training (10 epochs)
**File:** `epochs_10.yaml`
**Purpose:** See baseline performance with minimal training
**Changes from baseline:** 10 epochs (vs 30)

### 5. Extended Training (50 epochs)
**File:** `epochs_50.yaml`
**Purpose:** Test if longer training continues to improve
**Changes from baseline:** 50 epochs (vs 30), patience=10

---

## Batch Size Experiments (2)

### 6. Small Batch
**File:** `batch_64.yaml`
**Purpose:** Test if smaller batches improve generalization
**Changes from baseline:** batch_size = 64 (vs 128)

### 7. Large Batch
**File:** `batch_256.yaml`
**Purpose:** Test if larger batches train faster without hurting performance
**Changes from baseline:** batch_size = 256 (vs 128)

---

## Loss Function Experiments (1)

### 8. Focal Loss
**File:** `loss_focal.yaml`
**Purpose:** Test if focal loss handles class imbalance better
**Changes from baseline:** loss = 'focal' (vs 'ce')

---

## Data Augmentation Experiments (2)

### 9. No Augmentation
**File:** `aug_none.yaml`
**Purpose:** Test baseline performance without augmentation
**Changes from baseline:** mixup_alpha=0.0, random_erasing=0.0

### 10. Heavy Augmentation
**File:** `aug_heavy.yaml`
**Purpose:** Test if aggressive augmentation prevents overfitting
**Changes from baseline:** mixup_alpha=0.4, random_erasing=0.3

---

## Class Balancing Experiments (2)

### 11. No Balancing
**File:** `balance_no_weights.yaml`
**Purpose:** See how much class imbalance affects performance
**Changes from baseline:** use_class_weights=false, weighted_sampler=false

### 12. Weighted Sampler
**File:** `balance_weighted_sampler.yaml`
**Purpose:** Test oversampling minority classes during training
**Changes from baseline:** weighted_sampler=true

---

## Recommended Experiment Order

### Phase 1: Quick Tests (fastest)
1. `epochs_10.yaml` - Verify everything works (fast)
2. `baseline.yaml` - Establish reference point

### Phase 2: Core Hyperparameters (most impactful)
3. `lr_low.yaml` - Learning rate sweep
4. `lr_high.yaml` - Learning rate sweep
5. `epochs_50.yaml` - See if more training helps
6. `loss_focal.yaml` - Alternative loss function

### Phase 3: Training Details (optimization)
7. `batch_64.yaml` - Batch size effects
8. `batch_256.yaml` - Batch size effects
9. `aug_none.yaml` - Augmentation ablation
10. `aug_heavy.yaml` - Augmentation sweep

### Phase 4: Class Imbalance (if issues found)
11. `balance_no_weights.yaml` - Ablation study
12. `balance_weighted_sampler.yaml` - Oversampling approach

---

## Expected Insights

**Learning Rate:**
- Too low: Slow convergence, may not reach optimal performance in 30 epochs
- Too high: May overshoot optimal, unstable training
- Goal: Find sweet spot for fastest convergence to best performance

**Epochs:**
- 10 epochs: Quick baseline, likely underfit
- 30 epochs: Should be sufficient for convergence
- 50 epochs: Test if additional training helps or causes overfitting

**Batch Size:**
- Smaller (64): More gradient updates, potentially better generalization, slower
- Larger (256): Faster training, more stable gradients, may generalize worse

**Loss Function:**
- CE: Standard choice
- Focal: May help with hard-to-classify emotions (disgust, fear often underperform)

**Augmentation:**
- None: May overfit, good for seeing raw model capacity
- Moderate: Baseline approach
- Heavy: May prevent overfitting but could hurt if dataset is small

**Class Balancing:**
- Test impact on minority class performance (likely disgust, fear)
- Compare precision/recall tradeoffs per emotion

---

## Quick Run All Experiments

```bash
# Run all experiments overnight
python experiments/run_batch.py --data_root /path/to/data

# Or run just one category
python experiments/run_batch.py --data_root /path/to/data --filter lr_
python experiments/run_batch.py --data_root /path/to/data --filter epochs_
python experiments/run_batch.py --data_root /path/to/data --filter batch_
python experiments/run_batch.py --data_root /path/to/data --filter aug_
python experiments/run_batch.py --data_root /path/to/data --filter balance_
```

## After Running Experiments

```bash
# View comparison
python experiments/compare_results.py

# Export to CSV for analysis
python experiments/compare_results.py --export results.csv

# View test set results (if available)
python experiments/compare_results.py --split test
```
