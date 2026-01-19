# Ensemble Approach: Custom-ResNet18 with DeepFace/FER Override

This ensemble strategy uses Custom-ResNet18 as the base model but overrides its predictions when DeepFace and FER agree on a different emotion.

## Ensemble Logic

The decision process works as follows:

1. **Start with Custom-ResNet18 prediction** as the base
2. **Check if DeepFace and FER agree** on an emotion
3. **If they agree AND their prediction differs from Custom**: Override with their agreed prediction
4. **Otherwise**: Use Custom-ResNet18 prediction

### Example Scenarios

| Custom-ResNet18 | DeepFace | FER | Final Prediction | Reason |
|----------------|----------|-----|------------------|---------|
| happy | happy | happy | **happy** | All agree |
| happy | happy | sad | **happy** | Custom prediction (no DF+FER agreement) |
| happy | sad | happy | **happy** | Custom prediction (no DF+FER agreement) |
| happy | sad | sad | **sad** | Override (DF+FER agree, differ from Custom) |
| happy | sad | angry | **happy** | Custom prediction (all disagree) |

## Why This Approach?

This ensemble strategy leverages the strengths of multiple models:

- **Custom-ResNet18**: Trained specifically on your data, serves as the foundation
- **DeepFace + FER Agreement**: When two independent models agree, they provide a strong "second opinion"
- **Override Logic**: Corrects Custom-ResNet18 when it's likely wrong (two other models agree against it)

## Running the Ensemble Analysis

### Prerequisites

First, evaluate the models on Dataset 3:

```bash
python examples/evaluate_models_dataset3.py
```

### Run Ensemble Analysis

```bash
python examples/ensemble_custom_with_override.py
```

## Generated Outputs

All outputs are saved to `results/ensemble_evaluation/`:

### 1. Visualizations

#### ensemble_accuracy_comparison.png
- Bar chart comparing accuracy and F1-score across:
  - DeepFace-Emotion
  - FER
  - Custom-ResNet18
  - **Ensemble** (highlighted)

#### ensemble_per_emotion_f1.png
- Per-emotion F1-score comparison
- Shows which emotions benefit most from ensemble approach

#### ensemble_confusion_matrix.png
- Confusion matrix for the ensemble predictions
- Shows where the ensemble makes mistakes

#### ensemble_override_impact.png
- Two visualizations:
  - **Pie chart**: Breakdown of override outcomes
    - Helped (Custom wrong → Override right)
    - Hurt (Custom right → Override wrong)
    - Both wrong
    - Both right
  - **Bar chart**: Net gain from overrides

#### ensemble_agreement_patterns.png
- Distribution of agreement patterns:
  - All 3 models agree
  - Custom + DeepFace agree
  - Custom + FER agree
  - **DeepFace + FER agree (Override cases)**
  - All disagree

### 2. Text Report

#### ensemble_evaluation_report.txt
Comprehensive report including:
- Overall performance comparison table
- Improvement over Custom-ResNet18 alone
- Override analysis statistics
- Agreement pattern breakdown

## Interpreting Results

### If Ensemble Accuracy > Custom-ResNet18:
The override strategy is working! DeepFace and FER's agreement provides valuable corrections.

### If Ensemble Accuracy ≈ Custom-ResNet18:
The overrides roughly balance out (helped ≈ hurt). Custom-ResNet18 is already strong.

### If Ensemble Accuracy < Custom-ResNet18:
The overrides are hurting more than helping. DeepFace+FER agreement may not be reliable for this dataset.

## Key Metrics to Check

1. **Net Gain**: Number of helped cases minus hurt cases
   - Positive = Ensemble improves performance
   - Negative = Ensemble reduces performance

2. **Override Rate**: Percentage of predictions where DeepFace+FER agreed against Custom
   - High rate = Models frequently disagree
   - Low rate = Models usually align

3. **Per-Emotion Improvements**: Which emotions benefit from the ensemble?
   - Look at emotions where ensemble F1 > Custom F1

## Comparison with Simple Voting

This approach differs from simple majority voting:

| Approach | Decision Rule | Advantage |
|----------|---------------|-----------|
| **This Ensemble** | Custom + DF/FER override | Prioritizes your trained model, uses others as validators |
| **Majority Vote** | Most common prediction | All models weighted equally |
| **Weighted Vote** | Weighted by accuracy | Requires tuning weights |

## Next Steps

After analyzing the ensemble results:

1. **If ensemble helps**: Consider using it in production
2. **If specific emotions improve**: Use ensemble selectively for those emotions
3. **If ensemble hurts**: Stick with Custom-ResNet18 alone or try different ensemble strategies

## Alternative Ensemble Strategies

You can modify the script to try:

1. **All-agree only**: Only trust predictions when all 3 models agree
2. **Confidence-based**: Use prediction confidence scores to decide
3. **Majority voting**: Simple majority wins
4. **Weighted voting**: Weight by individual model accuracy

## Requirements

Same as the evaluation script:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
