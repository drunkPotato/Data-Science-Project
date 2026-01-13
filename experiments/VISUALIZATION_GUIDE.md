# Experiment Results Visualization Guide

This guide shows how to create publication-ready figures and tables from your experiment results.

## Tools Available

### 1. Publication-Ready Visualizations (`visualize_results.py`)
Generates high-quality PNG/PDF figures and Excel/CSV tables for your scientific paper.

### 2. Interactive Streamlit Dashboard (`streamlit_results_explorer.py`)
Interactive web interface for exploring results, comparing experiments, and exporting data.

---

## Quick Start

### Generate Publication-Ready Figures

```bash
# Generate all visualizations
python experiments/visualize_results.py

# Generate only for top 3 experiments (recommended)
python experiments/visualize_results.py --best_only

# Custom output directory
python experiments/visualize_results.py --output_dir my_paper_figures
```

**Output Location:** `experiments/results_figures/`

**Files Created:**
- `table_overall_comparison.csv/.xlsx` - Overall metrics for all experiments
- `table_test_f1_per_emotion.csv/.xlsx` - F1 scores per emotion
- `table_test_precision_per_emotion.csv/.xlsx` - Precision scores per emotion
- `table_test_recall_per_emotion.csv/.xlsx` - Recall scores per emotion
- `fig_overall_comparison.png/.pdf` - Bar chart comparing all experiments
- `fig_test_f1_heatmap.png/.pdf` - Heatmap of F1 scores per emotion
- `fig_test_precision_heatmap.png/.pdf` - Heatmap of precision scores
- `fig_test_recall_heatmap.png/.pdf` - Heatmap of recall scores
- `fig_lr_impact.png/.pdf` - Learning rate impact analysis
- `fig_batch_impact.png/.pdf` - Batch size impact analysis
- `fig_epochs_impact.png/.pdf` - Training duration impact analysis
- `fig_augmentation_impact.png/.pdf` - Augmentation strategy impact
- `fig_confusion_*.png/.pdf` - Confusion matrices for top experiments
- `fig_learning_curves_*.png/.pdf` - Training/validation curves

---

### Launch Interactive Dashboard

```bash
# Start Streamlit dashboard
streamlit run experiments/streamlit_results_explorer.py
```

**Dashboard Features:**
- Overview page with top performers
- Detailed experiment comparison
- Individual experiment deep-dive
- Per-emotion analysis
- Data export functionality
- Interactive plots with zoom/pan
- Configuration comparison tables

---

## Using Results in Your Paper

### For Microsoft Word

1. **Tables:**
   - Open `.xlsx` files in Excel
   - Copy and paste into Word
   - Or use "Insert > Table > From Excel"

2. **Figures:**
   - Use `.png` files for Word documents
   - Drag and drop into your document
   - Recommended: 300 DPI for publication quality

### For LaTeX

1. **Figures:**
   - Use `.pdf` files for best quality
   - Include with: `\includegraphics{fig_overall_comparison.pdf}`

2. **Tables:**
   - CSV files can be converted to LaTeX tables
   - Use online converters or pandas `to_latex()`

### For PowerPoint Presentations

1. **Use PNG figures** - better compatibility
2. **Resize as needed** - 300 DPI maintains quality
3. **Interactive exploration** - Use Streamlit dashboard during presentations

---

## Recommended Figures for Paper

### Essential Figures (Must Include):

1. **Overall Performance Comparison**
   - Use: `fig_overall_comparison.png`
   - Shows: Which configuration performs best overall

2. **Per-Emotion Heatmap**
   - Use: `fig_test_f1_heatmap.png`
   - Shows: Performance patterns across emotions and configurations

3. **Best Model Confusion Matrix**
   - Use: `fig_confusion_baseline_resnet18_test.png`
   - Shows: Where your model succeeds and fails

4. **Learning Curves (Best Model)**
   - Use: `fig_learning_curves_baseline_resnet18.png`
   - Shows: Training convergence and potential overfitting

### Optional Figures (If Space Allows):

5. **Hyperparameter Impact Analysis**
   - Use: `fig_lr_impact.png`, `fig_batch_impact.png`, etc.
   - Shows: Sensitivity to different hyperparameters

6. **Multiple Confusion Matrices**
   - Compare top 2-3 configurations
   - Shows: How different approaches make different errors

---

## Recommended Tables for Paper

### Essential Tables:

1. **Overall Results Table**
   - Use: `table_overall_comparison.xlsx`
   - Include: Top 5-7 configurations
   - Columns: Configuration name, key hyperparameters, accuracy, F1

2. **Per-Emotion Performance Table**
   - Use: `table_test_f1_per_emotion.xlsx`
   - Show: F1 scores for all emotions
   - Highlight: Best and worst performing emotions

### Optional Tables:

3. **Detailed Per-Emotion Metrics**
   - Use: `table_test_precision_per_emotion.xlsx` and `table_test_recall_per_emotion.xlsx`
   - For: Appendix or supplementary materials

---

## Tips for Publication Quality

### Figures:
- Use PDF format for LaTeX documents (vector graphics, scales perfectly)
- Use PNG format for Word/PowerPoint (300 DPI already set)
- Keep color schemes colorblind-friendly (tool uses RdYlGn and Viridis)
- Font sizes are optimized for 2-column IEEE format

### Tables:
- Round to 2-4 decimal places (already done in Excel files)
- Bold the best-performing values in Excel
- Add table captions explaining what's shown
- Reference table numbers in your text

### Writing:
- "As shown in Table 1, the baseline configuration achieved..."
- "Figure 2 illustrates the confusion matrix for..."
- "The heatmap in Figure 3 reveals that..."

---

## Troubleshooting

### Missing seaborn or matplotlib
```bash
pip install matplotlib seaborn pyyaml
```

### Missing plotly (for Streamlit)
```bash
pip install plotly streamlit
```

### Figures look pixelated in Word
- Make sure you're using the PNG files (not PDF)
- Don't resize too much in Word - insert at native size
- If needed, regenerate with higher DPI in the script

### Colors don't show in paper
- Many journals require grayscale for print
- Patterns/hatching can be added in the script if needed
- Color is fine for online/supplementary materials

---

## Customization

Edit `visualize_results.py` to customize:

- **Figure dimensions**: Change `figsize=(12, 6)` values
- **Color schemes**: Change `cmap='RdYlGn'` to other matplotlib colormaps
- **Font sizes**: Modify `plt.rcParams['font.size']` at the top
- **DPI**: Change `plt.rcParams['figure.dpi']` for higher/lower resolution
- **Number of experiments**: Use `--best_only` flag to limit output

---

## Need Help?

Common tasks:

**Export just one specific table:**
```python
python -c "
import pandas as pd
df = pd.read_csv('experiments/results_figures/table_overall_comparison.csv')
df.head(3).to_excel('top3_results.xlsx', index=False)
"
```

**Generate only confusion matrices:**
Edit `visualize_results.py` and comment out sections you don't need.

**Change experiment ordering:**
Results are sorted by test F1 score (best first). Edit the `sort` key in the script to change this.
