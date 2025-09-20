# Dataset Documentation

## Overview
This project uses two facial emotion recognition datasets for training and testing emotion detection models.

## Dataset 1: Faces_Dataset

### Structure
```
data/raw/Faces_Dataset/
├── train/                  # Training set (28,709 images)
│   ├── angry/             # Angry facial expressions
│   ├── disgusted/         # Disgusted facial expressions
│   ├── fearful/           # Fearful facial expressions
│   ├── happy/             # Happy facial expressions
│   ├── neutral/           # Neutral facial expressions
│   ├── sad/               # Sad facial expressions
│   └── surprised/         # Surprised facial expressions
└── test/                  # Test set (7,178 images)
    ├── angry/
    ├── disgusted/
    ├── fearful/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprised/
```

### Characteristics
- **Total Images**: 35,887 (28,709 train + 7,178 test)
- **Format**: PNG files
- **Emotions**: 7 categories (angry, disgusted, fearful, happy, neutral, sad, surprised)
- **Organization**: Folder-based labeling system
- **Naming**: Sequential (im0.png, im1.png, ...)

### Use Cases
- Training emotion classification models
- Evaluating model performance
- Benchmarking different architectures

## Dataset 2: Faces_Dataset_2_smaller_better_quality

### Structure
```
data/raw/Faces_Dataset_2_smaller_better_quality/
├── images/                # 172 high-quality facial images
└── emotions.csv          # Metadata file
```

### Metadata (emotions.csv)
- **set_id**: Unique identifier for each image
- **gender**: MALE/FEMALE
- **age**: Age of the person
- **country**: Country code (RU, PH, IN, etc.)

### Characteristics
- **Total Images**: 172
- **Quality**: Higher resolution/quality compared to Dataset 1
- **Metadata**: Rich demographic information
- **Format**: Various image formats with CSV metadata

### Use Cases
- Quality analysis and comparison
- Demographic bias testing
- Small-scale model validation

## Data Usage Guidelines

### For Training
- Use Faces_Dataset for main training/testing
- Use train/test split as provided
- Consider data augmentation for better generalization

### For Analysis
- Use Faces_Dataset_2 for quality benchmarks
- Analyze demographic representation
- Test model performance across different groups

### For Development
- Start with small subsets for rapid prototyping
- Use test sets sparingly to avoid overfitting
- Implement proper validation splits

## Data Quality Considerations

1. **Image Quality**: Dataset 2 has higher quality images
2. **Balance**: Check class distribution across emotions
3. **Diversity**: Dataset 2 provides demographic metadata for bias analysis
4. **Size**: Dataset 1 provides large-scale training data

## Preprocessing Recommendations

1. **Resize**: Standardize image dimensions (e.g., 224x224)
2. **Normalize**: Apply pixel value normalization
3. **Augmentation**: Use rotation, flip, brightness adjustments
4. **Face Detection**: Ensure faces are properly centered