# Emotion Detection from Facial Images

A machine learning project that detects emotions from facial images using pretrained models with an interactive visualization interface.

## Team
This project is developed by a team of 4 contributors.

## Project Overview
This project implements an end-to-end pipeline for emotion detection from facial images:

1. **Data Storage**: Store and manage facial image datasets
2. **Emotion Detection**: Use pretrained models to predict emotions from images
3. **Interactive Visualization**: Display results with navigation features

## Project Structure
```
├── data/
│   ├── raw/              # Original image datasets
│   └── processed/        # Preprocessed images
├── models/               # Pretrained and fine-tuned models
├── src/
│   ├── data_processing/  # Data preprocessing scripts
│   ├── emotion_detection/# Model inference code
│   └── visualization/    # Interactive visualization components
├── notebooks/            # Jupyter notebooks for exploration
├── tests/               # Unit tests
└── docs/                # Documentation
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- Git

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd data-science-project
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note**: If you encounter issues during installation (especially with TensorFlow/PyTorch), consider:
   - Using a virtual environment
   - Installing packages individually if conflicts arise
   - Checking system compatibility for GPU support

## Dataset Information

The project includes two facial emotion datasets:

### Dataset 1: Faces_Dataset
- **Training images**: 28,709 images
- **Test images**: 7,178 images
- **Emotions**: angry, disgusted, fearful, happy, neutral, sad, surprised
- **Format**: PNG images organized by emotion folders

### Dataset 2: Faces_Dataset_2_smaller_better_quality
- **Images**: 172 high-quality images
- **Metadata**: emotions.csv with demographic information (gender, age, country)
- **Format**: Images with CSV metadata

## Usage

### Running Emotion Detection
```bash
# Analyze single image
python examples/detect_emotions.py

# Or use the detector directly
from src.emotion_detection import EmotionDetector
detector = EmotionDetector()
result = detector.detect_emotion("data/raw/Faces_Dataset/test/happy/im0.png")
print(result['dominant_emotion'])
```

### Batch Processing
```bash
# Process entire folders
detector = EmotionDetector()
results = detector.detect_emotions_batch("data/raw/Faces_Dataset/test/happy")
detector.save_results(results, "data/processed/happy_emotions.csv")
```
