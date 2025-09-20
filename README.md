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

## Usage

### Data Preparation
Place your facial image datasets in the `data/raw/` directory.

### Running Emotion Detection
```bash
python src/emotion_detection/predict.py --input data/raw/your_images/
```

### Interactive Visualization
```bash
python src/visualization/app.py
```

## Development Workflow

### Branching Strategy
- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/<feature-name>`: Individual feature development
- `data/<data-task>`: Data processing tasks
- `viz/<visualization-feature>`: Visualization features

### Creating a Feature Branch
```bash
git checkout -b feature/your-feature-name
# Make your changes
git add .
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
```

## Contributing
1. Create a feature branch from `develop`
2. Make your changes
3. Add tests for new functionality
4. Submit a pull request to `develop`

## License
[Add your license here]