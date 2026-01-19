#!/usr/bin/env python3
"""
Evaluate DeepFace, FER, and Custom-ResNet18 models on Emotion Dataset 3 GREYSCALE test set
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from pathlib import Path
from src.emotion_detection import EmotionDetector


def main():
    # Greyscale Dataset 3 configuration
    dataset_path = Path(__file__).parent.parent / "data" / "raw" / "emotion dataset 3 greyscale"
    test_csv = dataset_path / "test_labels.csv"
    test_images_base = dataset_path / "DATASET" / "test"

    # Output configuration
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "dataset3_greyscale_model_comparison.csv"

    # Emotion label mapping (1-indexed)
    label_to_emotion = {
        1: 'surprised',
        2: 'fearful',
        3: 'disgusted',
        4: 'happy',
        5: 'sad',
        6: 'angry',
        7: 'neutral'
    }

    # Verify dataset exists
    if not test_csv.exists():
        print(f"ERROR: Test CSV not found at {test_csv}")
        print("Please run the greyscale conversion script first:")
        print("  python examples/create_greyscale_dataset3.py")
        return

    # Load test labels
    print("="*70)
    print("EVALUATING MODELS ON GREYSCALE EMOTION DATASET 3 TEST SET")
    print("="*70)
    print(f"\nLoading test labels from CSV...")
    df = pd.read_csv(test_csv)
    print(f"Found {len(df)} test images")

    # Initialize detector with all three models
    models_to_compare = ['DeepFace-Emotion', 'FER', 'Custom-ResNet18']
    print(f"\nInitializing models: {', '.join(models_to_compare)}")
    detector = EmotionDetector(models=models_to_compare)

    # Process each image
    print("\nProcessing greyscale test images...")
    all_results = []

    for idx, row in df.iterrows():
        image_name = row['image']
        label = row['label']
        true_emotion = label_to_emotion[label]

        # Construct full image path (images are in numbered folders)
        image_path = test_images_base / str(label) / image_name

        if not image_path.exists():
            print(f"WARNING: Image not found: {image_path}")
            continue

        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} images...")

        try:
            # Dataset 3 images are 48x48, already aligned faces, don't extract face
            result = detector.detect_emotion(str(image_path), extract_face=False)

            # Add true emotion to result
            result['true_emotion'] = true_emotion
            result['label'] = label

            all_results.append(result)

        except Exception as e:
            print(f"ERROR processing {image_name}: {e}")
            all_results.append({
                'image_path': str(image_path),
                'true_emotion': true_emotion,
                'label': label,
                'status': 'error',
                'error': str(e),
                'models': {}
            })

    print(f"\nProcessed {len(all_results)} images")

    # Save results to CSV
    print(f"\nSaving results to {output_file}...")
    save_results_with_labels(all_results, output_file)

    # Generate summary report
    print("\n" + "="*70)
    print("MODEL EVALUATION SUMMARY - GREYSCALE DATASET 3 TEST SET")
    print("="*70)

    comparison = detector.compare_models(all_results)
    print(f"\nTotal images processed: {comparison['total_images']}")

    for model_name, stats in comparison['models'].items():
        print(f"\n{model_name}:")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        print(f"  Successful predictions: {stats['successful_predictions']}")
        print(f"  Failed predictions: {stats['failed_predictions']}")

        if stats['emotion_distribution']:
            print("  Predicted emotion distribution:")
            for emotion in ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']:
                count = stats['emotion_distribution'].get(emotion, 0)
                percentage = count / stats['successful_predictions'] * 100 if stats['successful_predictions'] > 0 else 0
                print(f"    {emotion:>10}: {count:>4} ({percentage:>5.1f}%)")

    print("\n" + "="*70)
    print(f"Results saved to: {output_file}")
    print("\nNext step: Run visualization script to analyze model performance")
    print("  python examples/visualize_dataset3_greyscale_results.py")
    print("="*70)


def save_results_with_labels(results, output_path):
    """
    Save detection results to CSV with true emotion labels
    """
    flattened_results = []

    for result in results:
        base_row = {
            'image_path': result['image_path'],
            'true_emotion': result.get('true_emotion', 'unknown'),
            'label': result.get('label', -1),
            'status': result['status']
        }

        if result['status'] == 'success' and result.get('models'):
            for model_name, model_result in result['models'].items():
                if model_result['status'] == 'success':
                    row = base_row.copy()
                    row['model'] = model_name
                    row['dominant_emotion'] = model_result['dominant_emotion']

                    # Add individual emotion scores
                    for emotion, score in model_result['emotion_scores'].items():
                        row[f'{emotion}_score'] = score

                    flattened_results.append(row)
                else:
                    row = base_row.copy()
                    row['model'] = model_name
                    row['error'] = model_result.get('error', 'Unknown error')
                    row['status'] = 'error'
                    flattened_results.append(row)
        else:
            row = base_row.copy()
            row['model'] = 'all_models'
            row['error'] = result.get('error', 'Unknown error')
            flattened_results.append(row)

    df = pd.DataFrame(flattened_results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
