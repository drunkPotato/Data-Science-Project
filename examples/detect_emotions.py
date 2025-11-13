#!/usr/bin/env python3
"""
Example script for emotion detection from facial images.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.emotion_detection import EmotionDetector


def main():
    # Initialize the emotion detector with different emotion detection libraries
    models_to_compare = ['DeepFace-Emotion', 'FER', 'Custom-ResNet18']
    detector = EmotionDetector(models=models_to_compare)
    
    print(f"=== Multi-Model Emotion Detection Comparison ===")
    print(f"Models: {', '.join(models_to_compare)}")
    
    # Example 1: Detect emotion from multiple images
    print("\n=== Multiple Image Detection (First 10) ===")
    test_folder = "data/raw/Faces_Dataset/test/happy"
    
    if os.path.exists(test_folder):
        # Get first 10 images from happy folder
        image_files = []
        for file in sorted(os.listdir(test_folder)):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(test_folder, file))
            if len(image_files) >= 10:  # Stop at 10 images
                break
        
        if image_files:
            print(f"Processing {len(image_files)} images from happy folder...")
            
            for i, image_path in enumerate(image_files, 1):
                print(f"\n{'='*50}")
                print(f"IMAGE {i}: {os.path.basename(image_path)}")
                print('='*50)
                
                result = detector.detect_emotion(image_path)
                if result['status'] == 'success':
                    for model_name, model_result in result['models'].items():
                        if model_result['status'] == 'success':
                            print(f"\n{model_name} Model:")
                            print(f"  Dominant emotion: {model_result['dominant_emotion']}")
                            print("  All emotion scores:")
                            for emotion, score in model_result['emotion_scores'].items():
                                print(f"    {emotion}: {score:.2f}%")
                        else:
                            print(f"\n{model_name} Model: Error - {model_result['error']}")
                else:
                    print(f"Error processing image: {result.get('error', 'Unknown error')}")
        else:
            print("No image files found in happy folder")
    else:
        print(f"Test folder not found at {test_folder}")
        print("Please check the dataset path")
    
    # Example 2: Batch processing with model comparison
    print("\n=== Batch Processing with Model Comparison ===")
    input_folder = "data/raw/Faces_Dataset/test/happy"
    output_file = "data/processed/emotion_results_comparison.csv"
    base_folder = "data/raw/Faces_Dataset/test"
    
    emotion_folders = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    all_results = []
    
    if os.path.exists(base_folder):
        for emotion in emotion_folders:
            emotion_folder = os.path.join(base_folder, emotion)
            if os.path.exists(emotion_folder) and os.listdir(emotion_folder):
                print(f"\nProcessing {emotion} images...")
                try:
                    results = detector.detect_emotions_batch(emotion_folder)
                    all_results.extend(results)
                    
                    successful = len([r for r in results if r['status'] == 'success'])
                    failed = len([r for r in results if r['status'] == 'error'])
                    print(f"  {emotion}: {successful} successful, {failed} failed")
                    
                except Exception as e:
                    print(f"Error processing {emotion}: {e}")
            else:
                print(f"Skipping {emotion} - folder not found or empty")
        
        if all_results:
            detector.save_results(all_results, output_file)
            
            # Generate comparison report
            comparison = detector.compare_models(results)
            
            print(f"\n=== Model Comparison Report ===")
            print(f"Total images processed: {comparison['total_images']}")
            
            for model_name, stats in comparison['models'].items():
                print(f"\n{model_name} Model:")
                print(f"  Success rate: {stats['success_rate']:.2%}")
                print(f"  Successful predictions: {stats['successful_predictions']}")
                print(f"  Failed predictions: {stats['failed_predictions']}")
                print("  Emotion distribution:")
                for emotion, count in stats['emotion_distribution'].items():
                    percentage = count / stats['successful_predictions'] * 100 if stats['successful_predictions'] > 0 else 0
                    print(f"    {emotion}: {count} ({percentage:.1f}%)")
    else:
        print("No images found in data/raw folder")
        print("Please add some facial images to test the emotion detection")
    
    # Example 3: Single model for backward compatibility
    print(f"\n=== Single Model Example (Backward Compatibility) ===")
    single_detector = EmotionDetector(models='DeepFace-Emotion')
    
    if os.path.exists("data/raw/Faces_Dataset/test/happy/im0.png"):
        single_result = single_detector.detect_emotion("data/raw/Faces_Dataset/test/happy/im0.png")
        if single_result['status'] == 'success':
            deepface_result = single_result['models']['DeepFace-Emotion']
            if deepface_result['status'] == 'success':
                print(f"DeepFace only - Dominant emotion: {deepface_result['dominant_emotion']}")
            else:
                print(f"DeepFace error: {deepface_result['error']}")
        else:
            print("Error processing with single model")


if __name__ == "__main__":
    main()