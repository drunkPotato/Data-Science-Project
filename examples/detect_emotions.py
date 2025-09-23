#!/usr/bin/env python3
"""
Example script for emotion detection from facial images.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.emotion_detection import EmotionDetector


def main():
    # Initialize the emotion detector
    detector = EmotionDetector(model_name='VGG-Face')
    
    # Example 1: Detect emotion from a single image
    print("=== Single Image Detection ===")
    image_path = "data/raw/Faces_Dataset/test/happy/im0.png"
    
    if os.path.exists(image_path):
        result = detector.detect_emotion(image_path)
        if result['status'] == 'success':
            print(f"Dominant emotion: {result['dominant_emotion']}")
            print("All emotion scores:")
            for emotion, score in result['emotion_scores'].items():
                print(f"  {emotion}: {score:.2f}%")
        else:
            print(f"Error: {result['error']}")
    else:
        print(f"Sample image not found at {image_path}")
        print("Please add a sample image to data/raw/ folder")
    
    # Example 2: Batch processing
    print("\n=== Batch Processing ===")
    input_folder = "data/raw/Faces_Dataset/test/happy"
    output_file = "data/processed/emotion_results.csv"
    
    if os.path.exists(input_folder) and os.listdir(input_folder):
        try:
            results = detector.detect_emotions_batch(input_folder)
            detector.save_results(results, output_file)
            
            # Print summary
            successful = len([r for r in results if r['status'] == 'success'])
            failed = len([r for r in results if r['status'] == 'error'])
            print(f"\nSummary:")
            print(f"  Successfully processed: {successful} images")
            print(f"  Failed: {failed} images")
            
        except Exception as e:
            print(f"Error during batch processing: {e}")
    else:
        print("No images found in data/raw folder")
        print("Please add some facial images to test the emotion detection")


if __name__ == "__main__":
    main()