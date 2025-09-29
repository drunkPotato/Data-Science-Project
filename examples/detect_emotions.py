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
    
    # Example 2: Batch processing all emotions
    print("\n=== Batch Processing All Emotions ===")
    base_folder = "data/raw/Faces_Dataset/test"
    output_file = "data/processed/emotion_results.csv"
    
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
            
            total_successful = len([r for r in all_results if r['status'] == 'success'])
            total_failed = len([r for r in all_results if r['status'] == 'error'])
            print(f"\nOverall Summary:")
            print(f"  Total images processed: {len(all_results)}")
            print(f"  Successfully processed: {total_successful}")
            print(f"  Failed: {total_failed}")
        else:
            print("No images found to process")
    else:
        print("Dataset folder not found")
        print("Please ensure data/raw/Faces_Dataset/test/ exists")


if __name__ == "__main__":
    main()