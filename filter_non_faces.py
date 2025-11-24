#!/usr/bin/env python3
"""
Face Detection Filter Script

This script identifies images in the dataset that don't contain recognizable faces
and copies them to a separate folder for manual review.

Usage: python filter_non_faces.py
"""

import os
import shutil
from pathlib import Path
from deepface import DeepFace
import cv2
import numpy as np
from tqdm import tqdm

def setup_output_directory():
    """Create directory structure for flagged images."""
    flagged_dir = Path("data/flagged_non_faces")
    flagged_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for each emotion category
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    for emotion in emotions:
        (flagged_dir / emotion).mkdir(exist_ok=True)
    
    return flagged_dir

def is_face_detected_deepface(image_path):
    """
    Check if DeepFace can detect a face with enforce_detection=True.
    Returns (has_face, confidence_info)
    """
    try:
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion'],
            enforce_detection=True,
            detector_backend='opencv',
            silent=True
        )
        return True, "Face detected"
    except Exception as e:
        return False, str(e)

def is_face_detected_opencv(image_path):
    """
    Alternative face detection using OpenCV directly with confidence scoring.
    Returns (has_face, confidence)
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return False, "Could not load image"
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces with different scale factors for robustness
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Calculate a simple confidence based on face size relative to image
            img_area = img.shape[0] * img.shape[1]
            largest_face_area = max([w * h for (x, y, w, h) in faces])
            confidence = largest_face_area / img_area
            
            # Consider it a good face if it's at least 2% of the image
            if confidence > 0.02:
                return True, f"Face detected (confidence: {confidence:.3f})"
            else:
                return False, f"Face too small (confidence: {confidence:.3f})"
        else:
            return False, "No face detected"
            
    except Exception as e:
        return False, f"Error: {str(e)}"

def scan_dataset(dataset_path, method='both'):
    """
    Scan the entire dataset and identify non-face images.
    
    Args:
        dataset_path: Path to the dataset
        method: 'deepface', 'opencv', or 'both'
    """
    dataset_path = Path(dataset_path)
    flagged_dir = setup_output_directory()
    
    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    all_images = []
    
    for split in ['train', 'test']:
        split_path = dataset_path / split
        if split_path.exists():
            for emotion_dir in split_path.iterdir():
                if emotion_dir.is_dir():
                    for img_file in emotion_dir.iterdir():
                        if img_file.suffix.lower() in image_extensions:
                            all_images.append((img_file, emotion_dir.name, split))
    
    print(f"Found {len(all_images)} images to process")
    
    flagged_images = []
    results_log = []
    
    # Process each image
    for img_path, emotion, split in tqdm(all_images, desc="Scanning images"):
        results = {
            'path': str(img_path),
            'emotion': emotion,
            'split': split,
            'deepface_result': None,
            'opencv_result': None,
            'flagged': False
        }
        
        # Test with DeepFace
        if method in ['deepface', 'both']:
            df_has_face, df_info = is_face_detected_deepface(str(img_path))
            results['deepface_result'] = {'has_face': df_has_face, 'info': df_info}
        
        # Test with OpenCV
        if method in ['opencv', 'both']:
            cv_has_face, cv_info = is_face_detected_opencv(str(img_path))
            results['opencv_result'] = {'has_face': cv_has_face, 'info': cv_info}
        
        # Determine if image should be flagged
        if method == 'deepface':
            flag_image = not results['deepface_result']['has_face']
        elif method == 'opencv':
            flag_image = not results['opencv_result']['has_face']
        else:  # both
            # Flag if BOTH methods fail to detect a face
            df_fail = not results['deepface_result']['has_face']
            cv_fail = not results['opencv_result']['has_face']
            flag_image = df_fail and cv_fail
        
        results['flagged'] = flag_image
        results_log.append(results)
        
        # Copy flagged images
        if flag_image:
            flagged_images.append(img_path)
            dest_dir = flagged_dir / emotion
            dest_path = dest_dir / f"{split}_{img_path.name}"
            shutil.copy2(img_path, dest_path)
    
    # Save results summary
    save_results_summary(results_log, flagged_dir)
    
    print(f"\nScan complete!")
    print(f"Total images processed: {len(all_images)}")
    print(f"Flagged as non-faces: {len(flagged_images)}")
    print(f"Flagged images copied to: {flagged_dir}")
    
    return results_log, flagged_images

def save_results_summary(results_log, output_dir):
    """Save detailed results to a text file."""
    summary_file = output_dir / "detection_results.txt"
    
    with open(summary_file, 'w') as f:
        f.write("Face Detection Results Summary\n")
        f.write("=" * 50 + "\n\n")
        
        flagged_count = sum(1 for r in results_log if r['flagged'])
        total_count = len(results_log)
        
        f.write(f"Total images: {total_count}\n")
        f.write(f"Flagged images: {flagged_count}\n")
        f.write(f"Success rate: {((total_count - flagged_count) / total_count * 100):.2f}%\n\n")
        
        # Breakdown by emotion
        f.write("Breakdown by emotion:\n")
        f.write("-" * 30 + "\n")
        emotions = {}
        for result in results_log:
            emotion = result['emotion']
            if emotion not in emotions:
                emotions[emotion] = {'total': 0, 'flagged': 0}
            emotions[emotion]['total'] += 1
            if result['flagged']:
                emotions[emotion]['flagged'] += 1
        
        for emotion, stats in emotions.items():
            success_rate = ((stats['total'] - stats['flagged']) / stats['total'] * 100) if stats['total'] > 0 else 0
            f.write(f"{emotion}: {stats['flagged']}/{stats['total']} flagged ({success_rate:.1f}% success)\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("Detailed Results:\n")
        f.write("=" * 50 + "\n\n")
        
        for result in results_log:
            if result['flagged']:
                f.write(f"FLAGGED: {result['path']}\n")
                f.write(f"  Emotion: {result['emotion']}, Split: {result['split']}\n")
                if result['deepface_result']:
                    f.write(f"  DeepFace: {result['deepface_result']['info']}\n")
                if result['opencv_result']:
                    f.write(f"  OpenCV: {result['opencv_result']['info']}\n")
                f.write("\n")

def main():
    """Main execution function."""
    print("Face Detection Filter - Identifying Non-Face Images")
    print("=" * 60)
    
    dataset_path = "data/raw/Faces_Dataset"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' not found!")
        return
    
    print(f"Scanning dataset: {dataset_path}")
    print("Method: Using both DeepFace (enforce_detection=True) and OpenCV")
    print("Flagging criteria: Images where BOTH methods fail to detect faces")
    print()
    
    try:
        results_log, flagged_images = scan_dataset(dataset_path, method='both')
        print("\nTask completed successfully!")
        
    except KeyboardInterrupt:
        print("\nScan interrupted by user")
    except Exception as e:
        print(f"\nError during scan: {e}")

if __name__ == "__main__":
    main()