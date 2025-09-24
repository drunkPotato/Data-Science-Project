import os
from typing import Dict, List, Union
import pandas as pd
from deepface import DeepFace
import cv2
import numpy as np


class EmotionDetector:
    
    #Initialize detector
    def __init__(self, model_name: str = 'VGG-Face'):
        # Models: 'VGG-Face', 'Facenet', 'OpenFace', 'DeepFace'
        self.model_name = model_name
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    def detect_emotion(self, image_path: str) -> Dict:
        #Returns a dictionary with emotion predictions and confidence scores
        try:
            result = DeepFace.analyze(
                img_path=image_path,
                actions=['emotion'],
                enforce_detection=False
            )
            
            # Take the first face if multiple are detected
            if isinstance(result, list):
                result = result[0]
                
            return {
                'image_path': image_path,
                'dominant_emotion': result['dominant_emotion'],
                'emotion_scores': result['emotion'],
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'image_path': image_path,
                'error': str(e),
                'status': 'error'
            }
    
    def detect_emotions_batch(self, image_folder: str) -> List[Dict]:
        """
        Detect emotions from all images in a folder.
        
        Args:
            image_folder: Path to folder containing images
            
        Returns:
            List of dictionaries containing results for each image
        """
        results = []
        
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Folder not found: {image_folder}")
        
        # Get all image files
        image_files = []
        for file in os.listdir(image_folder):
            if any(file.lower().endswith(ext) for ext in self.supported_formats):
                image_files.append(os.path.join(image_folder, file))
        
        print(f"Processing {len(image_files)} images...")
        
        for i, image_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            result = self.detect_emotion(image_path)
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        """
        Save detection results to a CSV file.
        
        Args:
            results: List of detection results
            output_path: Path to save the CSV file
        """
        # Flatten the results for CSV format
        flattened_results = []
        
        for result in results:
            if result['status'] == 'success':
                row = {
                    'image_path': result['image_path'],
                    'dominant_emotion': result['dominant_emotion'],
                    'status': result['status']
                }
                # Add individual emotion scores
                for emotion, score in result['emotion_scores'].items():
                    row[f'{emotion}_score'] = score
                    
                flattened_results.append(row)
            else:
                flattened_results.append({
                    'image_path': result['image_path'],
                    'error': result.get('error', 'Unknown error'),
                    'status': result['status']
                })
        
        df = pd.DataFrame(flattened_results)
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")