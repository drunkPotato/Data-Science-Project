import os
from typing import Dict, List, Union
import pandas as pd
from deepface import DeepFace
import cv2
import numpy as np
from fer.fer import FER


class EmotionDetector:
    
    def __init__(self, models: Union[str, List[str]] = 'VGG-Face'):
        """
        Initialize the emotion detector.
        
        Args:
            models: Name(s) of the model(s) to use. Can be a single model or list of models.
                   Available models: 'DeepFace-Emotion', 'FER'
        """
        if isinstance(models, str):
            self.models = [models]
        else:
            self.models = models
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    def detect_emotion(self, image_path: str) -> Dict:
        """
        Detect emotion from a single image using all configured models.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing emotion predictions and confidence scores for each model
        """
        results = {
            'image_path': image_path,
            'models': {},
            'status': 'success'
        }
        
        for model in self.models:
            try:
                if model == 'DeepFace-Emotion':
                    # Use DeepFace emotion model
                    result = DeepFace.analyze(
                        img_path=image_path,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                elif model == 'FER':
                    # Use FER library for emotion detection
                    result = self._analyze_with_fer(image_path)
                else:
                    # Fallback to DeepFace
                    result = DeepFace.analyze(
                        img_path=image_path,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                
                # Handle both single face and multiple faces
                if isinstance(result, list):
                    result = result[0]
                    
                results['models'][model] = {
                    'dominant_emotion': result['dominant_emotion'],
                    'emotion_scores': result['emotion'],
                    'status': 'success'
                }
                
            except Exception as e:
                results['models'][model] = {
                    'error': str(e),
                    'status': 'error'
                }
                
        # Set overall status to error if all models failed
        if all(model_result['status'] == 'error' for model_result in results['models'].values()):
            results['status'] = 'error'
            
        return results
    
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
                base_row = {
                    'image_path': result['image_path'],
                    'status': result['status']
                }
                
                # Add results for each model
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
                flattened_results.append({
                    'image_path': result['image_path'],
                    'error': result.get('error', 'Unknown error'),
                    'status': result['status'],
                    'model': 'all_failed'
                })
        
        df = pd.DataFrame(flattened_results)
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    
    def compare_models(self, results: List[Dict]) -> Dict:
        """
        Compare performance of different models on the same dataset.
        
        Args:
            results: List of detection results from multiple models
            
        Returns:
            Dictionary containing comparison metrics
        """
        comparison = {
            'total_images': len(results),
            'models': {}
        }
        
        for model_name in self.models:
            successful = 0
            failed = 0
            dominant_emotions = []
            
            for result in results:
                if result['status'] == 'success' and model_name in result['models']:
                    model_result = result['models'][model_name]
                    if model_result['status'] == 'success':
                        successful += 1
                        dominant_emotions.append(model_result['dominant_emotion'])
                    else:
                        failed += 1
                else:
                    failed += 1
            
            # Calculate emotion distribution
            emotion_counts = {}
            for emotion in dominant_emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            comparison['models'][model_name] = {
                'successful_predictions': successful,
                'failed_predictions': failed,
                'success_rate': successful / (successful + failed) if (successful + failed) > 0 else 0,
                'emotion_distribution': emotion_counts
            }
        
        return comparison
    
    def _analyze_with_fer(self, image_path: str) -> Dict:
        """
        Analyze emotion using FER library as alternative to DeepFace.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Result dictionary in DeepFace-compatible format
        """
        try:            
            # Initialize FER detector (try different initialization approaches)
            try:
                emotion_detector = FER(mtcnn=True)
            except Exception:
                # Fallback initialization without mtcnn
                emotion_detector = FER()
            
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Detect emotions - try the newer API first
            try:
                emotions = emotion_detector.detect_emotions(img)
            except Exception as e:
                # Try alternative method names
                try:
                    emotions = emotion_detector.predict(img)
                except Exception:
                    emotions = emotion_detector(img)
            
            if not emotions:
                raise ValueError("No face detected in image")
            
            # Get the first face's emotions (FER returns list with emotion dict)
            face_emotions = emotions[0]['emotions']
            
            # Find dominant emotion
            dominant_emotion = max(face_emotions, key=face_emotions.get)
            
            # Convert to percentages to match DeepFace format
            emotion_scores = {k: v * 100 for k, v in face_emotions.items()}
            
            return {
                'dominant_emotion': dominant_emotion,
                'emotion': emotion_scores
            }
            
        except ImportError:
            raise ImportError("FER library not installed. Install with: pip install fer")
        except Exception as e:
            raise Exception(f"FER analysis failed: {str(e)}")