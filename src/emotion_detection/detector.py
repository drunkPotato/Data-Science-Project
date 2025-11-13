import os
from typing import Dict, List, Union
import pandas as pd


class EmotionDetector:
    
    def __init__(self, models: Union[str, List[str]] = 'DeepFace-Emotion'):
        """
        Initialize the emotion detector.
        
        Args:
            models: Name(s) of the model(s) to use. Can be a single model or list of models.
                   Available models: 'DeepFace-Emotion', 'FER', 'Custom-ResNet18'
        """
        if isinstance(models, str):
            self.models = [models]
        else:
            self.models = models
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Initialize custom model components when needed
        self.custom_model = None
        self.custom_transform = None
        self.custom_classes = None
        self.device = None
    
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
                    result = self._analyze_with_deepface(image_path)
                elif model == 'FER':
                    result = self._analyze_with_fer(image_path)
                elif model == 'Custom-ResNet18':
                    result = self._analyze_with_custom_model(image_path)
                else:
                    raise ValueError(f"Unknown model: {model}")
                
                results['models'][model] = {
                    'dominant_emotion': result['dominant_emotion'],
                    'emotion_scores': result['emotion_scores'],
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
    
    def _analyze_with_deepface(self, image_path: str) -> Dict:
        """Analyze emotion using DeepFace."""
        # Import only when needed to avoid conflicts
        from deepface import DeepFace
        
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        # Handle both single face and multiple faces
        if isinstance(result, list):
            result = result[0]
        
        # Map DeepFace emotion labels to match dataset format
        emotion_mapping = {
            'fear': 'fearful',
            'surprise': 'surprised', 
            'disgust': 'disgusted',
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'neutral': 'neutral'
        }
        
        # Apply mapping to emotion scores
        mapped_scores = {}
        for emotion, score in result['emotion'].items():
            mapped_emotion = emotion_mapping.get(emotion, emotion)
            mapped_scores[mapped_emotion] = score
        
        # Apply mapping to dominant emotion
        mapped_dominant = emotion_mapping.get(result['dominant_emotion'], result['dominant_emotion'])
            
        return {
            'dominant_emotion': mapped_dominant,
            'emotion_scores': mapped_scores
        }
    
    def _analyze_with_fer(self, image_path: str) -> Dict:
        """Analyze emotion using FER library."""
        # Import only when needed to avoid conflicts
        import cv2
        from fer.fer import FER
        
        # Initialize FER detector - try different face detection backends
        emotion_detector = None
        for use_mtcnn in [False, True]:  # Try without mtcnn first, then with mtcnn
            try:
                emotion_detector = FER(mtcnn=use_mtcnn)
                break
            except Exception:
                continue
        
        if emotion_detector is None:
            raise Exception("Failed to initialize FER detector with any configuration")
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Resize image if it's too small for face detection (FER needs larger images)
        if img.shape[0] < 100 or img.shape[1] < 100:
            # Resize to at least 224x224 using bicubic interpolation for better quality
            new_size = max(224, img.shape[0] * 5, img.shape[1] * 5)
            img = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_CUBIC)
        
        # Detect emotions
        try:
            emotions = emotion_detector.detect_emotions(img)
        except Exception:
            try:
                emotions = emotion_detector.predict(img)
            except Exception:
                emotions = emotion_detector(img)
        
        if not emotions:
            # Try converting image format (sometimes helps with face detection)
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                emotions = emotion_detector.detect_emotions(img_rgb)
            except Exception:
                pass
        
        if not emotions:
            raise ValueError("No face detected in image - tried multiple detection methods")
        
        # Get the first face's emotions
        face_emotions = emotions[0]['emotions']
        
        # Find dominant emotion
        dominant_emotion = max(face_emotions, key=face_emotions.get)
        
        # Convert to percentages to match DeepFace format
        emotion_scores = {k: v * 100 for k, v in face_emotions.items()}
        
        # Map FER emotion labels to match dataset format
        emotion_mapping = {
            'fear': 'fearful',
            'surprise': 'surprised', 
            'disgust': 'disgusted',
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'neutral': 'neutral'
        }
        
        # Apply mapping to emotion scores
        mapped_scores = {}
        for emotion, score in emotion_scores.items():
            mapped_emotion = emotion_mapping.get(emotion, emotion)
            mapped_scores[mapped_emotion] = score
        
        # Apply mapping to dominant emotion
        mapped_dominant = emotion_mapping.get(dominant_emotion, dominant_emotion)
        
        return {
            'dominant_emotion': mapped_dominant,
            'emotion_scores': mapped_scores
        }
    
    def _analyze_with_custom_model(self, image_path: str) -> Dict:
        """Analyze emotion using custom trained ResNet18 model."""
        # Load model if not already loaded
        if self.custom_model is None:
            self._load_custom_model()
        
        # Import only when needed to avoid conflicts
        import torch
        import torch.nn.functional as F
        from PIL import Image
        
        # Read and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.custom_transform(img).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            logits = self.custom_model(img_tensor)
            probs = F.softmax(logits, dim=1)
            
        # Convert to percentages and create emotion scores dict
        probs = probs.cpu().numpy()[0]
        emotion_scores = {
            emotion: float(prob * 100) 
            for emotion, prob in zip(self.custom_classes, probs)
        }
        
        # Find dominant emotion
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        
        return {
            'dominant_emotion': dominant_emotion,
            'emotion_scores': emotion_scores
        }
    
    def _load_custom_model(self):
        """Load the custom trained ResNet18 model."""
        # Import only when needed to avoid conflicts
        import torch
        from torchvision import transforms, models
        
        # Path to the trained model
        model_path = os.path.join(os.path.dirname(__file__), '../../runs/exp2_clean/best.pt')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Custom model not found at: {model_path}")
        
        # Load checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get model architecture and classes
        self.custom_classes = checkpoint['classes']
        num_classes = len(self.custom_classes)
        
        # Build model architecture
        model = models.resnet18(weights=None)
        in_feats = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3), 
            torch.nn.Linear(in_feats, num_classes)
        )
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        model.to(device)
        
        self.custom_model = model
        self.device = device
        
        # Define transform (same as training eval transform)
        self.custom_transform = transforms.Compose([
            transforms.Resize(int(224 * 1.15)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
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