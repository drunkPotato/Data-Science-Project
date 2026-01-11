import os
from typing import Dict, List, Union
import pandas as pd
import cv2
import numpy as np
from pathlib import Path


class EmotionDetector:
    def __init__(self, models: Union[str, List[str]] = 'DeepFace-Emotion'):
        if isinstance(models, str):
            self.models = [models]
        else:
            self.models = models

        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']

        # Cached models / components
        self.custom_model = None
        self.custom_transform = None
        self.custom_classes = None
        self.device = None

        self.fer_detector = None

        # Lazy init FER once
        if 'FER' in self.models:
            self._init_fer()
    
    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------

    def detect_emotion(self, image_path: str) -> Dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Process image directly without live stream array method
        results = {
            'image_path': image_path,
            'models': {},
            'status': 'success'
        }

        for model in self.models:
            try:
                if model == 'DeepFace-Emotion':
                    result = self._analyze_with_deepface_array(image)
                elif model == 'FER':
                    result = self._analyze_with_fer_array(image)
                elif model == 'Custom-ResNet18':
                    result = self._analyze_with_custom_model_array(image)
                else:
                    raise ValueError(f"Unknown model: {model}")

                results['models'][model] = {
                    'dominant_emotion': result['dominant_emotion'],
                    'emotion_scores': result['emotion_scores'],
                    'status': 'success'
                }

            except Exception as e:
                results['models'][model] = {
                    'status': 'error',
                    'error': str(e)
                }

        if all(m['status'] == 'error' for m in results['models'].values()):
            results['status'] = 'error'

        return results

    
    # ------------------------------------------------------------------
    # DeepFace
    # ------------------------------------------------------------------

    def _analyze_with_deepface_array(self, frame: np.ndarray) -> Dict:
        from deepface import DeepFace

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = DeepFace.analyze(
            img_path=rgb,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )

        if isinstance(result, list):
            result = result[0]

        mapping = {
            'fear': 'fearful',
            'surprise': 'surprised',
            'disgust': 'disgusted',
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'neutral': 'neutral'
        }

        scores = {mapping.get(k, k): v for k, v in result['emotion'].items()}
        dominant = mapping.get(result['dominant_emotion'], result['dominant_emotion'])

        return {
            'dominant_emotion': dominant,
            'emotion_scores': scores
        }
    
    # ------------------------------------------------------------------
    # FER
    # ------------------------------------------------------------------

    def _init_fer(self):
        try:
            from fer.fer import FER
        except ImportError:
            from fer import FER

        try:
            self.fer_detector = FER(mtcnn=False)
        except Exception:
            self.fer_detector = FER(mtcnn=True)

    def _analyze_with_fer_array(self, frame: np.ndarray) -> Dict:
        if self.fer_detector is None:
            raise RuntimeError("FER detector not initialized")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.fer_detector.detect_emotions(rgb)

        if not detections:
            raise ValueError("No face detected")

        emotions = detections[0]['emotions']
        dominant = max(emotions, key=emotions.get)

        mapping = {
            'fear': 'fearful',
            'surprise': 'surprised',
            'disgust': 'disgusted',
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'neutral': 'neutral'
        }

        scores = {mapping.get(k, k): v * 100 for k, v in emotions.items()}
        dominant = mapping.get(dominant, dominant)

        return {
            'dominant_emotion': dominant,
            'emotion_scores': scores
        }
    
    # ------------------------------------------------------------------
    # Custom ResNet18
    # ------------------------------------------------------------------

    def _analyze_with_custom_model_array(self, frame: np.ndarray) -> Dict:
        if self.custom_model is None:
            self._load_custom_model()

        import torch
        import torch.nn.functional as F
        from PIL import Image

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        tensor = self.custom_transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.custom_model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        scores = {
            emotion: float(prob * 100)
            for emotion, prob in zip(self.custom_classes, probs)
        }

        dominant = max(scores, key=scores.get)

        return {
            'dominant_emotion': dominant,
            'emotion_scores': scores
        }
    
    def _load_custom_model(self):
        import torch
        from torchvision import transforms, models

        model_path = os.path.join(os.path.dirname(__file__), '../../runs/exp2_clean/best.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)

        self.custom_classes = checkpoint['classes']
        num_classes = len(self.custom_classes)

        model = models.resnet18(weights=None)
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(model.fc.in_features, num_classes)
        )

        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        model.to(device)

        self.custom_model = model
        self.device = device

        self.custom_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
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
    
    def detect_emotions_video(self, video_path: str, frame_skip: int = 30, output_dir: str = None) -> Dict:
        """
        Detect emotions from video using frame-by-frame processing.
        
        Args:
            video_path: Path to the video file
            frame_skip: Process every nth frame (default: 5 for performance)
            output_dir: Directory to save extracted frames (optional)
            
        Returns:
            Dictionary containing timeline results and aggregated statistics
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Check if file is a supported video format
        file_ext = Path(video_path).suffix.lower()
        if file_ext not in self.supported_video_formats:
            raise ValueError(f"Unsupported video format: {file_ext}")
        
        print(f"Processing video: {video_path}")
        
        # Extract frames from video
        frames_data = self._extract_frames(video_path, frame_skip, output_dir)
        
        if not frames_data['frames']:
            raise ValueError("No frames extracted from video")
        
        print(f"Extracted {len(frames_data['frames'])} frames")
        
        # Process each frame for emotion detection
        timeline_results = []
        
        for i, frame_info in enumerate(frames_data['frames']):
            print(f"Processing frame {i+1}/{len(frames_data['frames'])}")
            
            try:
                # Detect emotions in the frame
                emotion_result = self.detect_emotion(frame_info['frame_path'])
                
                # Add timestamp information
                emotion_result['timestamp'] = frame_info['timestamp']
                emotion_result['frame_number'] = frame_info['frame_number']
                
                timeline_results.append(emotion_result)
                
            except Exception as e:
                print(f"Error processing frame {i+1}: {e}")
                timeline_results.append({
                    'timestamp': frame_info['timestamp'],
                    'frame_number': frame_info['frame_number'],
                    'status': 'error',
                    'error': str(e)
                })
        
        # Aggregate results
        aggregated_results = self._aggregate_video_results(timeline_results, frames_data)
        
        return {
            'video_path': video_path,
            'video_info': frames_data['video_info'],
            'timeline': timeline_results,
            'aggregated_results': aggregated_results,
            'processing_info': {
                'total_frames_processed': len(timeline_results),
                'frame_skip': frame_skip,
                'models_used': self.models
            }
        }
    
    def _extract_frames(self, video_path: str, frame_skip: int, output_dir: str = None) -> Dict:
        """Extract frames from video."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        video_info = {
            'fps': fps,
            'total_frames': total_frames,
            'duration_seconds': duration,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
        
        frames = []
        frame_count = 0
        
        # Create output directory for frames if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames according to frame_skip parameter
            if frame_count % frame_skip == 0:
                timestamp = frame_count / fps if fps > 0 else frame_count
                
                # Save frame to temporary location or specified directory
                if output_dir:
                    frame_filename = f"frame_{frame_count:06d}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                else:
                    # Use temporary file
                    import tempfile
                    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                    frame_path = temp_file.name
                    temp_file.close()
                
                # Save frame
                cv2.imwrite(frame_path, frame)
                
                frames.append({
                    'frame_number': frame_count,
                    'timestamp': timestamp,
                    'frame_path': frame_path
                })
            
            frame_count += 1
        
        cap.release()
        
        return {
            'video_info': video_info,
            'frames': frames
        }
    
    def _aggregate_video_results(self, timeline_results: List[Dict], frames_data: Dict) -> Dict:
        """Aggregate emotion detection results across the video timeline."""
        aggregated = {
            'emotion_timeline': {},
            'dominant_emotion_distribution': {},
            'average_confidence_scores': {},
            'emotion_changes': {}
        }
        
        # Process results for each model
        for model_name in self.models:
            model_timeline = []
            model_emotions = []
            confidence_scores = []
            
            for result in timeline_results:
                if (result.get('status') == 'success' and 
                    model_name in result.get('models', {}) and
                    result['models'][model_name].get('status') == 'success'):
                    
                    model_result = result['models'][model_name]
                    
                    timeline_point = {
                        'timestamp': result['timestamp'],
                        'frame_number': result['frame_number'],
                        'dominant_emotion': model_result['dominant_emotion'],
                        'confidence': max(model_result['emotion_scores'].values()),
                        'emotion_scores': model_result['emotion_scores']
                    }
                    
                    model_timeline.append(timeline_point)
                    model_emotions.append(model_result['dominant_emotion'])
                    confidence_scores.append(model_result['emotion_scores'])
            
            # Calculate emotion distribution
            emotion_distribution = {}
            for emotion in model_emotions:
                emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1
            
            # Calculate average confidence scores
            if confidence_scores:
                avg_scores = {}
                all_emotions = set()
                for scores in confidence_scores:
                    all_emotions.update(scores.keys())
                
                for emotion in all_emotions:
                    scores_for_emotion = [scores.get(emotion, 0) for scores in confidence_scores]
                    avg_scores[emotion] = np.mean(scores_for_emotion)
            else:
                avg_scores = {}
            
            # Detect emotion changes (when dominant emotion changes between consecutive frames)
            emotion_changes = []
            for i in range(1, len(model_timeline)):
                prev_emotion = model_timeline[i-1]['dominant_emotion']
                curr_emotion = model_timeline[i]['dominant_emotion']
                
                if prev_emotion != curr_emotion:
                    emotion_changes.append({
                        'timestamp': model_timeline[i]['timestamp'],
                        'frame_number': model_timeline[i]['frame_number'],
                        'from_emotion': prev_emotion,
                        'to_emotion': curr_emotion
                    })
            
            aggregated['emotion_timeline'][model_name] = model_timeline
            aggregated['dominant_emotion_distribution'][model_name] = emotion_distribution
            aggregated['average_confidence_scores'][model_name] = avg_scores
            aggregated['emotion_changes'][model_name] = emotion_changes
        
        return aggregated