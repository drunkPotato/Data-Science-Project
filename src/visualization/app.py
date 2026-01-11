import streamlit as st
import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from PIL import Image
import numpy as np
import cv2
import queue
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
from emotion_detection import EmotionDetector
import random
import time



# Page configuration
st.set_page_config(
    page_title="Emotion Detection App",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "Browse Dataset"

# Initialize global detector once and persist it
if 'detector' not in st.session_state:
    st.session_state.detector = EmotionDetector(models=['DeepFace-Emotion', 'FER', 'Custom-ResNet18'])

frame_queue = queue.Queue(maxsize=1)

def plot_model_comparison(results):
    """Create a side-by-side comparison of multiple models"""
    if 'models' not in results or len(results['models']) < 2:
        return None
    
    # Prepare data for comparison
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
    models = list(results['models'].keys())
    
    # Create subplot figure
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, model in enumerate(models):
        if results['models'][model]['status'] == 'success':
            emotion_scores = results['models'][model]['emotion_scores']
            scores = [emotion_scores.get(emotion, 0) for emotion in emotions]
            
            fig.add_trace(go.Bar(
                name=model,
                x=emotions,
                y=scores,
                marker_color=colors[i % len(colors)],
                text=[f'{score:.1f}%' for score in scores],
                textposition='auto',
                opacity=0.8
            ))
    
    fig.update_layout(
        title="Model Comparison",
        xaxis_title="Emotions",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 100]),
        barmode='group',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def load_sample_images():
    """Load sample images from dataset"""
    sample_images = []
    base_path = "data/raw/Faces_Dataset/test"

    if os.path.exists(base_path):
        emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        for emotion in emotions:
            emotion_path = os.path.join(base_path, emotion)
            if os.path.exists(emotion_path):

                images = sorted(os.listdir(emotion_path))[:20]  # first 20 per category

                for img in images:
                    if img.lower().endswith('.png'):
                        sample_images.append({
                            'path': os.path.join(emotion_path, img),
                            'emotion': emotion,
                            'filename': img
                        })

    random.shuffle(sample_images)
    return sample_images


def main():
    st.title("Multi-Model Emotion Recognition")
    
    # Navigation buttons with current mode highlighting
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        button_type = "primary" if st.session_state.current_mode == "Browse Dataset" else "secondary"
        if st.button("Browse Dataset", use_container_width=True, type=button_type, key="browse_btn"):
            st.session_state.current_mode = "Browse Dataset"
            st.rerun()
    
    with col2:
        button_type = "primary" if st.session_state.current_mode == "Batch Analysis" else "secondary"
        if st.button("Batch Analysis", use_container_width=True, type=button_type, key="batch_btn"):
            st.session_state.current_mode = "Batch Analysis"
            st.rerun()
    
    with col3:
        button_type = "primary" if st.session_state.current_mode == "Upload Image" else "secondary"
        if st.button("Upload Image", use_container_width=True, type=button_type, key="upload_btn"):
            st.session_state.current_mode = "Upload Image"
            st.rerun()
    
    with col4:
        button_type = "primary" if st.session_state.current_mode == "Video Analysis" else "secondary"
        if st.button("Video Analysis", use_container_width=True, type=button_type, key="video_btn"):
            st.session_state.current_mode = "Video Analysis"
            st.rerun()
    
    
    # Override the default red primary button color with a softer blue
    st.markdown("""
    <style>
    .stButton > button[kind="primary"] {
        background-color: #0066cc !important;
        border-color: #0066cc !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #004499 !important;
        border-color: #004499 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("---")  # Add separator line
    
    mode = st.session_state.current_mode
    
    if mode == "Upload Image":
        
        # Processing settings
        col1, col2 = st.columns([3, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                label_visibility="hidden"
            )
        
        with col2:
            st.subheader("Processing Settings")
            save_processed_image = st.checkbox(
                "Save processed image",
                value=False,
                help="Save the face-cropped image to disk for inspection"
            )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            with col2:
                st.subheader("Prediction Results")
                
                with st.spinner("Analyzing emotions with multiple models..."):
                    try:
                        # Save face-cropped image if requested
                        if save_processed_image:
                            import cv2
                            # Load and crop face from uploaded image
                            image = cv2.imread(temp_path)
                            cropped_face = st.session_state.detector._extract_face(image)
                            
                            # Create processed images directory
                            processed_dir = "temp_processed_images"
                            os.makedirs(processed_dir, exist_ok=True)
                            
                            # Save cropped face
                            processed_path = os.path.join(processed_dir, f"cropped_{uploaded_file.name}")
                            cv2.imwrite(processed_path, cropped_face)
                            
                            st.info(f"✅ Face-cropped image saved to: `{processed_path}`")
                        
                        result = st.session_state.detector.detect_emotion(temp_path)
                        
                        if result['status'] == 'success' and 'models' in result:
                            # Display model comparison chart
                            comparison_fig = plot_model_comparison(result)
                            if comparison_fig:
                                st.plotly_chart(comparison_fig, use_container_width=True)
                            
                            # Show results from each model
                            for model_name, model_result in result['models'].items():
                                if model_result['status'] == 'success':
                                    st.subheader(f"{model_name} Results")
                                    
                                    dominant_emotion = model_result['dominant_emotion']
                                    emotion_scores = model_result['emotion_scores']
                                    
                                    # Display dominant emotion and confidence
                                    confidence = emotion_scores[dominant_emotion]
                                    st.success(f"**Predicted:** {dominant_emotion.upper()} ({confidence:.1f}% confidence)")
                                    
                                    # Show top 3 emotions for this model
                                    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                                    cols = st.columns(3)
                                    for i, (emotion, score) in enumerate(sorted_emotions):
                                        with cols[i]:
                                            st.metric(f"{i+1}. {emotion.title()}", f"{score:.1f}%")
                                    
                                else:
                                    st.error(f"{model_name}: {model_result['error']}")
                            
                            # Show detailed comparison table
                            st.subheader("Detailed Model Comparison")
                            comparison_data = {}
                            for model_name, model_result in result['models'].items():
                                if model_result['status'] == 'success':
                                    comparison_data[model_name] = model_result['emotion_scores']
                            
                            if comparison_data:
                                comparison_df = pd.DataFrame(comparison_data).round(2)
                                comparison_df.index.name = 'Emotion'
                                st.dataframe(comparison_df, use_container_width=True)
                            
                        else:
                            st.error(f"Error: {result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                    
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
    
    elif mode == "Browse Dataset":
        st.header("Dataset Image Navigator")

        if 'sample_images' not in st.session_state:
            st.session_state.sample_images = load_sample_images()

        sample_images = st.session_state.sample_images
        
        if sample_images:
            # Initialize session state for current image index
            if 'current_image_idx' not in st.session_state:
                st.session_state.current_image_idx = 0
            if 'prediction_result' not in st.session_state:
                st.session_state.prediction_result = None
            
            # Ensure index is within bounds
            if st.session_state.current_image_idx >= len(sample_images):
                st.session_state.current_image_idx = 0
            
            current_img = sample_images[st.session_state.current_image_idx]
            
            st.markdown(f"**Image {st.session_state.current_image_idx + 1} of {len(sample_images)}**")
                
            # Create columns for image and prediction
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if os.path.exists(current_img['path']):
                    image = Image.open(current_img['path'])
                    st.image(image, use_container_width=True)
                
                # Navigation buttons
                nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
                
                with nav_col1:
                    if st.button("Previous", disabled=(st.session_state.current_image_idx == 0)):
                        st.session_state.current_image_idx -= 1
                        st.session_state.prediction_result = None
                        st.session_state.jump_input = ""
                        st.session_state.prev_jump_input = ""
                        st.rerun()
                
                with nav_col2:
                    st.markdown(f"**{st.session_state.current_image_idx + 1}/{len(sample_images)}**")
                
                with nav_col3:
                    if st.button("Next", disabled=(st.session_state.current_image_idx == len(sample_images) - 1)):
                        st.session_state.current_image_idx += 1
                        st.session_state.prediction_result = None
                        st.session_state.jump_input = ""
                        st.session_state.prev_jump_input = ""
                        st.rerun()
                
                with col2:
                    
                    # Predict button
                    if st.button("Predict Emotion", type="primary"):
                        with st.spinner("Analyzing emotion with multiple models..."):
                            result = st.session_state.detector.detect_emotion(current_img['path'])
                            st.session_state.prediction_result = result
                    
                    # Show prediction results if available
                    if st.session_state.prediction_result:
                        result = st.session_state.prediction_result
                        
                        if result['status'] == 'success' and 'models' in result:
                            # All models now use consistent emotion labels
                            emotion_mapping = {
                                'happy': 'happy',
                                'sad': 'sad', 
                                'angry': 'angry',
                                'surprised': 'surprised',
                                'fearful': 'fearful',
                                'disgusted': 'disgusted',
                                'neutral': 'neutral'
                            }
                            
                            # Show true label before the comparison
                            emotion_colors = {
                                'angry': '#FF4B4B',
                                'disgusted': '#8B4B8B', 
                                'fearful': '#4B4BFF',
                                'happy': '#4BFF4B',
                                'neutral': '#808080',
                                'sad': '#4B8BFF',
                                'surprised': '#FFB84B'
                            }
                            
                            emotion_color = emotion_colors.get(current_img['emotion'], '#808080')
                            st.markdown(f"""
                            <div style="
                                background-color: {emotion_color}; 
                                color: white; 
                                padding: 4px 12px; 
                                border-radius: 5px; 
                                text-align: center; 
                                font-size: 14px; 
                                font-weight: bold; 
                                margin: 5px 0;
                                display: inline-block;
                            ">
                                True Label: {current_img['emotion'].upper()}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display model comparison chart
                            comparison_fig = plot_model_comparison(result)
                            if comparison_fig:
                                st.plotly_chart(comparison_fig, use_container_width=True)
                            
                            # Show results from each model
                            for model_name, model_result in result['models'].items():
                                if model_result['status'] == 'success':
                                    dominant_emotion = model_result['dominant_emotion']
                                    emotion_scores = model_result['emotion_scores']
                                    confidence = emotion_scores[dominant_emotion]
                                    
                                    # All models now use the same emotion mapping
                                    true_emotion_mapped = emotion_mapping.get(current_img['emotion'], current_img['emotion'])
                                    
                                    # Show prediction result with color coding
                                    if dominant_emotion == true_emotion_mapped:
                                        st.success(f"**{model_name}**: {dominant_emotion.title()} ({confidence:.1f}%)")
                                    else:
                                        st.error(f"**{model_name}**: {dominant_emotion.title()} ({confidence:.1f}%)")
                                else:
                                    st.error(f"**{model_name}**: {model_result['error']}")
                            
                        else:
                            st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
                
            # Quick jump to image number
            st.markdown("---")
            if 'prev_jump_input' not in st.session_state:
                st.session_state.prev_jump_input = ""
            
            jump_to = st.text_input(f"Jump to image (1-{len(sample_images)}): Type number and press Enter", 
                                   placeholder="Enter image number...", key="jump_input")
            
            if jump_to != st.session_state.prev_jump_input and jump_to.isdigit():
                jump_num = int(jump_to)
                if 1 <= jump_num <= len(sample_images):
                    st.session_state.current_image_idx = jump_num - 1
                    st.session_state.prediction_result = None
                    st.session_state.prev_jump_input = jump_to
                    st.rerun()
            
            st.session_state.prev_jump_input = jump_to
                
        else:
            st.warning("No sample images found. Make sure the dataset is in the correct location.")
    
    elif mode == "Video Analysis":
        st.header("Video Emotion Analysis")
        st.write("Upload a video file to analyze emotions frame-by-frame using multiple models.")
        
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'],
            help="Upload a video file to analyze emotions frame-by-frame"
        )
        
        if uploaded_video is not None:
            # Display video details
            st.subheader("Video Information")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Save uploaded video temporarily
                temp_video_path = f"temp_{uploaded_video.name}"
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_video.getbuffer())
                
                # Display basic video info
                try:
                    cap = cv2.VideoCapture(temp_video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    
                    st.info(f"**Duration:** {duration:.1f} seconds | **FPS:** {fps:.1f} | **Resolution:** {width}x{height} | **Total Frames:** {frame_count}")
                    
                    # Show video preview
                    st.video(uploaded_video)
                    
                except Exception as e:
                    st.error(f"Error reading video file: {e}")
            
            with col2:
                st.subheader("Processing Settings")
                
                # Processing options
                frame_skip = st.slider(
                    "Frame Skip (process every Nth frame)",
                    min_value=1,
                    max_value=40,
                    value=30,
                    help="Higher values = faster processing, lower values = more detailed analysis"
                )
                
                save_frames = st.checkbox(
                    "Save extracted frames",
                    value=False,
                    help="Save individual frames to disk for inspection"
                )
                
                selected_models = st.multiselect(
                    "Select models for analysis",
                    options=['FER', 'DeepFace-Emotion', 'Custom-ResNet18'],
                    default=['FER', 'DeepFace-Emotion', 'Custom-ResNet18'],
                    help="Choose which emotion detection models to use"
                )
                
                if not selected_models:
                    st.warning("Please select at least one model.")
                
                # Process button
                if st.button("Analyze Video", type="primary", disabled=not selected_models):
                    try:
                        frames_output_dir = "temp_video_frames" if save_frames else None
                        
                        st.subheader("Processing Video...")
                        
                        # Create progress bars
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        with st.spinner("Extracting frames and analyzing emotions..."):
                            # Process video using global detector
                            results = st.session_state.detector.detect_emotions_video(
                                video_path=temp_video_path,
                                frame_skip=frame_skip,
                                output_dir=frames_output_dir
                            )
                            
                            progress_bar.progress(100)
                            status_text.text("Processing complete!")
                        
                        # Display results
                        st.subheader("Analysis Results")
                        
                        # Video processing summary
                        processing_info = results['processing_info']
                        video_info = results['video_info']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Frames Processed", processing_info['total_frames_processed'])
                        with col2:
                            st.metric("Processing Rate", f"{processing_info['total_frames_processed'] / video_info['duration_seconds']:.1f} FPS")
                        with col3:
                            st.metric("Frame Skip", processing_info['frame_skip'])
                        with col4:
                            st.metric("Models Used", len(processing_info['models_used']))
                        
                        
                        # Timeline visualization
                        st.subheader("Emotion Timeline")
                        
                        timeline_data = []
                        for result in results['timeline']:
                            if result.get('status') == 'success':
                                for model_name, model_result in result['models'].items():
                                    if model_result.get('status') == 'success':
                                        timeline_data.append({
                                            'timestamp': result['timestamp'],
                                            'model': model_name,
                                            'emotion': model_result['dominant_emotion'],
                                            'confidence': max(model_result['emotion_scores'].values())
                                        })
                        
                        if timeline_data:
                            timeline_df = pd.DataFrame(timeline_data)
                            
                            # Create timeline plot
                            fig_timeline = px.scatter(
                                timeline_df,
                                x='timestamp',
                                y='emotion',
                                color='model',
                                size='confidence',
                                title="Emotion Timeline Across Models",
                                labels={'timestamp': 'Time (seconds)', 'emotion': 'Detected Emotion'},
                                hover_data=['confidence']
                            )
                            
                            fig_timeline.update_layout(height=500)
                            st.plotly_chart(fig_timeline, use_container_width=True)
                        
                        
                    except Exception as e:
                        st.error(f"Error processing video: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                    
                    finally:
                        # Clean up temporary files
                        if os.path.exists(temp_video_path):
                            os.remove(temp_video_path)
                        
                        # Clean up frame directory if it was temporary
                        if not save_frames and frames_output_dir and os.path.exists(frames_output_dir):
                            import shutil
                            shutil.rmtree(frames_output_dir)
    


        
    
    elif mode == "Batch Analysis":
        st.header("Batch Analysis Results")
        
        # Check if results file exists (look for both single and multi-model results)
        multi_model_results_path = "data/processed/emotion_results_comparison.csv"
        single_model_results_path = "data/processed/emotion_results.csv"
        
        results_path = None
        if os.path.exists(multi_model_results_path):
            results_path = multi_model_results_path
            is_multi_model = True
        elif os.path.exists(single_model_results_path):
            results_path = single_model_results_path
            is_multi_model = False
        
        if results_path and os.path.exists(results_path):
            df = pd.read_csv(results_path)
            
            if not df.empty:                
                st.subheader("Analysis Summary")
                
                # Extract emotion from image path (assuming path contains emotion folder name)
                def extract_true_emotion(path):
                    # Extract emotion from path like "data/raw/Faces_Dataset/test/happy\im0.png"
                    path_parts = path.replace('\\', '/').split('/')
                    for part in path_parts:
                        if part in ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']:
                            return part
                    return 'unknown'
                
                # Add true emotion column
                df['true_emotion'] = df['image_path'].apply(extract_true_emotion)
                
                # All models now use consistent emotion labels
                emotion_mapping = {
                    'happy': 'happy',
                    'sad': 'sad', 
                    'angry': 'angry',
                    'surprised': 'surprised',
                    'fearful': 'fearful',
                    'disgusted': 'disgusted',
                    'neutral': 'neutral'
                }
                
                # Apply the same mapping to all models
                df['true_emotion_mapped'] = df['true_emotion'].map(emotion_mapping).fillna(df['true_emotion'])
                
                if is_multi_model:
                    # Multi-model analysis
                    st.subheader("Multi-Model Performance Comparison")
                    
                    # Get unique models
                    models = df['model'].unique()
                    
                    # Calculate metrics for each model
                    model_metrics = {}
                    for model in models:
                        model_df = df[df['model'] == model]
                        successful_df = model_df[model_df['status'] == 'success'].copy()
                        
                        if len(successful_df) > 0:
                            successful_df['correct'] = successful_df['dominant_emotion'] == successful_df['true_emotion_mapped']
                            accuracy = successful_df['correct'].mean() * 100
                        else:
                            accuracy = 0
                        
                        model_metrics[model] = {
                            'total': len(model_df),
                            'successful': len(successful_df),
                            'failed': len(model_df) - len(successful_df),
                            'accuracy': accuracy
                        }
                    
                    # Display model comparison metrics
                    cols = st.columns(len(models))
                    for i, model in enumerate(models):
                        with cols[i]:
                            st.metric(f"{model} Accuracy", f"{model_metrics[model]['accuracy']:.1f}%")
                            st.write(f"Success: {model_metrics[model]['successful']}")
                            st.write(f"Failed: {model_metrics[model]['failed']}")
                    
                    # Model comparison chart
                    st.subheader("Accuracy Comparison")
                    accuracy_data = [(model, metrics['accuracy']) for model, metrics in model_metrics.items()]
                    model_names, accuracies = zip(*accuracy_data)
                    
                    fig_comparison = go.Figure(data=[
                        go.Bar(
                            x=list(model_names),
                            y=list(accuracies),
                            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                            text=[f'{acc:.1f}%' for acc in accuracies],
                            textposition='auto',
                        )
                    ])
                    
                    fig_comparison.update_layout(
                        title="Model Accuracy Comparison",
                        xaxis_title="Model",
                        yaxis_title="Accuracy (%)",
                        yaxis=dict(range=[0, 100]),
                        height=400
                    )
                    
                    st.plotly_chart(fig_comparison, use_container_width=True)
                
                else:
                    # Single model analysis (original logic)
                    successful_df = df[df['status'] == 'success'].copy()
                    if len(successful_df) > 0:
                        successful_df['correct'] = successful_df['dominant_emotion'] == successful_df['true_emotion_mapped']
                        overall_accuracy = successful_df['correct'].mean() * 100
                    else:
                        overall_accuracy = 0
                    
                    # Overall statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Images", len(df))
                    with col2:
                        successful = len(successful_df)
                        st.metric("Successfully Processed", successful)
                    with col3:
                        failed = len(df) - successful
                        st.metric("Failed", failed)
                    with col4:
                        st.metric("Overall Accuracy", f"{overall_accuracy:.1f}%")
                
                # Per-emotion accuracy analysis
                if is_multi_model:
                    # Multi-model per-emotion analysis
                    models = df['model'].unique()
                    all_successful_df = df[df['status'] == 'success'].copy()
                    
                    if len(all_successful_df) > 0:
                        unique_true_emotions = all_successful_df['true_emotion'].unique()
                        
                        # Create comparison table for each emotion (for chart data only)
                        emotion_comparison = []
                        
                        for true_emotion in unique_true_emotions:
                            if true_emotion != 'unknown':
                                emotion_row = {'True Emotion': true_emotion.title()}
                                
                                for model in models:
                                    model_emotion_data = all_successful_df[
                                        (all_successful_df['true_emotion'] == true_emotion) & 
                                        (all_successful_df['model'] == model)
                                    ].copy()
                                    
                                    if len(model_emotion_data) > 0:
                                        model_emotion_data['correct'] = model_emotion_data['dominant_emotion'] == model_emotion_data['true_emotion_mapped']
                                        accuracy = model_emotion_data['correct'].mean() * 100
                                        emotion_row[f'{model} Accuracy'] = f"{accuracy:.1f}%"
                                        emotion_row[f'{model} Count'] = len(model_emotion_data)
                                    else:
                                        emotion_row[f'{model} Accuracy'] = "N/A"
                                        emotion_row[f'{model} Count'] = 0
                                
                                emotion_comparison.append(emotion_row)
                        
                        if emotion_comparison:
                            
                            # Create side-by-side accuracy chart by emotion
                            st.subheader("Per-Emotion Accuracy Comparison")
                            
                            emotions_for_chart = []
                            model_accuracies = {model: [] for model in models}
                            
                            for row in emotion_comparison:
                                emotions_for_chart.append(row['True Emotion'])
                                for model in models:
                                    acc_str = row.get(f'{model} Accuracy', 'N/A')
                                    if acc_str != 'N/A':
                                        accuracy_val = float(acc_str.rstrip('%'))
                                        model_accuracies[model].append(accuracy_val)
                                    else:
                                        model_accuracies[model].append(0)
                            
                            # Create grouped bar chart
                            fig_emotion_comparison = go.Figure()
                            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
                            
                            for i, model in enumerate(models):
                                fig_emotion_comparison.add_trace(go.Bar(
                                    name=model,
                                    x=emotions_for_chart,
                                    y=model_accuracies[model],
                                    marker_color=colors[i % len(colors)],
                                    text=[f'{acc:.1f}%' for acc in model_accuracies[model]],
                                    textposition='auto',
                                ))
                            
                            fig_emotion_comparison.update_layout(
                                title="Accuracy by Emotion Category - Model Comparison",
                                xaxis_title="True Emotion",
                                yaxis_title="Accuracy (%)",
                                yaxis=dict(range=[0, 100]),
                                barmode='group',
                                height=500,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )
                            
                            st.plotly_chart(fig_emotion_comparison, use_container_width=True)
                            
                            # Show which model performs better per emotion
                            st.subheader("Best Performing Model per Emotion")
                            best_performers = []
                            
                            for row in emotion_comparison:
                                emotion = row['True Emotion']
                                model_scores = {}
                                for model in models:
                                    acc_str = row.get(f'{model} Accuracy', 'N/A')
                                    if acc_str != 'N/A':
                                        model_scores[model] = float(acc_str.rstrip('%'))
                                
                                if model_scores:
                                    best_model = max(model_scores, key=model_scores.get)
                                    best_score = model_scores[best_model]
                                    
                                    # Check if there's a tie
                                    tied_models = [m for m, s in model_scores.items() if s == best_score]
                                    
                                    if len(tied_models) > 1:
                                        best_performers.append(f"**{emotion}**: Tie between {', '.join(tied_models)} ({best_score:.1f}%)")
                                    else:
                                        best_performers.append(f"**{emotion}**: {best_model} ({best_score:.1f}%)")
                            
                            for performer in best_performers:
                                st.write(performer)
                    
                
            else:
                st.info("No results found. Run the batch analysis first.")
        else:
            st.info("No analysis results found. Run the emotion detection script first.")
            
            if st.button("Run Sample Analysis"):
                with st.spinner("Running analysis on happy images..."):
                    # This would trigger the analysis
                    st.info("Please run `python examples/detect_emotions.py` in your terminal first.")

if __name__ == "__main__":
    main()