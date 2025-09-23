import streamlit as st
import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from emotion_detection import EmotionDetector

# Page configuration
st.set_page_config(
    page_title="Emotion Detection App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = EmotionDetector()

def plot_emotion_scores(emotion_scores):
    """Create a bar chart of emotion scores"""
    emotions = list(emotion_scores.keys())
    scores = list(emotion_scores.values())
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=emotions,
            y=scores,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF'],
            text=[f'{score:.1f}%' for score in scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Emotion Detection Results",
        xaxis_title="Emotions",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 100]),
        height=400
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
                images = sorted(os.listdir(emotion_path))[:20]  # Get first 20 images from each category
                for img in images:
                    if img.endswith('.png'):
                        sample_images.append({
                            'path': os.path.join(emotion_path, img),
                            'emotion': emotion,
                            'filename': img
                        })
    
    return sample_images

def main():
    st.title("Emotion Detection from Facial Images")
    st.markdown("Upload an image or select from sample dataset to detect emotions!")
    
    # Main navigation
    mode = st.selectbox(
        "Choose Mode",
        ["Browse Dataset", "Batch Analysis", "Upload Image"]
    )
    
    if mode == "Upload Image":
        st.header("Upload Your Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a facial image for emotion detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            with col2:
                st.subheader("Prediction Results")
                
                with st.spinner("Analyzing emotions..."):
                    try:
                        result = st.session_state.detector.detect_emotion(temp_path)
                        
                        if result['status'] == 'success':
                            dominant_emotion = result['dominant_emotion']
                            emotion_scores = result['emotion_scores']
                            
                            # Display dominant emotion
                            st.success(f"**Dominant Emotion:** {dominant_emotion.upper()}")
                            
                            # Plot emotion scores
                            fig = plot_emotion_scores(emotion_scores)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show detailed scores
                            st.subheader("Detailed Scores")
                            scores_df = pd.DataFrame([emotion_scores]).T
                            scores_df.columns = ['Confidence (%)']
                            scores_df = scores_df.round(2)
                            st.dataframe(scores_df, use_container_width=True)
                            
                        else:
                            st.error(f"Error: {result['error']}")
                            
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                    
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
    
    elif mode == "Browse Dataset":
        st.header("Dataset Image Navigator")
        
        sample_images = load_sample_images()
        
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
            
            st.subheader(f"Dataset Images")
            st.markdown(f"**Image {st.session_state.current_image_idx + 1} of {len(sample_images)}**")
                
            # Create columns for image and prediction
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"**Current Image: {current_img['filename']}**")
                if os.path.exists(current_img['path']):
                    image = Image.open(current_img['path'])
                    st.image(image, use_column_width=True)
                    st.markdown(f"**True Label:** {current_img['emotion'].title()}")
                
                # Navigation buttons
                nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
                
                with nav_col1:
                    if st.button("Previous", disabled=(st.session_state.current_image_idx == 0)):
                        st.session_state.current_image_idx -= 1
                        st.session_state.prediction_result = None
                        st.rerun()
                
                with nav_col2:
                    st.markdown(f"**{st.session_state.current_image_idx + 1}/{len(sample_images)}**")
                
                with nav_col3:
                    if st.button("Next", disabled=(st.session_state.current_image_idx == len(sample_images) - 1)):
                        st.session_state.current_image_idx += 1
                        st.session_state.prediction_result = None
                        st.rerun()
                
                with col2:
                    st.markdown("**Model Prediction:**")
                    
                    # Predict button
                    if st.button("Predict Emotion", type="primary"):
                        with st.spinner("Analyzing emotion..."):
                            result = st.session_state.detector.detect_emotion(current_img['path'])
                            st.session_state.prediction_result = result
                    
                    # Show prediction results if available
                    if st.session_state.prediction_result:
                        result = st.session_state.prediction_result
                        
                        if result['status'] == 'success':
                            dominant_emotion = result['dominant_emotion']
                            emotion_scores = result['emotion_scores']
                            
                            # Map dataset emotions to DeepFace emotions for comparison
                            emotion_mapping = {
                                'happy': 'happy',
                                'sad': 'sad', 
                                'angry': 'angry',
                                'surprised': 'surprise',
                                'fearful': 'fear',
                                'disgusted': 'disgust',
                                'neutral': 'neutral'
                            }
                            
                            # Get the mapped true emotion
                            true_emotion_mapped = emotion_mapping.get(current_img['emotion'], current_img['emotion'])
                            
                            # Show prediction result
                            if dominant_emotion == true_emotion_mapped:
                                st.success(f"Correct - **Predicted:** {dominant_emotion.title()}")
                            else:
                                st.error(f"Incorrect - **Predicted:** {dominant_emotion.title()}")
                            
                            # Show comparison
                            st.markdown(f"**Expected:** {current_img['emotion'].title()}")
                            st.markdown(f"**Got:** {dominant_emotion.title()}")
                            
                            # Show confidence score for dominant emotion
                            confidence = emotion_scores[dominant_emotion]
                            st.metric("Confidence", f"{confidence:.1f}%")
                            
                            # Show top 3 emotions
                            st.markdown("**Top 3 Predictions:**")
                            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                            for i, (emotion, score) in enumerate(sorted_emotions):
                                if i == 0:
                                    st.markdown(f"1st: {emotion}: {score:.1f}%")
                                elif i == 1:
                                    st.markdown(f"2nd: {emotion}: {score:.1f}%")
                                else:
                                    st.markdown(f"3rd: {emotion}: {score:.1f}%")
                            
                            # Mini bar chart
                            mini_fig = go.Figure(data=[
                                go.Bar(
                                    x=list(emotion_scores.keys()),
                                    y=list(emotion_scores.values()),
                                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF'],
                                    showlegend=False
                                )
                            ])
                            mini_fig.update_layout(
                                height=250,
                                margin=dict(l=0, r=0, t=20, b=0),
                                xaxis_title="Emotions",
                                yaxis_title="Score (%)",
                                title="Emotion Confidence Scores"
                            )
                            st.plotly_chart(mini_fig, use_container_width=True)
                            
                        else:
                            st.error(f"Analysis failed: {result['error']}")
                    else:
                        st.info("Click 'Predict Emotion' to analyze this image")
                
            # Quick jump to image number
            st.markdown("---")
            jump_col1, jump_col2 = st.columns([3, 1])
            with jump_col1:
                jump_to = st.number_input("Jump to image:", min_value=1, max_value=len(sample_images), value=st.session_state.current_image_idx + 1)
            with jump_col2:
                if st.button("Go to Image"):
                    st.session_state.current_image_idx = jump_to - 1
                    st.session_state.prediction_result = None
                    st.rerun()
                
        else:
            st.warning("No sample images found. Make sure the dataset is in the correct location.")
    
    elif mode == "Batch Analysis":
        st.header("Batch Analysis Results")
        
        # Check if results file exists
        results_path = "data/processed/emotion_results.csv"
        if os.path.exists(results_path):
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
                
                # Map dataset emotions to DeepFace emotions for comparison
                emotion_mapping = {
                    'happy': 'happy',
                    'sad': 'sad', 
                    'angry': 'angry',
                    'surprised': 'surprise',
                    'fearful': 'fear',
                    'disgusted': 'disgust',
                    'neutral': 'neutral'
                }
                
                # Map true emotions to match DeepFace output
                df['true_emotion_mapped'] = df['true_emotion'].map(emotion_mapping)
                
                # Calculate accuracy
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
                st.subheader("Accuracy by Emotion Category")
                
                if len(successful_df) > 0:
                    # Get unique emotions actually present in the data
                    unique_true_emotions = successful_df['true_emotion'].unique()
                    
                    st.write(f"**Found emotions in dataset:** {', '.join(unique_true_emotions)}")
                    
                    emotion_stats = []
                    
                    for true_emotion in unique_true_emotions:
                        if true_emotion != 'unknown':
                            # Get all images that are actually this emotion
                            emotion_data = successful_df[successful_df['true_emotion'] == true_emotion]
                            
                            if len(emotion_data) > 0:
                                correct_predictions = emotion_data['correct'].sum()
                                total_images = len(emotion_data)
                                accuracy = (correct_predictions / total_images) * 100
                                
                                # Most common incorrect prediction
                                incorrect_data = emotion_data[~emotion_data['correct']]
                                if len(incorrect_data) > 0:
                                    most_common_error = incorrect_data['dominant_emotion'].mode()
                                    most_common_error = most_common_error.iloc[0] if len(most_common_error) > 0 else 'N/A'
                                else:
                                    most_common_error = 'Perfect (no errors)'
                                
                                # Show prediction distribution for this emotion
                                prediction_counts = emotion_data['dominant_emotion'].value_counts()
                                prediction_breakdown = ', '.join([f"{pred}: {count}" for pred, count in prediction_counts.items()])
                                
                                emotion_stats.append({
                                    'True Emotion': true_emotion.title(),
                                    'Total Images': total_images,
                                    'Correct Predictions': correct_predictions,
                                    'Incorrect Predictions': total_images - correct_predictions,
                                    'Accuracy (%)': f"{accuracy:.1f}%",
                                    'Most Common Error': most_common_error,
                                    'All Predictions': prediction_breakdown
                                })
                    
                    # Display as table
                    if emotion_stats:
                        stats_df = pd.DataFrame(emotion_stats)
                        st.dataframe(stats_df, use_container_width=True)
                        
                        # Show detailed breakdown
                        st.subheader("Detailed Breakdown")
                        for _, row in stats_df.iterrows():
                            st.write(f"**{row['True Emotion']}**: {row['Correct Predictions']}/{row['Total Images']} correct ({row['Accuracy (%)']}) - Predictions: {row['All Predictions']}")
                        
                        # Accuracy bar chart
                        st.subheader("Accuracy Visualization")
                        if len(stats_df) > 0:
                            accuracy_data = [(row['True Emotion'], float(row['Accuracy (%)'].rstrip('%'))) for _, row in stats_df.iterrows()]
                            emotions_list, accuracies_list = zip(*accuracy_data)
                            
                            accuracy_fig = go.Figure(data=[
                                go.Bar(
                                    x=list(emotions_list),
                                    y=list(accuracies_list),
                                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF'],
                                    text=[f'{acc:.1f}%' for acc in accuracies_list],
                                    textposition='auto',
                                )
                            ])
                            
                            accuracy_fig.update_layout(
                                title="Prediction Accuracy by Emotion",
                                xaxis_title="Emotion",
                                yaxis_title="Accuracy (%)",
                                yaxis=dict(range=[0, 100]),
                                height=400
                            )
                            
                            st.plotly_chart(accuracy_fig, use_container_width=True)
                        
                        # Confusion matrix style analysis
                        st.subheader("Prediction vs True Emotion Analysis")
                        
                        # Create cross-tabulation
                        confusion_data = pd.crosstab(
                            successful_df['true_emotion'], 
                            successful_df['dominant_emotion'], 
                            margins=True
                        )
                        
                        st.markdown("**Confusion Matrix (True vs Predicted):**")
                        st.dataframe(confusion_data, use_container_width=True)
                        
                    else:
                        st.warning("No valid emotion data found.")
                else:
                    st.warning("No successful predictions to analyze.")
                
                # Emotion distribution
                st.subheader("Emotion Distribution")
                emotion_counts = df['dominant_emotion'].value_counts()
                fig_pie = px.pie(
                    values=emotion_counts.values,
                    names=emotion_counts.index,
                    title="Distribution of Detected Emotions"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Show sample results
                st.subheader("Sample Results")
                st.dataframe(df.head(20), use_container_width=True)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Full Results",
                    data=csv,
                    file_name="emotion_analysis_results.csv",
                    mime="text/csv"
                )
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