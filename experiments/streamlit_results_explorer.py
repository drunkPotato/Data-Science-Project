#!/usr/bin/env python3
"""
Interactive Streamlit dashboard for exploring experiment results.
Run with: streamlit run experiments/streamlit_results_explorer.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import yaml

st.set_page_config(
    page_title="Experiment Results Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_all_experiments(runs_dir="experiments/runs"):
    """Load all experiment data"""
    runs_path = Path(runs_dir)

    if not runs_path.exists():
        return []

    experiments = []

    for exp_dir in sorted(runs_path.iterdir()):
        if not exp_dir.is_dir():
            continue

        exp_data = {"name": exp_dir.name}

        # Load metrics
        best_metrics_file = exp_dir / "best_metrics.json"
        if best_metrics_file.exists():
            with open(best_metrics_file) as f:
                exp_data["val_metrics"] = json.load(f)

        test_metrics_file = exp_dir / "test_metrics.json"
        if test_metrics_file.exists():
            with open(test_metrics_file) as f:
                exp_data["test_metrics"] = json.load(f)

        # Load config
        config_file = exp_dir / "config.yaml"
        if config_file.exists():
            with open(config_file) as f:
                exp_data["config"] = yaml.safe_load(f)

        # Load history
        history_file = exp_dir / "history.json"
        if history_file.exists():
            with open(history_file) as f:
                exp_data["history"] = json.load(f)

        # Load confusion matrices
        val_cm = exp_dir / "best_confusion_matrix.npy"
        if val_cm.exists():
            exp_data["val_confusion_matrix"] = np.load(val_cm).tolist()

        test_cm = exp_dir / "test_confusion_matrix.npy"
        if test_cm.exists():
            exp_data["test_confusion_matrix"] = np.load(test_cm).tolist()

        if "val_metrics" in exp_data or "test_metrics" in exp_data:
            experiments.append(exp_data)

    return experiments

def plot_overall_metrics(experiments, metric="test_acc"):
    """Create bar chart of overall metrics"""
    data = []
    for exp in experiments:
        if "test_metrics" in exp:
            split, metric_name = metric.split('_', 1)
            metrics_key = f"{split}_metrics"
            if metrics_key in exp:
                value = exp[metrics_key].get(f"{split}_{metric_name}", 0) * 100
                data.append({
                    'Experiment': exp['name'],
                    'Value': value
                })

    if not data:
        return None

    df = pd.DataFrame(data)
    df = df.sort_values('Value', ascending=False)

    fig = px.bar(df, x='Experiment', y='Value',
                 title=f'{metric.replace("_", " ").title()} Comparison',
                 labels={'Value': f'{metric.split("_")[1].upper()} (%)'},
                 color='Value',
                 color_continuous_scale='RdYlGn')

    fig.update_layout(xaxis_tickangle=-45, height=500)

    return fig

def plot_per_emotion_comparison(experiments, emotion, metric="f1", split="test"):
    """Compare a specific emotion across experiments"""
    data = []
    metrics_key = f"{split}_metrics"

    for exp in experiments:
        if metrics_key in exp and "per_class" in exp[metrics_key]:
            per_class = exp[metrics_key]["per_class"]
            if emotion in per_class:
                value = per_class[emotion].get(metric, 0) * 100
                data.append({
                    'Experiment': exp['name'],
                    'Value': value
                })

    if not data:
        return None

    df = pd.DataFrame(data)
    df = df.sort_values('Value', ascending=False)

    fig = px.bar(df, x='Experiment', y='Value',
                 title=f'{emotion.title()} - {metric.upper()} Scores ({split.title()} Set)',
                 labels={'Value': f'{metric.upper()} (%)'},
                 color='Value',
                 color_continuous_scale='Viridis')

    fig.update_layout(xaxis_tickangle=-45, height=500)

    return fig

def plot_confusion_matrix(exp, split="test"):
    """Plot confusion matrix heatmap"""
    cm_key = f"{split}_confusion_matrix"
    metrics_key = f"{split}_metrics"

    if cm_key not in exp or metrics_key not in exp:
        return None

    cm = np.array(exp[cm_key])
    emotions = sorted(list(exp[metrics_key]["per_class"].keys()))

    # Normalize by row
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=[e.title() for e in emotions],
        y=[e.title() for e in emotions],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        colorbar=dict(title="Normalized<br>Rate")
    ))

    fig.update_layout(
        title=f"Confusion Matrix: {exp['name']} ({split.title()} Set)",
        xaxis_title="Predicted Emotion",
        yaxis_title="True Emotion",
        height=600,
        width=650
    )

    return fig

def plot_learning_curves(exp):
    """Plot training history"""
    if "history" not in exp:
        return None

    history = exp["history"]

    if not history:
        return None

    epochs = list(range(1, len(history) + 1))
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    train_acc = [h['train_acc'] * 100 for h in history]
    val_acc = [h['val_acc'] * 100 for h in history]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Loss Curves', 'Accuracy Curves')
    )

    # Loss subplot
    fig.add_trace(
        go.Scatter(x=epochs, y=train_loss, name='Train Loss', mode='lines+markers'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=val_loss, name='Val Loss', mode='lines+markers'),
        row=1, col=1
    )

    # Accuracy subplot
    fig.add_trace(
        go.Scatter(x=epochs, y=train_acc, name='Train Acc', mode='lines+markers'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=val_acc, name='Val Acc', mode='lines+markers'),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)

    fig.update_layout(height=400, title_text=f"Training History: {exp['name']}")

    return fig

def plot_emotion_heatmap(experiments, metric="f1", split="test"):
    """Heatmap of all emotions across experiments"""
    metrics_key = f"{split}_metrics"

    # Get emotions
    emotions = None
    for exp in experiments:
        if metrics_key in exp and "per_class" in exp[metrics_key]:
            emotions = sorted(list(exp[metrics_key]["per_class"].keys()))
            break

    if not emotions:
        return None

    # Build matrix
    data = []
    exp_names = []
    for exp in experiments:
        if metrics_key in exp and "per_class" in exp[metrics_key]:
            row = [exp[metrics_key]["per_class"].get(emotion, {}).get(metric, 0) * 100
                   for emotion in emotions]
            data.append(row)
            exp_names.append(exp["name"])

    if not data:
        return None

    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=[e.title() for e in emotions],
        y=exp_names,
        colorscale='RdYlGn',
        text=np.round(data, 2),
        texttemplate='%{text}%',
        textfont={"size": 9},
        colorbar=dict(title=f"{metric.upper()}<br>Score (%)")
    ))

    fig.update_layout(
        title=f"Per-Emotion {metric.upper()} Scores ({split.title()} Set)",
        xaxis_title="Emotion",
        yaxis_title="Experiment",
        height=max(400, len(exp_names) * 30)
    )

    return fig

def main():
    st.title("Experiment Results Explorer")
    st.markdown("Interactive dashboard for analyzing emotion detection experiment results")

    # Load data
    experiments = load_all_experiments()

    if not experiments:
        st.error("No experiments found! Make sure experiments have been run.")
        st.info("Run: `python experiments/run_batch.py --data_root /path/to/data`")
        return

    st.success(f"Loaded {len(experiments)} experiments")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select View",
        ["Overview", "Detailed Comparison", "Individual Experiment", "Per-Emotion Analysis", "Export Data"]
    )

    # Sort experiments by test F1
    experiments_sorted = sorted(
        [e for e in experiments if "test_metrics" in e],
        key=lambda x: x["test_metrics"]["test_macro_f1"],
        reverse=True
    )

    if page == "Overview":
        st.header("Experiment Overview")

        # Top 3 performers
        st.subheader("Top 3 Performing Experiments")

        for i, exp in enumerate(experiments_sorted[:3], 1):
            test_acc = exp["test_metrics"]["test_acc"] * 100
            test_f1 = exp["test_metrics"]["test_macro_f1"] * 100

            st.markdown(f"**{i}. {exp['name']}**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Test Accuracy", f"{test_acc:.2f}%")
            with col2:
                st.metric("Test Macro F1", f"{test_f1:.2f}%")

        st.markdown("---")

        # Overall metrics comparison
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Test Accuracy Comparison")
            fig_acc = plot_overall_metrics(experiments_sorted, "test_acc")
            if fig_acc:
                st.plotly_chart(fig_acc, use_container_width=True)

        with col2:
            st.subheader("Test Macro F1 Comparison")
            fig_f1 = plot_overall_metrics(experiments_sorted, "test_macro_f1")
            if fig_f1:
                st.plotly_chart(fig_f1, use_container_width=True)

        # Per-emotion heatmap
        st.subheader("Per-Emotion Performance Heatmap")
        metric_choice = st.selectbox("Select Metric", ["f1", "precision", "recall"])
        fig_heatmap = plot_emotion_heatmap(experiments_sorted, metric=metric_choice, split="test")
        if fig_heatmap:
            st.plotly_chart(fig_heatmap, use_container_width=True)

    elif page == "Detailed Comparison":
        st.header("Detailed Experiment Comparison")

        # Select experiments to compare
        exp_names = [e["name"] for e in experiments_sorted]
        selected_names = st.multiselect(
            "Select experiments to compare (max 5)",
            exp_names,
            default=exp_names[:3] if len(exp_names) >= 3 else exp_names
        )

        if not selected_names:
            st.warning("Please select at least one experiment")
            return

        selected_exps = [e for e in experiments_sorted if e["name"] in selected_names]

        # Configuration comparison table
        st.subheader("Configuration Comparison")

        config_data = []
        for exp in selected_exps:
            if "config" in exp:
                cfg = exp["config"]
                config_data.append({
                    'Experiment': exp['name'],
                    'Epochs': cfg["training"]["epochs"],
                    'Learning Rate': f"{cfg['training']['lr']:.0e}",
                    'Batch Size': cfg["training"]["batch_size"],
                    'Loss': cfg["loss"]["type"].upper(),
                    'Mixup Alpha': cfg["augmentation"]["mixup_alpha"],
                    'Random Erasing': cfg["augmentation"]["random_erasing"],
                    'Class Weights': 'Yes' if cfg["loss"]["use_class_weights"] else 'No'
                })

        if config_data:
            st.dataframe(pd.DataFrame(config_data), use_container_width=True)

        # Metrics comparison table
        st.subheader("Performance Metrics Comparison")

        metrics_data = []
        for exp in selected_exps:
            row = {'Experiment': exp['name']}

            if "test_metrics" in exp:
                row['Test Acc'] = f"{exp['test_metrics']['test_acc']*100:.2f}%"
                row['Test F1'] = f"{exp['test_metrics']['test_macro_f1']*100:.2f}%"

            if "val_metrics" in exp:
                row['Val Acc'] = f"{exp['val_metrics']['val_acc']*100:.2f}%"
                row['Val F1'] = f"{exp['val_metrics']['val_macro_f1']*100:.2f}%"

            metrics_data.append(row)

        if metrics_data:
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

        # Learning curves comparison
        st.subheader("Learning Curves")

        for exp in selected_exps:
            with st.expander(f"{exp['name']} Learning Curves"):
                fig_curves = plot_learning_curves(exp)
                if fig_curves:
                    st.plotly_chart(fig_curves, use_container_width=True)
                else:
                    st.info("No training history available")

    elif page == "Individual Experiment":
        st.header("Individual Experiment Analysis")

        # Select experiment
        exp_names = [e["name"] for e in experiments_sorted]
        selected_name = st.selectbox("Select Experiment", exp_names)

        exp = next((e for e in experiments if e["name"] == selected_name), None)

        if not exp:
            return

        # Show metrics
        st.subheader("Overall Metrics")

        col1, col2, col3, col4 = st.columns(4)

        if "test_metrics" in exp:
            with col1:
                st.metric("Test Accuracy", f"{exp['test_metrics']['test_acc']*100:.2f}%")
            with col2:
                st.metric("Test Macro F1", f"{exp['test_metrics']['test_macro_f1']*100:.2f}%")

        if "val_metrics" in exp:
            with col3:
                st.metric("Val Accuracy", f"{exp['val_metrics']['val_acc']*100:.2f}%")
            with col4:
                st.metric("Val Macro F1", f"{exp['val_metrics']['val_macro_f1']*100:.2f}%")

        # Per-emotion metrics
        st.subheader("Per-Emotion Metrics")

        if "test_metrics" in exp and "per_class" in exp["test_metrics"]:
            emotion_data = []
            for emotion, metrics in exp["test_metrics"]["per_class"].items():
                emotion_data.append({
                    'Emotion': emotion.title(),
                    'Precision': f"{metrics['precision']*100:.2f}%",
                    'Recall': f"{metrics['recall']*100:.2f}%",
                    'F1': f"{metrics['f1']*100:.2f}%"
                })

            st.dataframe(pd.DataFrame(emotion_data), use_container_width=True)

        # Confusion matrix
        st.subheader("Confusion Matrix")

        split_choice = st.radio("Select Split", ["test", "val"])
        fig_cm = plot_confusion_matrix(exp, split=split_choice)
        if fig_cm:
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info(f"No confusion matrix available for {split_choice} set")

        # Learning curves
        st.subheader("Training History")
        fig_curves = plot_learning_curves(exp)
        if fig_curves:
            st.plotly_chart(fig_curves, use_container_width=True)
        else:
            st.info("No training history available")

        # Configuration
        if "config" in exp:
            st.subheader("Configuration")
            with st.expander("Show Full Configuration"):
                st.json(exp["config"])

    elif page == "Per-Emotion Analysis":
        st.header("Per-Emotion Analysis")

        # Get all emotions
        emotions = set()
        for exp in experiments:
            if "test_metrics" in exp and "per_class" in exp["test_metrics"]:
                emotions.update(exp["test_metrics"]["per_class"].keys())

        emotions = sorted(list(emotions))

        selected_emotion = st.selectbox("Select Emotion", [e.title() for e in emotions])
        emotion_lower = selected_emotion.lower()

        metric_choice = st.selectbox("Select Metric", ["f1", "precision", "recall"])

        # Plot comparison
        fig = plot_per_emotion_comparison(experiments_sorted, emotion_lower, metric=metric_choice, split="test")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Show detailed table
        st.subheader(f"{selected_emotion} Performance Details")

        emotion_data = []
        for exp in experiments_sorted:
            if "test_metrics" in exp and "per_class" in exp["test_metrics"]:
                per_class = exp["test_metrics"]["per_class"]
                if emotion_lower in per_class:
                    emotion_data.append({
                        'Experiment': exp['name'],
                        'Precision': f"{per_class[emotion_lower]['precision']*100:.2f}%",
                        'Recall': f"{per_class[emotion_lower]['recall']*100:.2f}%",
                        'F1': f"{per_class[emotion_lower]['f1']*100:.2f}%"
                    })

        if emotion_data:
            st.dataframe(pd.DataFrame(emotion_data), use_container_width=True)

    elif page == "Export Data":
        st.header("Export Results")

        st.markdown("""
        Export experiment results in various formats for use in papers and presentations.
        """)

        # Export overall metrics
        st.subheader("Export Overall Metrics")

        overall_data = []
        for exp in experiments_sorted:
            row = {'Experiment': exp['name']}

            if "config" in exp:
                cfg = exp["config"]
                row['Epochs'] = cfg["training"]["epochs"]
                row['Learning_Rate'] = cfg["training"]["lr"]
                row['Batch_Size'] = cfg["training"]["batch_size"]
                row['Loss'] = cfg["loss"]["type"]

            if "test_metrics" in exp:
                row['Test_Accuracy'] = exp['test_metrics']['test_acc']
                row['Test_Macro_F1'] = exp['test_metrics']['test_macro_f1']

            if "val_metrics" in exp:
                row['Val_Accuracy'] = exp['val_metrics']['val_acc']
                row['Val_Macro_F1'] = exp['val_metrics']['val_macro_f1']

            overall_data.append(row)

        if overall_data:
            df_overall = pd.DataFrame(overall_data)

            csv_overall = df_overall.to_csv(index=False)
            st.download_button(
                label="Download Overall Metrics (CSV)",
                data=csv_overall,
                file_name="experiment_overall_metrics.csv",
                mime="text/csv"
            )

        # Export per-emotion metrics
        st.subheader("Export Per-Emotion Metrics")

        emotion_data = []
        for exp in experiments_sorted:
            if "test_metrics" in exp and "per_class" in exp["test_metrics"]:
                for emotion, metrics in exp["test_metrics"]["per_class"].items():
                    emotion_data.append({
                        'Experiment': exp['name'],
                        'Emotion': emotion,
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'F1': metrics['f1']
                    })

        if emotion_data:
            df_emotion = pd.DataFrame(emotion_data)

            csv_emotion = df_emotion.to_csv(index=False)
            st.download_button(
                label="Download Per-Emotion Metrics (CSV)",
                data=csv_emotion,
                file_name="experiment_per_emotion_metrics.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
