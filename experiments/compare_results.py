#!/usr/bin/env python3

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tabulate import tabulate

def load_experiment_results(runs_dir):
    runs_dir = Path(runs_dir)

    if not runs_dir.exists():
        print(f"Error: Runs directory not found: {runs_dir}")
        return []

    experiments = []

    for exp_dir in sorted(runs_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        exp_data = {"name": exp_dir.name}

        best_metrics = exp_dir / "best_metrics.json"
        if best_metrics.exists():
            with open(best_metrics) as f:
                exp_data["val_metrics"] = json.load(f)

        test_metrics = exp_dir / "test_metrics.json"
        if test_metrics.exists():
            with open(test_metrics) as f:
                exp_data["test_metrics"] = json.load(f)

        config_file = exp_dir / "config.yaml"
        if config_file.exists():
            import yaml
            with open(config_file) as f:
                exp_data["config"] = yaml.safe_load(f)

        history_file = exp_dir / "history.json"
        if history_file.exists():
            with open(history_file) as f:
                exp_data["history"] = json.load(f)

        if "val_metrics" in exp_data or "test_metrics" in exp_data:
            experiments.append(exp_data)

    return experiments

def print_overall_summary(experiments):
    print("\n" + "="*100)
    print("OVERALL EXPERIMENT SUMMARY")
    print("="*100 + "\n")

    if not experiments:
        print("No experiment results found.\n")
        return

    rows = []
    for exp in experiments:
        row = {"Experiment": exp["name"]}

        if "val_metrics" in exp:
            row["Val Acc"] = f"{exp['val_metrics']['val_acc']:.4f}"
            row["Val F1"] = f"{exp['val_metrics']['val_macro_f1']:.4f}"
        else:
            row["Val Acc"] = "N/A"
            row["Val F1"] = "N/A"

        if "test_metrics" in exp:
            row["Test Acc"] = f"{exp['test_metrics']['test_acc']:.4f}"
            row["Test F1"] = f"{exp['test_metrics']['test_macro_f1']:.4f}"
        else:
            row["Test Acc"] = "N/A"
            row["Test F1"] = "N/A"

        if "config" in exp:
            cfg = exp["config"]
            row["Epochs"] = cfg["training"]["epochs"]
            row["LR"] = cfg["training"]["lr"]
            row["BS"] = cfg["training"]["batch_size"]
            row["Loss"] = cfg["loss"]["type"]

        rows.append(row)

    print(tabulate(rows, headers="keys", tablefmt="grid"))
    print()

def print_per_class_comparison(experiments, split="val"):
    print("\n" + "="*100)
    print(f"PER-CLASS METRICS COMPARISON ({split.upper()} SET)")
    print("="*100 + "\n")

    if not experiments:
        return

    emotions = None
    metrics_key = f"{split}_metrics"

    for exp in experiments:
        if metrics_key in exp and "per_class" in exp[metrics_key]:
            emotions = list(exp[metrics_key]["per_class"].keys())
            break

    if not emotions:
        print(f"No per-class metrics found for {split} set.\n")
        return

    for metric in ["precision", "recall", "f1"]:
        print(f"\n{metric.upper()}:")
        print("-" * 100)

        rows = []
        for emotion in emotions:
            row = {"Emotion": emotion}
            for exp in experiments:
                if metrics_key in exp and "per_class" in exp[metrics_key]:
                    value = exp[metrics_key]["per_class"].get(emotion, {}).get(metric, 0)
                    row[exp["name"][:20]] = f"{value:.4f}"
                else:
                    row[exp["name"][:20]] = "N/A"
            rows.append(row)

        print(tabulate(rows, headers="keys", tablefmt="grid"))
        print()

def print_config_comparison(experiments):
    print("\n" + "="*100)
    print("CONFIGURATION COMPARISON")
    print("="*100 + "\n")

    if not experiments:
        return

    rows = []
    for exp in experiments:
        if "config" not in exp:
            continue

        cfg = exp["config"]
        row = {
            "Experiment": exp["name"],
            "Epochs": cfg["training"]["epochs"],
            "Batch Size": cfg["training"]["batch_size"],
            "LR": cfg["training"]["lr"],
            "Weight Decay": cfg["training"]["weight_decay"],
            "Loss": cfg["loss"]["type"],
            "Label Smooth": cfg["loss"]["label_smoothing"],
            "Mixup": cfg["augmentation"]["mixup_alpha"],
            "Random Erase": cfg["augmentation"]["random_erasing"],
            "Class Weights": cfg["loss"]["use_class_weights"],
            "Weighted Sampler": cfg["data"]["weighted_sampler"]
        }
        rows.append(row)

    print(tabulate(rows, headers="keys", tablefmt="grid"))
    print()

def export_to_csv(experiments, output_file):
    print(f"\nExporting results to {output_file}...")

    rows = []
    for exp in experiments:
        row = {"experiment": exp["name"]}

        if "config" in exp:
            cfg = exp["config"]
            row["epochs"] = cfg["training"]["epochs"]
            row["batch_size"] = cfg["training"]["batch_size"]
            row["lr"] = cfg["training"]["lr"]
            row["weight_decay"] = cfg["training"]["weight_decay"]
            row["loss"] = cfg["loss"]["type"]
            row["label_smoothing"] = cfg["loss"]["label_smoothing"]
            row["mixup_alpha"] = cfg["augmentation"]["mixup_alpha"]
            row["random_erasing"] = cfg["augmentation"]["random_erasing"]
            row["use_class_weights"] = cfg["loss"]["use_class_weights"]
            row["weighted_sampler"] = cfg["data"]["weighted_sampler"]

        if "val_metrics" in exp:
            row["val_acc"] = exp["val_metrics"]["val_acc"]
            row["val_macro_f1"] = exp["val_metrics"]["val_macro_f1"]

            for emotion, metrics in exp["val_metrics"]["per_class"].items():
                row[f"val_{emotion}_precision"] = metrics["precision"]
                row[f"val_{emotion}_recall"] = metrics["recall"]
                row[f"val_{emotion}_f1"] = metrics["f1"]

        if "test_metrics" in exp:
            row["test_acc"] = exp["test_metrics"]["test_acc"]
            row["test_macro_f1"] = exp["test_metrics"]["test_macro_f1"]

            for emotion, metrics in exp["test_metrics"]["per_class"].items():
                row[f"test_{emotion}_precision"] = metrics["precision"]
                row[f"test_{emotion}_recall"] = metrics["recall"]
                row[f"test_{emotion}_f1"] = metrics["f1"]

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Exported {len(rows)} experiments to {output_file}\n")

def find_best_per_emotion(experiments, split="val"):
    print("\n" + "="*100)
    print(f"BEST PERFORMING EXPERIMENT PER EMOTION ({split.upper()} SET)")
    print("="*100 + "\n")

    if not experiments:
        return

    emotions = None
    metrics_key = f"{split}_metrics"

    for exp in experiments:
        if metrics_key in exp and "per_class" in exp[metrics_key]:
            emotions = list(exp[metrics_key]["per_class"].keys())
            break

    if not emotions:
        print(f"No per-class metrics found for {split} set.\n")
        return

    rows = []
    for emotion in emotions:
        best_f1 = -1
        best_exp = None

        for exp in experiments:
            if metrics_key in exp and "per_class" in exp[metrics_key]:
                f1 = exp[metrics_key]["per_class"].get(emotion, {}).get("f1", 0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_exp = exp

        if best_exp:
            row = {
                "Emotion": emotion,
                "Best Experiment": best_exp["name"],
                "F1": f"{best_f1:.4f}",
                "Precision": f"{best_exp[metrics_key]['per_class'][emotion]['precision']:.4f}",
                "Recall": f"{best_exp[metrics_key]['per_class'][emotion]['recall']:.4f}"
            }
            rows.append(row)

    print(tabulate(rows, headers="keys", tablefmt="grid"))
    print()

def main():
    parser = argparse.ArgumentParser(description="Compare emotion detection experiment results")
    parser.add_argument("--runs_dir", type=str,
                       default="experiments/runs",
                       help="Directory containing experiment runs")
    parser.add_argument("--export", type=str, default=None,
                       help="Export results to CSV file")
    parser.add_argument("--split", type=str, default="val",
                       choices=["val", "test"],
                       help="Which split to show per-class metrics for")

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    runs_dir = base_dir / args.runs_dir

    experiments = load_experiment_results(runs_dir)

    if not experiments:
        print(f"\nNo experiments found in {runs_dir}")
        print("Run some experiments first using run_experiment.py or run_batch.py\n")
        sys.exit(0)

    print_overall_summary(experiments)
    print_config_comparison(experiments)
    print_per_class_comparison(experiments, split=args.split)
    find_best_per_emotion(experiments, split=args.split)

    if args.export:
        export_file = base_dir / args.export
        export_to_csv(experiments, export_file)

if __name__ == "__main__":
    main()
