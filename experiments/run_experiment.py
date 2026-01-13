#!/usr/bin/env python3

import os
import sys
import yaml
import json
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def config_to_args(config):
    args = []

    args.extend(['--model', config['model']['arch']])
    args.extend(['--dropout', str(config['model']['dropout'])])
    if not config['model']['pretrained']:
        args.append('--no_pretrain')

    args.extend(['--epochs', str(config['training']['epochs'])])
    args.extend(['--batch-size', str(config['training']['batch_size'])])
    args.extend(['--eval_batch_size', str(config['training']['eval_batch_size'])])
    args.extend(['--lr', str(config['training']['lr'])])
    args.extend(['--weight_decay', str(config['training']['weight_decay'])])
    args.extend(['--freeze_epochs', str(config['training']['freeze_epochs'])])
    args.extend(['--onecycle_pct_start', str(config['training']['onecycle_pct_start'])])

    args.extend(['--loss', config['loss']['type']])
    args.extend(['--label_smoothing', str(config['loss']['label_smoothing'])])
    args.extend(['--focal_gamma', str(config['loss']['focal_gamma'])])
    args.extend(['--use_class_weights', str(config['loss']['use_class_weights']).lower()])

    args.extend(['--mixup_alpha', str(config['augmentation']['mixup_alpha'])])
    args.extend(['--random_erasing', str(config['augmentation']['random_erasing'])])

    args.extend(['--img_size', str(config['data']['img_size'])])
    args.extend(['--weighted_sampler', str(config['data']['weighted_sampler']).lower()])

    args.extend(['--patience', str(config['other']['patience'])])
    args.extend(['--seed', str(config['other']['seed'])])
    args.extend(['--workers', str(config['other']['workers'])])

    return args

def run_experiment(config_path, data_root, force=False):
    config = load_config(config_path)
    exp_name = config['name']

    base_dir = Path(__file__).parent.parent
    out_dir = base_dir / "experiments" / "runs" / exp_name

    if out_dir.exists() and not force:
        print(f"Experiment '{exp_name}' already exists at {out_dir}")
        print("Use --force to overwrite, or choose a different config name")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, indent=2)

    args = config_to_args(config)
    args.extend(['--data_root', str(data_root)])
    args.extend(['--out_dir', str(out_dir)])

    train_script = base_dir / "models" / "private_modle.py"

    cmd = [sys.executable, str(train_script)] + args

    print(f"\n{'='*80}")
    print(f"Starting experiment: {exp_name}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"Output directory: {out_dir}")
    print(f"{'='*80}\n")

    start_time = datetime.now()

    try:
        result = subprocess.run(cmd, check=True)
        status = "success"
        error = None
    except subprocess.CalledProcessError as e:
        status = "failed"
        error = str(e)
        print(f"\nExperiment failed with error: {error}")
    except KeyboardInterrupt:
        status = "interrupted"
        error = "User interrupted"
        print(f"\nExperiment interrupted by user")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    experiment_log = {
        "name": exp_name,
        "config_path": str(config_path),
        "description": config.get('description', ''),
        "status": status,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "output_dir": str(out_dir),
        "error": error
    }

    log_file = base_dir / "experiments" / "experiment_log.jsonl"
    with open(log_file, 'a') as f:
        f.write(json.dumps(experiment_log) + '\n')

    print(f"\n{'='*80}")
    print(f"Experiment completed: {exp_name}")
    print(f"Status: {status}")
    print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"Results saved to: {out_dir}")
    print(f"{'='*80}\n")

    return experiment_log

def main():
    parser = argparse.ArgumentParser(description="Run emotion detection training experiment from config")
    parser.add_argument("config", type=str, help="Path to experiment config YAML file")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--force", action="store_true", help="Overwrite existing experiment results")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    if not os.path.exists(args.data_root):
        print(f"Error: Data root not found: {args.data_root}")
        sys.exit(1)

    run_experiment(args.config, args.data_root, args.force)

if __name__ == "__main__":
    main()
