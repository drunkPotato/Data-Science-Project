#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
import yaml

sys.path.append(str(Path(__file__).parent))
from run_experiment import run_experiment

def run_batch_experiments(config_dir, data_root, force=False, filter_prefix=None):
    config_dir = Path(config_dir)

    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}")
        sys.exit(1)

    config_files = sorted(config_dir.glob("*.yaml"))

    if filter_prefix:
        config_files = [f for f in config_files if f.stem.startswith(filter_prefix)]

    if not config_files:
        print(f"No config files found in {config_dir}")
        if filter_prefix:
            print(f"with prefix '{filter_prefix}'")
        sys.exit(1)

    print(f"\nFound {len(config_files)} experiments to run:")
    for f in config_files:
        with open(f) as cf:
            config = yaml.safe_load(cf)
        print(f"  - {f.stem}: {config.get('description', 'No description')}")

    print("\n" + "="*80)
    input("Press Enter to start batch experiments (Ctrl+C to cancel)...")
    print("="*80 + "\n")

    results = []
    for i, config_file in enumerate(config_files, 1):
        print(f"\n{'#'*80}")
        print(f"# Experiment {i}/{len(config_files)}: {config_file.stem}")
        print(f"{'#'*80}\n")

        try:
            result = run_experiment(config_file, data_root, force)
            results.append(result)
        except KeyboardInterrupt:
            print("\n\nBatch interrupted by user")
            break
        except Exception as e:
            print(f"\n\nExperiment {config_file.stem} failed with error: {e}")
            results.append({
                "name": config_file.stem,
                "status": "error",
                "error": str(e)
            })

    print(f"\n\n{'='*80}")
    print("BATCH SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {len(config_files)}")
    print(f"Completed: {len(results)}")

    successful = sum(1 for r in results if r and r.get('status') == 'success')
    failed = sum(1 for r in results if r and r.get('status') in ['failed', 'error'])
    interrupted = sum(1 for r in results if r and r.get('status') == 'interrupted')

    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Interrupted: {interrupted}")
    print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(description="Run multiple experiments from config directory")
    parser.add_argument("--config_dir", type=str,
                       default="experiments/configs",
                       help="Directory containing experiment config YAML files")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Path to dataset root")
    parser.add_argument("--force", action="store_true",
                       help="Overwrite existing experiment results")
    parser.add_argument("--filter", type=str, default=None,
                       help="Only run configs with names starting with this prefix")

    args = parser.parse_args()

    run_batch_experiments(args.config_dir, args.data_root, args.force, args.filter)

if __name__ == "__main__":
    main()
