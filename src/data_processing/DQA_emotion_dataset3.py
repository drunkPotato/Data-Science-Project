#!/usr/bin/env python3
"""
Data Quality Assessment (DQA) for Emotion Dataset 3

This script performs comprehensive quality checks on the emotion dataset 3
which uses CSV labels and numbered folders structure.
"""

import sys
import os
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class EmotionDataset3DQA:
    def __init__(self, dataset_path, check_duplicates=True):
        self.dataset_path = Path(dataset_path)
        self.splits = ['train', 'test']
        self.check_duplicates = check_duplicates
        self.supported_formats = {'.png', '.jpg', '.jpeg'}
        self.corrupted_files = []
        self.file_formats = defaultdict(int)
        self.missing_files = []
        self.extra_files = []

        # Emotion label mapping (1-indexed)
        self.label_to_emotion = {
            1: 'angry',
            2: 'disgusted',
            3: 'fearful',
            4: 'happy',
            5: 'sad',
            6: 'surprised',
            7: 'neutral'
        }

    def get_file_hash(self, file_path):
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return None

    def load_csv_labels(self, split):
        """Load CSV labels for a split"""
        csv_path = self.dataset_path / f"{split}_labels.csv"
        if not csv_path.exists():
            print(f"ERROR: Labels CSV not found: {csv_path}")
            return None

        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            print(f"ERROR loading CSV: {e}")
            return None

    def check_csv_structure(self):
        """Check CSV files structure and content"""
        results = {}

        for split in self.splits:
            csv_path = self.dataset_path / f"{split}_labels.csv"
            results[split] = {
                'csv_exists': csv_path.exists(),
                'has_required_columns': False,
                'label_distribution': {},
                'total_records': 0,
                'missing_labels': 0,
                'invalid_labels': 0
            }

            if not results[split]['csv_exists']:
                continue

            df = self.load_csv_labels(split)
            if df is None:
                continue

            # Check columns
            required_cols = ['image', 'label']
            results[split]['has_required_columns'] = all(col in df.columns for col in required_cols)
            results[split]['total_records'] = len(df)

            if not results[split]['has_required_columns']:
                print(f"WARNING: {split}_labels.csv missing required columns: {required_cols}")
                continue

            # Check for missing labels
            results[split]['missing_labels'] = df['label'].isna().sum()

            # Check for invalid labels (not in 1-7 range)
            valid_labels = set(range(1, 8))
            results[split]['invalid_labels'] = (~df['label'].isin(valid_labels)).sum()

            # Label distribution
            label_counts = df['label'].value_counts().sort_index()
            for label, count in label_counts.items():
                emotion_name = self.label_to_emotion.get(label, f'unknown_{label}')
                results[split]['label_distribution'][emotion_name] = count

        return results

    def check_image_files(self, split):
        """Check if image files exist and are readable"""
        df = self.load_csv_labels(split)
        if df is None:
            return {}

        results = {
            'total_expected': len(df),
            'total_found': 0,
            'missing_files': [],
            'extra_files': [],
            'corrupted_files': [],
            'file_formats': defaultdict(int),
            'images_by_folder': defaultdict(int)
        }

        # Check images referenced in CSV
        dataset_split_path = self.dataset_path / 'DATASET' / split

        print(f"  Checking {len(df)} images in {split} split...")
        processed = 0

        for idx, row in df.iterrows():
            image_name = row['image']
            label = row['label']
            emotion_name = self.label_to_emotion.get(label, 'unknown')

            # Images are stored in numbered folders matching their label
            image_path = dataset_split_path / str(label) / image_name

            if image_path.exists() and image_path.is_file():
                results['total_found'] += 1
                results['images_by_folder'][label] += 1

                # Check file format
                file_ext = image_path.suffix.lower()
                results['file_formats'][file_ext] += 1

                # Check readability
                try:
                    with Image.open(image_path) as img:
                        img.verify()
                except Exception as e:
                    results['corrupted_files'].append(str(image_path))
            else:
                results['missing_files'].append(image_name)

            processed += 1
            if processed % 1000 == 0:
                print(f"    Processed {processed}/{len(df)} images...")

        # Check for extra files not in CSV
        for label_folder in dataset_split_path.iterdir():
            if label_folder.is_dir() and label_folder.name.isdigit():
                for img_file in label_folder.iterdir():
                    if img_file.is_file():
                        if img_file.name not in df['image'].values:
                            results['extra_files'].append(str(img_file))

        return results

    def check_duplicates_overall(self):
        """Check for duplicate images across all splits"""
        file_hashes = defaultdict(list)
        total_files = 0
        processed_files = 0

        print("Starting duplicate detection across all splits...")

        for split in self.splits:
            df = self.load_csv_labels(split)
            if df is None:
                continue

            dataset_split_path = self.dataset_path / 'DATASET' / split

            for idx, row in df.iterrows():
                image_name = row['image']
                label = row['label']
                image_path = dataset_split_path / str(label) / image_name

                if image_path.exists() and image_path.is_file():
                    total_files += 1
                    processed_files += 1
                    file_hash = self.get_file_hash(image_path)
                    if file_hash:
                        file_hashes[file_hash].append(image_path)

                    if processed_files % 1000 == 0:
                        print(f"  Processed {processed_files:,} files...")

        print(f"Duplicate detection completed. Total files processed: {total_files:,}")

        unique_files = len(file_hashes)
        duplicate_count = total_files - unique_files

        # Find actual duplicate groups
        duplicate_groups = {k: v for k, v in file_hashes.items() if len(v) > 1}

        return {
            'total_files': total_files,
            'unique_files': unique_files,
            'duplicates': duplicate_count,
            'duplicate_groups': duplicate_groups
        }

    def analyze_class_balance(self, csv_results):
        """Analyze class balance across all splits"""
        total_counts = defaultdict(int)

        for split_result in csv_results.values():
            if 'label_distribution' in split_result:
                for emotion, count in split_result['label_distribution'].items():
                    total_counts[emotion] += count

        total_images = sum(total_counts.values())
        if total_images == 0:
            return {}

        balance_analysis = {}
        for emotion, count in total_counts.items():
            percentage = (count / total_images) * 100
            balance_analysis[emotion] = {
                'count': count,
                'percentage': percentage,
                'severely_underrepresented': percentage < 5,
                'severely_overrepresented': percentage > 50
            }

        return balance_analysis

    def generate_report(self):
        """Generate comprehensive DQA report"""
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("DATA QUALITY ASSESSMENT: EMOTION DATASET 3")
        report_lines.append("="*70)
        report_lines.append("")

        # Check CSV structure
        print("Checking CSV label files...")
        csv_results = self.check_csv_structure()

        report_lines.append("CSV LABEL FILES:")
        report_lines.append("-" * 70)
        for split in self.splits:
            result = csv_results[split]
            report_lines.append(f"\n{split.upper()} SPLIT:")

            if not result['csv_exists']:
                report_lines.append("  STATUS: MISSING - CSV file not found")
            else:
                status = "PASS" if result['has_required_columns'] else "FAIL"
                report_lines.append(f"  STATUS: {status}")
                report_lines.append(f"  Total records: {result['total_records']}")

                if result['missing_labels'] > 0:
                    report_lines.append(f"  [WARNING] Missing labels: {result['missing_labels']}")

                if result['invalid_labels'] > 0:
                    report_lines.append(f"  [WARNING] Invalid labels (not 1-7): {result['invalid_labels']}")

                if result['label_distribution']:
                    report_lines.append("  Label distribution:")
                    for emotion in ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']:
                        count = result['label_distribution'].get(emotion, 0)
                        report_lines.append(f"    {emotion:>10}: {count:>6} images")

        report_lines.append("")

        # Check image files
        print("\nChecking image files...")
        image_results = {}
        for split in self.splits:
            image_results[split] = self.check_image_files(split)

        report_lines.append("IMAGE FILES VERIFICATION:")
        report_lines.append("-" * 70)

        all_images_ok = True
        for split in self.splits:
            result = image_results[split]
            report_lines.append(f"\n{split.upper()} SPLIT:")
            report_lines.append(f"  Expected images: {result['total_expected']}")
            report_lines.append(f"  Found images: {result['total_found']}")

            if result['missing_files']:
                report_lines.append(f"  [WARNING] Missing files: {len(result['missing_files'])}")
                all_images_ok = False
                if len(result['missing_files']) <= 10:
                    for f in result['missing_files']:
                        report_lines.append(f"    - {f}")
                else:
                    for f in result['missing_files'][:10]:
                        report_lines.append(f"    - {f}")
                    report_lines.append(f"    ... and {len(result['missing_files'])-10} more")

            if result['extra_files']:
                report_lines.append(f"  [WARNING] Extra files not in CSV: {len(result['extra_files'])}")
                all_images_ok = False

            if result['corrupted_files']:
                report_lines.append(f"  [WARNING] Corrupted files: {len(result['corrupted_files'])}")
                all_images_ok = False
                for f in result['corrupted_files'][:10]:
                    report_lines.append(f"    - {f}")
            else:
                report_lines.append("  [PASS] All images are readable")

            if result['file_formats']:
                report_lines.append("  File formats:")
                for fmt, count in sorted(result['file_formats'].items()):
                    percentage = (count / result['total_found']) * 100 if result['total_found'] > 0 else 0
                    report_lines.append(f"    {fmt}: {count} files ({percentage:.1f}%)")

        report_lines.append("")

        # Duplicate check
        if self.check_duplicates:
            print("\nChecking for duplicates...")
            dup_results = self.check_duplicates_overall()

            report_lines.append("DUPLICATE DETECTION:")
            report_lines.append("-" * 70)
            report_lines.append(f"Total files: {dup_results['total_files']}")
            report_lines.append(f"Unique files: {dup_results['unique_files']}")
            report_lines.append(f"Duplicates: {dup_results['duplicates']}")

            if dup_results['duplicates'] > 0:
                report_lines.append(f"[WARNING] Found {len(dup_results['duplicate_groups'])} duplicate groups:")
                for idx, (hash_val, files) in enumerate(list(dup_results['duplicate_groups'].items())[:5]):
                    report_lines.append(f"  Group {idx+1}: {len(files)} copies")
                    for f in files:
                        report_lines.append(f"    - {f}")
                if len(dup_results['duplicate_groups']) > 5:
                    report_lines.append(f"  ... and {len(dup_results['duplicate_groups'])-5} more groups")
            else:
                report_lines.append("[PASS] No duplicates found")

            report_lines.append("")

        # Class balance analysis
        balance_analysis = self.analyze_class_balance(csv_results)
        if balance_analysis:
            report_lines.append("CLASS BALANCE ANALYSIS:")
            report_lines.append("-" * 70)
            imbalance_warnings = []

            for emotion in ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']:
                if emotion in balance_analysis:
                    data = balance_analysis[emotion]
                    report_lines.append(f"  {emotion:>10}: {data['count']:>6} images ({data['percentage']:>5.1f}%)")

                    if data['severely_underrepresented']:
                        imbalance_warnings.append(f"'{emotion}' severely underrepresented (<5%)")
                    elif data['severely_overrepresented']:
                        imbalance_warnings.append(f"'{emotion}' severely overrepresented (>50%)")

            if imbalance_warnings:
                report_lines.append("  [WARNING] Class imbalances detected:")
                for warning in imbalance_warnings:
                    report_lines.append(f"    - {warning}")
            else:
                report_lines.append("  [PASS] No severe class imbalances detected")

            report_lines.append("")

        # Overall summary
        report_lines.append("OVERALL SUMMARY:")
        report_lines.append("-" * 70)

        all_ok = True
        for split_result in csv_results.values():
            if not split_result['csv_exists'] or not split_result['has_required_columns']:
                all_ok = False

        if not all_images_ok:
            all_ok = False

        if all_ok:
            report_lines.append("[PASS] Dataset structure is valid")
        else:
            report_lines.append("[FAIL] Dataset has quality issues")

        report_lines.append("")
        report_lines.append("="*70)

        return "\n".join(report_lines)

def main():
    check_duplicates = '--duplicates' in sys.argv

    dataset_path = Path(__file__).parent.parent.parent / "data" / "raw" / "emotion dataset 3"

    if not dataset_path.exists():
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        return

    print(f"Starting DQA for Emotion Dataset 3 at: {dataset_path}")
    print("="*70)

    if check_duplicates:
        print("Note: Duplicate checking is enabled. This may take several minutes.")

    dqa = EmotionDataset3DQA(dataset_path, check_duplicates=check_duplicates)
    report = dqa.generate_report()

    # Print to console
    print("\n" + report)

    # Save to file
    output_path = Path(__file__).parent / "DQA_emotion_dataset3_summary.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved to: {output_path}")

if __name__ == "__main__":
    main()
