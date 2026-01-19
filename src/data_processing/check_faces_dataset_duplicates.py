#!/usr/bin/env python3
"""
Check Faces_Dataset_backup for duplicate images and train-test leakage
"""

import hashlib
import os
from pathlib import Path
from collections import defaultdict
from PIL import Image
import sys


def get_file_hash(file_path):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"ERROR reading {file_path}: {e}")
        return None


def scan_directory(base_path, split_name):
    """Scan directory and compute hashes for all images"""
    print(f"\nScanning {split_name} set...")

    file_hashes = {}
    emotion_counts = defaultdict(int)
    total_files = 0
    errors = 0

    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

    for emotion in emotions:
        emotion_path = base_path / emotion
        if not emotion_path.exists():
            print(f"  WARNING: {emotion} folder not found")
            continue

        image_files = list(emotion_path.glob('*'))
        emotion_counts[emotion] = len(image_files)

        for img_file in image_files:
            if img_file.is_file():
                total_files += 1

                # Verify it's a valid image
                try:
                    with Image.open(img_file) as img:
                        img.verify()
                except Exception as e:
                    print(f"  ERROR: Invalid image {img_file}: {e}")
                    errors += 1
                    continue

                # Compute hash
                file_hash = get_file_hash(img_file)
                if file_hash:
                    if file_hash in file_hashes:
                        file_hashes[file_hash].append(str(img_file))
                    else:
                        file_hashes[file_hash] = [str(img_file)]
                else:
                    errors += 1

                # Progress indicator
                if total_files % 500 == 0:
                    print(f"  Processed {total_files} files...")

    print(f"  Total files: {total_files}")
    print(f"  Errors: {errors}")
    print(f"  Emotion distribution:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"    {emotion:>10}: {count:>5} images")

    return file_hashes, total_files, errors


def find_duplicates(file_hashes):
    """Find duplicate files within a set"""
    duplicates = {hash_val: files for hash_val, files in file_hashes.items() if len(files) > 1}
    return duplicates


def find_cross_set_duplicates(train_hashes, test_hashes):
    """Find files that appear in both train and test sets"""
    train_hash_set = set(train_hashes.keys())
    test_hash_set = set(test_hashes.keys())

    common_hashes = train_hash_set & test_hash_set

    leakage = {}
    for hash_val in common_hashes:
        leakage[hash_val] = {
            'train': train_hashes[hash_val],
            'test': test_hashes[hash_val]
        }

    return leakage


def extract_emotion_from_path(path):
    """Extract emotion from file path"""
    path_parts = path.replace('\\', '/').split('/')
    for part in path_parts:
        if part in ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']:
            return part
    return 'unknown'


def generate_report(train_hashes, test_hashes, train_total, test_total,
                   train_errors, test_errors, train_dupes, test_dupes, leakage):
    """Generate comprehensive duplicate detection report"""

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("FACES_DATASET_BACKUP - DUPLICATE DETECTION REPORT")
    report_lines.append("="*80)
    report_lines.append("")

    # Overall statistics
    report_lines.append("DATASET STATISTICS:")
    report_lines.append("-"*80)
    report_lines.append(f"Train set:")
    report_lines.append(f"  Total images: {train_total}")
    report_lines.append(f"  Unique images: {len(train_hashes)}")
    report_lines.append(f"  Read errors: {train_errors}")
    report_lines.append("")
    report_lines.append(f"Test set:")
    report_lines.append(f"  Total images: {test_total}")
    report_lines.append(f"  Unique images: {len(test_hashes)}")
    report_lines.append(f"  Read errors: {test_errors}")
    report_lines.append("")

    # Train set duplicates
    report_lines.append("TRAIN SET DUPLICATES:")
    report_lines.append("-"*80)
    if train_dupes:
        report_lines.append(f"Found {len(train_dupes)} duplicate groups in train set:")
        report_lines.append("")

        for idx, (hash_val, files) in enumerate(list(train_dupes.items())[:10], 1):
            report_lines.append(f"  Duplicate Group {idx}: {len(files)} copies")
            for file_path in files:
                emotion = extract_emotion_from_path(file_path)
                filename = Path(file_path).name
                report_lines.append(f"    [{emotion}] {filename}")
            report_lines.append("")

        if len(train_dupes) > 10:
            report_lines.append(f"  ... and {len(train_dupes) - 10} more duplicate groups")
            report_lines.append("")

        # Count total duplicate files
        total_train_dupes = sum(len(files) - 1 for files in train_dupes.values())
        report_lines.append(f"  Total duplicate files (excluding first copy): {total_train_dupes}")
    else:
        report_lines.append("[PASS] No duplicates found in train set")
    report_lines.append("")

    # Test set duplicates
    report_lines.append("TEST SET DUPLICATES:")
    report_lines.append("-"*80)
    if test_dupes:
        report_lines.append(f"Found {len(test_dupes)} duplicate groups in test set:")
        report_lines.append("")

        for idx, (hash_val, files) in enumerate(list(test_dupes.items())[:10], 1):
            report_lines.append(f"  Duplicate Group {idx}: {len(files)} copies")
            for file_path in files:
                emotion = extract_emotion_from_path(file_path)
                filename = Path(file_path).name
                report_lines.append(f"    [{emotion}] {filename}")
            report_lines.append("")

        if len(test_dupes) > 10:
            report_lines.append(f"  ... and {len(test_dupes) - 10} more duplicate groups")
            report_lines.append("")

        # Count total duplicate files
        total_test_dupes = sum(len(files) - 1 for files in test_dupes.values())
        report_lines.append(f"  Total duplicate files (excluding first copy): {total_test_dupes}")
    else:
        report_lines.append("[PASS] No duplicates found in test set")
    report_lines.append("")

    # Train-test leakage
    report_lines.append("TRAIN-TEST LEAKAGE:")
    report_lines.append("-"*80)
    if leakage:
        report_lines.append(f"[WARNING] Found {len(leakage)} images appearing in BOTH train and test sets!")
        report_lines.append("")
        report_lines.append("This is DATA LEAKAGE and will inflate test accuracy metrics.")
        report_lines.append("")

        # Show first 10 leakage cases
        for idx, (hash_val, locations) in enumerate(list(leakage.items())[:10], 1):
            report_lines.append(f"  Leakage Case {idx}:")
            report_lines.append(f"    Train files ({len(locations['train'])}):")
            for file_path in locations['train']:
                emotion = extract_emotion_from_path(file_path)
                filename = Path(file_path).name
                report_lines.append(f"      [{emotion}] {filename}")
            report_lines.append(f"    Test files ({len(locations['test'])}):")
            for file_path in locations['test']:
                emotion = extract_emotion_from_path(file_path)
                filename = Path(file_path).name
                report_lines.append(f"      [{emotion}] {filename}")
            report_lines.append("")

        if len(leakage) > 10:
            report_lines.append(f"  ... and {len(leakage) - 10} more leakage cases")
            report_lines.append("")

        # Breakdown by emotion
        report_lines.append("  Leakage by emotion:")
        emotion_leakage = defaultdict(int)
        for hash_val, locations in leakage.items():
            # Count unique emotions affected
            for file_path in locations['train'] + locations['test']:
                emotion = extract_emotion_from_path(file_path)
                emotion_leakage[emotion] += 1

        for emotion, count in sorted(emotion_leakage.items()):
            report_lines.append(f"    {emotion:>10}: {count:>3} leaked images")

    else:
        report_lines.append("[PASS] No train-test leakage detected!")
        report_lines.append("Train and test sets are properly separated.")
    report_lines.append("")

    # Summary
    report_lines.append("SUMMARY:")
    report_lines.append("-"*80)

    issues = []
    if train_dupes:
        total_train_dupes = sum(len(files) - 1 for files in train_dupes.values())
        issues.append(f"Train set: {len(train_dupes)} duplicate groups ({total_train_dupes} extra files)")
    if test_dupes:
        total_test_dupes = sum(len(files) - 1 for files in test_dupes.values())
        issues.append(f"Test set: {len(test_dupes)} duplicate groups ({total_test_dupes} extra files)")
    if leakage:
        issues.append(f"Train-Test Leakage: {len(leakage)} images in both sets [CRITICAL]")

    if issues:
        report_lines.append("[WARNING] Issues found:")
        for issue in issues:
            report_lines.append(f"  - {issue}")
    else:
        report_lines.append("[PASS] No duplicates or leakage detected!")
        report_lines.append("Dataset is clean and properly split.")

    report_lines.append("")
    report_lines.append("="*80)

    return "\n".join(report_lines)


def main():
    # Dataset paths
    dataset_path = Path(__file__).parent.parent.parent / "data" / "raw" / "Faces_Dataset_backup"
    train_path = dataset_path / "train"
    test_path = dataset_path / "test"

    print("="*80)
    print("FACES_DATASET_BACKUP - DUPLICATE DETECTION")
    print("="*80)
    print(f"\nDataset path: {dataset_path}")

    # Check if paths exist
    if not dataset_path.exists():
        print(f"\nERROR: Dataset not found at {dataset_path}")
        print("Please check the path and try again.")
        return

    if not train_path.exists() or not test_path.exists():
        print(f"\nERROR: Train or test folder not found")
        print(f"Train path: {train_path} (exists: {train_path.exists()})")
        print(f"Test path: {test_path} (exists: {test_path.exists()})")
        return

    # Scan train set
    train_hashes, train_total, train_errors = scan_directory(train_path, "train")

    # Scan test set
    test_hashes, test_total, test_errors = scan_directory(test_path, "test")

    # Find duplicates within each set
    print("\n" + "="*80)
    print("ANALYZING DUPLICATES...")
    print("="*80)

    train_dupes = find_duplicates(train_hashes)
    test_dupes = find_duplicates(test_hashes)

    print(f"\nTrain set duplicates: {len(train_dupes)} groups")
    print(f"Test set duplicates: {len(test_dupes)} groups")

    # Find train-test leakage
    print("\nChecking for train-test leakage...")
    leakage = find_cross_set_duplicates(train_hashes, test_hashes)

    if leakage:
        print(f"[WARNING] Found {len(leakage)} images in BOTH train and test sets!")
    else:
        print("[PASS] No train-test leakage detected")

    # Generate report
    print("\n" + "="*80)
    print("GENERATING REPORT...")
    print("="*80)

    report = generate_report(
        train_hashes, test_hashes,
        train_total, test_total,
        train_errors, test_errors,
        train_dupes, test_dupes,
        leakage
    )

    # Print report
    print("\n" + report)

    # Save report
    output_path = Path(__file__).parent / "faces_dataset_duplicate_report.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
