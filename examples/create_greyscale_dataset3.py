#!/usr/bin/env python3
"""
Create a greyscale copy of Emotion Dataset 3 test set for comparison
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import pandas as pd
from pathlib import Path
import shutil


def main():
    # Source dataset paths
    dataset_path = Path(__file__).parent.parent / "data" / "raw" / "emotion dataset 3"
    test_csv = dataset_path / "test_labels.csv"
    test_images_base = dataset_path / "DATASET" / "test"

    # Destination paths for greyscale version
    greyscale_dataset_path = Path(__file__).parent.parent / "data" / "raw" / "emotion dataset 3 greyscale"
    greyscale_test_csv = greyscale_dataset_path / "test_labels.csv"
    greyscale_images_base = greyscale_dataset_path / "DATASET" / "test"

    # Create output directories
    greyscale_dataset_path.mkdir(parents=True, exist_ok=True)
    greyscale_images_base.mkdir(parents=True, exist_ok=True)

    # Create numbered folders (1-7 for each emotion)
    for label_num in range(1, 8):
        label_folder = greyscale_images_base / str(label_num)
        label_folder.mkdir(parents=True, exist_ok=True)

    # Verify source dataset exists
    if not test_csv.exists():
        print(f"ERROR: Test CSV not found at {test_csv}")
        return

    # Load test labels
    print("="*70)
    print("CREATING GREYSCALE VERSION OF EMOTION DATASET 3 TEST SET")
    print("="*70)
    print(f"\nSource: {test_images_base}")
    print(f"Destination: {greyscale_images_base}")

    df = pd.read_csv(test_csv)
    print(f"\nProcessing {len(df)} test images...")

    processed = 0
    skipped = 0
    errors = 0

    for idx, row in df.iterrows():
        image_name = row['image']
        label = row['label']

        # Source and destination paths
        src_path = test_images_base / str(label) / image_name
        dst_path = greyscale_images_base / str(label) / image_name

        if not src_path.exists():
            print(f"WARNING: Source image not found: {src_path}")
            errors += 1
            continue

        try:
            # Read image
            img = cv2.imread(str(src_path))

            if img is None:
                print(f"WARNING: Could not read image: {src_path}")
                errors += 1
                continue

            # Check if already greyscale
            if len(img.shape) == 2:
                # Already greyscale, just copy
                shutil.copy2(src_path, dst_path)
                skipped += 1
            elif len(img.shape) == 3 and img.shape[2] == 1:
                # Single channel but 3D array, just copy
                shutil.copy2(src_path, dst_path)
                skipped += 1
            else:
                # Convert to greyscale
                grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(str(dst_path), grey_img)
                processed += 1

            # Progress indicator
            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{len(df)} images...")

        except Exception as e:
            print(f"ERROR processing {image_name}: {e}")
            errors += 1

    # Copy the CSV file
    shutil.copy2(test_csv, greyscale_test_csv)

    print("\n" + "="*70)
    print("GREYSCALE CONVERSION COMPLETE")
    print("="*70)
    print(f"Total images: {len(df)}")
    print(f"Converted to greyscale: {processed}")
    print(f"Already greyscale (copied): {skipped}")
    print(f"Errors: {errors}")
    print(f"\nGreyscale dataset saved to:")
    print(f"  {greyscale_dataset_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
