#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Quality Assessment (DQA) Summary for Faces_Dataset

This script checks if the Faces_Dataset has the expected 7 emotion folders
in each of its three splits: train, test, val.
"""

import sys
import os
import hashlib
from pathlib import Path
from collections import defaultdict
from PIL import Image

class FacesDatasetDQA:
    def __init__(self, dataset_path, check_duplicates=False):
        
        self.dataset_path = Path(dataset_path)
        self.expected_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        self.splits = ['train', 'test', 'val']
        self.check_duplicates = check_duplicates
        self.supported_formats = {'.png', '.jpg', '.jpeg'}
        self.corrupted_files = []
        self.file_formats = defaultdict(int)

    
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
    
    def check_duplicates_overall(self):
        """Check for duplicate images across all splits"""
        file_hashes = defaultdict(list)
        total_files = 0
        processed_files = 0
        
        print("Starting duplicate detection across all splits...")
        
        for split in self.splits:
            split_path = self.dataset_path / split
            
            if not split_path.exists() or not split_path.is_dir():
                continue
                
            for emotion in self.expected_emotions:
                emotion_path = split_path / emotion
                
                if emotion_path.exists() and emotion_path.is_dir():
                    for file_path in emotion_path.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                            total_files += 1
                            processed_files += 1
                            file_hash = self.get_file_hash(file_path)
                            if file_hash:
                                file_hashes[file_hash].append(file_path)
                            
                            # Progress update every 1000 files
                            if processed_files % 1000 == 0:
                                print(f"  Processed {processed_files:,} files...")
        
        print(f"Duplicate detection completed. Total files processed: {total_files:,}")
        
        # Count duplicates
        unique_files = len(file_hashes)
        duplicate_count = total_files - unique_files
        
        return {
            'total_files': total_files,
            'unique_files': unique_files,
            'duplicates': duplicate_count
        }
    
    def check_structure(self):
        """Check if dataset has the expected 7 emotion folders in each split"""
        results = {}
        
        for split in self.splits:
            split_path = self.dataset_path / split
            results[split] = {
                'split_exists': split_path.exists() and split_path.is_dir(),
                'emotions_found': [],
                'emotions_missing': [],
                'has_all_7_emotions': False,
                'picture_counts': {}
            }
            
            if results[split]['split_exists']:
                for emotion in self.expected_emotions:
                    emotion_path = split_path / emotion
                    if emotion_path.exists() and emotion_path.is_dir():
                        results[split]['emotions_found'].append(emotion)
                    else:
                        results[split]['emotions_missing'].append(emotion)
                
                results[split]['has_all_7_emotions'] = len(results[split]['emotions_found']) == 7
                
                # Count pictures in this split
                results[split]['picture_counts'] = self.count_pictures_in_split(split)
                
        return results
    
    def count_pictures_in_split(self, split):
        """Count pictures in each emotion folder for a given split"""
        split_path = self.dataset_path / split
        counts = {}
        processed_in_split = 0
        
        if split_path.exists() and split_path.is_dir():
            for emotion in self.expected_emotions:
                emotion_path = split_path / emotion
                count = 0
                if emotion_path.exists() and emotion_path.is_dir():
                    for file_path in emotion_path.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                            count += 1
                            processed_in_split += 1
                            # Track file formats
                            self.file_formats[file_path.suffix.lower()] += 1
                            # Check if image is readable
                            self.check_image_readability(file_path)
                            
                            # Progress update every 1000 files (across all splits)
                            total_processed = sum(self.file_formats.values())
                            if total_processed % 1000 == 0 and total_processed > 0:
                                print(f"  Processed {total_processed:,} files for readability check...")
                                
                counts[emotion] = count
        
        return counts
    
    def check_image_readability(self, file_path):
        """Check if an image file can be opened and is not corrupted"""
        try:
            with Image.open(file_path) as img:
                img.verify()  # Verify the image integrity
        except Exception as e:
            self.corrupted_files.append(str(file_path))
    
    def analyze_class_balance(self, structure_results):
        """Analyze class balance across all splits and flag severe imbalances"""
        total_counts = defaultdict(int)
        
        # Aggregate counts across all splits
        for split_result in structure_results.values():
            if 'picture_counts' in split_result:
                for emotion, count in split_result['picture_counts'].items():
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
                'severely_underrepresented': percentage < 5,  # Less than 5%
                'severely_overrepresented': percentage > 50   # More than 50%
            }
        
        return balance_analysis
    
    def check_file_format_consistency(self):
        """Check file format consistency across the dataset"""
        format_analysis = {
            'formats_found': dict(self.file_formats),
            'is_consistent': len(self.file_formats) <= 1,
            'dominant_format': max(self.file_formats.items(), key=lambda x: x[1])[0] if self.file_formats else None,
            'total_files': sum(self.file_formats.values())
        }
        return format_analysis
    
    def generate_report(self):
        """Generate a readable text report"""
        print("Starting dataset structure analysis and readability check...")
        structure_results = self.check_structure()
        print("Structure analysis completed.")
        
        report_lines = []
        report_lines.append(f"Expected emotions (7): {', '.join(self.expected_emotions)}")
        report_lines.append("")
        
        all_splits_ok = True
        
        for split in self.splits:
            result = structure_results[split]
            report_lines.append(f"{split.upper()} SPLIT:")
            report_lines.append("-" * 20)
            
            if not result['split_exists']:
                report_lines.append("  STATUS: MISSING - Split folder does not exist")
                all_splits_ok = False
            else:
                status = "PASS" if result['has_all_7_emotions'] else "FAIL"
                report_lines.append(f"  STATUS: {status}")
                report_lines.append(f"  Emotions found: {len(result['emotions_found'])}/7")
                
                if result['emotions_found']:
                    report_lines.append(f"  Present: {', '.join(sorted(result['emotions_found']))}")
                
                if result['emotions_missing']:
                    report_lines.append(f"  Missing: {', '.join(sorted(result['emotions_missing']))}")
                    all_splits_ok = False
                
                # Add picture counts
                if result['picture_counts']:
                    report_lines.append("  Picture counts:")
                    total_pictures = 0
                    for emotion in sorted(self.expected_emotions):
                        count = result['picture_counts'].get(emotion, 0)
                        total_pictures += count
                        report_lines.append(f"    {emotion:>10}: {count:>6} pictures")
                    report_lines.append(f"    {'TOTAL':>10}: {total_pictures:>6} pictures")
                
        
        # Overall duplicate check (if enabled)
        if self.check_duplicates:
            print("Checking for duplicates across all splits...")
            overall_duplicates = self.check_duplicates_overall()
            report_lines.append("OVERALL DUPLICATE CHECK:")
            report_lines.append("-" * 25)
            report_lines.append(f"Total files across all splits: {overall_duplicates['total_files']}")
            report_lines.append(f"Unique files across all splits: {overall_duplicates['unique_files']}")
            report_lines.append(f"Duplicates across all splits: {overall_duplicates['duplicates']}")
            if overall_duplicates['duplicates'] > 0:
                report_lines.append("[WARNING] Duplicates found across splits!")
            else:
                report_lines.append("[PASS] No duplicates found across splits")
            report_lines.append("")
        
        # File format consistency check
        format_analysis = self.check_file_format_consistency()
        report_lines.append("FILE FORMAT ANALYSIS:")
        report_lines.append("-" * 20)
        if format_analysis['total_files'] > 0:
            report_lines.append(f"Total files processed: {format_analysis['total_files']}")
            report_lines.append(f"File formats found: {list(format_analysis['formats_found'].keys())}")
            for fmt, count in format_analysis['formats_found'].items():
                percentage = (count / format_analysis['total_files']) * 100
                report_lines.append(f"  {fmt}: {count} files ({percentage:.1f}%)")
            
            if format_analysis['is_consistent']:
                report_lines.append("[PASS] All files use consistent format")
            else:
                report_lines.append(f"[WARNING] Multiple formats detected. Dominant: {format_analysis['dominant_format']}")
        else:
            report_lines.append("No files found for format analysis")
        report_lines.append("")
        
        # Class balance analysis
        balance_analysis = self.analyze_class_balance(structure_results)
        if balance_analysis:
            report_lines.append("CLASS BALANCE ANALYSIS:")
            report_lines.append("-" * 25)
            imbalance_warnings = []
            
            for emotion in sorted(self.expected_emotions):
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
        
        # Corrupted files check
        if self.corrupted_files:
            report_lines.append("CORRUPTED FILES DETECTED:")
            report_lines.append("-" * 25)
            report_lines.append(f"[WARNING] Found {len(self.corrupted_files)} corrupted/unreadable files:")
            for corrupted_file in self.corrupted_files[:10]:  # Show first 10
                report_lines.append(f"  - {corrupted_file}")
            if len(self.corrupted_files) > 10:
                report_lines.append(f"  ... and {len(self.corrupted_files) - 10} more")
            report_lines.append("")
        else:
            report_lines.append("IMAGE READABILITY CHECK:")
            report_lines.append("-" * 25)
            report_lines.append("[PASS] All image files are readable and not corrupted")
            report_lines.append("")
        
        # Overall summary
        report_lines.append("OVERALL SUMMARY:")
        report_lines.append("-" * 20)
        if all_splits_ok:
            report_lines.append("[PASS] All splits have the required 7 emotion folders")
        else:
            report_lines.append("[FAIL] Some splits are missing or incomplete")
        
        return "\n".join(report_lines)

def main():
    
    # Check for command line arguments
    check_duplicates = '--duplicates' in sys.argv
    
    dataset_path = Path(__file__).parent.parent.parent / "data" / "raw" / "Faces_Dataset"
    
    if not dataset_path.exists():
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        return
    
    if check_duplicates:
        print("Note: Duplicate checking is enabled. This may take several minutes.")
    
    
    dqa = FacesDatasetDQA(dataset_path, check_duplicates=check_duplicates)
    report = dqa.generate_report()
    
    # Save report to text file in the same folder as the script
    output_path = Path(__file__).parent / "DQA_summary.txt"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

if __name__ == "__main__":
    main()