"""
Unified Data Organization Script for MAD-AD
===========================================

This script provides a unified workflow to:
1. Process multiple datasets (IXI, MOOD, BraTS)
2. Organize them into the required folder structure
3. Merge training and validation sets from multiple sources
4. Set up test data from BraTS

Usage:
    python organize_datasets.py --config datasets_config.json --output_dir ./data

Or run interactively:
    python organize_datasets.py --interactive
"""

import os
import argparse
import json
import shutil
from pathlib import Path
from glob import glob
import numpy as np


def merge_datasets(source_dirs, target_dir, split='train'):
    """
    Merge multiple processed datasets into a single directory.
    
    Args:
        source_dirs: List of source directories to merge
        target_dir: Target directory for merged data
        split: 'train', 'val', or 'test'
    """
    os.makedirs(target_dir, exist_ok=True)
    
    file_count = 0
    for source_dir in source_dirs:
        source_split_dir = os.path.join(source_dir, split)
        if not os.path.exists(source_split_dir):
            print(f"Warning: {source_split_dir} does not exist, skipping...")
            continue
        
        # Copy all files from source to target
        files = glob(os.path.join(source_split_dir, '*.png'))
        print(f"Copying {len(files)} files from {source_split_dir}...")
        
        for file_path in files:
            filename = os.path.basename(file_path)
            target_path = os.path.join(target_dir, filename)
            
            # Handle filename conflicts by adding dataset prefix
            if os.path.exists(target_path):
                dataset_name = os.path.basename(source_dir)
                new_filename = f"{dataset_name}_{filename}"
                target_path = os.path.join(target_dir, new_filename)
            
            shutil.copy2(file_path, target_path)
            file_count += 1
    
    print(f"Total {file_count} files copied to {target_dir}")
    return file_count


def validate_dataset_structure(data_dir):
    """
    Validate that the dataset has the required structure.
    
    Expected structure:
    data_dir/
        train/
            {id}-slice_{idx}-{modality}.png
            {id}-slice_{idx}-brainmask.png
        val/
            {id}-slice_{idx}-{modality}.png
            {id}-slice_{idx}-brainmask.png
        test/
            {id}-slice_{idx}-{modality}.png
            {id}-slice_{idx}-brainmask.png
            {id}-slice_{idx}-segmentation.png
    """
    required_splits = ['train', 'val', 'test']
    issues = []
    
    for split in required_splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            issues.append(f"Missing directory: {split_dir}")
            continue
        
        # Check for image files
        image_files = glob(os.path.join(split_dir, '*-T*.png'))
        brainmask_files = glob(os.path.join(split_dir, '*-brainmask.png'))
        
        if len(image_files) == 0:
            issues.append(f"No image files found in {split_dir}")
        
        if len(brainmask_files) == 0:
            issues.append(f"No brainmask files found in {split_dir}")
        
        # For test set, also check segmentation
        if split == 'test':
            seg_files = glob(os.path.join(split_dir, '*-segmentation.png'))
            if len(seg_files) == 0:
                issues.append(f"No segmentation files found in {split_dir}")
    
    if issues:
        print("Dataset validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("Dataset validation passed!")
        return True


def generate_dataset_statistics(data_dir, output_file='dataset_stats.json'):
    """Generate and save statistics about the organized dataset."""
    stats = {}
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        # Count different file types
        image_files = glob(os.path.join(split_dir, '*-T*.png'))
        brainmask_files = glob(os.path.join(split_dir, '*-brainmask.png'))
        seg_files = glob(os.path.join(split_dir, '*-segmentation.png'))
        
        # Extract modalities
        modalities = set()
        for img in image_files:
            basename = os.path.basename(img)
            # Extract modality (T1, T2, T1CE, FLAIR)
            if '-T1-' in basename or basename.endswith('-T1.png'):
                modalities.add('T1')
            elif '-T2-' in basename or basename.endswith('-T2.png'):
                modalities.add('T2')
            elif '-T1CE-' in basename or basename.endswith('-T1CE.png'):
                modalities.add('T1CE')
            elif '-FLAIR-' in basename or basename.endswith('-FLAIR.png'):
                modalities.add('FLAIR')
        
        stats[split] = {
            'total_images': len(image_files),
            'total_brainmasks': len(brainmask_files),
            'total_segmentations': len(seg_files),
            'modalities': sorted(list(modalities))
        }
    
    # Save statistics
    stats_path = os.path.join(data_dir, output_file)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset Statistics:")
    print(json.dumps(stats, indent=2))
    print(f"\nStatistics saved to: {stats_path}")
    
    return stats


def organize_datasets(config, output_dir):
    """
    Organize multiple datasets according to configuration.
    
    Args:
        config: Dictionary with dataset configuration
        output_dir: Final output directory
    """
    print("="*60)
    print("MAD-AD Dataset Organization")
    print("="*60)
    
    # Create output directory structure
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Process each dataset source
    train_sources = []
    val_sources = []
    test_sources = []
    
    for dataset_name, dataset_info in config.get('datasets', {}).items():
        print(f"\nProcessing {dataset_name}...")
        processed_dir = dataset_info.get('processed_dir')
        
        if not os.path.exists(processed_dir):
            print(f"  Warning: {processed_dir} does not exist, skipping...")
            continue
        
        # Determine which splits to use from this dataset
        use_for = dataset_info.get('use_for', ['train', 'val', 'test'])
        
        if 'train' in use_for:
            train_sources.append(processed_dir)
        if 'val' in use_for:
            val_sources.append(processed_dir)
        if 'test' in use_for:
            test_sources.append(processed_dir)
    
    # Merge datasets
    print("\n" + "="*60)
    print("Merging training data...")
    print("="*60)
    merge_datasets(train_sources, train_dir, split='train')
    
    print("\n" + "="*60)
    print("Merging validation data...")
    print("="*60)
    merge_datasets(val_sources, val_dir, split='val')
    
    print("\n" + "="*60)
    print("Merging test data...")
    print("="*60)
    merge_datasets(test_sources, test_dir, split='test')
    
    # Validate dataset structure
    print("\n" + "="*60)
    print("Validating dataset structure...")
    print("="*60)
    validate_dataset_structure(output_dir)
    
    # Generate statistics
    print("\n" + "="*60)
    print("Generating dataset statistics...")
    print("="*60)
    generate_dataset_statistics(output_dir)
    
    print("\n" + "="*60)
    print("Dataset organization complete!")
    print("="*60)


def interactive_mode():
    """Interactive mode for dataset organization."""
    print("="*60)
    print("MAD-AD Dataset Organization - Interactive Mode")
    print("="*60)
    
    config = {'datasets': {}}
    
    # Ask about IXI
    print("\n1. IXI Dataset")
    use_ixi = input("   Do you have processed IXI data? (y/n): ").lower() == 'y'
    if use_ixi:
        ixi_dir = input("   Enter path to processed IXI data: ").strip()
        use_for = input("   Use for (train/val/test, comma-separated): ").strip().split(',')
        use_for = [s.strip() for s in use_for]
        config['datasets']['IXI'] = {
            'processed_dir': ixi_dir,
            'use_for': use_for
        }
    
    # Ask about MOOD
    print("\n2. MOOD Dataset")
    use_mood = input("   Do you have processed MOOD data? (y/n): ").lower() == 'y'
    if use_mood:
        mood_dir = input("   Enter path to processed MOOD data: ").strip()
        use_for = input("   Use for (train/val/test, comma-separated): ").strip().split(',')
        use_for = [s.strip() for s in use_for]
        config['datasets']['MOOD'] = {
            'processed_dir': mood_dir,
            'use_for': use_for
        }
    
    # Ask about BraTS
    print("\n3. BraTS Dataset")
    use_brats = input("   Do you have processed BraTS data? (y/n): ").lower() == 'y'
    if use_brats:
        brats_dir = input("   Enter path to processed BraTS data: ").strip()
        use_for = input("   Use for (train/val/test, comma-separated): ").strip().split(',')
        use_for = [s.strip() for s in use_for]
        config['datasets']['BraTS'] = {
            'processed_dir': brats_dir,
            'use_for': use_for
        }
    
    # Output directory
    print("\n4. Output Configuration")
    output_dir = input("   Enter output directory for final organized data: ").strip()
    
    # Save config
    config_path = os.path.join(output_dir, 'organization_config.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n   Configuration saved to: {config_path}")
    
    # Run organization
    proceed = input("\n   Proceed with organization? (y/n): ").lower() == 'y'
    if proceed:
        organize_datasets(config, output_dir)
    else:
        print("   Organization cancelled.")


def main():
    parser = argparse.ArgumentParser(description='Organize datasets for MAD-AD')
    parser.add_argument('--config', type=str,
                       help='Path to JSON configuration file')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Output directory for organized data')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--validate_only', action='store_true',
                       help='Only validate existing dataset structure')
    
    args = parser.parse_args()
    
    if args.validate_only:
        print("Validating dataset structure...")
        validate_dataset_structure(args.output_dir)
        generate_dataset_statistics(args.output_dir)
        return
    
    if args.interactive:
        interactive_mode()
    elif args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        organize_datasets(config, args.output_dir)
    else:
        print("Error: Please provide either --config or --interactive")
        parser.print_help()


if __name__ == '__main__':
    main()
