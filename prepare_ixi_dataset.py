"""
Preprocess IXI Dataset for MAD-AD Training
==========================================

This script preprocesses IXI T1 and T2 MRI images:
1. Registers images to MNI152 1mm space
2. Skull strips to create brain masks
3. Normalizes intensities
4. Extracts axial slices as PNG images
5. Organizes into train/val splits

Requirements:
- nibabel
- nipype
- ANTs (ANTsPy or command line)
- scikit-image
- numpy
- PIL

Usage:
    python prepare_ixi_dataset.py --ixi_root /path/to/IXI --output_dir ./data --modality T1 --train_ratio 0.8
"""

import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import nibabel as nib
from glob import glob
from tqdm import tqdm
import json

try:
    import ants
    USE_ANTS = True
except ImportError:
    USE_ANTS = False
    print("Warning: ANTsPy not found. Will use nibabel only (no registration).")

from scipy.ndimage import zoom
from skimage import exposure


def normalize_intensity(volume, percentile_low=1, percentile_high=99):
    """Normalize intensity using percentile-based clipping."""
    volume = volume.astype(np.float32)
    p_low = np.percentile(volume[volume > 0], percentile_low)
    p_high = np.percentile(volume[volume > 0], percentile_high)
    volume = np.clip(volume, p_low, p_high)
    volume = (volume - p_low) / (p_high - p_low + 1e-8)
    return volume


def create_simple_brain_mask(volume, threshold_percentile=20):
    """Create a simple brain mask using Otsu-like thresholding."""
    volume_norm = volume / (np.max(volume) + 1e-8)
    threshold = np.percentile(volume_norm[volume_norm > 0], threshold_percentile)
    mask = (volume_norm > threshold).astype(np.uint8)
    
    # Simple morphological operations to clean up mask
    from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
    mask = binary_fill_holes(mask)
    mask = binary_erosion(mask, iterations=2)
    mask = binary_dilation(mask, iterations=3)
    
    return mask.astype(np.uint8)


def register_to_mni(image_path, output_path, mni_template=None):
    """
    Register image to MNI152 1mm space using ANTs.
    
    Args:
        image_path: Path to input NIfTI image
        output_path: Path to save registered image
        mni_template: Path to MNI template (if None, uses ANTs default)
    
    Returns:
        Path to registered image, Path to brain mask
    """
    if not USE_ANTS:
        print(f"Skipping registration for {image_path} (ANTsPy not available)")
        # Just copy and normalize
        img = nib.load(image_path)
        data = img.get_fdata()
        data_norm = normalize_intensity(data)
        
        # Save normalized image
        nib.save(nib.Nifti1Image(data_norm, img.affine), output_path)
        
        # Create brain mask
        mask = create_simple_brain_mask(data)
        mask_path = output_path.replace('.nii.gz', '_brainmask.nii.gz')
        nib.save(nib.Nifti1Image(mask, img.affine), mask_path)
        
        return output_path, mask_path
    
    # Load image with ANTs
    img = ants.image_read(str(image_path))
    
    # Load or get MNI template
    if mni_template is None:
        # Use ANTs default MNI template
        mni_template = ants.get_ants_data('mni')
    
    template = ants.image_read(str(mni_template))
    
    # Perform registration
    print(f"Registering {os.path.basename(image_path)} to MNI space...")
    registration = ants.registration(
        fixed=template,
        moving=img,
        type_of_transform='SyN',  # Symmetric normalization
        verbose=False
    )
    
    # Apply transformation
    registered_img = registration['warpedmovout']
    
    # Normalize intensity
    data = registered_img.numpy()
    data_norm = normalize_intensity(data)
    registered_img_norm = ants.from_numpy(data_norm, 
                                          origin=registered_img.origin,
                                          spacing=registered_img.spacing,
                                          direction=registered_img.direction)
    
    # Save registered image
    ants.image_write(registered_img_norm, str(output_path))
    
    # Create brain mask using registered image
    mask = create_simple_brain_mask(data_norm)
    mask_img = ants.from_numpy(mask.astype(np.float32),
                               origin=registered_img.origin,
                               spacing=registered_img.spacing,
                               direction=registered_img.direction)
    
    mask_path = output_path.replace('.nii.gz', '_brainmask.nii.gz')
    ants.image_write(mask_img, str(mask_path))
    
    return output_path, mask_path


def extract_axial_slices(volume_path, mask_path, output_dir, subject_id, modality, target_size=256):
    """
    Extract axial slices from 3D volume and save as PNG images.
    
    Args:
        volume_path: Path to registered NIfTI volume
        mask_path: Path to brain mask NIfTI
        output_dir: Directory to save slices
        subject_id: Subject identifier
        modality: Imaging modality (T1 or T2)
        target_size: Target image size (default: 256x256)
    """
    # Load volume and mask
    volume_img = nib.load(volume_path)
    volume = volume_img.get_fdata()
    
    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract middle slices (typically contain more brain)
    n_slices = volume.shape[2]
    start_slice = int(n_slices * 0.2)  # Skip first 20%
    end_slice = int(n_slices * 0.8)    # Skip last 20%
    
    saved_count = 0
    
    for slice_idx in range(start_slice, end_slice):
        slice_2d = volume[:, :, slice_idx]
        mask_2d = mask[:, :, slice_idx]
        
        # Skip slices with very little brain content
        if np.sum(mask_2d) < 1000:  # Threshold for minimum brain pixels
            continue
        
        # Resize to target size
        from skimage.transform import resize
        slice_resized = resize(slice_2d, (target_size, target_size), 
                              mode='constant', preserve_range=True, anti_aliasing=True)
        mask_resized = resize(mask_2d, (target_size, target_size), 
                             mode='constant', preserve_range=True, anti_aliasing=False)
        
        # Convert to uint8
        slice_uint8 = (slice_resized * 255).astype(np.uint8)
        mask_uint8 = (mask_resized > 0.5).astype(np.uint8) * 255
        
        # Apply mask to image
        slice_uint8[mask_uint8 == 0] = 0
        
        # Save as PNG
        filename = f"{subject_id}-slice_{slice_idx:03d}-{modality}.png"
        mask_filename = f"{subject_id}-slice_{slice_idx:03d}-brainmask.png"
        
        Image.fromarray(slice_uint8, mode='L').save(os.path.join(output_dir, filename))
        Image.fromarray(mask_uint8, mode='L').save(os.path.join(output_dir, mask_filename))
        
        saved_count += 1
    
    return saved_count


def process_ixi_dataset(ixi_root, output_dir, modality='T1', train_ratio=0.8, 
                       use_registration=True, mni_template=None, image_size=256):
    """
    Process IXI dataset and organize into train/val splits.
    
    Args:
        ixi_root: Root directory of IXI dataset
        output_dir: Output directory for processed data
        modality: T1 or T2
        train_ratio: Ratio of training data (default: 0.8)
        use_registration: Whether to register to MNI space
        mni_template: Path to MNI template (if None, uses default)
        image_size: Target image size
    """
    # Find all images for specified modality
    pattern = os.path.join(ixi_root, f"**/*{modality}*.nii*")
    image_files = glob(pattern, recursive=True)
    
    if len(image_files) == 0:
        raise ValueError(f"No {modality} images found in {ixi_root}")
    
    print(f"Found {len(image_files)} {modality} images")
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    temp_dir = os.path.join(output_dir, 'temp_registered')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Split train/val
    np.random.seed(42)
    n_train = int(len(image_files) * train_ratio)
    indices = np.random.permutation(len(image_files))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # Process each image
    stats = {'train': 0, 'val': 0}
    
    for idx, image_path in enumerate(tqdm(image_files, desc="Processing IXI images")):
        subject_id = os.path.basename(image_path).split('.')[0]
        
        # Determine train or val
        split = 'train' if idx in train_indices else 'val'
        split_dir = train_dir if split == 'train' else val_dir
        
        try:
            # Register to MNI (if enabled)
            if use_registration and USE_ANTS:
                registered_path = os.path.join(temp_dir, f"{subject_id}_registered.nii.gz")
                reg_img_path, mask_path = register_to_mni(image_path, registered_path, mni_template)
            else:
                # Just normalize and create mask
                img = nib.load(image_path)
                data = img.get_fdata()
                data_norm = normalize_intensity(data)
                
                reg_img_path = os.path.join(temp_dir, f"{subject_id}_normalized.nii.gz")
                nib.save(nib.Nifti1Image(data_norm, img.affine), reg_img_path)
                
                mask = create_simple_brain_mask(data)
                mask_path = os.path.join(temp_dir, f"{subject_id}_brainmask.nii.gz")
                nib.save(nib.Nifti1Image(mask, img.affine), mask_path)
            
            # Extract slices
            n_slices = extract_axial_slices(
                reg_img_path, mask_path, split_dir, 
                subject_id, modality, target_size=image_size
            )
            
            stats[split] += n_slices
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # Save dataset statistics
    stats_file = os.path.join(output_dir, f'ixi_{modality.lower()}_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset preparation complete!")
    print(f"Training slices: {stats['train']}")
    print(f"Validation slices: {stats['val']}")
    print(f"Statistics saved to: {stats_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Prepare IXI dataset for MAD-AD')
    parser.add_argument('--ixi_root', type=str, required=True,
                       help='Root directory of IXI dataset')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Output directory for processed data')
    parser.add_argument('--modality', type=str, choices=['T1', 'T2'], default='T1',
                       help='MRI modality to process')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of training data (default: 0.8)')
    parser.add_argument('--no_registration', action='store_true',
                       help='Skip registration to MNI space (faster but less standardized)')
    parser.add_argument('--mni_template', type=str, default=None,
                       help='Path to MNI template (optional)')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Target image size (default: 256)')
    
    args = parser.parse_args()
    
    process_ixi_dataset(
        ixi_root=args.ixi_root,
        output_dir=args.output_dir,
        modality=args.modality,
        train_ratio=args.train_ratio,
        use_registration=not args.no_registration,
        mni_template=args.mni_template,
        image_size=args.image_size
    )


if __name__ == '__main__':
    main()
