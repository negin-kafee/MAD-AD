"""
Preprocess BraTS Dataset for MAD-AD Testing
===========================================

This script preprocesses BraTS (Brain Tumor Segmentation) dataset for testing:
1. Registers images to MNI152 1mm space (optional)
2. Creates brain masks
3. Normalizes intensities
4. Extracts axial slices as PNG images with segmentation masks
5. Organizes into test folder

BraTS contains multi-modal brain MRI (T1, T1CE, T2, FLAIR) with tumor segmentation masks.
This is used for TESTING anomaly detection (tumors are anomalies).

Requirements:
- nibabel
- nipype (optional, for registration)
- ANTs (ANTsPy, optional)
- scikit-image
- numpy
- PIL

Usage:
    python prepare_brats_dataset.py --brats_root /path/to/BraTS --output_dir ./data --modality T1
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
    non_zero = volume[volume > 0]
    if len(non_zero) == 0:
        return volume
    
    p_low = np.percentile(non_zero, percentile_low)
    p_high = np.percentile(non_zero, percentile_high)
    volume = np.clip(volume, p_low, p_high)
    volume = (volume - p_low) / (p_high - p_low + 1e-8)
    return volume


def create_brain_mask_from_image(volume):
    """Create a brain mask from the image itself."""
    volume_norm = volume / (np.max(volume) + 1e-8)
    non_zero = volume_norm[volume_norm > 0]
    if len(non_zero) == 0:
        return np.zeros_like(volume, dtype=np.uint8)
    
    threshold = np.percentile(non_zero, 10)
    mask = (volume_norm > threshold).astype(np.uint8)
    
    from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
    mask = binary_fill_holes(mask)
    mask = binary_erosion(mask, iterations=1)
    mask = binary_dilation(mask, iterations=2)
    
    return mask.astype(np.uint8)


def load_brats_case(case_dir, modality='t1'):
    """
    Load BraTS case with specified modality and segmentation.
    
    BraTS naming convention:
    - {case_id}_{modality}.nii.gz (e.g., BraTS20_Training_001_t1.nii.gz)
    - {case_id}_seg.nii.gz (segmentation)
    
    Args:
        case_dir: Directory containing the case files
        modality: One of ['t1', 't1ce', 't2', 'flair']
    
    Returns:
        image_path, segmentation_path
    """
    modality = modality.lower()
    
    # Find image file
    image_files = glob(os.path.join(case_dir, f"*_{modality}.nii*"))
    if len(image_files) == 0:
        image_files = glob(os.path.join(case_dir, f"*{modality.upper()}.nii*"))
    if len(image_files) == 0:
        return None, None
    
    image_path = image_files[0]
    
    # Find segmentation file
    seg_files = glob(os.path.join(case_dir, "*seg.nii*"))
    if len(seg_files) == 0:
        print(f"Warning: No segmentation found for {case_dir}")
        return image_path, None
    
    seg_path = seg_files[0]
    
    return image_path, seg_path


def process_segmentation_mask(seg_volume):
    """
    Process BraTS segmentation mask.
    
    BraTS labels:
    - 0: Background
    - 1: Necrotic/Core
    - 2: Edema
    - 4: Enhancing tumor (in some versions)
    
    We combine all tumor regions into a binary mask.
    """
    # Convert all tumor labels to 1
    binary_mask = (seg_volume > 0).astype(np.uint8)
    return binary_mask


def register_to_mni(image_path, output_path, mni_template=None):
    """Register image to MNI152 1mm space using ANTs."""
    if not USE_ANTS:
        # Just normalize without registration
        img = nib.load(image_path)
        data = img.get_fdata()
        data_norm = normalize_intensity(data)
        
        nib.save(nib.Nifti1Image(data_norm, img.affine), output_path)
        
        mask = create_brain_mask_from_image(data)
        mask_path = output_path.replace('.nii.gz', '_brainmask.nii.gz')
        nib.save(nib.Nifti1Image(mask, img.affine), mask_path)
        
        return output_path, mask_path
    
    # Load image with ANTs
    img = ants.image_read(str(image_path))
    
    # Load or get MNI template
    if mni_template is None:
        mni_template = ants.get_ants_data('mni')
    
    template = ants.image_read(str(mni_template))
    
    # Perform registration
    print(f"Registering {os.path.basename(image_path)} to MNI space...")
    registration = ants.registration(
        fixed=template,
        moving=img,
        type_of_transform='SyN',
        verbose=False
    )
    
    registered_img = registration['warpedmovout']
    
    # Normalize intensity
    data = registered_img.numpy()
    data_norm = normalize_intensity(data)
    registered_img_norm = ants.from_numpy(data_norm, 
                                          origin=registered_img.origin,
                                          spacing=registered_img.spacing,
                                          direction=registered_img.direction)
    
    ants.image_write(registered_img_norm, str(output_path))
    
    # Create brain mask
    mask = create_brain_mask_from_image(data_norm)
    mask_img = ants.from_numpy(mask.astype(np.float32),
                               origin=registered_img.origin,
                               spacing=registered_img.spacing,
                               direction=registered_img.direction)
    
    mask_path = output_path.replace('.nii.gz', '_brainmask.nii.gz')
    ants.image_write(mask_img, str(mask_path))
    
    return output_path, mask_path


def register_segmentation(seg_path, transform_params, output_path):
    """Apply same transformation to segmentation mask."""
    if not USE_ANTS:
        # Just save as-is
        img = nib.load(seg_path)
        data = img.get_fdata()
        data_binary = process_segmentation_mask(data)
        nib.save(nib.Nifti1Image(data_binary, img.affine), output_path)
        return output_path
    
    # This would require saving and reusing transformation from registration
    # For simplicity, we'll just process the segmentation in the original space
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()
    seg_binary = process_segmentation_mask(seg_data)
    nib.save(nib.Nifti1Image(seg_binary, seg_img.affine), output_path)
    return output_path


def extract_axial_slices_with_segmentation(volume_path, mask_path, seg_path, 
                                           output_dir, subject_id, modality, target_size=256):
    """
    Extract axial slices from 3D volume with segmentation masks.
    
    Only saves slices that contain some tumor (segmentation > 0) or healthy tissue.
    """
    # Load volume, mask, and segmentation
    volume_img = nib.load(volume_path)
    volume = volume_img.get_fdata()
    
    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata()
    
    seg_img = nib.load(seg_path) if seg_path else None
    seg = seg_img.get_fdata() if seg_img else np.zeros_like(volume)
    seg_binary = process_segmentation_mask(seg)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract slices that contain brain content
    n_slices = volume.shape[2]
    start_slice = int(n_slices * 0.1)  # Skip first 10%
    end_slice = int(n_slices * 0.9)    # Skip last 10%
    
    saved_count = 0
    
    for slice_idx in range(start_slice, end_slice):
        slice_2d = volume[:, :, slice_idx]
        mask_2d = mask[:, :, slice_idx]
        seg_2d = seg_binary[:, :, slice_idx]
        
        # Skip slices with very little brain content
        if np.sum(mask_2d) < 500:
            continue
        
        # Resize to target size
        from skimage.transform import resize
        slice_resized = resize(slice_2d, (target_size, target_size), 
                              mode='constant', preserve_range=True, anti_aliasing=True)
        mask_resized = resize(mask_2d, (target_size, target_size), 
                             mode='constant', preserve_range=True, anti_aliasing=False)
        seg_resized = resize(seg_2d, (target_size, target_size), 
                            mode='constant', preserve_range=True, anti_aliasing=False, order=0)
        
        # Convert to uint8
        slice_uint8 = (slice_resized * 255).astype(np.uint8)
        mask_uint8 = (mask_resized > 0.5).astype(np.uint8) * 255
        seg_uint8 = (seg_resized > 0.5).astype(np.uint8) * 255
        
        # Apply mask to image
        slice_uint8[mask_uint8 == 0] = 0
        
        # Save as PNG
        filename = f"{subject_id}-slice_{slice_idx:03d}-{modality}.png"
        mask_filename = f"{subject_id}-slice_{slice_idx:03d}-brainmask.png"
        seg_filename = f"{subject_id}-slice_{slice_idx:03d}-segmentation.png"
        
        Image.fromarray(slice_uint8, mode='L').save(os.path.join(output_dir, filename))
        Image.fromarray(mask_uint8, mode='L').save(os.path.join(output_dir, mask_filename))
        Image.fromarray(seg_uint8, mode='L').save(os.path.join(output_dir, seg_filename))
        
        saved_count += 1
    
    return saved_count


def process_brats_dataset(brats_root, output_dir, modality='T1', 
                          use_registration=False, mni_template=None, image_size=256):
    """
    Process BraTS dataset and organize into test folder.
    
    Args:
        brats_root: Root directory of BraTS dataset
        output_dir: Output directory for processed data
        modality: MRI modality (T1, T1CE, T2, FLAIR)
        use_registration: Whether to register to MNI space
        mni_template: Path to MNI template
        image_size: Target image size
    """
    modality_lower = modality.lower()
    
    # Find all case directories
    # BraTS structure typically: BraTS_root/BraTS20_Training_XXX/ or similar
    case_dirs = []
    for root, dirs, files in os.walk(brats_root):
        # Check if this directory contains NIfTI files
        if any(f.endswith('.nii.gz') or f.endswith('.nii') for f in files):
            case_dirs.append(root)
    
    # Remove duplicate parent directories
    case_dirs = sorted(list(set(case_dirs)))
    
    print(f"Found {len(case_dirs)} case directories in BraTS dataset")
    
    if len(case_dirs) == 0:
        raise ValueError(f"No valid case directories found in {brats_root}")
    
    # Create output directories
    test_dir = os.path.join(output_dir, 'test')
    temp_dir = os.path.join(output_dir, 'temp_registered')
    
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process each case
    stats = {'test': 0, 'cases_processed': 0}
    
    for case_idx, case_dir in enumerate(tqdm(case_dirs, desc="Processing BraTS cases")):
        # Load case data
        image_path, seg_path = load_brats_case(case_dir, modality_lower)
        
        if image_path is None:
            print(f"Skipping {case_dir}: {modality} not found")
            continue
        
        case_name = os.path.basename(case_dir)
        subject_id = f"BraTS_{case_idx:04d}"
        
        try:
            # Register to MNI (if enabled)
            if use_registration and USE_ANTS:
                registered_path = os.path.join(temp_dir, f"{subject_id}_registered.nii.gz")
                reg_img_path, mask_path = register_to_mni(image_path, registered_path, mni_template)
                
                # Register segmentation with same transform
                if seg_path:
                    seg_registered_path = os.path.join(temp_dir, f"{subject_id}_seg.nii.gz")
                    seg_path = register_segmentation(seg_path, None, seg_registered_path)
            else:
                # Just normalize and create mask
                img = nib.load(image_path)
                data = img.get_fdata()
                data_norm = normalize_intensity(data)
                
                reg_img_path = os.path.join(temp_dir, f"{subject_id}_normalized.nii.gz")
                nib.save(nib.Nifti1Image(data_norm, img.affine), reg_img_path)
                
                mask = create_brain_mask_from_image(data)
                mask_path = os.path.join(temp_dir, f"{subject_id}_brainmask.nii.gz")
                nib.save(nib.Nifti1Image(mask, img.affine), mask_path)
                
                # Process segmentation
                if seg_path:
                    seg_img = nib.load(seg_path)
                    seg_data = seg_img.get_fdata()
                    seg_binary = process_segmentation_mask(seg_data)
                    seg_norm_path = os.path.join(temp_dir, f"{subject_id}_seg.nii.gz")
                    nib.save(nib.Nifti1Image(seg_binary, seg_img.affine), seg_norm_path)
                    seg_path = seg_norm_path
            
            # Extract slices with segmentation
            n_slices = extract_axial_slices_with_segmentation(
                reg_img_path, mask_path, seg_path,
                test_dir, subject_id, modality, target_size=image_size
            )
            
            stats['test'] += n_slices
            stats['cases_processed'] += 1
            
        except Exception as e:
            print(f"Error processing {case_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save dataset statistics
    stats_file = os.path.join(output_dir, f'brats_{modality.lower()}_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset preparation complete!")
    print(f"Test slices: {stats['test']}")
    print(f"Cases processed: {stats['cases_processed']}")
    print(f"Statistics saved to: {stats_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Prepare BraTS dataset for MAD-AD testing')
    parser.add_argument('--brats_root', type=str, required=True,
                       help='Root directory of BraTS dataset')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Output directory for processed data')
    parser.add_argument('--modality', type=str, 
                       choices=['T1', 'T1CE', 'T2', 'FLAIR'], default='T1',
                       help='MRI modality to process')
    parser.add_argument('--use_registration', action='store_true',
                       help='Enable registration to MNI space (requires ANTsPy)')
    parser.add_argument('--mni_template', type=str, default=None,
                       help='Path to MNI template (optional)')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Target image size (default: 256)')
    
    args = parser.parse_args()
    
    process_brats_dataset(
        brats_root=args.brats_root,
        output_dir=args.output_dir,
        modality=args.modality,
        use_registration=args.use_registration,
        mni_template=args.mni_template,
        image_size=args.image_size
    )


if __name__ == '__main__':
    main()
