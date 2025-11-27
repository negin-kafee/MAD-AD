# Dataset Preparation Guide for MAD-AD

This guide explains how to prepare your own MRI datasets (IXI, MOOD, BraTS) for training and testing the MAD-AD model.

## 📋 Overview

The MAD-AD model requires data to be organized in a specific structure:

```
data/
├── train/
│   ├── {subject_id}-slice_{idx}-{modality}.png
│   ├── {subject_id}-slice_{idx}-brainmask.png
│   └── ...
├── val/
│   ├── {subject_id}-slice_{idx}-{modality}.png
│   ├── {subject_id}-slice_{idx}-brainmask.png
│   └── ...
└── test/
    ├── {subject_id}-slice_{idx}-{modality}.png
    ├── {subject_id}-slice_{idx}-brainmask.png
    ├── {subject_id}-slice_{idx}-segmentation.png
    └── ...
```

## 🔧 Prerequisites

### Required Python Packages

Install the required packages:

```bash
pip install nibabel numpy pillow scikit-image scipy tqdm
```

### Optional (for MNI Registration)

For better standardization, install ANTsPy for registration to MNI space:

```bash
pip install antspyx
```

Or install ANTs command-line tools:
- **macOS**: `brew install ants`
- **Linux**: Follow [ANTs installation guide](https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS)

## 📊 Dataset Sources

### 1. IXI Dataset
- **Source**: [IXI Dataset](https://brain-development.org/ixi-dataset/)
- **Description**: Brain MRI from healthy subjects (T1, T2, PD, MRA, DTI)
- **Use**: Training and validation (healthy/normal data)

### 2. MOOD Dataset
- **Source**: [MOOD Challenge](http://medicalood.dkfz.de/)
- **Description**: Medical Out-of-Distribution dataset with brain and abdominal CT/MRI
- **Use**: Training and validation (healthy/normal data)

### 3. BraTS Dataset
- **Source**: [BraTS Challenge](http://braintumorsegmentation.org/)
- **Description**: Brain MRI with tumor segmentation masks (T1, T1CE, T2, FLAIR)
- **Use**: Testing (tumors are anomalies to detect)

## 🚀 Step-by-Step Workflow

### Step 1: Download Raw Datasets

Download your chosen datasets to your local machine:

```bash
# Example directory structure
/path/to/datasets/
├── IXI/
│   ├── IXI001-T1.nii.gz
│   ├── IXI001-T2.nii.gz
│   └── ...
├── MOOD/
│   └── brain/
│       └── *.nii.gz
└── BraTS/
    ├── BraTS20_Training_001/
    │   ├── BraTS20_Training_001_t1.nii.gz
    │   ├── BraTS20_Training_001_t2.nii.gz
    │   ├── BraTS20_Training_001_seg.nii.gz
    │   └── ...
    └── ...
```

### Step 2: Preprocess IXI Dataset

Process IXI T1 and/or T2 images:

```bash
# Process IXI T1
python prepare_ixi_dataset.py \
    --ixi_root /path/to/IXI \
    --output_dir ./processed_data/ixi_t1 \
    --modality T1 \
    --train_ratio 0.8 \
    --image_size 256

# Process IXI T2
python prepare_ixi_dataset.py \
    --ixi_root /path/to/IXI \
    --output_dir ./processed_data/ixi_t2 \
    --modality T2 \
    --train_ratio 0.8 \
    --image_size 256
```

**Options:**
- `--ixi_root`: Root directory containing IXI NIfTI files
- `--output_dir`: Where to save processed data
- `--modality`: T1 or T2
- `--train_ratio`: Fraction for training (rest goes to validation)
- `--no_registration`: Skip MNI registration (faster, but less standardized)
- `--image_size`: Target image size (default: 256)

### Step 3: Preprocess MOOD Dataset

Process MOOD brain MRI images:

```bash
python prepare_mood_dataset.py \
    --mood_root /path/to/MOOD \
    --output_dir ./processed_data/mood \
    --modality T1 \
    --train_ratio 0.8 \
    --image_size 256
```

**Options:**
- `--mood_root`: Root directory containing MOOD data
- `--output_dir`: Where to save processed data
- `--modality`: Default modality label (MOOD may not specify)
- `--train_ratio`: Fraction for training
- `--use_registration`: Enable MNI registration (requires ANTsPy)
- `--no_filter`: Disable automatic brain-only filtering

### Step 4: Preprocess BraTS Dataset

Process BraTS images for testing:

```bash
python prepare_brats_dataset.py \
    --brats_root /path/to/BraTS \
    --output_dir ./processed_data/brats \
    --modality T1 \
    --image_size 256
```

**Options:**
- `--brats_root`: Root directory containing BraTS cases
- `--output_dir`: Where to save processed data
- `--modality`: T1, T1CE, T2, or FLAIR
- `--use_registration`: Enable MNI registration
- `--image_size`: Target image size (default: 256)

### Step 5: Organize All Datasets

Merge and organize all processed datasets:

#### Option A: Using Configuration File

1. Create a configuration file (see `datasets_config_example.json`):

```json
{
  "datasets": {
    "IXI_T1": {
      "processed_dir": "./processed_data/ixi_t1",
      "use_for": ["train", "val"]
    },
    "IXI_T2": {
      "processed_dir": "./processed_data/ixi_t2",
      "use_for": ["train", "val"]
    },
    "MOOD": {
      "processed_dir": "./processed_data/mood",
      "use_for": ["train", "val"]
    },
    "BraTS": {
      "processed_dir": "./processed_data/brats",
      "use_for": ["test"]
    }
  }
}
```

2. Run the organization script:

```bash
python organize_datasets.py \
    --config datasets_config.json \
    --output_dir ./data
```

#### Option B: Interactive Mode

```bash
python organize_datasets.py --interactive
```

Follow the prompts to specify your datasets and output location.

### Step 6: Validate Dataset

Validate the final dataset structure:

```bash
python organize_datasets.py \
    --validate_only \
    --output_dir ./data
```

This will check that all required files are present and generate statistics.

## 📁 Expected Final Structure

After all steps, your data directory should look like:

```
data/
├── train/
│   ├── IXI001-slice_045-T1.png
│   ├── IXI001-slice_045-brainmask.png
│   ├── IXI002-slice_050-T2.png
│   ├── IXI002-slice_050-brainmask.png
│   ├── MOOD_0001-slice_040-T1.png
│   ├── MOOD_0001-slice_040-brainmask.png
│   └── ...
├── val/
│   ├── IXI500-slice_048-T1.png
│   ├── IXI500-slice_048-brainmask.png
│   └── ...
├── test/
│   ├── BraTS_0001-slice_070-T1.png
│   ├── BraTS_0001-slice_070-brainmask.png
│   ├── BraTS_0001-slice_070-segmentation.png
│   └── ...
└── dataset_stats.json
```

## 🎯 Training the Model

Once your data is prepared, train MAD-AD:

```bash
# For T1 images
torchrun train-MAD-AD.py \
    --modality T1 \
    --model UNet_L \
    --mask-ratio 0.75 \
    --image-size 256 \
    --augmentation True \
    --data-root ./data/ \
    --ckpt-every 20 \
    --epochs 200

# For T2 images
torchrun train-MAD-AD.py \
    --modality T2 \
    --model UNet_L \
    --mask-ratio 0.75 \
    --image-size 256 \
    --augmentation True \
    --data-root ./data/ \
    --ckpt-every 20 \
    --epochs 200
```

## 🧪 Testing the Model

Evaluate on BraTS test set:

```bash
python evaluate-MAD-AD.py \
    --data-root ./data/ \
    --model-path ./MAD-AD_T1_UNet_L/001-UNet_L/checkpoints/best_mse.pt \
    --ddim-steps 10
```

## 📝 Important Notes

### Data Quality
- **Training data should only contain healthy/normal scans** (IXI, MOOD)
- **Test data should contain anomalies** (BraTS with tumors)
- Remove any corrupted or low-quality images

### Registration
- MNI registration improves standardization but is optional
- Without registration, the model can still work but may be less robust
- Registration requires ANTsPy or ANTs command-line tools

### Image Size
- Default is 256×256 pixels
- Can be changed but must match during training and testing
- Larger sizes require more memory

### Modalities
- Process each modality separately (T1, T2, etc.)
- Train separate models for each modality
- BraTS provides all modalities, choose what matches your training data

### Computational Requirements
- Dataset processing can take several hours depending on size
- Registration is the most time-consuming step
- Recommended: Process on a machine with multiple cores

## 🐛 Troubleshooting

### Issue: "No images found"
- Check that NIfTI files have `.nii` or `.nii.gz` extensions
- Verify the dataset directory structure
- Use `--no_filter` flag for MOOD if brain filtering is too aggressive

### Issue: "ANTsPy not found"
- Install ANTsPy: `pip install antspyx`
- Or use `--no_registration` flag to skip registration

### Issue: "Out of memory"
- Process datasets in smaller batches
- Reduce `--image_size` parameter
- Close other applications

### Issue: Filename conflicts during organization
- The script automatically handles conflicts by adding dataset prefixes
- Check the output for any warnings

## 📧 Support

For questions or issues:
1. Check the main README.md
2. Review the original paper
3. Check dataset source documentation

## 🔗 References

- **MAD-AD Paper**: [arXiv:2502.16943](https://arxiv.org/abs/2502.16943)
- **IXI Dataset**: https://brain-development.org/ixi-dataset/
- **MOOD Challenge**: http://medicalood.dkfz.de/
- **BraTS Challenge**: http://braintumorsegmentation.org/
