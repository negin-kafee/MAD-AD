# Summary of Dataset Preparation Tools

This document summarizes all the tools created for adapting MAD-AD to your custom datasets (IXI, MOOD, BraTS).

## 📦 Files Created

### Main Processing Scripts

1. **`prepare_ixi_dataset.py`**
   - Processes IXI T1 and T2 MRI images
   - Performs MNI registration (optional)
   - Creates brain masks
   - Extracts and normalizes axial slices
   - Splits into train/val sets

2. **`prepare_mood_dataset.py`**
   - Processes MOOD brain MRI dataset
   - Filters brain scans from abdominal scans
   - Creates brain masks
   - Extracts and normalizes axial slices
   - Splits into train/val sets

3. **`prepare_brats_dataset.py`**
   - Processes BraTS dataset for testing
   - Handles tumor segmentation masks
   - Supports all BraTS modalities (T1, T1CE, T2, FLAIR)
   - Creates test set with anomalies

4. **`organize_datasets.py`**
   - Merges multiple processed datasets
   - Organizes into final train/val/test structure
   - Validates dataset completeness
   - Generates dataset statistics
   - Supports interactive and config-based modes

### Automation & Configuration

5. **`prepare_datasets.sh`**
   - Automated end-to-end pipeline
   - Processes all datasets in sequence
   - Generates configuration automatically
   - Validates final output
   - Color-coded progress reporting

6. **`datasets_config_example.json`**
   - Example configuration file
   - Shows how to specify multiple datasets
   - Defines which datasets to use for train/val/test

### Documentation

7. **`DATASET_PREPARATION.md`**
   - Comprehensive preparation guide
   - Step-by-step instructions
   - Troubleshooting section
   - Dataset source information

8. **`QUICKSTART.md`**
   - Quick reference guide
   - Common commands
   - Training and evaluation examples
   - Tips and best practices

9. **`requirements_data_prep.txt`**
   - Additional Python dependencies for data processing
   - Installation instructions for optional tools

10. **`SUMMARY.md`** (this file)
    - Overview of all created files
    - Usage workflow
    - Key features

## 🎯 Complete Workflow

### Option 1: Automated (Recommended for beginners)

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -r requirements_data_prep.txt

# 2. Run automated pipeline
./prepare_datasets.sh \
    --ixi-root /path/to/IXI \
    --mood-root /path/to/MOOD \
    --brats-root /path/to/BraTS \
    --output-dir ./data \
    --modality T1

# 3. Train model
torchrun train-MAD-AD.py \
    --data-root ./data \
    --modality T1 \
    --model UNet_L

# 4. Evaluate model
python evaluate-MAD-AD.py \
    --data-root ./data \
    --model-path ./MAD-AD_T1_UNet_L/001-UNet_L/checkpoints/best_mse.pt
```

### Option 2: Manual (More control)

```bash
# 1. Process each dataset individually
python prepare_ixi_dataset.py --ixi_root /path/to/IXI --output_dir ./processed/ixi_t1 --modality T1
python prepare_mood_dataset.py --mood_root /path/to/MOOD --output_dir ./processed/mood
python prepare_brats_dataset.py --brats_root /path/to/BraTS --output_dir ./processed/brats --modality T1

# 2. Organize datasets
python organize_datasets.py --config datasets_config.json --output_dir ./data

# 3. Validate
python organize_datasets.py --validate_only --output_dir ./data

# 4. Train and evaluate (same as Option 1)
```

### Option 3: Interactive (Easiest for exploration)

```bash
# Interactive dataset organization
python organize_datasets.py --interactive
```

## ✨ Key Features

### Data Processing
- ✅ MNI registration support (optional, requires ANTsPy)
- ✅ Automatic brain masking
- ✅ Intensity normalization
- ✅ Axial slice extraction
- ✅ Configurable image sizes
- ✅ Train/val/test splitting

### Dataset Support
- ✅ IXI (T1, T2)
- ✅ MOOD (brain MRI)
- ✅ BraTS (all modalities with segmentation)
- ✅ Extensible to other datasets

### Organization
- ✅ Automatic merging of multiple datasets
- ✅ Conflict resolution for duplicate filenames
- ✅ Validation of final structure
- ✅ Statistical reporting

### Usability
- ✅ Fully automated pipeline
- ✅ Manual control options
- ✅ Interactive mode
- ✅ Comprehensive documentation
- ✅ Error handling and reporting

## 📋 Requirements

### Essential
- Python 3.11+
- nibabel (NIfTI file handling)
- numpy, scipy (numerical operations)
- Pillow (image processing)
- scikit-image (image transformations)
- tqdm (progress bars)

### Optional but Recommended
- ANTsPy or ANTs CLI (MNI registration)
  - Better standardization
  - More robust preprocessing
  - Can skip with `--no_registration` flag

### Training Requirements
- PyTorch with CUDA support
- All packages from main `requirements.txt`

## 🎓 Expected Dataset Sizes

After processing, expect approximately:

- **IXI T1**: ~5,000-8,000 slices (train+val)
- **IXI T2**: ~5,000-8,000 slices (train+val)
- **MOOD**: Variable (depends on subset)
- **BraTS**: ~10,000-15,000 slices (test)

Total storage: ~1-3 GB for processed PNG files

## 💡 Best Practices

1. **Start with a subset**: Test pipeline on small sample first
2. **Use registration**: Better results with MNI registration
3. **Quality check**: Review some output images manually
4. **Save configs**: Keep your `datasets_config.json` for reproducibility
5. **Version control**: Track which datasets and versions you used
6. **Monitor disk space**: Processing creates temporary files
7. **Backup raw data**: Keep original NIfTI files

## 🔍 Validation Checklist

Before training, verify:

- [ ] `data/train/` contains images and brainmasks (no segmentations)
- [ ] `data/val/` contains images and brainmasks (no segmentations)
- [ ] `data/test/` contains images, brainmasks, AND segmentations
- [ ] File naming follows pattern: `{id}-slice_{idx}-{modality}.png`
- [ ] Brain masks are binary (0 and 255)
- [ ] Segmentations show tumor regions in test set
- [ ] `dataset_stats.json` shows reasonable numbers
- [ ] Images are 256×256 (or your specified size)

## 🚨 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Import errors | Install `requirements_data_prep.txt` |
| ANTs not found | Install ANTsPy or use `--no_registration` |
| Out of memory | Process datasets one at a time |
| No images found | Check dataset paths and file extensions |
| Slow processing | Disable registration or use smaller subset |
| Validation fails | Check file naming and directory structure |

## 📞 Getting Help

1. Read `DATASET_PREPARATION.md` for detailed instructions
2. Check `QUICKSTART.md` for common commands
3. Review error messages carefully
4. Validate dataset structure with `--validate_only`
5. Check the main README.md and paper

## 🎉 Success Indicators

You're ready to train when:

1. ✅ All processing scripts completed without errors
2. ✅ Dataset validation passes
3. ✅ Statistics show reasonable numbers
4. ✅ Manual inspection of a few images looks good
5. ✅ Train/val have only healthy data
6. ✅ Test has anomalous data (tumors)

## 🔄 Next Steps After Setup

1. **Train model**: Use `train-MAD-AD.py` with your data
2. **Monitor training**: Check validation loss convergence
3. **Evaluate**: Test on BraTS to measure anomaly detection
4. **Tune**: Adjust hyperparameters if needed
5. **Deploy**: Use trained model for anomaly detection

## 📚 Additional Resources

- Original MAD-AD paper: https://arxiv.org/abs/2502.16943
- IXI dataset: https://brain-development.org/ixi-dataset/
- MOOD challenge: http://medicalood.dkfz.de/
- BraTS challenge: http://braintumorsegmentation.org/
- ANTs software: https://github.com/ANTsX/ANTs

---

**Created**: November 2025
**Purpose**: Adapt MAD-AD for custom MRI datasets (IXI, MOOD, BraTS)
**Status**: Ready for use ✅
