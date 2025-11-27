# MAD-AD Custom Dataset Quick Reference

## 🚀 Quick Start (Automated)

If you have all datasets ready:

```bash
./prepare_datasets.sh \
    --ixi-root /path/to/IXI \
    --mood-root /path/to/MOOD \
    --brats-root /path/to/BraTS \
    --output-dir ./data \
    --modality T1
```

## 📝 Manual Step-by-Step

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements_data_prep.txt

# Optional but recommended
pip install antspyx
```

### 2. Process Each Dataset

**IXI (T1 and T2):**
```bash
python prepare_ixi_dataset.py \
    --ixi_root /path/to/IXI \
    --output_dir ./processed_data/ixi_t1 \
    --modality T1 \
    --image_size 256

python prepare_ixi_dataset.py \
    --ixi_root /path/to/IXI \
    --output_dir ./processed_data/ixi_t2 \
    --modality T2 \
    --image_size 256
```

**MOOD:**
```bash
python prepare_mood_dataset.py \
    --mood_root /path/to/MOOD \
    --output_dir ./processed_data/mood \
    --image_size 256
```

**BraTS:**
```bash
python prepare_brats_dataset.py \
    --brats_root /path/to/BraTS \
    --output_dir ./processed_data/brats \
    --modality T1 \
    --image_size 256
```

### 3. Organize All Datasets
```bash
python organize_datasets.py \
    --config datasets_config_example.json \
    --output_dir ./data
```

### 4. Validate
```bash
python organize_datasets.py --validate_only --output_dir ./data
```

## 🎯 Training

**Single GPU:**
```bash
python train-MAD-AD.py \
    --modality T1 \
    --model UNet_L \
    --data-root ./data \
    --image-size 256 \
    --augmentation True \
    --epochs 200 \
    --global-batch-size 32
```

**Multi-GPU (Recommended):**
```bash
torchrun --nproc_per_node=2 train-MAD-AD.py \
    --modality T1 \
    --model UNet_L \
    --data-root ./data \
    --image-size 256 \
    --augmentation True \
    --epochs 200 \
    --global-batch-size 96
```

## 🧪 Evaluation

```bash
python evaluate-MAD-AD.py \
    --data-root ./data \
    --model-path ./MAD-AD_T1_UNet_L/001-UNet_L/checkpoints/best_mse.pt \
    --ddim-steps 10 \
    --batch-size 16
```

## 📊 Dataset Structure

Your final `./data` directory should contain:

```
data/
├── train/          # Healthy subjects from IXI + MOOD
├── val/            # Healthy subjects from IXI + MOOD
├── test/           # BraTS with tumors (anomalies)
└── dataset_stats.json
```

## 🔧 Common Options

### Model Sizes
- `UNet_XS`: Smallest, fastest
- `UNet_S`: Small
- `UNet_M`: Medium
- `UNet_L`: Large (recommended, used in paper)
- `UNet_XL`: Largest

### Modalities
- `T1`: T1-weighted
- `T2`: T2-weighted
- `T1CE`: T1 contrast-enhanced (BraTS)
- `FLAIR`: FLAIR (BraTS)

### Image Sizes
- `256`: Default, balanced
- `128`: Faster, lower quality
- `512`: Higher quality, more memory

## 💡 Tips

1. **Start small**: Test with a subset of data first
2. **Use registration**: Better results with MNI registration (requires ANTsPy)
3. **Monitor training**: Check validation loss regularly
4. **Save checkpoints**: Use `--ckpt-every 20` to save regularly
5. **Augmentation**: Enable for better generalization

## 🐛 Troubleshooting

**Out of memory during training:**
- Reduce `--global-batch-size`
- Use smaller model (`UNet_M` or `UNet_S`)
- Reduce `--image-size`

**No images found:**
- Check dataset paths
- Verify NIfTI file extensions (`.nii` or `.nii.gz`)
- Use `--no_filter` for MOOD if needed

**Registration fails:**
- Install ANTsPy: `pip install antspyx`
- Or use `--no_registration` flag (faster but less standardized)

**Low performance:**
- Ensure test set has actual anomalies (tumors)
- Check data quality and preprocessing
- Try different DDIM steps (5, 10, 20)

## 📚 Key Files

- `DATASET_PREPARATION.md`: Detailed guide
- `prepare_ixi_dataset.py`: IXI preprocessing
- `prepare_mood_dataset.py`: MOOD preprocessing
- `prepare_brats_dataset.py`: BraTS preprocessing
- `organize_datasets.py`: Dataset organization
- `prepare_datasets.sh`: Automated pipeline
- `datasets_config_example.json`: Configuration template

## 📈 Expected Results

From the paper (BraTS dataset):
- AUROC: ~0.95
- Dice Score: ~0.70-0.80
- AP: ~0.85

Results may vary based on:
- Dataset quality
- Preprocessing choices
- Model size
- Training duration

## 🔗 Resources

- **Paper**: https://arxiv.org/abs/2502.16943
- **IXI**: https://brain-development.org/ixi-dataset/
- **MOOD**: http://medicalood.dkfz.de/
- **BraTS**: http://braintumorsegmentation.org/
