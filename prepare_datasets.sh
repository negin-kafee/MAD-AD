#!/bin/bash

# Quick Start Script for MAD-AD Dataset Preparation
# This script automates the entire dataset preparation pipeline

set -e  # Exit on error

echo "=========================================="
echo "MAD-AD Dataset Preparation - Quick Start"
echo "=========================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
OUTPUT_DIR="./data"
PROCESSED_DIR="./processed_data"
IMAGE_SIZE=256
TRAIN_RATIO=0.8
USE_REGISTRATION=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ixi-root)
            IXI_ROOT="$2"
            shift 2
            ;;
        --mood-root)
            MOOD_ROOT="$2"
            shift 2
            ;;
        --brats-root)
            BRATS_ROOT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --image-size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --modality)
            MODALITY="$2"
            shift 2
            ;;
        --use-registration)
            USE_REGISTRATION=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --ixi-root DIR        Path to IXI dataset root"
            echo "  --mood-root DIR       Path to MOOD dataset root"
            echo "  --brats-root DIR      Path to BraTS dataset root"
            echo "  --output-dir DIR      Output directory (default: ./data)"
            echo "  --image-size SIZE     Image size (default: 256)"
            echo "  --modality MOD        Modality for BraTS (T1/T2/T1CE/FLAIR)"
            echo "  --use-registration    Enable MNI registration (requires ANTsPy)"
            echo "  --help                Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --ixi-root /data/IXI --brats-root /data/BraTS --modality T1"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if at least one dataset is provided
if [ -z "$IXI_ROOT" ] && [ -z "$MOOD_ROOT" ] && [ -z "$BRATS_ROOT" ]; then
    echo -e "${RED}Error: Please provide at least one dataset root directory${NC}"
    echo "Use --help for usage information"
    exit 1
fi

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROCESSED_DIR"

# Step 1: Process IXI if provided
if [ -n "$IXI_ROOT" ]; then
    echo -e "${BLUE}=========================================="
    echo "Step 1: Processing IXI Dataset"
    echo -e "==========================================${NC}"
    
    for mod in T1 T2; do
        echo -e "${GREEN}Processing IXI ${mod}...${NC}"
        
        REG_FLAG=""
        if [ "$USE_REGISTRATION" = false ]; then
            REG_FLAG="--no_registration"
        fi
        
        python prepare_ixi_dataset.py \
            --ixi_root "$IXI_ROOT" \
            --output_dir "${PROCESSED_DIR}/ixi_${mod,,}" \
            --modality "$mod" \
            --train_ratio "$TRAIN_RATIO" \
            --image_size "$IMAGE_SIZE" \
            $REG_FLAG
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ IXI ${mod} processing complete${NC}"
        else
            echo -e "${YELLOW}⚠ IXI ${mod} processing encountered issues${NC}"
        fi
    done
else
    echo -e "${YELLOW}Skipping IXI dataset (not provided)${NC}"
fi

# Step 2: Process MOOD if provided
if [ -n "$MOOD_ROOT" ]; then
    echo -e "${BLUE}=========================================="
    echo "Step 2: Processing MOOD Dataset"
    echo -e "==========================================${NC}"
    
    REG_FLAG=""
    if [ "$USE_REGISTRATION" = true ]; then
        REG_FLAG="--use_registration"
    fi
    
    python prepare_mood_dataset.py \
        --mood_root "$MOOD_ROOT" \
        --output_dir "${PROCESSED_DIR}/mood" \
        --modality "T1" \
        --train_ratio "$TRAIN_RATIO" \
        --image_size "$IMAGE_SIZE" \
        $REG_FLAG
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ MOOD processing complete${NC}"
    else
        echo -e "${YELLOW}⚠ MOOD processing encountered issues${NC}"
    fi
else
    echo -e "${YELLOW}Skipping MOOD dataset (not provided)${NC}"
fi

# Step 3: Process BraTS if provided
if [ -n "$BRATS_ROOT" ]; then
    echo -e "${BLUE}=========================================="
    echo "Step 3: Processing BraTS Dataset"
    echo -e "==========================================${NC}"
    
    # Default to T1 if not specified
    if [ -z "$MODALITY" ]; then
        MODALITY="T1"
    fi
    
    REG_FLAG=""
    if [ "$USE_REGISTRATION" = true ]; then
        REG_FLAG="--use_registration"
    fi
    
    python prepare_brats_dataset.py \
        --brats_root "$BRATS_ROOT" \
        --output_dir "${PROCESSED_DIR}/brats" \
        --modality "$MODALITY" \
        --image_size "$IMAGE_SIZE" \
        $REG_FLAG
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ BraTS processing complete${NC}"
    else
        echo -e "${YELLOW}⚠ BraTS processing encountered issues${NC}"
    fi
else
    echo -e "${YELLOW}Skipping BraTS dataset (not provided)${NC}"
fi

# Step 4: Generate configuration file
echo -e "${BLUE}=========================================="
echo "Step 4: Generating Configuration"
echo -e "==========================================${NC}"

CONFIG_FILE="${OUTPUT_DIR}/datasets_config.json"
echo "{" > "$CONFIG_FILE"
echo '  "datasets": {' >> "$CONFIG_FILE"

FIRST_DATASET=true

if [ -n "$IXI_ROOT" ]; then
    for mod in T1 T2; do
        if [ "$FIRST_DATASET" = false ]; then
            echo "," >> "$CONFIG_FILE"
        fi
        cat >> "$CONFIG_FILE" << EOF
    "IXI_${mod}": {
      "processed_dir": "${PROCESSED_DIR}/ixi_${mod,,}",
      "use_for": ["train", "val"],
      "description": "IXI ${mod}-weighted MRI images"
    }
EOF
        FIRST_DATASET=false
    done
fi

if [ -n "$MOOD_ROOT" ]; then
    if [ "$FIRST_DATASET" = false ]; then
        echo "," >> "$CONFIG_FILE"
    fi
    cat >> "$CONFIG_FILE" << EOF
    "MOOD": {
      "processed_dir": "${PROCESSED_DIR}/mood",
      "use_for": ["train", "val"],
      "description": "MOOD brain MRI images"
    }
EOF
    FIRST_DATASET=false
fi

if [ -n "$BRATS_ROOT" ]; then
    if [ "$FIRST_DATASET" = false ]; then
        echo "," >> "$CONFIG_FILE"
    fi
    cat >> "$CONFIG_FILE" << EOF
    "BraTS": {
      "processed_dir": "${PROCESSED_DIR}/brats",
      "use_for": ["test"],
      "description": "BraTS brain MRI with tumor segmentation"
    }
EOF
fi

cat >> "$CONFIG_FILE" << EOF

  }
}
EOF

echo -e "${GREEN}✓ Configuration saved to ${CONFIG_FILE}${NC}"

# Step 5: Organize datasets
echo -e "${BLUE}=========================================="
echo "Step 5: Organizing Datasets"
echo -e "==========================================${NC}"

python organize_datasets.py \
    --config "$CONFIG_FILE" \
    --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dataset organization complete${NC}"
else
    echo -e "${RED}✗ Dataset organization failed${NC}"
    exit 1
fi

# Step 6: Validate
echo -e "${BLUE}=========================================="
echo "Step 6: Validating Final Dataset"
echo -e "==========================================${NC}"

python organize_datasets.py \
    --validate_only \
    --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Validation passed${NC}"
else
    echo -e "${RED}✗ Validation failed${NC}"
    exit 1
fi

# Summary
echo ""
echo -e "${GREEN}=========================================="
echo "Dataset Preparation Complete!"
echo -e "==========================================${NC}"
echo ""
echo "Your organized dataset is ready at: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "1. Review the dataset statistics in ${OUTPUT_DIR}/dataset_stats.json"
echo "2. Train the model using:"
echo "   torchrun train-MAD-AD.py --data-root ${OUTPUT_DIR} --modality T1"
echo "3. Evaluate the model using:"
echo "   python evaluate-MAD-AD.py --data-root ${OUTPUT_DIR} --model-path <model_path>"
echo ""
