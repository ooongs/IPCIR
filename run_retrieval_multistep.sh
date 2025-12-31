#!/bin/bash

# Run retrieval for multiple inference steps
# This script tests each step's generated images

# Configuration
MODEL_TYPE="sdxl"  # or "flux"
OUTPUT_BASE_PATH="./output"
INFERENCE_STEPS=(1 4 8 16 32)
GPU_ID=0

# CIRCO dataset configuration
DATASET_PATH="/home/jinzhenxiong/Imagine-and-Seek/data/CIRCO"
LAYOUT_PATH="/home/jinzhenxiong/IPCIR/CIRCO/ipcir_layout.json"
TYPE="G"
EVAL_TYPE="LDRE-G"

# Fusion parameters
S_W=1.0
T_W=1.0
A_W=1.0
FUSION_WEIGHT=0.3

# Multi-scale testing (optional, comment out to disable)
MULTI_SCALE="--test_multiple_scales --scale_start 0.0 --scale_end 1.0 --scale_step 0.1"

echo "========================================================================"
echo "Multi-Step Retrieval Testing"
echo "========================================================================"
echo "Model Type:       ${MODEL_TYPE}"
echo "Inference Steps:  ${INFERENCE_STEPS[@]}"
echo "GPU:              ${GPU_ID}"
echo "Multi-scale:      ${MULTI_SCALE:-disabled}"
echo "========================================================================"
echo ""

# Run retrieval for each step
for step in "${INFERENCE_STEPS[@]}"; do
    AUG_DIR="${OUTPUT_BASE_PATH}/proxy_images_${MODEL_TYPE}_step${step}"
    SUBMISSION_NAME="circo_aug_${MODEL_TYPE}_step${step}"

    echo ""
    echo "========================================================================"
    echo "Processing Step ${step}"
    echo "========================================================================"
    echo "Aug Dir:          ${AUG_DIR}"
    echo "Submission Name:  ${SUBMISSION_NAME}"
    echo ""

    # Check if directory exists
    if [ ! -d "${AUG_DIR}/combined" ]; then
        echo "WARNING: Directory ${AUG_DIR}/combined not found, skipping..."
        continue
    fi

    # Check number of images
    IMG_COUNT=$(find ${AUG_DIR}/combined -name "*.jpg" 2>/dev/null | wc -l)
    echo "Found ${IMG_COUNT} images in ${AUG_DIR}/combined"

    if [ ${IMG_COUNT} -eq 0 ]; then
        echo "WARNING: No images found, skipping..."
        continue
    fi

    # Run retrieval
    echo "Starting retrieval..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python src/retrieval_circo.py \
        --submission-name ${SUBMISSION_NAME} \
        --dataset-path ${DATASET_PATH} \
        --s_w ${S_W} --t_w ${T_W} --a_w ${A_W} --fusion_weight ${FUSION_WEIGHT} \
        --aug_dir ${AUG_DIR} \
        --layout_path ${LAYOUT_PATH} \
        --type ${TYPE} --eval-type ${EVAL_TYPE} --with_aug \
        ${MULTI_SCALE}

    if [ $? -eq 0 ]; then
        echo "✓ Step ${step} completed successfully!"
    else
        echo "✗ Step ${step} failed!"
    fi

    echo ""
done

echo ""
echo "========================================================================"
echo "All steps completed!"
echo "========================================================================"
echo ""
echo "Submission files:"
ls -lh data/test_submissions/circo/circo_aug_${MODEL_TYPE}_step* 2>/dev/null || echo "No submission files found"
