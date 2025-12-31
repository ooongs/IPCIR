#!/bin/bash

# Callback-based multi-step proxy image generation script
# Runs max_steps ONCE and saves intermediate results at specified steps

# Configuration
MODEL_TYPE="sdxl"  # or "flux"
JSON_FILE="./test1.json"
OUTPUT_BASE_PATH="./output"
NUM_PROMPTS=5
IMG_PER_PROMPT=1
MAX_INFERENCE_STEPS=32
SAVE_STEPS="1 4 8 16 32"  # Steps at which to save intermediate results

# Multi-GPU settings (set GPU_NUM to number of GPUs you want to use)
GPU_NUM=1

echo "========================================================================"
echo "Callback-based Multi-Step Proxy Image Generation"
echo "========================================================================"
echo "Model Type:           ${MODEL_TYPE}"
echo "Max Inference Steps:  ${MAX_INFERENCE_STEPS}"
echo "Save at Steps:        ${SAVE_STEPS}"
echo "Number of GPUs:       ${GPU_NUM}"
echo "Prompts per ID:       ${NUM_PROMPTS}"
echo "Images per prompt:    ${IMG_PER_PROMPT}"
echo ""
echo "Note: Running ${MAX_INFERENCE_STEPS} steps ONCE and saving intermediate results"
echo "      This is much faster than running each step separately!"
echo "========================================================================"

# Launch processes for each GPU
for ((i=0; i<${GPU_NUM}; i++)); do
    echo "Starting GPU ${i}..."
    CUDA_VISIBLE_DEVICES=${i} python generate_proxy_multistep_callback.py \
        --model_type ${MODEL_TYPE} \
        --json_file ${JSON_FILE} \
        --output_base_path ${OUTPUT_BASE_PATH} \
        --num_prompts ${NUM_PROMPTS} \
        --img_per_prompt ${IMG_PER_PROMPT} \
        --max_inference_steps ${MAX_INFERENCE_STEPS} \
        --save_steps ${SAVE_STEPS} \
        --idx ${i} \
        --gpu_num ${GPU_NUM} &
done

# Wait for all background processes to complete
wait

echo ""
echo "========================================================================"
echo "All processes completed!"
echo "========================================================================"
echo ""
echo "Generated directories:"
for step in ${SAVE_STEPS}; do
    DIR="${OUTPUT_BASE_PATH}/proxy_images_${MODEL_TYPE}_step${step}"
    if [ -d "${DIR}" ]; then
        COUNT=$(find ${DIR}/combined -name "*.jpg" 2>/dev/null | wc -l)
        echo "  - ${DIR}/combined (${COUNT} images)"
    fi
done
