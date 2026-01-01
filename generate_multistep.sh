#!/bin/bash

# Multi-step proxy image generation script for SDXL Base
# Each inference step is run SEPARATELY (1, 4, 8, 16, 32)

# Configuration
MODEL_PATH="stabilityai/stable-diffusion-xl-base-1.0"
JSON_FILE="./test1.json"
OUTPUT_BASE_PATH="./output"
NUM_PROMPTS=5
IMG_PER_PROMPT=1
INFERENCE_STEPS="1 4 8 16 32"
GUIDANCE_SCALE=7.5

# Multi-GPU settings (set GPU_NUM to number of GPUs you want to use)
GPU_NUM=1

echo "========================================================================"
echo "SDXL Base Multi-Step Proxy Image Generation (Separate Execution)"
echo "========================================================================"
echo "Model:            ${MODEL_PATH}"
echo "Inference Steps:  ${INFERENCE_STEPS}"
echo "Guidance Scale:   ${GUIDANCE_SCALE}"
echo "Number of GPUs:   ${GPU_NUM}"
echo "Prompts per ID:   ${NUM_PROMPTS}"
echo "Images per prompt: ${IMG_PER_PROMPT}"
echo ""
echo "Note: Each step runs separately (e.g., step 1=1 inference, step 32=32 inferences)"
echo "      Total steps: 1 + 4 + 8 + 16 + 32 = 61 steps per image"
echo "========================================================================"

# Launch processes for each GPU
for ((i=0; i<${GPU_NUM}; i++)); do
    echo "Starting GPU ${i}..."
    CUDA_VISIBLE_DEVICES=${i} python generate_proxy_multistep.py \
        --model_path ${MODEL_PATH} \
        --json_file ${JSON_FILE} \
        --output_base_path ${OUTPUT_BASE_PATH} \
        --num_prompts ${NUM_PROMPTS} \
        --img_per_prompt ${IMG_PER_PROMPT} \
        --inference_steps ${INFERENCE_STEPS} \
        --guidance_scale ${GUIDANCE_SCALE} \
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
for step in ${INFERENCE_STEPS}; do
    DIR="${OUTPUT_BASE_PATH}/proxy_images_sdxl_step${step}"
    if [ -d "${DIR}" ]; then
        COUNT=$(find ${DIR}/combined -name "*.jpg" 2>/dev/null | wc -l)
        echo "  - ${DIR}/combined (${COUNT} images)"
    fi
done
