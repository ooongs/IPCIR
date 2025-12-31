#!/bin/bash

# SDXL Proxy Image Generation Script
# This script runs image generation on multiple GPUs in parallel

# Configuration
MODEL_TYPE="flux"
SDXL_PATH="/home/jinzhenxiong/temp/stabilityai/sdxl-turbo"
FLUX_PATH="/home/jinzhenxiong/pretrain/black-forest-labs/FLUX.1-schnell"
JSON_FILE="/home/jinzhenxiong/Imagine-and-Seek/data/CIRCO/annotations/test.json"
OUTPUT_PATH="./output/proxy_images_${MODEL_TYPE}"
NUM_GPUS=3
IMG_PER_PROMPT=1
NUM_PROMPTS=5
NUM_INFERENCE_STEPS=4
GUIDANCE_SCALE=0.0

# GPU 0
CUDA_VISIBLE_DEVICES=5 python generate_proxy.py \
    --model_type $MODEL_TYPE \
    --sdxl_path $SDXL_PATH \
    --flux_path $FLUX_PATH \
    --json_file $JSON_FILE \
    --output_path $OUTPUT_PATH \
    --img_per_prompt $IMG_PER_PROMPT \
    --num_prompts $NUM_PROMPTS \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --idx 0 \
    --gpu_num $NUM_GPUS &

# GPU 1
CUDA_VISIBLE_DEVICES=6 python generate_proxy.py \
    --model_type $MODEL_TYPE \
    --sdxl_path $SDXL_PATH \
    --flux_path $FLUX_PATH \
    --json_file $JSON_FILE \
    --output_path $OUTPUT_PATH \
    --img_per_prompt $IMG_PER_PROMPT \
    --num_prompts $NUM_PROMPTS \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --idx 1 \
    --gpu_num $NUM_GPUS &

# GPU 2
CUDA_VISIBLE_DEVICES=7 python generate_proxy.py \
    --model_type $MODEL_TYPE \
    --sdxl_path $SDXL_PATH \
    --flux_path $FLUX_PATH \
    --json_file $JSON_FILE \
    --output_path $OUTPUT_PATH \
    --img_per_prompt $IMG_PER_PROMPT \
    --num_prompts $NUM_PROMPTS \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --idx 2 \
    --gpu_num $NUM_GPUS &

# # GPU 3
# CUDA_VISIBLE_DEVICES=7 python generate_proxy.py \
#     --model_type $MODEL_TYPE \
#     --sdxl_path $SDXL_PATH \
#     --flux_path $FLUX_PATH \
#     --json_file $JSON_FILE \
#     --output_path $OUTPUT_PATH \
#     --img_per_prompt $IMG_PER_PROMPT \
#     --num_prompts $NUM_PROMPTS \
#     --num_inference_steps $NUM_INFERENCE_STEPS \
#     --guidance_scale $GUIDANCE_SCALE \
#     --idx 3 \
#     --gpu_num $NUM_GPUS &

# Wait for all processes to complete
wait
echo "All proxy generation processes completed!"
