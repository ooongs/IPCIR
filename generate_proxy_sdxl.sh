#!/bin/bash

# SDXL Proxy Image Generation Script
# This script runs image generation on multiple GPUs in parallel

# Configuration
SDXL_PATH="/data/jinzhenxiong/temp/stabilityai/sdxl-turbo"
JSON_FILE="/home/jinzhenxiong/Imagine-and-Seek/data/CIRCO/annotations/test.json"
OUTPUT_PATH="./output/proxy_images_sdxl"
NUM_GPUS=4
IMG_PER_PROMPT=1
NUM_PROMPTS=5
NUM_INFERENCE_STEPS=1
GUIDANCE_SCALE=0.0

# GPU 0
CUDA_VISIBLE_DEVICES=0 python generate_proxy_sdxl.py \
    --sdxl_path $SDXL_PATH \
    --json_file $JSON_FILE \
    --output_path $OUTPUT_PATH \
    --img_per_prompt $IMG_PER_PROMPT \
    --num_prompts $NUM_PROMPTS \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --idx 0 \
    --gpu_num $NUM_GPUS &

# GPU 1
CUDA_VISIBLE_DEVICES=1 python generate_proxy_sdxl.py \
    --sdxl_path $SDXL_PATH \
    --json_file $JSON_FILE \
    --output_path $OUTPUT_PATH \
    --img_per_prompt $IMG_PER_PROMPT \
    --num_prompts $NUM_PROMPTS \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --idx 1 \
    --gpu_num $NUM_GPUS &

# GPU 2
CUDA_VISIBLE_DEVICES=2 python generate_proxy_sdxl.py \
    --sdxl_path $SDXL_PATH \
    --json_file $JSON_FILE \
    --output_path $OUTPUT_PATH \
    --img_per_prompt $IMG_PER_PROMPT \
    --num_prompts $NUM_PROMPTS \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --idx 2 \
    --gpu_num $NUM_GPUS &

# GPU 3
CUDA_VISIBLE_DEVICES=3 python generate_proxy_sdxl.py \
    --sdxl_path $SDXL_PATH \
    --json_file $JSON_FILE \
    --output_path $OUTPUT_PATH \
    --img_per_prompt $IMG_PER_PROMPT \
    --num_prompts $NUM_PROMPTS \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --idx 3 \
    --gpu_num $NUM_GPUS &

# Wait for all processes to complete
wait
echo "All SDXL proxy generation processes completed!"
