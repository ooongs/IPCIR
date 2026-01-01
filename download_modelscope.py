#!/usr/bin/env python
"""
ModelScope를 사용해서 Step1X-Edit-v1p2 모델 다운로드
"""
import os
from modelscope.hub.snapshot_download import snapshot_download

# 다운로드 경로 설정
cache_dir = "/home/jinzhenxiong/pretrain"
os.makedirs(cache_dir, exist_ok=True)

print("="*60)
print("ModelScope를 사용해서 모델 다운로드 중...")
print(f"저장 경로: {cache_dir}")
print("="*60)

# ModelScope에 없으면 HuggingFace에서 다운로드 (ModelScope SDK 사용)
from modelscope.hub.api import HubApi

api = HubApi()
allow = [
    "model_index.json",
    "scheduler/*",

    "tokenizer/*",
    "tokenizer_2/*",

    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",

    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",

    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",

    # 보통 SDXL에서는 0.9 VAE(vae_1_0)를 많이 씀. 필요에 맞게 하나만 선택
    "vae_1_0/config.json",
    "vae_1_0/diffusion_pytorch_model.fp16.safetensors",
    # 또는 아래 두 줄로 기본 vae를 받기
    # "vae/config.json",
    # "vae/diffusion_pytorch_model.fp16.safetensors",
]

ignore = [
    "**/*.msgpack",          # Flax/JAX
    "**/*.onnx*",            # ONNX (+ onnx_data)
    "**/openvino_model.*",   # OpenVINO
    "**/diffusion_pytorch_model.safetensors",  # fp32 (fp16만 쓸 거면 제외)
    "sd_xl_base_1.0.safetensors",
    "sd_xl_base_1.0_0.9vae.safetensors",
    "*.png", "*.md", ".gitattributes"
]
# ModelScope가 HuggingFace를 프록시로 사용
model_dir = snapshot_download(
    # 'stabilityai/stable-diffusion-xl-base-1.0',  # ModelScope 미러 시도
    "madebyollin/sdxl-vae-fp16-fix",
    cache_dir=cache_dir,
    # allow_file_pattern=allow,
    # ignore_file_pattern=ignore
)

print(f"\n✓ 완료! 모델 경로: {model_dir}")





