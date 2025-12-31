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

try:
    # ModelScope에서 다운로드 시도
    # HuggingFace 모델을 ModelScope 미러에서 다운로드
    model_dir = snapshot_download(
        'black-forest-labs/FLUX.1-schnell',
        cache_dir=cache_dir,
        revision='main'
    )
    
    print("\n" + "="*60)
    print(f"✓ 다운로드 완료!")
    print(f"모델 경로: {model_dir}")
    print("="*60)
    
except Exception as e:
    print("\n" + "="*60)
    print(f"✗ ModelScope 다운로드 실패: {e}")
    print("\n다른 방법 시도 중...")
    print("="*60)
    
    # ModelScope에 없으면 HuggingFace에서 다운로드 (ModelScope SDK 사용)
    from modelscope.hub.api import HubApi
    
    api = HubApi()
    # ModelScope가 HuggingFace를 프록시로 사용
    model_dir = snapshot_download(
        'black-forest-labs/FLUX.1-schnell',  # ModelScope 미러 시도
        cache_dir=cache_dir
    )
    
    print(f"\n✓ 완료! 모델 경로: {model_dir}")





