# Multi-Step Proxy Image Generation Guide

여러 inference step (1, 4, 8, 16, 32)에 대해 proxy 이미지를 생성하는 가이드입니다.

## 📁 생성된 파일들

1. **generate_proxy_multistep.py** - Python 스크립트
2. **generate_multistep.sh** - 실행용 셸 스크립트

## 🚀 사용 방법

### 방법 1: 셸 스크립트 사용 (권장)

```bash
# 스크립트 편집하여 설정 변경
vim generate_multistep.sh

# 실행
bash generate_multistep.sh
```

### 방법 2: Python 직접 실행

```bash
# 단일 GPU
CUDA_VISIBLE_DEVICES=0 python generate_proxy_multistep.py \
    --model_type sdxl \
    --json_file ./test1.json \
    --output_base_path ./output \
    --num_prompts 5 \
    --img_per_prompt 1 \
    --inference_steps 1 4 8 16 32

# 다중 GPU (GPU 0, 1 사용)
CUDA_VISIBLE_DEVICES=0 python generate_proxy_multistep.py \
    --inference_steps 1 4 8 16 32 --idx 0 --gpu_num 2 &
CUDA_VISIBLE_DEVICES=1 python generate_proxy_multistep.py \
    --inference_steps 1 4 8 16 32 --idx 1 --gpu_num 2 &
wait
```

## 📂 출력 디렉토리 구조

```
output/
├── proxy_images_sdxl_step1/
│   └── combined/
│       ├── combined_123456_0.jpg
│       ├── combined_123456_1.jpg
│       └── ...
├── proxy_images_sdxl_step4/
│   └── combined/
│       └── ...
├── proxy_images_sdxl_step8/
│   └── combined/
│       └── ...
├── proxy_images_sdxl_step16/
│   └── combined/
│       └── ...
└── proxy_images_sdxl_step32/
    └── combined/
        └── ...
```

## ⚙️ 주요 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--model_type` | 모델 타입 (sdxl 또는 flux) | sdxl |
| `--sdxl_path` | SDXL 모델 경로 | /home/jinzhenxiong/temp/stabilityai/sdxl-turbo |
| `--flux_path` | Flux 모델 경로 | /home/jinzhenxiong/pretrain/black-forest-labs/FLUX.1-schnell |
| `--json_file` | 입력 JSON 파일 경로 | ./test1.json |
| `--output_base_path` | 출력 베이스 디렉토리 | ./output |
| `--num_prompts` | ID당 사용할 프롬프트 개수 | 5 |
| `--img_per_prompt` | 프롬프트당 생성할 이미지 개수 | 1 |
| `--inference_steps` | 사용할 inference step 리스트 | 1 4 8 16 32 |
| `--guidance_scale` | Guidance scale | 0.0 |
| `--idx` | GPU 인덱스 (멀티 GPU 사용시) | 0 |
| `--gpu_num` | 총 GPU 개수 (멀티 GPU 사용시) | 1 |

## 🔍 retrieval_circo.py와 연동

생성된 이미지를 retrieval에 사용하려면:

```bash
# Step 1 이미지로 테스트
CUDA_VISIBLE_DEVICES=0 python src/retrieval_circo.py \
    --submission-name circo_aug_step1 \
    --aug_dir ./output/proxy_images_sdxl_step1 \
    --type G --eval-type LDRE-G --with_aug

# Step 4 이미지로 테스트
CUDA_VISIBLE_DEVICES=0 python src/retrieval_circo.py \
    --submission-name circo_aug_step4 \
    --aug_dir ./output/proxy_images_sdxl_step4 \
    --type G --eval-type LDRE-G --with_aug

# ... 나머지 step도 동일
```

## 💡 팁

1. **재개 기능**: 스크립트는 이미 생성된 이미지를 자동으로 건너뜁니다. 중단된 경우 다시 실행하면 이어서 진행됩니다.

2. **멀티 GPU**: `generate_multistep.sh`에서 `GPU_NUM`을 변경하여 여러 GPU를 활용할 수 있습니다.

3. **특정 step만 생성**: 원하는 step만 지정할 수 있습니다:
   ```bash
   python generate_proxy_multistep.py --inference_steps 1 4
   ```

4. **FLUX 모델 사용**:
   ```bash
   python generate_proxy_multistep.py --model_type flux
   ```

## 📊 예상 소요 시간

SDXL-Turbo 기준 (대략적인 추정):
- Step 1: ~0.2초/이미지
- Step 4: ~0.5초/이미지
- Step 8: ~0.8초/이미지
- Step 16: ~1.5초/이미지
- Step 32: ~2.5초/이미지

800개 ID × 5개 이미지 = 4000개 이미지 기준:
- Step 1: ~13분
- Step 4: ~33분
- Step 8: ~53분
- Step 16: ~100분
- Step 32: ~167분
- **전체 (1+4+8+16+32)**: ~6-7시간 (단일 GPU)

## 🧹 생성된 이미지 삭제 (재생성 시)

```bash
# 특정 step 삭제
rm -rf ./output/proxy_images_sdxl_step1

# 모든 step 삭제
rm -rf ./output/proxy_images_sdxl_step*
```
