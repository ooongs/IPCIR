# Callback-based Multi-Step Image Generation Guide

`callback_on_step_end`를 사용하여 **한 번의 실행**으로 여러 step의 중간 결과를 저장하는 가이드입니다.

## 🚀 핵심 개선점

### 기존 방식 (`generate_proxy_multistep.py`)
- Step 1, 4, 8, 16, 32를 **각각 별도로** 실행
- 총 실행 횟수: 1 + 4 + 8 + 16 + 32 = **61 steps**
- 800개 ID × 5개 프롬프트 = 4000개 × 61 steps = **244,000 step 실행**

### 새로운 방식 (`generate_proxy_multistep_callback.py`) ⭐
- Step 32를 **한 번만** 실행하면서 중간 step (1, 4, 8, 16, 32)의 latent를 저장
- 총 실행 횟수: **32 steps만!**
- 800개 ID × 5개 프롬프트 = 4000개 × 32 steps = **128,000 step 실행**
- **약 2배 빠름!** 🚀

## 📁 생성된 파일들

1. **generate_proxy_multistep_callback.py** - Callback 기반 생성 스크립트
2. **test_multistep_callback.py** - 테스트 스크립트
3. **generate_callback.sh** - 실행용 셸 스크립트

## 🧪 테스트 실행

먼저 테스트 스크립트로 동작을 확인해보세요:

```bash
# Flux 모델 테스트
python test_multistep_callback.py --model flux

# SDXL 모델 테스트
python test_multistep_callback.py --model sdxl

# 둘 다 테스트
python test_multistep_callback.py --model both
```

테스트 결과는 `./test_output/` 디렉토리에 저장됩니다:
```
test_output/
├── flux_multistep/
│   ├── step_01.png
│   ├── step_04.png
│   ├── step_08.png
│   ├── step_16.png
│   ├── step_32.png
│   └── final.png
└── sdxl_multistep/
    ├── step_01.png
    ├── step_04.png
    ├── step_08.png
    ├── step_16.png
    ├── step_32.png
    └── final.png
```

## 🚀 본격 사용

### 방법 1: 셸 스크립트 사용 (권장)

```bash
# 스크립트 편집하여 설정 변경
vim generate_callback.sh

# 실행
bash generate_callback.sh
```

### 방법 2: Python 직접 실행

```bash
# 단일 GPU
CUDA_VISIBLE_DEVICES=0 python generate_proxy_multistep_callback.py \
    --model_type sdxl \
    --json_file ./test1.json \
    --output_base_path ./output \
    --num_prompts 5 \
    --img_per_prompt 1 \
    --max_inference_steps 32 \
    --save_steps 1 4 8 16 32

# 다중 GPU (GPU 0, 1 사용)
CUDA_VISIBLE_DEVICES=0 python generate_proxy_multistep_callback.py \
    --max_inference_steps 32 --save_steps 1 4 8 16 32 --idx 0 --gpu_num 2 &
CUDA_VISIBLE_DEVICES=1 python generate_proxy_multistep_callback.py \
    --max_inference_steps 32 --save_steps 1 4 8 16 32 --idx 1 --gpu_num 2 &
wait
```

## 📂 출력 디렉토리 구조

```
output/
├── proxy_images_sdxl_step1/
│   └── combined/
│       ├── combined_123456_0.jpg
│       └── ...
├── proxy_images_sdxl_step4/
│   └── combined/
├── proxy_images_sdxl_step8/
│   └── combined/
├── proxy_images_sdxl_step16/
│   └── combined/
└── proxy_images_sdxl_step32/
    └── combined/
```

## ⚙️ 주요 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--model_type` | 모델 타입 (sdxl 또는 flux) | sdxl |
| `--json_file` | 입력 JSON 파일 경로 | ./test1.json |
| `--output_base_path` | 출력 베이스 디렉토리 | ./output |
| `--num_prompts` | ID당 사용할 프롬프트 개수 | 5 |
| `--img_per_prompt` | 프롬프트당 생성할 이미지 개수 | 1 |
| `--max_inference_steps` | 최대 inference step (실제 실행 횟수) | 32 |
| `--save_steps` | 저장할 step 리스트 | 1 4 8 16 32 |
| `--guidance_scale` | Guidance scale | 0.0 |

## 🔍 작동 원리

### Callback 함수의 동작

```python
class LatentSaver:
    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        current_step = step_index + 1

        # step 1, 4, 8, 16, 32에서만 저장
        if current_step in target_steps:
            latents = callback_kwargs["latents"]

            # VAE decode: latent → image
            image = pipe.vae.decode(latents).sample

            # 저장
            pil_image.save(f"step_{current_step}.jpg")
```

### 실행 과정

1. **Step 1**: latent 디코드 → `step1/combined_xxx_0.jpg` 저장
2. **Step 2-3**: 건너뜀
3. **Step 4**: latent 디코드 → `step4/combined_xxx_0.jpg` 저장
4. **Step 5-7**: 건너뜀
5. **Step 8**: latent 디코드 → `step8/combined_xxx_0.jpg` 저장
6. ...
7. **Step 32**: latent 디코드 → `step32/combined_xxx_0.jpg` 저장 + 최종 이미지

## 📊 성능 비교

### 단일 프롬프트 기준 (SDXL-Turbo)

| 방식 | Step 실행 횟수 | 예상 시간 |
|------|---------------|----------|
| 별도 실행 | 1+4+8+16+32 = 61 | ~13초 |
| Callback 방식 | 32 | ~6초 |
| **개선** | **-47%** | **2.2배 빠름** |

### 전체 데이터셋 (800 IDs × 5 prompts)

| 방식 | 총 Step 수 | 예상 시간 |
|------|-----------|----------|
| 별도 실행 | 244,000 | ~14시간 |
| Callback 방식 | 128,000 | **~7시간** |
| **개선** | **-47%** | **2배 빠름** |

## 💡 장점

1. ✅ **속도**: 2배 빠름
2. ✅ **일관성**: 같은 noise trajectory를 공유하므로 step 간 비교가 공정
3. ✅ **메모리**: 모델을 한 번만 로드
4. ✅ **코드 간결성**: 한 번의 파이프라인 호출

## ⚠️ 주의사항

1. **VAE Decode 오버헤드**: Step마다 VAE decode가 추가로 실행되므로 약간의 오버헤드 발생 (하지만 전체적으로는 여전히 빠름)

2. **메모리 사용**: 중간 step에서 VAE decode를 하므로 약간 더 많은 메모리 사용 (보통 문제 없음)

3. **Flux 모델**: `_unpack_latents` 메서드를 사용하여 latent를 올바르게 처리

4. **SDXL 모델**: `scaling_factor`를 사용하여 latent를 스케일 조정

## 🧹 생성된 이미지 삭제 (재생성 시)

```bash
# 특정 step 삭제
rm -rf ./output/proxy_images_sdxl_step1

# 모든 step 삭제
rm -rf ./output/proxy_images_sdxl_step*

# Callback 방식으로 생성한 캐시 삭제
rm ./output/proxy_images_sdxl_step*/aug_features_*.npz
```

## 🔄 기존 방식과의 호환성

생성된 이미지 파일 구조와 이름이 동일하므로, `retrieval_circo.py`에서 동일하게 사용 가능합니다:

```bash
# Step 1 이미지로 테스트
python src/retrieval_circo.py \
    --aug_dir ./output/proxy_images_sdxl_step1 \
    --with_aug
```

## 📝 요약

- **Callback 방식**: 32 step을 한 번 실행하면서 중간 결과 저장 → **빠름, 효율적**
- **별도 실행 방식**: 각 step을 독립적으로 실행 → 느림, 비효율적

**추천**: Callback 방식 사용! 🚀
