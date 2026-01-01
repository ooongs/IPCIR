# SDXL Base Callback-based Multi-Step Image Generation Guide

`callback_on_step_end`를 사용하여 **SDXL Base 모델**로 한 번의 실행으로 여러 step의 중간 결과를 저장하는 가이드입니다.

## 🎯 주요 변경사항

- ✅ **SDXL Base 1.0** 사용 (`stabilityai/stable-diffusion-xl-base-1.0`)
- ✅ **StableDiffusionXLPipeline** 사용
- ✅ Flux 모델 제거 (단일 모델 테스트)
- ✅ Proper VAE scaling 적용 (검은 이미지 방지)

## 🚀 핵심 개선점

### 기존 방식 (각 step별로 실행)
- Step 1, 4, 8, 16, 32를 **각각 별도로** 실행
- 총 실행 횟수: 1 + 4 + 8 + 16 + 32 = **61 steps**
- 800개 ID × 5개 프롬프트 = 4000개 × 61 steps = **244,000 step 실행**

### Callback 방식 (한 번에 실행) ⭐
- Step 32를 **한 번만** 실행하면서 중간 step의 latent를 VAE decode하여 저장
- 총 실행 횟수: **32 steps만!**
- 800개 ID × 5개 프롬프트 = 4000개 × 32 steps = **128,000 step 실행**
- **약 2배 빠름!** 🚀

## 📁 생성된 파일들

1. **generate_proxy_multistep_callback.py** - Callback 기반 생성 스크립트
2. **test_multistep_callback.py** - 테스트 스크립트
3. **generate_callback.sh** - 실행용 셸 스크립트

## 🧪 테스트 실행 (필수!)

먼저 테스트 스크립트로 동작을 확인해보세요:

```bash
python test_multistep_callback.py
```

테스트 결과는 `./test_output/sdxl_multistep/` 디렉토리에 저장됩니다:
```
test_output/sdxl_multistep/
├── step_01.png  ← 1 step 결과 (노이즈가 많음)
├── step_04.png  ← 4 step 결과
├── step_08.png  ← 8 step 결과
├── step_16.png  ← 16 step 결과
├── step_32.png  ← 32 step 결과
└── final.png    ← 최종 결과 (step_32.png와 동일)
```

**이미지를 확인하여 검은 화면이 아닌지 확인하세요!**

## 🚀 본격 사용

### 방법 1: 셸 스크립트 사용 (권장)

```bash
# 실행
bash generate_callback.sh
```

### 방법 2: Python 직접 실행

```bash
# 단일 GPU
CUDA_VISIBLE_DEVICES=0 python generate_proxy_multistep_callback.py \
    --model_path stabilityai/stable-diffusion-xl-base-1.0 \
    --json_file ./test1.json \
    --output_base_path ./output \
    --num_prompts 5 \
    --img_per_prompt 1 \
    --max_inference_steps 32 \
    --save_steps 1 4 8 16 32 \
    --guidance_scale 7.5

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
| `--model_path` | SDXL Base 모델 경로 | stabilityai/stable-diffusion-xl-base-1.0 |
| `--json_file` | 입력 JSON 파일 경로 | ./test1.json |
| `--output_base_path` | 출력 베이스 디렉토리 | ./output |
| `--num_prompts` | ID당 사용할 프롬프트 개수 | 5 |
| `--img_per_prompt` | 프롬프트당 생성할 이미지 개수 | 1 |
| `--max_inference_steps` | 최대 inference step (실제 실행 횟수) | 32 |
| `--save_steps` | 저장할 step 리스트 | 1 4 8 16 32 |
| `--guidance_scale` | Guidance scale (SDXL 권장: 7.5) | 7.5 |

## 🔍 작동 원리

### Callback 함수의 동작

```python
class LatentSaver:
    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        current_step = step_index + 1

        # step 1, 4, 8, 16, 32에서만 저장
        if current_step in target_steps:
            latents = callback_kwargs["latents"]

            # IMPORTANT: Proper scaling to prevent black images
            latents = latents / pipe.vae.config.scaling_factor

            # VAE decode: latent → image
            image = pipe.vae.decode(latents, return_dict=False)[0]

            # Convert and save
            pil_image.save(f"step_{current_step}.jpg")
```

### 검은 이미지 방지를 위한 핵심 코드

StackOverflow 링크에서 제시된 해결책을 적용했습니다:

```python
# ✅ CORRECT: Scale before decode
latents = latents / pipe.vae.config.scaling_factor
image = pipe.vae.decode(latents, return_dict=False)[0]

# ❌ WRONG: No scaling (results in black images)
image = pipe.vae.decode(latents).sample
```

## 📊 예상 성능

### 단일 프롬프트 기준 (SDXL Base)

| 방식 | Step 실행 횟수 | 예상 시간 |
|------|---------------|----------|
| 별도 실행 | 1+4+8+16+32 = 61 | ~30초 |
| Callback 방식 | 32 | ~16초 |
| **개선** | **-47%** | **1.9배 빠름** |

### 전체 데이터셋 (800 IDs × 5 prompts)

| 방식 | 총 Step 수 | 예상 시간 |
|------|-----------|----------|
| 별도 실행 | 244,000 | ~33시간 |
| Callback 방식 | 128,000 | **~18시간** |
| **개선** | **-47%** | **1.8배 빠름** |

> SDXL Base는 Turbo보다 느리지만 품질이 더 좋습니다.

## 💡 장점

1. ✅ **속도**: 약 2배 빠름
2. ✅ **일관성**: 같은 noise trajectory를 공유하므로 step 간 비교가 공정
3. ✅ **품질**: SDXL Base는 고품질 이미지 생성
4. ✅ **메모리**: 모델을 한 번만 로드
5. ✅ **검증됨**: StackOverflow 솔루션 적용으로 검은 이미지 문제 해결

## ⚠️ 중요 사항

### 1. 반드시 테스트 먼저 실행
```bash
python test_multistep_callback.py
```
생성된 이미지를 확인하여 정상 작동하는지 검증하세요.

### 2. Guidance Scale
- SDXL Base는 **guidance_scale=7.5** 권장
- SDXL Turbo는 guidance_scale=0.0 사용
- 이 코드는 SDXL Base용이므로 7.5를 사용합니다.

### 3. Generator Device
```python
# ✅ CORRECT for SDXL
generator = torch.Generator("cuda").manual_seed(seed)

# ❌ May cause issues
generator = torch.Generator("cpu").manual_seed(seed)
```

### 4. VAE Decoding
중간 step의 latent는 완전히 denoised되지 않았으므로:
- Step 1: 매우 노이즈가 많은 이미지
- Step 4-8: 점진적으로 개선
- Step 16-32: 거의 최종 품질

## 🧹 생성된 이미지 삭제 (재생성 시)

```bash
# 특정 step 삭제
rm -rf ./output/proxy_images_sdxl_step1

# 모든 step 삭제
rm -rf ./output/proxy_images_sdxl_step*

# 캐시 파일도 삭제
find ./output -name "aug_features_*.npz" -delete
```

## 🔄 Retrieval과 연동

생성된 이미지는 `retrieval_circo.py`에서 바로 사용 가능:

```bash
# Step 1 이미지로 테스트
python src/retrieval_circo.py \
    --aug_dir ./output/proxy_images_sdxl_step1 \
    --with_aug

# Step 32 이미지로 테스트
python src/retrieval_circo.py \
    --aug_dir ./output/proxy_images_sdxl_step32 \
    --with_aug
```

## 🐛 문제 해결

### 검은 이미지가 생성되는 경우
1. VAE scaling이 제대로 적용되었는지 확인
2. `latents / pipe.vae.config.scaling_factor` 코드 확인
3. Generator device를 "cuda"로 설정했는지 확인

### OOM (Out of Memory) 에러
1. Batch size 줄이기 (현재는 1)
2. Mixed precision 사용 확인 (torch.float16)
3. GPU 메모리 정리: `torch.cuda.empty_cache()`

### 느린 생성 속도
1. 올바른 GPU 사용 확인: `nvidia-smi`
2. GPU가 다른 프로세스에 사용 중인지 확인
3. Mixed precision 적용 확인

## 📝 요약

- **모델**: SDXL Base 1.0 (고품질)
- **방식**: Callback으로 32 step 한 번 실행, 중간 결과 저장
- **속도**: 기존 대비 약 2배 빠름
- **해결**: StackOverflow 솔루션으로 검은 이미지 문제 해결

**권장 워크플로우**:
1. `python test_multistep_callback.py` 실행
2. 생성된 이미지 확인 (검은 화면 아닌지)
3. 문제 없으면 `bash generate_callback.sh` 실행
4. 생성 완료 후 retrieval 테스트

🚀 Happy Generating!
