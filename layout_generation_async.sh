VLLM_MODEL_NAME="Qwen/Qwen1.5-32B-Chat" python generate_layout_async.py \
    --output_path ./generate_layout_output/ \
    --llm_path /home/llq/WorkSpace/LLM/Qwen/Qwen1.5-32B-Chat \
    --model_type vllm_openai_async \
    --batch_size 32 \
    --mode circo_test \
    --dataset_path /home/llq/WorkSpace/code/Course/VL/dataset/CIRCO

CUDA_VISIBLE_DEVICES=3 python generate_proxy_migc_elite.py \
    --layout_file /home/llq/WorkSpace/code/Course/VL/Imagine-and-Seek/generate_layout_output/output_proxy_layout_0.json \
    --image_source /home/llq/WorkSpace/code/Course/VL/dataset/CIRCO/COCO2017_unlabeled/unlabeled2017 \
    --output_path ./proxy_generation_output/ \
    --sd1x_path /home/llq/WorkSpace/code/Course/VL/Imagine-and-Seek/weights/realisticVisionV60B1_v60B1VAE.safetensors \
    --aug_caption "high quality image" \
    --img_per_mode 3 \
    --MIGCsteps 25 \
    --guidance_scale 7.5 \
    --NaiveFuserSteps 50

