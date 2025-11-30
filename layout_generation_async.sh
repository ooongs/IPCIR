CUDA_VISIBLE_DEIVCES=1,3 VLLM_MODEL_NAME="Qwen/Qwen1.5-32B-Chat" python generate_layout_async.py \
    --output_path ./generate_layout_output/ \
    --llm_path /home/llq/WorkSpace/LLM/Qwen/Qwen1.5-32B-Chat \
    --model_type vllm_openai_async \
    --batch_size 32 \
    --mode circo_test \
    --dataset_path /home/llq/WorkSpace/code/Course/VL/dataset/CIRCO 