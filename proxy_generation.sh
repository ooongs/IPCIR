CUDA_VISIBLE_DEVICES=2 python generate_proxy_migc_elite.py \
    --layout_file /home/llq/WorkSpace/code/Course/VL/Imagine-and-Seek/generate_layout_output/output_proxy_layout_0.json \
    --image_source /home/llq/WorkSpace/code/Course/VL/dataset/CIRCO/COCO2017_unlabeled/unlabeled2017 \
    --output_path ./proxy_generation_output/ \
    --sd1x_path /home/llq/WorkSpace/code/Course/VL/Imagine-and-Seek/weights/realisticVisionV60B1_v60B1VAE.safetensors \
    --aug_caption "high quality image" \
    --img_per_mode 3 \
    --MIGCsteps 25 \
    --guidance_scale 7.5 \
    --NaiveFuserSteps 50
