python src/retrieval_circo.py \
    --submission-name circo_aug_G --dataset-path /home/llq/WorkSpace/code/Course/VL/dataset/CIRCO \
    --s_w 1.0 --t_w 1.0 --a_w 1.0 --fusion_weight 0.3 \
    --aug_dir /home/llq/WorkSpace/code/Course/VL/Imagine-and-Seek/circo_test/images \
    --layout_path /home/llq/WorkSpace/code/Course/VL/Imagine-and-Seek/run1_SC_numN3/generate_layout_output/output_proxy_layout_0.json \
    --type G --eval-type LDRE-G --with_aug 