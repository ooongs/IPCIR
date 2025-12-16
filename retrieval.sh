CUDA_VISIBLE_DEVICES=0 python src/retrieval_circo.py \
    --submission-name circo_aug_G --dataset-path /home/jinzhenxiong/Imagine-and-Seek/data/CIRCO \
    --s_w 1.0 --t_w 1.0 --a_w 1.0 --fusion_weight 0.3 \
    --aug_dir /home/jinzhenxiong/mac-file-upload/circo_test/images \
    --layout_path /home/jinzhenxiong/mac-file-upload/circo_test/images/ipcir_layout.json \
    --type G --eval-type LDRE-G --with_aug 