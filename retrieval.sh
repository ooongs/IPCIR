CUDA_VISIBLE_DEVICES=6 python src/retrieval_circo.py \
    --submission-name circo_aug_G --dataset-path /home/jinzhenxiong/Imagine-and-Seek/data/CIRCO \
<<<<<<< HEAD
    --s_w 1.0 --t_w 1.0 --a_w 1.0 --fusion_weight 0.5 \
    --aug_dir /home/jinzhenxiong/IPCIR/output/proxy_images_flux \
    --layout_path /home/jinzhenxiong/IPCIR/CIRCO/ipcir_layout.json \
    --type G --eval-type LDRE-G --with_aug 
=======
    --s_w 1.0 --t_w 1.0 --a_w 1.0 --fusion_weight 0.3 \
    --aug_dir /home/jinzhenxiong/mac-file-upload/circo_test/images \
    --layout_path /home/jinzhenxiong/mac-file-upload/circo_test/images/ipcir_layout.json \
    --type G --eval-type LDRE-G --with_aug \
    --test_multiple_scales --scale_start 0.0 --scale_end 1.0 --scale_step 0.1 
>>>>>>> e826c3c11c4b75c72b67846d3b3e4acf7022a14d
