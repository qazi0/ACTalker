save_dir="/apdcephfs_cq8/share_1367250/ethanyichen/gf_work/data/vasa/train_model_test"
tmp_dir="/apdcephfs_cq8/share_1367250/ethanyichen/gf_work/data/vasa/tmp"

CUDA_VISIBLE_DEVICES=0 python ./evaluation.py \
        --save_dir $save_dir \
        --data_dir $tmp_dir