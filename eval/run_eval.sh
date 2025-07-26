# ps -ef|grep "eval_end2end_dit.py"|grep -v grep|cut -c 9-16|xargs kill -9
## 端到端测试


config='../config/eval-hydit-E0.yaml'
model_path='/apdcephfs_cq8/share_1367250/ethanyichen/gf_work/data/cq5_checkpoint/svd.ckpt'
video_dir="/apdcephfs_cq8/share_1367250/ethanyichen/svd_work/data/evaluation/0801/svd"
image_dir="/apdcephfs_cq8/share_1367250/ethanyichen/data/test-data-all/image-sqare"
exm='0801-384k'

num=142
gpu_id=0

eval_audio(){

    save_dir='/apdcephfs_cq8/share_1367250/ethanyichen/svd_work/data/autoeval/'$exm
    tmp_dir='/apdcephfs_cq8/share_1367250/ethanyichen/svd_work/data/tmp'$save_dir
    # CUDA_VISIBLE_DEVICES=$gpu_id python concate.py  \
    #             --save_dir $save_dir \
    #             --model_path  $model_path \
    #             --video_dir $video_dir \
    #             --image_dir $image_dir \
    #             --num $num
    

    # CUDA_VISIBLE_DEVICES=$gpu_id python ./eval_fid.py \
    #     --batch_size 32 \
    #     --save_dir $save_dir \
    #     --model_path $model_path \
    #     --faceid \

    CUDA_VISIBLE_DEVICES=$gpu_id python ./evaluation.py \
        --model_path $model_path \
        --save_dir $save_dir \
        --data_dir $tmp_dir
}


run_pipeline() {
    eval_audio
}
run_pipeline

