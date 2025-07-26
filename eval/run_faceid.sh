gpu_id=1

save_dir='/apdcephfs_jn/share_302243908/ethanyichen/svd_work/data/svd_infer/E7-94K-loopy/visuals/000000-576-250-2.0-2.0-7.5-1.25-motion12-motion24-area1.2-overlap0-shift7-noise1.0-cropFalse-0.9-bfrFalse-teethTrue-interframeTrue/loopy_test'
# save_dir='/apdcephfs_jn/share_302243908/ethanyichen/svd_work/data/svd_infer/loopy-test/visuals/000000-576-250-2.0-2.0-7.5-1.25-motion12-motion24-area1.2-overlap0-shift7-noise1.0-cropFalse-0.9-bfrFalse-teethTrue-interframeTrue/loopy_test'
# save_dir="/apdcephfs/share_302508626/ethanyichen/svd_work/data/svd_infer/E5-212k-loopy/visuals/000000-576-250-2.0-2.0-7.5-1.25-motion12-area1.2-overlap0-shift7-noise1.0-cropFalse-0.9-bfrFalse-teethTrue-interframeTrue/testset"
save_dir="/apdcephfs_jn/share_302243908/ethanyichen/svd_work/data/svd_infer/E7-182K-loopy/visuals/000000-576-500-2.0-2.0-7.5-1.25-motion12-motion24-area1.2-overlap0-shift7-noise1.0-cropFalse-0.9-bfrFalse-teethTrue-interframeTrue/test"
save_dir="/apdcephfs/share_302508626/ethanyichen/svd_work/data/svd_infer/E8-118K-loopy/visuals/000000-576-500-2.0-2.0-7.5-1.25-motion12-motion24-area1.2-overlap0-shift7-noise1.0-cropFalse-0.9-bfrFalse-teethTrue-interframeTrue/loopy_test"
source_imgae_dir='/apdcephfs_jn/share_302243908/ethanyichen/svd_work/data/loopy_test/image'

CUDA_VISIBLE_DEVICES=$gpu_id python ./evaluation_faceid.py \
        --save_dir $save_dir \
        --source_imgae_dir $source_imgae_dir \
        # --det_head 

