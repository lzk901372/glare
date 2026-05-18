#!/bin/bash

# Simple generation script for single test case
epoch=401
seed=42
exp_name="IC010_epoch_600_lr_5e-4_bs_256_LIA_listening_RealTalkListening_SeamlessV3Listening_num-prev-frames_10_10sec"

# Test configuration
# actor="Altman"
REF_PATH="/data/zikai/Codes/float_yumin_incontext/assets/sam_altman_512x512.jpg"
AUD_PATH="/data/zikai/Codes/float_yumin_incontext/assets/aud-sample-vs-1.wav"
REF_R_S_PATH="/data/zikai/Codes/float_yumin_incontext/data/influencers/LIA/altman_v2.pth"

# Generation parameters
num_prev_frames=10
num_ref_frames=125
wav2vec_sec=10

a_cfg_scale=2
e_cfg_scale=1
r_cfg_scale=1
s_cfg_scale=1
ref_cfg_scale=1
i_cfg_scale=1


# Setup paths
ref_name=$(basename $REF_R_S_PATH .pth)
CHECKPOINT_PATH="checkpoints/${exp_name}/${epoch}.pth"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# mkdir -p "checkpoints/${exp_name}/results_${actor}"
result_dir="checkpoints/${exp_name}/results"
mkdir -p $result_dir
RESULT_PATH="${result_dir}/epoch${epoch}_${seed}_a_${a_cfg_scale}_e_${e_cfg_scale}_r_${r_cfg_scale}_s_${s_cfg_scale}_i_${i_cfg_scale}_ref_${ref_cfg_scale}_num-prev_${num_prev_frames}_${TIMESTAMP}.mp4"

# Run generation
CUDA_VISIBLE_DEVICES=0 python generate.py \
    --ref_path ${REF_PATH} \
    --aud_path ${AUD_PATH} \
    --seed ${seed} \
    --a_cfg_scale ${a_cfg_scale} \
    --e_cfg_scale ${e_cfg_scale} \
    --r_cfg_scale ${r_cfg_scale} \
    --s_cfg_scale ${s_cfg_scale} \
    --ref_cfg_scale ${ref_cfg_scale} \
    --i_cfg_scale ${i_cfg_scale} \
    --ckpt_path ${CHECKPOINT_PATH} \
    --res_video_path ${RESULT_PATH} \
    --save_latent \
    --no_crop \
    --num_prev_frames ${num_prev_frames} \
    --num_ref_frames ${num_ref_frames} \
    --wav2vec_sec ${wav2vec_sec} \
    --ref_r_s_path ${REF_R_S_PATH}
