#!/bin/bash

# Data settings
EVAL_SET="RealTalkListening"
VIDEO_LIST="RealTalkListening_video_eval.txt"
VIDEO_DIR="/data/zikai/Data/RealTalkListening/Original"
FRAMES_DIR="/data/zikai/Data/RealTalkListening/Original_first_frame"
AUDIO_DIR="/data/zikai/Data/RealTalkListening/Original_wav"
GT_DIR="./videos_eval/${EVAL_SET}/GT"
REF_R_S_PATH="/data/zikai/Codes/float_yumin_incontext/data/influencers/LIA/altman_v2.pth"

# Model settings
EXP_NAME="IC010_epoch_600_lr_5e-4_bs_256_LIA_listening_RealTalkListening_SeamlessV3Listening_num-prev-frames_10_10sec"
EPOCH=600
SEED=42

num_prev_frames=10
num_ref_frames=125
wav2vec_sec=10
a_cfg_scale=2
e_cfg_scale=1
r_cfg_scale=1
s_cfg_scale=1
ref_cfg_scale=1
i_cfg_scale=1


# Output settings
OUTPUT_DIR="./videos_eval/${EVAL_SET}/${EXP_NAME}"
mkdir -p ${OUTPUT_DIR}
CKPT_PATH="./checkpoints/${EXP_NAME}/${EPOCH}.pth"
mkdir -p ${GT_DIR}


# Run generation
python tools/generate_videos.py \
    --video_list ${VIDEO_LIST} \
    --video_dir ${VIDEO_DIR} \
    --ref_dir ${FRAMES_DIR} \
    --audio_dir ${AUDIO_DIR} \
    --res_dir ${OUTPUT_DIR} \
    --ckpt_path ${CKPT_PATH} \
    --a_cfg_scale ${a_cfg_scale} \
    --e_cfg_scale ${e_cfg_scale} \
    --r_cfg_scale ${r_cfg_scale} \
    --s_cfg_scale ${s_cfg_scale} \
    --ref_cfg_scale ${ref_cfg_scale} \
    --i_cfg_scale ${i_cfg_scale} \
    --seed ${SEED} \
    --num_prev_frames ${num_prev_frames} \
    --num_ref_frames ${num_ref_frames} \
    --wav2vec_sec ${wav2vec_sec} \
    --ref_r_s_path ${REF_R_S_PATH}


# Create GT videos by making symbolic links
for video_id in $(cat ${VIDEO_LIST}); do
    cp ${VIDEO_DIR}/${video_id}.mp4 ${GT_DIR}/${video_id}.mp4
done