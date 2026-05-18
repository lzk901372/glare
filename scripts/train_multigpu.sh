#!/bin/bash
# set -euo pipefail

# Training configuration
gpus=0,1,2,3 # A1: (0,1) or (3,4). A2: (0,1) or (2,3)
epoch=600
batch_size=64 # batch_size per GPU (total effective batch_size = batch_size * num_gpus * gradient_accumulation_steps)
datasets="RealTalkListening SeamlessV3Listening"  # SeamlessV3Listening" #"HDTF RAVDESS" # "TheSkinDeep"
mode="listening"
lr=5e-4 #1e-3 # 5e-4
gc=1.0 # gradient clipping
dim_w=512
dim_m=20
vel_loss_weight=1.0
gradient_accumulation_steps=1 # gradient accumulation steps (1 = no accumulation)
lia_version='LIA' # highres | LIA_v3 | LIA_v4 | LIA
lia_path="./checkpoints/float.pth" # highres
# lia_path="./data/checkpoints/lia_v4.pth"
data_root_dir="/data/zikai/Data/H5_files/"
r_s_scale=1.0 # 10.0 for LIA-X
num_prev_frames=10
num_ref_frames=125
wav2vec_sec=10
num_processes=$(echo $gpus | tr ',' '\n' | wc -l)
# num_processes=1

# echo "Pass"

datasets_str=$(echo $datasets | tr ' ' '_')
total_batch_size=$((batch_size * num_processes * gradient_accumulation_steps))
exp_name="IC010_epoch_${epoch}_lr_${lr}_bs_${total_batch_size}_${lia_version}_${mode}_${datasets_str}_num-prev-frames_${num_prev_frames}_${wav2vec_sec}sec"

# echo "Pass 2"

netstat -tulpn | grep 29500
# sudo ss -tulpn | grep 29500

# echo "Pass 2.5"

# Find available ports in the range
for port in {29500..29510}; do
    if ! netstat -tulpn | grep -q ":$port "; then
        echo "Port $port is available"
        break
    fi
done

# echo "Pass 3"

# Set environment variables for proper GPU mode detection
export CUDA_VISIBLE_DEVICES=$gpus
# export CUDA_VISIBLE_DEVICES='-1'
export ACCELERATE_NUM_PROCESSES=$num_processes
# export ACCELERATE_NUM_PROCESSES=1

# Launch training
if [ $num_processes -eq 1 ]; then
    python train_multigpu.py \
        --epochs $epoch \
        --batch_size $batch_size \
        --datasets $datasets \
        --lr $lr \
        --gc $gc \
        --save_every_n_epochs 25 \
        --root_dir $data_root_dir \
        --exp_name $exp_name \
        --vel_loss_weight $vel_loss_weight \
        --lia_path $lia_path \
        --lia_version $lia_version \
        --dim_w $dim_w \
        --dim_m $dim_m \
        --r_s_scale $r_s_scale \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --save_intermediate_video \
        --mode $mode \
        --num_ref_frames $num_ref_frames \
        --num_prev_frames $num_prev_frames \
        --wav2vec_sec $wav2vec_sec
else
    accelerate launch \
        --num_processes $num_processes \
        --multi_gpu \
        --gpu_ids $gpus \
        --main_process_port $port \
        train_multigpu.py \
        --epochs $epoch \
        --batch_size $batch_size \
        --datasets $datasets \
        --lr $lr \
        --gc $gc \
        --save_every_n_epochs 25 \
        --root_dir $data_root_dir \
        --exp_name $exp_name \
        --vel_loss_weight $vel_loss_weight \
        --lia_path $lia_path \
        --lia_version $lia_version \
        --dim_w $dim_w \
        --dim_m $dim_m \
        --r_s_scale $r_s_scale \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --save_intermediate_video \
        --mode $mode \
        --num_ref_frames $num_ref_frames \
        --num_prev_frames $num_prev_frames \
        --wav2vec_sec $wav2vec_sec
fi
