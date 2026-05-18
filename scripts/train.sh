#!/bin/bash

epoch=1000
batch_size=256
datasets="RAVDESS"
# datasets="HDTF RAVDESS"
# datasets="HDTF"
lr=2e-3
gamma=1.5
gc=1.0 # gradient clipping
# dataset="HDTF"

datasets_str=$(echo $datasets | tr ' ' '_')
# exp_name="epoch_${epoch}_${dataset}_cfv_debug-start_from_pretrained_rsbugfixed"
exp_name="epoch_${epoch}_lr_${lr}_bs_${batch_size}_gamma_${gamma}_gc_${gc}_wo-wr_${datasets_str}"

python train.py \
    --epochs $epoch \
    --batch_size $batch_size \
    --datasets $datasets \
    --lr $lr \
    --gamma $gamma \
    --gc $gc \
    --exp_name $exp_name


# epoch=24
# exp_name="epoch_${epoch}_combined"

# python train.py \
#     --epochs $epoch \
#     --batch_size 2 \
#     --exp_name $exp_name
