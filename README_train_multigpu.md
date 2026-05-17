# Model Training with `train_multigpu.py`

This document only covers how to launch training with `train_multigpu.py` (single-GPU and multi-GPU).

## 1. Environment Setup

Run this in the project root:

```bash
pip install -r requirements.txt
```

Recommended checks before training:

- Ensure GPUs are available (`nvidia-smi`)
- Ensure the training data directory is accessible (for `--root_dir`)
- Ensure pretrained weights exist (default: `--lia_path ./checkpoints/float.pth`)

## 2. Minimal Training Command (Single GPU)

```bash
python train_multigpu.py \
  --epochs 100 \
  --batch_size 8 \
  --datasets RealTalkListening SeamlessV3Listening \
  --root_dir /path/to/H5_files \
  --exp_name exp_single_gpu
```

Notes:

- `--datasets` accepts multiple datasets separated by spaces
- Checkpoints are saved to `checkpoints/<exp_name>/`
- Training uses `fp16` mixed precision automatically (managed by `Accelerator`)

## 3. Multi-GPU Training Command (Recommended)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --num_processes 4 \
  --multi_gpu \
  --gpu_ids 0,1,2,3 \
  --main_process_port 29500 \
  train_multigpu.py \
  --epochs 600 \
  --batch_size 64 \
  --datasets RealTalkListening SeamlessV3Listening \
  --lr 5e-4 \
  --gc 1.0 \
  --root_dir /path/to/H5_files \
  --exp_name exp_multi_gpu \
  --lia_path ./checkpoints/float.pth \
  --lia_version LIA \
  --dim_w 512 \
  --dim_m 20 \
  --r_s_scale 1.0 \
  --gradient_accumulation_steps 1 \
  --mode listening \
  --num_ref_frames 125 \
  --num_prev_frames 10 \
  --wav2vec_sec 10 \
  --save_every_n_epochs 25 \
  --save_intermediate_video
```

Notes:

- If the port is occupied, change `--main_process_port` (e.g., `29501`)
- Effective total batch size:
  `batch_size * num_processes * gradient_accumulation_steps`
- You can also refer to `scripts/train_multigpu.sh` for the multi-GPU launch flow

## 4. Common Arguments

- `--epochs`: number of training epochs
- `--batch_size`: batch size per GPU
- `--datasets`: list of training datasets (multiple allowed)
- `--root_dir`: dataset root directory
- `--exp_name`: experiment name (defines checkpoint subdirectory)
- `--lr`: learning rate
- `--optimizer`: optimizer type, default is `AdamW` (optional: `Adam`)
- `--weight_decay`: weight decay
- `--gradient_accumulation_steps`: gradient accumulation steps
- `--gc`: gradient clipping threshold
- `--save_every_n_epochs`: save a checkpoint every N epochs
- `--keep_last_n_checkpoints`: keep only the latest N checkpoints
- `--save_intermediate_video`: save intermediate inference videos during training
- `--lia_path` / `--lia_version`: LIA weight path and version
- `--mode`: `talking` or `listening`

## 5. Logs and Outputs

Training outputs are stored in:

- `checkpoints/<exp_name>/training.log`
- `checkpoints/<exp_name>/*.pth` (periodic checkpoints)
- `checkpoints/<exp_name>/e2e.pth` (final export at end of training)
- `checkpoints/<exp_name>/results/*.mp4` (if intermediate video saving is enabled)

`wandb` is initialized on the main process, and the default run name is `float_<exp_name>`.

