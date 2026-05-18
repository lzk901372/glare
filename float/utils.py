import tempfile
import subprocess
import torch
import torchvision
import os
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import json
import numpy as np
from loguru import logger
import glob

def save_video(vid_target_recon: torch.Tensor, video_path: str, audio_path: str, fps: int = 25) -> str:
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        temp_filename = temp_video.name
        vid = vid_target_recon.permute(0, 2, 3, 1)
        vid = vid.detach().clamp(-1, 1).cpu()
        vid = ((vid + 1) / 2 * 255).type('torch.ByteTensor')
        torchvision.io.write_video(temp_filename, vid, fps=fps)
        if audio_path is not None:
            with open(os.devnull, 'wb') as f:
                command =  "ffmpeg -i {} -i {} -c:v copy -c:a aac {} -y".format(temp_filename, audio_path, video_path)
                subprocess.call(command, shell=True, stdout=f, stderr=f)
            if os.path.exists(video_path):
                os.remove(temp_filename)
        else:
            os.rename(temp_filename, video_path)
        return video_path


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0

    def step(self):
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Warmup phase: linear increase
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing phase
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_last_lr(self):
        return [self.optimizer.param_groups[0]['lr']]

def extract_latents(
        dataset,
        output_path: str,
        batch_size: int = 256,
        rank: int = 0,
        world_size: int = 1,
        lia_path: str = "checkpoints/float.pth",
        lia_version: str = "highres"
    ):
    """
    Extract motion latents from a dataset using the FLOAT motion autoencoder.

    Args:
        dataset: Dataset object that yields image tensors
        output_path (str): Path to save the extracted latents
        batch_size (int): Batch size for processing
        rank (int): Current GPU rank (0-indexed)
        world_size (int): Total number of GPUs
    """

    if lia_version == "LIA_X":
        from float.models.LIA_X.generator import Generator
        motion_autoencoder = Generator(size=512, motion_dim=40, scale=2)
        motion_autoencoder.load_state_dict(torch.load("./checkpoints/lia-x.pt", weights_only=True), strict=True)
        logger.info(f"Rank {rank}: Loaded LIA-X checkpoint from ./checkpoints/lia-x.pt")
    else:
        from float.models.generator import Generator
        state_dict = torch.load(lia_path)
        motion_autoencoder = Generator(size=512, style_dim=512, motion_dim=20)
        motion_autoencoder_state_dict = {k.replace("motion_autoencoder.", ""): v for k, v in state_dict.items() if k.startswith("motion_autoencoder.")}
        motion_autoencoder.load_state_dict(motion_autoencoder_state_dict)
        logger.info(f"Rank {rank}: Loaded LIA checkpoint from {lia_path}")
    motion_autoencoder.cuda()
    motion_autoencoder.eval()

    # Optimize DataLoader for better GPU utilization
    num_workers = min(8, os.cpu_count())  # Use more workers but cap at CPU count
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=4,  # Prefetch more batches
        drop_last=False
    )

    logger.info(f"Rank {rank}: dataset size: {len(dataset)}")
    logger.info(f"Rank {rank}: dataloader size: {len(dataloader)}")
    logger.info(f"Rank {rank}: using {num_workers} workers with prefetch_factor=4")

    # Modify output path to include rank
    output_base, output_ext = os.path.splitext(output_path)
    rank_output_path = f"{output_base}_rank{rank}{output_ext}"

    total_num_frame = 0
    r_s_list = []

    # Use autocast for potential speedup (if available)
    try:
        from torch.cuda.amp import autocast
        use_amp = True
    except ImportError:
        use_amp = False

    # Reduce save frequency to decrease I/O overhead
    save_interval = 2000  # Save every 2000 batches instead of 1000

    # Timing variables
    total_data_loading_time = 0
    total_gpu_processing_time = 0
    total_transfer_time = 0
    timing_window = 100  # Report timing every N batches

    with torch.no_grad():
        batch_start_time = time.time()

        for idx, frames in tqdm(enumerate(dataloader), desc=f"Rank {rank}", total=len(dataloader)):
            # Measure data loading time
            data_loading_time = time.time() - batch_start_time
            total_data_loading_time += data_loading_time

            # Measure GPU transfer time
            transfer_start = time.time()
            frames = frames.cuda(non_blocking=True)
            torch.cuda.synchronize()  # Ensure transfer is complete
            transfer_time = time.time() - transfer_start
            total_transfer_time += transfer_time

            # Measure GPU processing time
            processing_start = time.time()
            if lia_version == "LIA_X":
                s_r, _, _ = motion_autoencoder.enc(frames, input_target=None)
                r_s_lambda = motion_autoencoder.enc.enc_r2t(s_r)
                r_s = motion_autoencoder.dec.direction(r_s_lambda)
            elif use_amp:
                with autocast():
                    s_r, _, _ = motion_autoencoder.enc(frames, input_target=None)
                    r_s_lambda = motion_autoencoder.enc.fc(s_r)
                    r_s = motion_autoencoder.dec.direction(r_s_lambda)
            else:
                s_r, _, _ = motion_autoencoder.enc(frames, input_target=None)
                r_s_lambda = motion_autoencoder.enc.fc(s_r)
                r_s = motion_autoencoder.dec.direction(r_s_lambda)

            torch.cuda.synchronize()  # Ensure processing is complete
            processing_time = time.time() - processing_start
            total_gpu_processing_time += processing_time

            r_s_list.append(r_s.detach().cpu())
            total_num_frame += frames.shape[0]

            # Report timing statistics
            if idx % timing_window == 0 and idx > 0:
                avg_data_loading = total_data_loading_time / (idx + 1)
                avg_gpu_processing = total_gpu_processing_time / (idx + 1)
                avg_transfer = total_transfer_time / (idx + 1)
                total_batch_time = avg_data_loading + avg_transfer + avg_gpu_processing

                logger.info(f"Rank {rank} - Batch {idx}:")
                logger.info(f"  Data loading: {avg_data_loading*1000:.1f}ms ({avg_data_loading/total_batch_time*100:.1f}%)")
                logger.info(f"  GPU transfer: {avg_transfer*1000:.1f}ms ({avg_transfer/total_batch_time*100:.1f}%)")
                logger.info(f"  GPU processing: {avg_gpu_processing*1000:.1f}ms ({avg_gpu_processing/total_batch_time*100:.1f}%)")
                logger.info(f"  Total per batch: {total_batch_time*1000:.1f}ms")
                logger.info(f"  Throughput: {batch_size/total_batch_time:.1f} images/sec")

                # GPU utilization estimate
                gpu_busy_ratio = avg_gpu_processing / total_batch_time
                logger.info(f"  GPU busy ratio: {gpu_busy_ratio*100:.1f}%")
                if gpu_busy_ratio < 0.8:
                    logger.info(f"  ⚠️  GPU underutilized - bottleneck in data loading!")

            # Less frequent intermediate saves to reduce I/O overhead
            if idx % save_interval == 0 and idx > 0:
                logger.info(f"Rank {rank}: Processed {total_num_frame} frames")
                torch.save(torch.cat(r_s_list, dim=0), rank_output_path)

            # Memory management - clear cache periodically
            if idx % 1000 == 0:
                torch.cuda.empty_cache()

            # Prepare for next iteration timing
            batch_start_time = time.time()

    # Final timing report
    total_batches = len(dataloader)
    if total_batches > 0:
        avg_data_loading = total_data_loading_time / total_batches
        avg_gpu_processing = total_gpu_processing_time / total_batches
        avg_transfer = total_transfer_time / total_batches
        total_batch_time = avg_data_loading + avg_transfer + avg_gpu_processing

        logger.info(f"\nRank {rank} - FINAL TIMING REPORT:")
        logger.info(f"  Total batches: {total_batches}")
        logger.info(f"  Avg data loading: {avg_data_loading*1000:.1f}ms ({avg_data_loading/total_batch_time*100:.1f}%)")
        logger.info(f"  Avg GPU transfer: {avg_transfer*1000:.1f}ms ({avg_transfer/total_batch_time*100:.1f}%)")
        logger.info(f"  Avg GPU processing: {avg_gpu_processing*1000:.1f}ms ({avg_gpu_processing/total_batch_time*100:.1f}%)")
        logger.info(f"  Avg total per batch: {total_batch_time*1000:.1f}ms")
        logger.info(f"  Overall throughput: {batch_size/total_batch_time:.1f} images/sec")
        logger.info(f"  GPU utilization: {avg_gpu_processing/total_batch_time*100:.1f}%")

    r_s = torch.cat(r_s_list, dim=0)
    logger.info(f"Rank {rank}: Final tensor shape: {r_s.shape}")
    torch.save(r_s, rank_output_path)
    logger.info(f"Rank {rank}: Saved latents to {rank_output_path}")

def read_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def read_txt(txt_path):
    with open(txt_path, 'r') as f:
        return [line.rstrip() for line in f.readlines()]

def get_landmarks_relative_to_bbox(bbox_first_frame, detection_json):
    x_min, y_min, x_max, y_max = bbox_first_frame
    landmarks_relative = []
    for det in detection_json:
        landmarks = np.array(det['landmark_68'])
        landmarks_relative_x = (landmarks[:, 0] - x_min) / (x_max - x_min) # Normalize to [0, 1]
        landmarks_relative_y = (landmarks[:, 1] - y_min) / (y_max - y_min) # Normalize to [0, 1]
        landmarks_relative.append(np.concatenate([landmarks_relative_x, landmarks_relative_y], axis=0))
    landmarks_relative = np.stack(landmarks_relative)
    return landmarks_relative

def compute_alpha_stats(
    alpha: torch.Tensor,
    use_alpha_mean: bool = True,
    use_alpha_std: bool = True,
    use_speed_mean: bool = True,
    use_speed_std: bool = True,
    use_accel_mean: bool = True,
    use_accel_std: bool = True,
    device: str = None
):
    """
    alpha: (T, 20)
    all return shape: (20,)
    """

    assert alpha.ndim == 2, "alpha must be a 2D tensor"
    T = alpha.shape[0]
    alpha_dict = {}

    # 1) alpha statistics
    if use_alpha_mean:
        alpha_mean_over_time = alpha.mean(dim=0)                       # (K,)
        alpha_dict['alpha_mean'] = alpha_mean_over_time
    if use_alpha_std:
        alpha_std_over_time  = alpha.std(dim=0, unbiased=False)        # (K,)
        alpha_dict['alpha_std'] = alpha_std_over_time

    # 2) first derivative statistics
    if use_speed_mean or use_speed_std:
        assert T >= 2, "T must be greater than or equal to 2"
        d1 = alpha[1:, :] - alpha[:-1, :]                       # (T-1, K)
        abs_d1 = d1.abs()
        if use_speed_mean:
            abs_speed_mean_over_time = abs_d1.mean(dim=0)                                 # (K,)
            alpha_dict['speed_mean'] = abs_speed_mean_over_time
        if use_speed_std:
            abs_speed_std_over_time  = abs_d1.std(dim=0, unbiased=False)                  # (K,)
            alpha_dict['speed_std'] = abs_speed_std_over_time

    # 3) second derivative statistics (|Δ²alpha_t| statistics)
    if use_accel_mean or use_accel_std:
        assert T >= 3, "T must be greater than or equal to 3"
        # Δ²alpha_t = alpha_{t+1} - 2*alpha_t + alpha_{t-1}
        d2 = alpha[2:, :] - 2*alpha[1:-1, :] + alpha[:-2, :]  # (T-2, K)
        abs_d2 = d2.abs()
        if use_accel_mean:
            abs_accel_mean_over_time = abs_d2.mean(dim=0)                                 # (K,)
            alpha_dict['accel_mean'] = abs_accel_mean_over_time
        if use_accel_std:
            abs_accel_std_over_time  = abs_d2.std(dim=0, unbiased=False)                  # (K,)
            alpha_dict['accel_std'] = abs_accel_std_over_time

    if device is not None:
        alpha_dict = {k: v.to(device) for k, v in alpha_dict.items()}

    return alpha_dict

def get_files(path_pattern, root=None, return_rel_path=False):
    """
    path_pattern: list of paths or a single path pattern
    root: root directory. if root is provided, path_pattern *should* be relative to root
    return_rel_path: if True, return relative paths to root
    """
    if return_rel_path:
        assert root is not None, "root must be provided if return_rel_path is True"
    
    # Normalize to list
    if isinstance(path_pattern, str):
        path_pattern = [path_pattern]
    
    if root:
        path_pattern = [os.path.join(root, p) for p in path_pattern]

    files = []
    for p in path_pattern:
        files.extend(glob.glob(p))
    
    if return_rel_path:
        files = [os.path.relpath(f, root) for f in files]
    
    if isinstance(files, list):
        return sorted(files) if files else [] # Make sure the order is consistent
    else:
        raise ValueError(f"Unsupported files type: {type(files)}")