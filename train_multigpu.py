import os
import glob
import time
from float.models.FLOAT import FLOAT
from float.data import build_dataset
from float.options.train_options import TrainOptions
from float.utils import save_video, WarmupCosineScheduler, compute_alpha_stats
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from loguru import logger
import wandb
import math
from pathlib import Path
from accelerate import Accelerator, DistributedDataParallelKwargs

# Constants
CONSOLE_LOG_INTERVAL = 10
WARMUP_RATIO = 0.1
MIN_LR_RATIO = 0.1


def setup_logging(checkpoint_dir):
    """Setup logging to both console and file."""
    log_file = f"{checkpoint_dir}/training.log"
    logger.remove()  # Remove default handler
    logger.add(log_file, rotation="10 MB", retention="7 days", level="INFO")  # File handler
    logger.add(lambda msg: print(msg, end=""), level="INFO")  # Console handler
    logger.info(f"Logging to file: {log_file}")


def setup_datasets_and_dataloaders(opt, transform):
    """Setup training and validation datasets and dataloaders."""

    curr_window_size = int(opt.wav2vec_sec * opt.fps)
    window_size = int(opt.wav2vec_sec * opt.fps + opt.num_prev_frames + opt.num_ref_frames)
    logger.info(f"window_size: {window_size}")

    # Training dataset
    dataset_train = build_dataset(
        root_dir=opt.root_dir,
        dataset_names=opt.datasets,
        transform=transform,
        window_size=window_size,
        curr_window_size=curr_window_size,
        split='train',
        audio_transform=None,
        audio_sample_rate=16000,
        fps=25,
        use_speaking_scores=opt.use_speaking_scores,
        lia_version=opt.lia_version,
        lia_path=opt.lia_path,
        use_alpha_mean=opt.use_alpha_mean,
        use_alpha_std=opt.use_alpha_std,
        use_speed_mean=opt.use_speed_mean,
        use_speed_std=opt.use_speed_std,
        use_accel_mean=opt.use_accel_mean,
        use_accel_std=opt.use_accel_std
    )
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Validation dataset
    dataset_val = build_dataset(
        root_dir=opt.root_dir,
        dataset_names=opt.datasets,
        transform=transform,
        window_size=window_size,
        curr_window_size=curr_window_size,
        split='val',
        audio_transform=None,
        audio_sample_rate=16000,
        fps=25,
        use_speaking_scores=opt.use_speaking_scores,
        lia_version=opt.lia_version,
        lia_path=opt.lia_path,
        use_alpha_mean=opt.use_alpha_mean,
        use_alpha_std=opt.use_alpha_std,
        use_speed_mean=opt.use_speed_mean,
        use_speed_std=opt.use_speed_std,
        use_accel_mean=opt.use_accel_mean,
        use_accel_std=opt.use_accel_std
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(f"Training samples: {len(dataset_train)}")
    logger.info(f"Validation samples: {len(dataset_val)}")

    return dataloader_train, dataloader_val


def setup_model(opt):
    """Initialize and configure the model."""
    model = FLOAT(opt)
    state_dict = torch.load(opt.lia_path)

    # Load and freeze motion autoencoder
    if opt.lia_version == "LIA_X":
        # pretrained LIA weights are already loaded in FLOAT
        pass
    else:
        motion_autoencoder_state_dict = {
            k.replace("motion_autoencoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("motion_autoencoder.")
        }
        model.motion_autoencoder.load_state_dict(motion_autoencoder_state_dict, strict=True)
        logger.info(f"Loaded LIA weights from {opt.lia_path}")
    model.motion_autoencoder.requires_grad_(False)
    model.motion_autoencoder.eval()

    # Configure trainable components
    model.fmt.train()
    model.audio_encoder.train()
    model.to(opt.device)
    model.fmt.initialize_weights()
    model.emotion_encoder.requires_grad_(False)

    return model


def setup_optimizer(model, opt):
    """Setup optimizer only."""
    # Combine FMT parameters and audio projection parameters
    optimizer_params = list(model.fmt.parameters()) \
        + list(model.audio_encoder.audio_projection.parameters()) \
        + list(model.intensity_embedder.parameters()) \
        + list(model.channel_projection.parameters()) \
        + list(model.reaction_classifier.parameters())

    if opt.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(optimizer_params, lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer == "Adam":
        optimizer = torch.optim.Adam(optimizer_params, lr=opt.lr)
    else:
        raise ValueError(f"Invalid optimizer: {opt.optimizer}")

    return optimizer


def setup_scheduler(optimizer, opt, dataloader_train):
    """Setup learning rate scheduler using prepared dataloader length."""
    # Learning rate scheduler with warmup
    # Account for gradient accumulation - scheduler.step() is only called when sync_gradients is True
    total_steps = (len(dataloader_train) * opt.epochs) // opt.gradient_accumulation_steps
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, min_lr=opt.lr * MIN_LR_RATIO)

    logger.info(f"Batches per epoch per process: {len(dataloader_train)}")
    logger.info(f"Gradient accumulation steps: {opt.gradient_accumulation_steps}")
    logger.info(f"Total scheduler steps: {total_steps} (total batches: {len(dataloader_train) * opt.epochs})")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Initial learning rate: {opt.lr}")
    logger.info(f"Minimum learning rate: {opt.lr * MIN_LR_RATIO}")

    return scheduler


def load_checkpoint_if_exists(accelerator, model, optimizer, scheduler, checkpoint_dir, dataloader_train, opt):
    """Load the latest checkpoint if it exists."""
    start_epoch = 0
    global_step = 0

    checkpoint_files = glob.glob(f"{checkpoint_dir}/*.pth")
    checkpoint_files = [f for f in checkpoint_files if os.path.dirname(f) == checkpoint_dir]

    if not checkpoint_files:
        return start_epoch, global_step

    checkpoint_files.sort(key=lambda f: int(os.path.basename(f).split('.')[0]) if os.path.basename(f).split('.')[0].isdigit() else -1)
    latest_checkpoint = checkpoint_files[-1]
    logger.info(f"Found existing checkpoint: {latest_checkpoint}")

    # Load checkpoint
    checkpoint = torch.load(latest_checkpoint, map_location=accelerator.device)
    accelerator.unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    global_step = checkpoint['global_step']

    # Restore scheduler state
    if 'scheduler_state' in checkpoint:
        # Account for gradient accumulation when calculating completed scheduler steps
        completed_steps = (start_epoch * len(dataloader_train)) // opt.gradient_accumulation_steps
        scheduler.current_step = completed_steps
        logger.info(f"Set scheduler current_step to {completed_steps} based on {start_epoch} completed epochs")

    logger.info(f"Resuming from epoch {start_epoch}, global step {global_step}")
    return start_epoch, global_step


def log_batch_metrics(accelerator, batch_idx, times, loss_dict, current_lr, grad_norm, global_step):
    """Log batch-level metrics and timing information."""
    if batch_idx % CONSOLE_LOG_INTERVAL == 0 and accelerator.is_main_process:
        data_load_time, forward_time, backward_time, total_time = times
        logger.info(
            f"Batch {batch_idx}: data_load={data_load_time:.3f}s, "
            f"forward={forward_time:.3f}s, backward={backward_time:.3f}s, "
            f"total={total_time:.3f}s, lr={current_lr:.6f}, grad_norm={grad_norm:.3f}"
        )
        wandb.log({
            "train_loss": loss_dict['loss'].item(),
            "train_loss_ot": loss_dict['loss_ot'].item(),
            "train_loss_vel": loss_dict['loss_vel'].item(),
            "lr": current_lr,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        }, step=global_step)


def save_checkpoint(accelerator, model, optimizer, scheduler, opt, epoch, global_step, loss, checkpoint_dir, e2e=False):
    """Save training checkpoint."""
    if e2e:
        checkpoint_path = f"{checkpoint_dir}/e2e.pth"
        model_state_dict = accelerator.unwrap_model(model).state_dict()
        model_state_dict_e2e = {
            k: v for k, v in model_state_dict.items() if
            k.startswith("fmt")
            or k.startswith("motion_autoencoder")
            or k.startswith("audio_encoder.audio_projection")
            or k.startswith("intensity_embedder")
            or k.startswith("channel_projection")
            or k.startswith("reaction_classifier")
            or k == "audio_encoder.wav2vec2.masked_spec_embed"
        }
        torch.save(model_state_dict_e2e, checkpoint_path)
    else:
        checkpoint_path = f"{checkpoint_dir}/{epoch+1:03d}.pth"
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': accelerator.unwrap_model(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state': {'current_step': scheduler.current_step},
            'opt': opt,
            'loss': loss.item() if loss is not None else None
        }
        torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")


def save_inference_video(accelerator, model, data_inference, checkpoint_dir, epoch, behavior_control):
    """Generate and save inference video."""
    if data_inference is None:
        return

    model.eval()
    os.makedirs(f"{checkpoint_dir}/results", exist_ok=True)

    with torch.no_grad():
        if behavior_control:
            for (ref_video_idx, behavior) in enumerate(["neutral", "smile", "motion"]):
                inference_video_path = f"{checkpoint_dir}/results/inference_video_{epoch+1:03d}_ref_video_{behavior}.mp4"
                alpha_dict = compute_alpha_stats(data_inference[f"ref_video_{ref_video_idx}"])
                inference_video = accelerator.unwrap_model(model).inference_from_latent(data_inference, alpha_dict=alpha_dict)['d_hat']
                save_video(inference_video, inference_video_path, audio_path=data_inference['audio_path'], fps=25)
                logger.info(f"Saved inference video: {inference_video_path}")
        else:
            inference_video_path = f"{checkpoint_dir}/results/inference_video_{epoch+1:03d}.mp4"
            inference_video = accelerator.unwrap_model(model).inference_from_latent(data_inference)['d_hat']
            save_video(inference_video, inference_video_path, audio_path=data_inference['audio_path'], fps=25)
            logger.info(f"Saved inference video: {inference_video_path}")

    # Clean up GPU memory
    del inference_video
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Reset training mode
    accelerator.unwrap_model(model).fmt.train()
    accelerator.unwrap_model(model).audio_encoder.train()


def cleanup_old_checkpoints(checkpoint_dir, opt):
    """Remove old checkpoints, keeping only the last n."""
    checkpoint_files = glob.glob(f"{checkpoint_dir}/*.pth")
    checkpoint_files = [f for f in checkpoint_files if os.path.dirname(f) == checkpoint_dir]
    checkpoint_files.sort(key=lambda f: int(os.path.basename(f).split('.')[0]) if os.path.basename(f).split('.')[0].isdigit() else -1)

    if len(checkpoint_files) > opt.keep_last_n_checkpoints:
        files_to_remove = checkpoint_files[:-opt.keep_last_n_checkpoints]
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                logger.info(f"Removed old checkpoint: {file_path}")
            except OSError as e:
                logger.warning(f"Error removing {file_path}: {e}")


def run_validation(accelerator, model, dataloader_val, opt, global_step):
    """Run validation loop and log metrics."""
    model.eval()
    val_losses = []
    val_losses_ot = []
    val_losses_vel = []
    val_losses_reaction = []

    with torch.no_grad():
        for batch in dataloader_val:
            r_s, audio, alpha_dict = batch['r_s'], batch['audio'], batch['alpha_dict']
            intensity_scores, reaction_scores = batch['intensity'], batch['reaction']
            loss_dict = model(r_s=r_s, audio=audio, alpha_dict=alpha_dict, intensity_scores=intensity_scores, reaction_scores=reaction_scores)
            val_losses.append(loss_dict['loss'].item())
            val_losses_ot.append(loss_dict['loss_ot'].item())
            val_losses_vel.append(loss_dict['loss_vel'].item())
            val_losses_reaction.append(loss_dict['loss_reaction'].item())
    
    # Calculate averages
    avg_val_loss = sum(val_losses) / len(val_losses)
    avg_val_loss_ot = sum(val_losses_ot) / len(val_losses_ot)
    avg_val_loss_vel = sum(val_losses_vel) / len(val_losses_vel)
    avg_val_loss_reaction = sum(val_losses_reaction) / len(val_losses_reaction)

    # Gather validation losses from all processes
    val_losses_tensor = torch.tensor([avg_val_loss, avg_val_loss_ot, avg_val_loss_vel, avg_val_loss_reaction], device=accelerator.device)
    gathered_losses = accelerator.gather(val_losses_tensor)

    if accelerator.is_main_process:
        gathered_losses = gathered_losses.view(accelerator.num_processes, 4)
        final_val_loss = gathered_losses[:, 0].mean().item()
        final_val_loss_ot = gathered_losses[:, 1].mean().item()
        final_val_loss_vel = gathered_losses[:, 2].mean().item()
        final_val_loss_reaction = gathered_losses[:, 3].mean().item()

        wandb.log({
            "val_loss": final_val_loss,
            "val_loss_ot": final_val_loss_ot,
            "val_loss_vel": final_val_loss_vel,
            "val_loss_reaction": final_val_loss_reaction,
        }, step=global_step)
        logger.info(f"Validation - loss: {final_val_loss:.4f}, loss_ot: {final_val_loss_ot:.4f}, loss_vel: {final_val_loss_vel:.4f}, loss_reaction: {final_val_loss_reaction:.4f}")

    # Reset training mode
    accelerator.unwrap_model(model).fmt.train()
    accelerator.unwrap_model(model).audio_encoder.train()


def train_one_epoch(accelerator, model, optimizer, scheduler, dataloader_train, opt, epoch, global_step):
    """Train for one epoch."""
    # Set epoch for distributed training sampler
    if hasattr(dataloader_train.sampler, 'set_epoch'):
        dataloader_train.sampler.set_epoch(epoch)
    elif hasattr(dataloader_train.batch_sampler, 'sampler') and hasattr(dataloader_train.batch_sampler.sampler, 'set_epoch'):
        dataloader_train.batch_sampler.sampler.set_epoch(epoch)

    epoch_start_time = time.perf_counter()

    for batch_idx, batch in enumerate(dataloader_train):
        batch_start_time = time.perf_counter()

        # Data loading
        data_load_start = time.perf_counter()
        r_s, audio, alpha_dict, metadata = batch['r_s'], batch['audio'], batch['alpha_dict'], batch['metadata']
        intensity_scores, reaction_scores = batch['intensity'], batch['reaction']
        data_load_time = time.perf_counter() - data_load_start

        # Forward and backward pass with gradient accumulation
        forward_start = time.perf_counter()
        with accelerator.accumulate(model):
            loss_dict = model(
                r_s=r_s, 
                audio=audio, 
                alpha_dict=alpha_dict,
                intensity_scores=intensity_scores,
                reaction_scores=reaction_scores
            )
            loss = loss_dict['loss']
            forward_time = time.perf_counter() - forward_start

            # Backward pass
            backward_start = time.perf_counter()
            accelerator.backward(loss)

            # Gradient clipping and optimizer step only when accumulating is complete
            if accelerator.sync_gradients:
                # grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=opt.gc)
                params_to_clip = [
                    p
                    for group in optimizer.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
                grad_norm = accelerator.clip_grad_norm_(params_to_clip, max_norm=opt.gc)
                optimizer.step()
                optimizer.zero_grad()

            backward_time = time.perf_counter() - backward_start

        # Update learning rate and global step only when accumulating is complete
        if accelerator.sync_gradients:
            scheduler.step()
            global_step += 1

        current_lr = scheduler.get_last_lr()[0]

        total_batch_time = time.perf_counter() - batch_start_time

        # Log metrics (only on sync steps to avoid confusion)
        if accelerator.sync_gradients:
            times = (data_load_time, forward_time, backward_time, total_batch_time)
            grad_norm = grad_norm if 'grad_norm' in locals() else 0.0
            log_batch_metrics(accelerator, batch_idx, times, loss_dict, current_lr, grad_norm, global_step)

    epoch_time = time.perf_counter() - epoch_start_time
    logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")

    if accelerator.is_main_process:
        wandb.log({"epoch_time": epoch_time}, step=global_step)

    return global_step, loss


def main():

    # Get number of processes from environment or command line
    # For single GPU, we'll have num_processes=1
    num_processes = int(os.environ.get('ACCELERATE_NUM_PROCESSES', '1'))

    opt = TrainOptions().parse()

    if num_processes == 1:
        # Single GPU mode - initialize without distributed settings
        accelerator = Accelerator(
            mixed_precision="fp16",
            gradient_accumulation_steps=opt.gradient_accumulation_steps
        )
    else:
        # Setup
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        print(f"ddp_kwargs: {ddp_kwargs}")
        # Multi GPU mode - use distributed settings
        accelerator = Accelerator(
            mixed_precision="fp16",
            gradient_accumulation_steps=opt.gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs]
        )

    rank, world = accelerator.process_index, accelerator.num_processes
    opt.rank, opt.ngpus = rank, world
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    behavior_control = opt.use_alpha_mean or opt.use_alpha_std or opt.use_speed_mean or opt.use_speed_std or opt.use_accel_mean or opt.use_accel_std
    logger.info(f"Options: {opt}")
    logger.info(f"Behavior control: {behavior_control}")

    # Create checkpoint directory and setup logging
    checkpoint_dir = f"checkpoints/{opt.exp_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    setup_logging(checkpoint_dir)

    # Setup data
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataloader_train, dataloader_val = setup_datasets_and_dataloaders(opt, transform)

    # Load inference data
    if opt.save_intermediate_video:
        if behavior_control:
            inference_path = Path(opt.root_dir) / "behavior_control" / f"data_inference_{opt.mode}_{opt.lia_version}.pth"
        else:
            # inference_path = Path(opt.root_dir) / f"data_inference_{opt.lia_version}.pth"
            inference_path = Path("/data/zikai/Codes/float_incontext/checkpoints/data_inference.pth")

        if inference_path.is_file():
            data_inference = torch.load(inference_path, map_location=opt.device)
        else:
            raise ValueError("No inference sample found. Run tools/prepare_inference_sample.py to create it. "
                             "Or set --save-intermediate-video to False.")

    # Setup model
    model = setup_model(opt)

    # Setup optimizer BEFORE preparing with accelerator (needs unwrapped model)
    optimizer = setup_optimizer(model, opt)

    # Prepare model, optimizer, and dataloaders with accelerator
    model, optimizer, dataloader_train, dataloader_val = accelerator.prepare(
        model, optimizer, dataloader_train, dataloader_val
    )

    # Setup scheduler AFTER preparing dataloaders (needs correct dataloader length)
    scheduler = setup_scheduler(optimizer, opt, dataloader_train)

    # Prepare scheduler with accelerator
    scheduler = accelerator.prepare(scheduler)

    # Load checkpoint if exists
    start_epoch, global_step = load_checkpoint_if_exists(
        accelerator, model, optimizer, scheduler, checkpoint_dir, dataloader_train, opt
    )

    # Initialize wandb
    run_name = f"float_{opt.exp_name}"
    if start_epoch > 0:
        run_name += f"_resume_epoch_{start_epoch}"
    if accelerator.is_main_process:
        wandb.init(project="float", name=run_name)

    # Initialize gradients to ensure clean start
    optimizer.zero_grad()

    # Ensure all processes are synchronized before training starts
    accelerator.wait_for_everyone()

    # Training loop
    for epoch in range(start_epoch, opt.epochs):
        logger.info(f"Epoch {epoch+1}/{opt.epochs}")

        global_step, loss = train_one_epoch(
            accelerator, model, optimizer, scheduler, dataloader_train, opt, epoch, global_step
        )

        if torch.isnan(loss):
            raise ValueError("Loss is nan")

        # Save checkpoint and inference video
        if accelerator.is_main_process and (epoch % opt.save_every_n_epochs == 0 or epoch == opt.epochs - 1):
            save_checkpoint(accelerator, model, optimizer, scheduler, opt, epoch, global_step, loss, checkpoint_dir)
            if opt.save_intermediate_video:
                save_inference_video(accelerator, model, data_inference, checkpoint_dir, epoch, behavior_control)

        # Wait for main process to finish saving before continuing
        accelerator.wait_for_everyone()

        # Cleanup old checkpoints
        if accelerator.is_main_process:
            cleanup_old_checkpoints(checkpoint_dir, opt)

        # Wait for cleanup to complete
        accelerator.wait_for_everyone()

        # Validation
        run_validation(accelerator, model, dataloader_val, opt, global_step)
        accelerator.wait_for_everyone()

    # Save final e2e checkpoint
    if accelerator.is_main_process:
        save_checkpoint(accelerator, model, optimizer, scheduler, opt, epoch, global_step, loss, checkpoint_dir, e2e=True)

    # Final synchronization to ensure all processes complete together
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
