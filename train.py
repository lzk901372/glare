import os
import glob
import time
from float.models.FLOAT import FLOAT
from float.data import build_dataset
from float.options.train_options import TrainOptions
from float.utils import save_video
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from loguru import logger
import wandb
import math
from float.utils import WarmupCosineScheduler

def main():
    opt = TrainOptions().parse()
    opt.rank, opt.ngpus  = 0, 1
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"opt: {opt}")

    # Create checkpoint directory
    checkpoint_dir = f"checkpoints/{opt.exp_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup logging to both console and file
    log_file = f"{checkpoint_dir}/training.log"
    logger.remove()  # Remove default handler
    logger.add(log_file, rotation="10 MB", retention="7 days", level="INFO")  # File handler
    logger.add(lambda msg: print(msg, end=""), level="INFO")  # Console handler
    logger.info(f"Logging to file: {log_file}")

    # Data
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    logger.info(f"window_size: {int(opt.wav2vec_sec * opt.fps + opt.num_prev_frames)}")

    dataset_train = build_dataset(
        root_dir=opt.root_dir,
        dataset_names=opt.datasets,
        transform=transform,
        window_size=int(opt.wav2vec_sec * opt.fps + opt.num_prev_frames),
        split='train',
        audio_transform=None,
        audio_sample_rate=16000,
        fps=25,
    )
    dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True)
    dataset_val = build_dataset(
        root_dir=opt.root_dir,
        dataset_names=opt.datasets,
        transform=transform,
        window_size=int(opt.wav2vec_sec * opt.fps + opt.num_prev_frames),
        split='val',
        audio_transform=None,
        audio_sample_rate=16000,
        fps=25,
    )
    dataloader_val = DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=False)

    # Inference sample
    if os.path.isfile("data_inference.pth"):
        data_inference = torch.load("data_inference.pth", map_location=opt.device)
    else:
        data_inference = None
        raise ValueError("No inference sample found. Run tools/prepare_inference_sample.py to create it.")

    logger.info(f"len(dataset_train): {len(dataset_train)}")
    logger.info(f"len(dataset_val): {len(dataset_val)}")

    # Model
    model = FLOAT(opt)
    state_dict = torch.load("checkpoints/float.pth")

    motion_autoencoder_state_dict = {k.replace("motion_autoencoder.",""): v for k, v in state_dict.items() if k.startswith("motion_autoencoder.")}
    model.motion_autoencoder.load_state_dict(motion_autoencoder_state_dict, strict=True)
    model.motion_autoencoder.requires_grad_(False)

    model.fmt.train()
    model.audio_encoder.train() ### comment out for debugging
    model.to(opt.device)
    model.fmt.initialize_weights() ### comment out for debugging
    model.emotion_encoder.requires_grad_(False)

    # Optimizer
    # Combine FMT parameters and audio projection parameters
    optimizer_params = list(model.fmt.parameters()) + list(model.audio_encoder.audio_projection.parameters())

    if opt.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(optimizer_params, lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer == "Adam":
        optimizer = torch.optim.Adam(optimizer_params, lr=opt.lr)
    else:
        raise ValueError(f"Invalid optimizer: {opt.optimizer}")

    # Learning rate scheduler with warmup
    total_steps = len(dataloader_train) * opt.epochs
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, min_lr=opt.lr / 10.)

    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Initial learning rate: {opt.lr}")

    # Initialize training state
    start_epoch = 0
    global_step = 0

    # Define helper function for extracting epoch numbers from checkpoint filenames
    def extract_epoch(filename):
        try:
            # Extract epoch number from filename like "001.pth" -> 1
            basename = os.path.basename(filename)
            epoch_str = basename.split('.')[0]  # Remove .pth extension
            return int(epoch_str)
        except (ValueError, IndexError):
            # If filename doesn't follow expected pattern, return -1 to put it at the beginning
            logger.warning(f"Could not extract epoch number from filename: {filename}")
            return -1

    # Try to load existing checkpoint for resuming
    latest_checkpoint = None
    checkpoint_files = glob.glob(f"{checkpoint_dir}/*.pth")
    # Filter out files in subdirectories and only keep files directly in checkpoint_dir
    checkpoint_files = [f for f in checkpoint_files if os.path.dirname(f) == checkpoint_dir]
    if checkpoint_files:
        checkpoint_files.sort(key=extract_epoch)
        latest_checkpoint = checkpoint_files[-1]
        logger.info(f"Found existing checkpoint: {latest_checkpoint}")

        # Load checkpoint
        checkpoint = torch.load(latest_checkpoint, map_location=opt.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']

        # Restore scheduler state
        if 'scheduler_state' in checkpoint:
            # Calculate the correct scheduler position based on training progress
            # If we've completed `start_epoch` epochs out of `opt.epochs` total epochs,
            # the scheduler should be at the corresponding position in the total schedule
            completed_steps = start_epoch * len(dataloader_train)
            scheduler.current_step = completed_steps
            logger.info(f"Set scheduler current_step to {completed_steps} based on {start_epoch} completed epochs")

        logger.info(f"Resuming from epoch {start_epoch}, global step {global_step}")

    # Initialize wandb after we know if we're resuming
    run_name = f"float_{opt.exp_name}"
    if start_epoch > 0:
        run_name += f"_resume_epoch_{start_epoch}"
    wandb.init(project="float", name=run_name)
    # wandb.init(project="debug", name=run_name)

    for epoch in range(start_epoch, opt.epochs):
        logger.info(f"Epoch {epoch+1}/{opt.epochs}")
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(dataloader_train):
            batch_start_time = time.time()

            # Time data loading and transfer
            data_load_start = time.time()
            r_s, audio, metadata = batch['r_s'], batch['audio'], batch['metadata']
            r_s = r_s.to(opt.device)
            audio = audio.to(opt.device)
            data_load_time = time.time() - data_load_start

            # Time forward pass
            forward_start = time.time()
            loss_dict = model(r_s, audio)
            loss = loss_dict['loss']
            loss_ot = loss_dict['loss_ot']
            loss_vel = loss_dict['loss_vel']
            forward_time = time.time() - forward_start

            # Time backward pass
            backward_start = time.time()
            optimizer.zero_grad()
            loss.backward()

            # Compute gradient magnitude before clipping
            grad_norm_before = clip_grad_norm_(optimizer_params, max_norm=float('inf'))

            # Clip gradients to prevent explosion
            max_grad_norm = opt.gc  # You can adjust this value
            clip_grad_norm_(optimizer_params, max_norm=max_grad_norm)

            # Compute gradient magnitude after clipping
            grad_norm_after = clip_grad_norm_(optimizer_params, max_norm=float('inf'))

            optimizer.step()
            backward_time = time.time() - backward_start

            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            total_batch_time = time.time() - batch_start_time
            # Log step-level metrics
            global_step += 1
            # Log timing information every 10 batches
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}: data_load={data_load_time:.3f}s, forward={forward_time:.3f}s, backward={backward_time:.3f}s, total={total_batch_time:.3f}s, lr={current_lr:.6f}, grad_norm_before={grad_norm_before:.3f}, grad_norm_after={grad_norm_after:.3f}")
                wandb.log({
                    "train_loss": loss.item(),
                    "train_loss_ot": loss_ot.item(),
                    "train_loss_vel": loss_vel.item(),
                    "lr": current_lr,
                    "grad_norm_before": grad_norm_before.item() if isinstance(grad_norm_before, torch.Tensor) else grad_norm_before,
                    "grad_norm_after": grad_norm_after.item() if isinstance(grad_norm_after, torch.Tensor) else grad_norm_after,
                }, step=global_step)

        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")

        # Log epoch-level metrics
        wandb.log({
            "epoch_time": epoch_time
        }, step=global_step)

        if epoch % opt.save_every_n_epochs == 0 or epoch == opt.epochs - 1:
            # Save checkpoint with all necessary information for resuming
            checkpoint_path = f"{checkpoint_dir}/{epoch+1:03d}.pth"
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state': {
                    'current_step': scheduler.current_step
                },
                'opt': opt,
                'loss': loss.item() if 'loss' in locals() else None
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

            # Save inference result
            if data_inference is not None:
                model.eval()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                os.makedirs(f"{checkpoint_dir}/results", exist_ok=True)
                with torch.no_grad():
                    inference_video_path = f"{checkpoint_dir}/results/inference_video_{epoch+1:03d}.mp4"
                    inference_video = model.inference_from_latent(data_inference)['d_hat']
                    save_video(inference_video, inference_video_path, audio_path=data_inference['audio_path'], fps=25)
                    logger.info(f"Saved inference video: {inference_video_path}")

                # Release GPU memory after inference
                del inference_video
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                model.fmt.train()
                model.audio_encoder.train()

        # Keep only the last n checkpoints
        checkpoint_files = glob.glob(f"{checkpoint_dir}/*.pth")
        # Filter out files in subdirectories and only keep files directly in checkpoint_dir
        checkpoint_files = [f for f in checkpoint_files if os.path.dirname(f) == checkpoint_dir]

        # Sort by epoch number, with error handling for malformed filenames
        checkpoint_files.sort(key=extract_epoch)

        # Remove older checkpoints if we have more than 5
        if len(checkpoint_files) > opt.keep_last_n_checkpoints:
            files_to_remove = checkpoint_files[:-opt.keep_last_n_checkpoints]
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                    logger.info(f"Removed old checkpoint: {file_path}")
                except OSError as e:
                    logger.info(f"Error removing {file_path}: {e}")

        # Validation
        model.eval()
        val_losses = []
        val_losses_ot = []
        val_losses_vel = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader_val):
                r_s, audio, metadata = batch['r_s'], batch['audio'], batch['metadata']
                r_s = r_s.to(opt.device)
                audio = audio.to(opt.device)

                loss_dict = model(r_s=r_s, audio=audio)
                val_losses.append(loss_dict['loss'].item())
                val_losses_ot.append(loss_dict['loss_ot'].item())
                val_losses_vel.append(loss_dict['loss_vel'].item())

        # Log average validation losses
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_loss_ot = sum(val_losses_ot) / len(val_losses_ot)
        avg_val_loss_vel = sum(val_losses_vel) / len(val_losses_vel)
        wandb.log({
            "val_loss": avg_val_loss,
            "val_loss_ot": avg_val_loss_ot,
            "val_loss_vel": avg_val_loss_vel,\
        }, step=global_step)
        logger.info(f"Validation loss: {avg_val_loss:.4f}, loss_ot: {avg_val_loss_ot:.4f}, loss_vel: {avg_val_loss_vel:.4f}")

        # Reset model to training mode after validation
        model.fmt.train()
        model.audio_encoder.train()

if __name__ == "__main__":
    main()