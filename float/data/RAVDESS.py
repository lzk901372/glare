import os
import torch
import numpy as np
from float.utils import compute_alpha_stats
from .base import BaseDataset
from PIL import Image
import torchaudio
import random
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
from loguru import logger
import h5py

class RAVDESSDataset(BaseDataset):
    def __init__(
        self,
        root_dir,
        window_size=60,
        curr_window_size=50,
        hdf5_window_size=60,
        transform=None,
        audio_transform=None,
        audio_sample_rate=16000,
        fps=25,
        split='train',  # 'train' or 'val'
        train_ratio=0.9,  # Ratio for train split
        random_seed=42,  # For reproducible splits
        wav2vec_model_path='./checkpoints/wav2vec2-base-960h',
        use_speaking_scores=False,
        lia_version='highres',
        lia_path='./checkpoints/float.pth',
        use_alpha_mean=True,
        use_alpha_std=True,
        use_speed_mean=True,
        use_speed_std=True,
        use_accel_mean=True,
        use_accel_std=True,
    ):
        """
        Args:
            root_dir (str): Root directory of the dataset
            window_size (int): Number of frames to sample in each window
            curr_window_size (int): Number of frames in the current window
            hdf5_window_size (int): Number of frames per chunk in the HDF5 file (default: 60)
            transform (callable, optional): Optional transform to be applied on video frames
            audio_transform (callable, optional): Optional transform to be applied on audio
            audio_sample_rate (int): Sample rate for audio
            fps (int): Frames per second of the video
            split (str): 'train' or 'val' to specify which split to use
            train_ratio (float): Ratio of data to use for training (default: 0.9)
            random_seed (int): Random seed for reproducible splits (default: 42)
            wav2vec_model_path (str): Path to the wav2vec model
            use_speaking_scores (bool): Whether to use speaking scores (default: False)
            lia_version (str): Version of LIA (default: 'highres')
            lia_path (str): Path to LIA checkpoint
            use_alpha_mean (bool): Whether to use alpha mean
            use_alpha_std (bool): Whether to use alpha std
            use_speed_mean (bool): Whether to use speed mean
            use_speed_std (bool): Whether to use speed std
            use_accel_mean (bool): Whether to use accel mean
            use_accel_std (bool): Whether to use accel std
        """
        self.hdf5_path = Path(root_dir) / f'hdf5_{lia_version}' / 'RAVDESS_talking_preprocessed.h5'
        super().__init__(
            root_dir,
            window_size,
            curr_window_size,
            hdf5_window_size,
            transform,
            audio_transform,
            audio_sample_rate,
            fps,
            split,
            train_ratio,
            random_seed,
            wav2vec_model_path,
            use_speaking_scores,
            lia_version=lia_version,
            lia_path=lia_path,
            use_alpha_mean=use_alpha_mean,
            use_alpha_std=use_alpha_std,
            use_speed_mean=use_speed_mean,
            use_speed_std=use_speed_std,
            use_accel_mean=use_accel_mean,
            use_accel_std=use_accel_std,
        )
        self.set_audio_length = self.window_size * self.audio_sample_rate // self.fps  # 38400
        assert self.hdf5_window_size == self.window_size, f"RAVDESS supports only hdf5_window_size == window_size == 60"
        ###
        # neutral_pose_path = Path(root_dir) / "neutral_pose" / "RAVDESS" / "v1" / f'latents_{lia_version}.pth'
        # if os.path.isfile(neutral_pose_path):
        #     self.neutral_pose = torch.load(neutral_pose_path)
        #     logger.info(f"Loaded neutral pose for RAVDESS")
        ###

    def get_actor_id_from_sample_id(self, sample_id):
        return "_".join(sample_id.split('_')[:2])

    def get_filename_from_sample_id(self, sample_id):
        return sample_id.split('_')[2] + ".mp4"

    def chunk_path_to_ref_video_path(self, chunk_path):
        # chunk_path: Actor_24_01-01-05-02-02-02-24/chunk_000 -> {root}/Actor_24/01-01-05-02-02-02-24.mp4
        # actor_id: Actor_24
        # filename: 01-01-05-02-02-02-24.mp4
        # chunk_id: chunk_000

        sample_id = chunk_path.split('/')[0]
        chunk_idx = int(chunk_path.split('/')[1].split('_')[1])
        start_frame_idx = chunk_idx * 60
        actor_id = self.get_actor_id_from_sample_id(sample_id)
        filename = self.get_filename_from_sample_id(sample_id)
        return f"{actor_id}/{filename}", start_frame_idx

    def index_dataset(self):
        """
        Index RAVDESS dataset by finding all batch-video chunks with sufficient frames.
        The dataset structure expected:
        ├── RAVDESS (flat structure in HDF5):
        │   ├── r_s/
        │   │   ├── Actor_01_01-01-01-01-01-01-01/chunk_000  # Motion latents (shape: [60, motion_dim])
        │   │   ├── Actor_01_01-01-01-01-01-01-01/chunk_001
        │   │   ├── Actor_24_02-02-06-01-02-02-24/chunk_000
        │   │   └── ...
        │   └── audio/
        │       ├── Actor_01_01-01-01-01-01-01-01/chunk_000  # Audio data
        │       ├── Actor_01_01-01-01-01-01-01-01/chunk_001
        │       └── ...
        """
        all_subdirs = []
        all_identities = []
        sample_chunk_num = {}
        with h5py.File(self.hdf5_path, 'r') as f:
            for sample_id in f['r_s'].keys():
                actor_id = self.get_actor_id_from_sample_id(sample_id)
                all_identities.append(actor_id)
                chunks = f[f'r_s/{sample_id}'].keys()
                sample_chunk_num[sample_id] = len(chunks)
                for i, chunk in enumerate(chunks):
                    if f[f'r_s/{sample_id}/{chunk}'][()].shape[0] != 60:
                        continue
                    all_subdirs.append(f"{sample_id}/{chunk}")
        self.sample_chunk_num = sample_chunk_num
        all_identities = sorted(list(set(all_identities)))

        # Create reproducible train/val split
        random.seed(self.random_seed)
        random.shuffle(all_identities)
        random.shuffle(all_subdirs)

        split_idx = int(len(all_identities) * self.train_ratio)

        if self.split == 'train':
            selected_identities = all_identities[:split_idx]
            self.subdirs = [x for x in all_subdirs if self.get_actor_id_from_sample_id(x.split('/')[0]) in selected_identities]
        elif self.split == 'val':
            selected_identities = all_identities[split_idx:]
            self.subdirs = [x for x in all_subdirs if self.get_actor_id_from_sample_id(x.split('/')[0]) in selected_identities]
        else:
            raise ValueError(f"Split must be 'train' or 'val', got {self.split}")

        print(f"RAVDESS: Created {self.split} split with {len(self.subdirs)} chunks "
              f"({len(selected_identities)} identities, {len(all_identities)} total identities)")

    def __len__(self):
        return len(self.subdirs)

    def __getitem__(self, index):
        chunk_path = self.subdirs[index]
        sample_id = chunk_path.split('/')[0]
        chunk_name = chunk_path.split('/')[1]
        chunk_id = int(chunk_name.split('_')[1])
        actor_id = self.get_actor_id_from_sample_id(sample_id)

        if chunk_id + 1 >= self.sample_chunk_num[sample_id] - 1:
            # chunk_id = 0
            with h5py.File(self.hdf5_path, 'r') as f:
                audio = f[f'audio/{chunk_path}'][:]
                audio = torch.from_numpy(audio).float().squeeze(0)
                r_s = f[f'r_s/{chunk_path}'][:]
                r_s = torch.from_numpy(r_s).float()
        else:
            next_chunk_path = f"{sample_id}/chunk_{chunk_id + 1:03d}"
            with h5py.File(self.hdf5_path, 'r') as f:
                # 1st chunk
                audio1 = f[f'audio/{chunk_path}'][:]
                audio1 = torch.from_numpy(audio1).float().squeeze(0)
                r_s1 = f[f'r_s/{chunk_path}'][:]
                r_s1 = torch.from_numpy(r_s1).float()

                # 2nd chunk
                audio2 = f[f'audio/{next_chunk_path}'][:]
                audio2 = torch.from_numpy(audio2).float().squeeze(0)
                r_s2 = f[f'r_s/{next_chunk_path}'][:]
                r_s2 = torch.from_numpy(r_s2).float()

                audio = torch.cat([audio1, audio2], dim=0)
                r_s = torch.cat([r_s1, r_s2], dim=0)

                start_frame = random.randint(0, r_s.shape[0] - self.window_size)
                end_frame = start_frame + self.window_size
                r_s = r_s[start_frame:end_frame, :]

                audio_start = start_frame * self.audio_sample_rate // self.fps
                audio_end = audio_start + self.window_size * self.audio_sample_rate // self.fps
                audio = audio[audio_start:audio_end]

        alpha = torch.matmul(r_s, self.lia_Q)
        alpha_dict = compute_alpha_stats(
            alpha[-self.curr_window_size:, :],
            use_alpha_mean=self.use_alpha_mean,
            use_alpha_std=self.use_alpha_std,
            use_speed_mean=self.use_speed_mean,
            use_speed_std=self.use_speed_std,
            use_accel_mean=self.use_accel_mean,
            use_accel_std=self.use_accel_std,
        )

        if self.audio_transform:
            audio = self.audio_transform(audio)

        assert audio.shape[0] == int(self.set_audio_length), f"sample_id: {sample_id}, chunk_name: {chunk_name}, audio.shape: {audio.shape}, sr: {self.audio_sample_rate}, window_size: {self.window_size}, fps: {self.fps}"

        ref_video_path, start_frame_idx = self.chunk_path_to_ref_video_path(chunk_path)
        item = {
            'r_s': r_s, # [T, 512]
            'audio': audio, # [T*sr/25]
            'alpha_dict': alpha_dict,
            'is_speaking': True,
            'metadata': {
                'num_frame': r_s.shape[0],
                'ref_video_path': ref_video_path,
                'start_frame_idx': start_frame_idx,
            },
        }
        # if hasattr(self, "neutral_pose"):
        #     item['r_s_refvid_0'] = self.neutral_pose[actor_id]
        if self.use_speaking_scores:
            item['speak_scores'] = torch.ones_like(r_s[:, :1])
        return item

    def get_metadata(self, subdir):
        metadata = {}

        return metadata
