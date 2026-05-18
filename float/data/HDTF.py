import torch
from float.utils import compute_alpha_stats
from .base import BaseDataset
from pathlib import Path
import random
import h5py
import pandas as pd
from loguru import logger
import os

class HDTFDataset(BaseDataset):
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
        split='train',
        train_ratio=0.9,
        random_seed=42,
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
        self.hdf5_path = Path(root_dir) / f'hdf5_{lia_version}' / 'HDTF_talking_preprocessed.h5'
        # Filter out videos with hyperiqa>=50 and low self-reconstruction quality
        self.self_reconstruction_quality_path = Path(root_dir) / "self-reconstruction_psnr" / "HDTF_visual_quality.csv"
        df_self_reconstruction_quality = pd.read_csv(self.self_reconstruction_quality_path)
        filtered_self_reconstruction = list(df_self_reconstruction_quality[df_self_reconstruction_quality["PSNR_dB"] > 23]["vid_name"]) # > 23dB
        self.video_ids = [os.path.basename(v).replace(".mp4", "") for v in filtered_self_reconstruction]
        logger.info(f"HDTFDataset: {len(self.video_ids)} videos after filtering")
        ###
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

    def chunk_path_to_ref_video_path(self, chunk_path):
        # chunk_path: WRA_EricCantor_000/chunk_000
        # filename: WRA_EricCantor_000.mp4
        # chunk_id: chunk_000

        sample_id = chunk_path.split('/')[0]
        chunk_idx = int(chunk_path.split('/')[1].split('_')[1])
        start_frame_idx = chunk_idx * 60
        return sample_id, start_frame_idx

    def __len__(self):
        return len(self.subdirs)

    def index_dataset(self):
        """
        Index HDTF dataset by finding all batch-video chunks with sufficient frames.
        The dataset structure expected:
        ├── HDTF (flat structure in HDF5):
        │   ├── r_s/
        │   │   ├── RD_Radio10_000/chunk_000  # Motion latents (shape: [60, motion_dim])
        │   │   ├── RD_Radio10_000/chunk_001
        │   │   ├── RD_Radio10_000/chunk_002
        │   │   └── ...
        │   └── audio/
        │       ├── RD_Radio10_000/chunk_000  # Audio data
        │       ├── RD_Radio10_000/chunk_001
        │       └── ...
        """
        all_subdirs = []
        all_identities = []
        identities_chunk_num = {}
        with h5py.File(self.hdf5_path, 'r') as f:
            for actor_id in f['r_s'].keys():
                if actor_id not in self.video_ids:
                    continue
                all_identities.append(actor_id)
                chunks = sorted(list(f[f'r_s/{actor_id}'].keys()))
                num_curr_chunks = len(chunks)
                identities_chunk_num[actor_id] = num_curr_chunks
                if num_curr_chunks < self.min_num_chunks:
                    continue
                for i, chunk in enumerate(chunks):
                    # Requires at least min_num_chunks chunks
                    # index of the last chunk to include >= num_curr_chunks
                    if i + self.min_num_chunks - 1 >= num_curr_chunks:
                        continue
                    if f[f'r_s/{actor_id}/{chunk}'][()].shape[0] != self.hdf5_window_size:
                        raise ValueError(f"Chunk {chunk} has shape {f[f'r_s/{actor_id}/{chunk}'][()].shape[0]} != 60")
                    all_subdirs.append(f"{actor_id}/{chunk}")
        self.identities_chunk_num = identities_chunk_num

        # Create reproducible train/val split
        random.seed(self.random_seed)
        random.shuffle(all_identities)
        random.shuffle(all_subdirs)

        split_idx = int(len(all_identities) * self.train_ratio)

        if self.split == 'train':
            selected_identities = all_identities[:split_idx]
            self.subdirs = [x for x in all_subdirs if x.split('/')[0] in selected_identities]
        elif self.split == 'val':
            selected_identities = all_identities[split_idx:]
            self.subdirs = [x for x in all_subdirs if x.split('/')[0] in selected_identities]
        else:
            raise ValueError(f"Split must be 'train' or 'val', got {self.split}")

        print(f"HDTF: Created {self.split} split with {len(selected_identities)} videos "
              f"(total videos: {len(all_identities)}, total chunks: {len(all_subdirs)})")

    def __getitem__(self, index):
        chunk_path = self.subdirs[index]
        actor_id = chunk_path.split('/')[0]
        chunk_name = chunk_path.split('/')[1]
        chunk_id = int(chunk_name.split('_')[1])

        if chunk_id + self.min_num_chunks - 1 >= self.identities_chunk_num[actor_id] - 1:
            # When the index of the last chunk to include >= index of the last possible chunk
            with h5py.File(self.hdf5_path, 'r') as f:
                audio = []
                r_s = []
                for idx_chunk in range(self.min_num_chunks):
                    chunk_path = f"{actor_id}/chunk_{chunk_id + idx_chunk:03d}"
                    if idx_chunk == 0:
                        start_chunk_path = chunk_path
                    audio_now = f[f'audio/{chunk_path}'][:]
                    audio_now = torch.from_numpy(audio_now).float().squeeze(0)
                    r_s_now = f[f'r_s/{chunk_path}'][:]
                    r_s_now = torch.from_numpy(r_s_now).float()
                    audio.append(audio_now)
                    r_s.append(r_s_now)

                audio = torch.cat(audio, dim=0)
                r_s = torch.cat(r_s, dim=0)
                audio = audio[:self.window_size * self.audio_sample_rate // self.fps]
                r_s = r_s[:self.window_size, :]
        else:
            # When the index of the last chunk to include < index of the last possible chunk
            with h5py.File(self.hdf5_path, 'r') as f:
                audio = []
                r_s = []
                # Add the next chunk to allow the random start frame within the first chunk
                for idx_chunk in range(self.min_num_chunks + 1):
                    chunk_path = f"{actor_id}/chunk_{chunk_id + idx_chunk:03d}"
                    if idx_chunk == 0:
                        start_chunk_path = chunk_path
                    audio_now = f[f'audio/{chunk_path}'][:]
                    audio_now = torch.from_numpy(audio_now).float().squeeze(0)
                    r_s_now = f[f'r_s/{chunk_path}'][:]
                    r_s_now = torch.from_numpy(r_s_now).float()
                    audio.append(audio_now)
                    r_s.append(r_s_now)

                audio = torch.cat(audio, dim=0)
                r_s = torch.cat(r_s, dim=0)

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

        assert audio.shape[0] == int(self.set_audio_length), f"actor_id: {actor_id}, chunk_name: {chunk_name}, audio.shape: {audio.shape}, sr: {self.audio_sample_rate}, window_size: {self.window_size}, fps: {self.fps}"

        ref_video_path, start_frame_idx = self.chunk_path_to_ref_video_path(start_chunk_path)
        item = {
            'r_s': r_s, # [T, 512]
            'audio': audio, # [T*sr/25]
            'alpha_dict': alpha_dict,
            'is_speaking': True,
            'metadata': {
                'num_frame': r_s.shape[0],
                'ref_video_path': ref_video_path,
                'start_frame_idx': start_frame_idx,
            }
        }
        if self.use_speaking_scores:
            item['speak_scores'] = torch.ones_like(r_s[:, :1])
        return item

    def get_metadata(self, subdir):
        metadata = {
            'source': subdir.parent.name,
        }

        return metadata
