import torch
from .base import BaseDataset
import random
from pathlib import Path
from loguru import logger
import h5py
import os
import pandas as pd
import json
from float.utils import compute_alpha_stats

def get_video_id_from_rel_video_path(rel_video_path):
    return rel_video_path.split("/")[-1].split(".")[0]

def read_lines(path):
    return [line.strip() for line in open(path, "r").readlines()]

class VideoDataset(BaseDataset):
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
            lia_path (str): Path to LIA checkpoint (default: './checkpoints/float.pth')
            use_alpha_mean (bool): Whether to use alpha mean (default: True)
            use_alpha_std (bool): Whether to use alpha std (default: True)
            use_speed_mean (bool): Whether to use speed mean (default: True)
            use_speed_std (bool): Whether to use speed std (default: True)
            use_accel_mean (bool): Whether to use accel mean (default: True)
            use_accel_std (bool): Whether to use accel std (default: True)
        """
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


    def index_dataset(self):
        """
        Index The dataset by finding all batch-video chunks with sufficient frames.
        The dataset structure expected:
        ├── {DatasetName} (flat structure in HDF5):
        │   ├── r_s/
        │   │   ├── video_id_000/chunk_000  # Motion latents (shape: [60, motion_dim])
        │   │   ├── video_id_000/chunk_001
        │   │   ├── video_id_000/chunk_002
        │   │   └── ...
        │   └── audio/
        │       ├── video_id_000/chunk_000  # Audio data
        │       ├── video_id_000/chunk_001
        │       └── ...
        """
        all_subdirs = [] # List of chunk_paths (e.g. "video_id_000/chunk_000")
        all_videos = []  # List of video_ids (e.g. "video_id_000")
        video_chunk_num = {}

        with h5py.File(self.hdf5_path, 'r') as f:
            # Iterate through all chunk paths in the flat structure
            # Parse chunk_path like "video_id_000/chunk_000"
            for video_id in f['r_s'].keys():
                if f"{video_id}" not in self.video_ids:
                    logger.debug(f"video_id: {video_id} not in self.video_ids")
                    continue
                chunks = sorted(list(f['r_s'][video_id].keys()))
                num_curr_chunks = len(chunks)
                if num_curr_chunks < self.min_num_chunks:
                    # logger.debug(f"video_id: {video_id} has less than {self.min_num_chunks} chunks, skipping")
                    continue
                for i, chunk_id in enumerate(chunks):
                    # Requires at least min_num_chunks chunks
                    # index of the last chunk to include >= num_curr_chunks
                    if i + self.min_num_chunks - 1 >= num_curr_chunks:
                        # logger.debug(f"video_id: {video_id} has more than {self.min_num_chunks} chunks, skipping")
                        continue
                    chunk_path = f"{video_id}/{chunk_id}"
                    if f'scores/{chunk_path}' in f:
                        if f[f'scores/{chunk_path}'][:].shape[0] != self.hdf5_window_size:
                            print(f"Warning: Scores too short for {chunk_path} (expected 60, got {f[f'scores/{chunk_path}'][:].shape[0]})")
                            continue
                    all_subdirs.append(chunk_path)
                all_videos.append(f"{video_id}")

                # Count chunks per batch/video pair
                if video_id not in video_chunk_num:
                    video_chunk_num[video_id] = 0
                video_chunk_num[video_id] += 1

        all_videos = sorted(list(set(all_videos))) # List of video_ids
        self.video_chunk_num = video_chunk_num
        print(f"len(all_videos): {len(all_videos)}")
        print(f"len(all_subdirs): {len(all_subdirs)}")
        print(f"all_videos: {all_videos[:10]}")
        print(f"all_subdirs: {all_subdirs[:10]}")

        # Create reproducible train/val split
        random.seed(self.random_seed)
        random.shuffle(all_videos)
        random.shuffle(all_subdirs)

        split_idx = int(len(all_videos) * self.train_ratio)

        if self.split == 'train':
            selected_videos = set(all_videos[:split_idx])  # List of video_ids
            self.subdirs = [x for x in all_subdirs if x.split('/')[0] in selected_videos] # List of chunk_paths (e.g. "video_id_000/chunk_000")
        elif self.split == 'val':
            selected_videos = set(all_videos[split_idx:])
            self.subdirs = [x for x in all_subdirs if x.split('/')[0] in selected_videos] # List of chunk_paths (e.g. "video_id_000/chunk_000")
        else:
            raise ValueError(f"Split must be 'train' or 'val', got {self.split}")

        logger.info(f"Created {self.split} split with {len(self.subdirs)} videos "
              f"(total videos: {len(all_videos)}, total chunks: {len(all_subdirs)})")
        logger.info(f"split: {self.split}, selected_videos: {list(selected_videos)[:10]}")

    def __len__(self):
        return len(self.subdirs)

    def __getitem__(self, index):
        chunk_path = self.subdirs[index]
        video_id = chunk_path.split('/')[0]
        chunk_id = int(chunk_path.split('/')[-1].split('_')[1])

        if chunk_id + self.min_num_chunks - 1 >= self.video_chunk_num[f"{video_id}"] - 1:
            # When the index of the last chunk to include >= index of the last possible chunk
            with h5py.File(self.hdf5_path, 'r') as f:
                audio = []
                r_s = []

                reaction_scores = []
                intensity_scores = []

                for idx_chunk in range(self.min_num_chunks):
                    chunk_path = f"{video_id}/chunk_{chunk_id + idx_chunk:03d}"
                    if idx_chunk == 0:
                        start_chunk_path = chunk_path
                    audio_now = f[f'audio/{chunk_path}'][:]
                    audio_now = torch.from_numpy(audio_now).float().squeeze(0)
                    r_s_now = f[f'r_s/{chunk_path}'][:]
                    r_s_now = torch.from_numpy(r_s_now).float()
                    audio.append(audio_now)
                    r_s.append(r_s_now)
                    
                    if f"reaction/{chunk_path}" in f:
                        reaction_scores_now = f[f'reaction/{chunk_path}'][:]
                        reaction_scores_now = torch.from_numpy(reaction_scores_now).float()
                        reaction_scores.append(reaction_scores_now)
                    if f"intensity/{chunk_path}" in f:
                        intensity_scores_now = f[f'intensity/{chunk_path}'][:]
                        intensity_scores_now = torch.from_numpy(intensity_scores_now).float()
                        intensity_scores.append(intensity_scores_now)

                audio = torch.cat(audio, dim=0)
                r_s = torch.cat(r_s, dim=0)

                if f"reaction/{chunk_path}" in f:
                    reaction_scores = torch.cat(reaction_scores, dim=0)
                if f"intensity/{chunk_path}" in f:
                    intensity_scores = torch.cat(intensity_scores, dim=0)
                
                audio = audio[:self.window_size * self.audio_sample_rate // self.fps]
                r_s = r_s[:self.window_size, :]

                if f"reaction/{chunk_path}" in f:
                    reaction_scores = reaction_scores[:self.window_size]
                if f"intensity/{chunk_path}" in f:
                    intensity_scores = intensity_scores[:self.window_size]
                # logger.info(f"From start. audio.shape: {audio.shape}, r_s.shape: {r_s.shape}")
                #
                if f'scores/{chunk_path}' in f:
                    speak_scores = f[f'scores/{chunk_path}'][:]
                    speak_scores = torch.from_numpy(speak_scores).float()
        else:
            # When the index of the last chunk to include < index of the last possible chunk
            with h5py.File(self.hdf5_path, 'r') as f:
                audio = []
                r_s = []
                speak_scores = []

                reaction_scores = []
                intensity_scores = []

                # Add the next chunk to allow the random start frame within the first chunk
                for idx_chunk in range(self.min_num_chunks + 1):
                    chunk_path = f"{video_id}/chunk_{chunk_id + idx_chunk:03d}"
                    if idx_chunk == 0:
                        start_chunk_path = chunk_path
                    audio_now = f[f'audio/{chunk_path}'][:]
                    audio_now = torch.from_numpy(audio_now).float().squeeze(0)
                    r_s_now = f[f'r_s/{chunk_path}'][:]
                    r_s_now = torch.from_numpy(r_s_now).float()
                    audio.append(audio_now)
                    r_s.append(r_s_now)
                    if f'scores/{chunk_path}' in f:
                        speak_scores_now = f[f'scores/{chunk_path}'][:]
                        speak_scores_now = torch.from_numpy(speak_scores_now).float()
                        speak_scores.append(speak_scores_now)
                    
                    if f"reaction/{chunk_path}" in f:
                        reaction_scores_now = f[f'reaction/{chunk_path}'][:]
                        reaction_scores_now = torch.from_numpy(reaction_scores_now).float()
                        reaction_scores.append(reaction_scores_now)
                    if f"intensity/{chunk_path}" in f:
                        intensity_scores_now = f[f'intensity/{chunk_path}'][:]
                        intensity_scores_now = torch.from_numpy(intensity_scores_now).float()
                        intensity_scores.append(intensity_scores_now)

                audio = torch.cat(audio, dim=0)
                r_s = torch.cat(r_s, dim=0)
                if f'scores/{chunk_path}':
                    speak_scores = torch.cat(speak_scores, dim=0)
                
                if f"reaction/{chunk_path}" in f:
                    reaction_scores = torch.cat(reaction_scores, dim=0)
                if f"intensity/{chunk_path}" in f:
                    intensity_scores = torch.cat(intensity_scores, dim=0)

                # sample window
                start_frame = random.randint(0, r_s.shape[0] - self.window_size)
                end_frame = start_frame + self.window_size
                r_s = r_s[start_frame:end_frame, :] # [T, 512]
                if f'scores/{chunk_path}':
                    speak_scores = speak_scores[start_frame:end_frame]
                
                if f"reaction/{chunk_path}" in f:
                    reaction_scores = reaction_scores[start_frame:end_frame]
                if f"intensity/{chunk_path}" in f:
                    intensity_scores = intensity_scores[start_frame:end_frame]

                audio_start = start_frame * self.audio_sample_rate // self.fps
                audio_end = audio_start + self.window_size * self.audio_sample_rate // self.fps
                audio = audio[audio_start:audio_end]
                # logger.info(f"From middle. audio.shape: {audio.shape}, r_s.shape: {r_s.shape}")

        # compute alpha
        alpha = torch.matmul(r_s, self.lia_Q) # [T, 20]
        alpha_dict = compute_alpha_stats(
            alpha[-self.curr_window_size:, :],
            use_alpha_mean=self.use_alpha_mean,
            use_alpha_std=self.use_alpha_std,
            use_speed_mean=self.use_speed_mean,
            use_speed_std=self.use_speed_std,
            use_accel_mean=self.use_accel_mean,
            use_accel_std=self.use_accel_std
        )

        if self.audio_transform:
            audio = self.audio_transform(audio)

        assert audio.shape[0] == int(self.set_audio_length), f"video_id: {video_id}, chunk_name: {chunk_path}, audio.shape: {audio.shape}, sr: {self.audio_sample_rate}, window_size: {self.window_size}, fps: {self.fps}"

        ref_video_path, start_frame_idx = self.chunk_path_to_ref_video_path(start_chunk_path)
        item = {
            'r_s': r_s, # [T, 512]
            'alpha_dict': alpha_dict,
            'audio': audio, # [T*sr/25]
            'reaction': reaction_scores, # [T, 1]
            'intensity': intensity_scores.unsqueeze(-1), # [T, 1]
            'metadata': {
                'num_frame': r_s.shape[0],
                'ref_video_path': ref_video_path,
                'start_frame_idx': start_frame_idx,
            }
        }
        if self.use_speaking_scores:
            if 'speak_scores' in locals():
                # logger.info(f"speak_scores: {speak_scores}")
                item['speak_scores'] = speak_scores.unsqueeze(1) # [T, 1]
            else:
                item['speak_scores'] = None
        return item

    def chunk_path_to_ref_video_path(self, chunk_path):
        # chunk_path: video_id_000/chunk_000 -> video_id_000.mp4
        # video_id: video_id_000
        # chunk_id: chunk_000
        video_id = chunk_path.split('/')[0]
        chunk_idx = int(chunk_path.split('/')[1].split('_')[1])
        start_frame_idx = chunk_idx * 60
        return f"{video_id}.mp4", start_frame_idx

    def get_metadata(self, subdir):
        return {}


class Hallo3Dataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'Hallo3_talking_preprocessed.h5'
        # Filter out videos with hyperiqa>=50 and low self-reconstruction quality
        self.self_reconstruction_quality_path = Path(root_dir) / "self-reconstruction_psnr" / "Hallo3_visual_quality.csv"
        df_self_reconstruction_quality = pd.read_csv(self.self_reconstruction_quality_path)
        filtered_self_reconstruction = list(df_self_reconstruction_quality[df_self_reconstruction_quality["PSNR_dB"] > 23]["vid_name"]) # > 23dB
        self.video_ids = [v.replace(".mp4", "") for v in filtered_self_reconstruction]
        logger.info(f"Hallo3Dataset: {len(self.video_ids)} videos after filtering")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = True
        if self.use_speaking_scores and item['speak_scores'] is None:
            item['speak_scores'] = torch.ones_like(item['r_s'][:, :1])
        return item

# class TheSkinDeepListeningDataset(VideoDataset):
#     def __init__(self, *args, **kwargs):
#         root_dir = kwargs.get("root_dir")
#         self.hdf5_path = Path(root_dir) / f'hdf5_{self.lia_version}' / 'TheSkinDeep_listening_preprocessed.h5'
#         self.pseudo_label_path = Path(root_dir) / 'TheSkinDeep_listening_labler_pseudo_labels_train_ratio_0.9_val_ratio_0.1_POS_TH_0.06.csv' # ~30%
#         self.more_than_one_person_paths = read_lines(Path(root_dir) / 'the-skin-deep_more_than_one.txt')
#         self.more_than_one_person_paths = [os.path.basename(line).split(".")[0] for line in self.more_than_one_person_paths]
#         self.rel_video_paths = [line.split(",")[0].strip() for line in open(self.pseudo_label_path, "r").readlines()[1:]]
#         self.video_ids = [
#             get_video_id_from_rel_video_path(rel_video_path)
#             for rel_video_path in self.rel_video_paths
#             if os.path.basename(rel_video_path).split(".")[0] not in self.more_than_one_person_paths
#         ]
#         logger.info(f"len(more_than_one_person_paths): {len(list(self.more_than_one_person_paths))}")
#         logger.info(f"more_than_one_person_paths: {list(self.more_than_one_person_paths)[:10]}")
#         logger.info(f"rel_video_paths: {self.rel_video_paths[:10]}")
#         logger.info(f"TheSkinDeepListeningDataset: {len(self.rel_video_paths)} videos, {len(self.video_ids)} videos with single speaker")
#         super().__init__(*args, **kwargs)

#     def __getitem__(self, index):
#         item = super().__getitem__(index)
#         item['is_speaking'] = False
#         if self.use_speaking_scores and item['speak_scores'] is None:
#             item['speak_scores'] = -torch.ones_like(item['r_s'][:, :1])
#         return item

# class TheSkinDeepTalkingDataset(VideoDataset):
#     def __init__(self, *args, **kwargs):
#         root_dir = kwargs.get("root_dir")
#         self.hdf5_path = Path(root_dir) / f'hdf5_{self.lia_version}' / 'TheSkinDeep_talking_preprocessed.h5'
#         self.pseudo_label_path = Path(root_dir) / 'TheSkinDeep_talking_labler_pseudo_labels_train_ratio_0.9_val_ratio_0.1_POS_TH_0.06.csv'
#         self.rel_video_paths = [line.split(",")[0].strip() for line in open(self.pseudo_label_path, "r").readlines()[1:]]
#         self.video_ids = [get_video_id_from_rel_video_path(rel_video_path) for rel_video_path in self.rel_video_paths]
#         super().__init__(*args, **kwargs)

#     def __getitem__(self, index):
#         item = super().__getitem__(index)
#         item['is_speaking'] = True
#         if self.use_speaking_scores and item['speak_scores'] is None:
#             item['speak_scores'] = torch.ones_like(item['r_s'][:, :1])
#         return item

class PsychologyIsListeningDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'PsychologyIs_listening_preprocessed.h5'
        self.more_than_one_person_paths = read_lines(Path(root_dir) / 'more_than_one_speakers' / 'PsychologyIs.txt')
        self.more_than_one_person_paths = [os.path.basename(line).split(".")[0] for line in self.more_than_one_person_paths]
        # Filter out videos with hyperiqa>=50 and low self-reconstruction quality
        self.self_reconstruction_quality_path = Path(root_dir) / "self-reconstruction_psnr" / "PsychologyIs_visual_quality.csv"
        df_self_reconstruction_quality = pd.read_csv(self.self_reconstruction_quality_path)
        filtered_self_reconstruction = set(df_self_reconstruction_quality[df_self_reconstruction_quality["PSNR_dB"] > 23]["vid_name"]) # > 23dB
        filtered_self_reconstruction = {v.replace(".mp4", "") for v in filtered_self_reconstruction}
        ###
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        logger.info(f"PsychologyIsListeningDataset: {len(self.video_ids)} videos")
        self.video_ids = [
            video_id for video_id in self.video_ids
            if video_id not in self.more_than_one_person_paths and
            "ge7P5zl5koY" not in video_id and
            video_id in filtered_self_reconstruction
        ]
        logger.info(f"PsychologyIsListeningDataset: {len(self.video_ids)} videos after filtering")
        ### Filter out videos with low visual quality
        selected_video_ids = set(pd.read_csv(Path(root_dir) / 'csvs_iqa40' / 'PsychologyIs_visual_quality.csv')['video_id'].tolist())
        self.video_ids = [video_id for video_id in self.video_ids if video_id in selected_video_ids]
        logger.info(f"Selected {len(self.video_ids)} video ids from {len(selected_video_ids)} selected video ids. {len(self.video_ids) / len(selected_video_ids) * 100:.2f}%")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = False
        if self.use_speaking_scores and item['speak_scores'] is None:
            item['speak_scores'] = -torch.ones_like(item['r_s'][:, :1])
        return item

class AmplifyMeHubListeningDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'AmplifyMeHub_listening_preprocessed.h5'
        self.more_than_one_person_paths = read_lines(Path(root_dir) / 'more_than_one_speakers' / 'AmplifyMeHub.txt')
        self.more_than_one_person_paths = [os.path.basename(line).split(".")[0] for line in self.more_than_one_person_paths]
        # Filter out videos with hyperiqa>=50 and low self-reconstruction quality
        self.self_reconstruction_quality_path = Path(root_dir) / "self-reconstruction_psnr" / "AmplifyMeHub_visual_quality.csv"
        df_self_reconstruction_quality = pd.read_csv(self.self_reconstruction_quality_path)
        filtered_self_reconstruction = set(df_self_reconstruction_quality[df_self_reconstruction_quality["PSNR_dB"] > 23]["vid_name"]) # > 23dB
        filtered_self_reconstruction = {v.replace(".mp4", "") for v in filtered_self_reconstruction}
        ###
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        logger.info(f"AmplifyMeHubListeningDataset: {len(self.video_ids)} videos")
        self.video_ids = [
            video_id for video_id in self.video_ids
            if video_id not in self.more_than_one_person_paths and
            video_id in filtered_self_reconstruction
        ]
        logger.info(f"AmplifyMeHubListeningDataset: {len(self.video_ids)} videos after filtering")
        ### Filter out videos with low visual quality
        selected_video_ids = set(pd.read_csv(Path(root_dir) / 'csvs_iqa40' / 'AmplifyMeHub_visual_quality.csv')['video_id'].tolist())
        self.video_ids = [video_id for video_id in self.video_ids if video_id in selected_video_ids]
        logger.info(f"Selected {len(self.video_ids)} video ids from {len(selected_video_ids)} selected video ids. {len(self.video_ids) / len(selected_video_ids) * 100:.2f}%")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = False
        if self.use_speaking_scores and item['speak_scores'] is None:
            item['speak_scores'] = -torch.ones_like(item['r_s'][:, :1])
        return item

class TheWellnessTheoryListeningDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'TheWellnessTheory_listening_preprocessed.h5'
        self.more_than_one_person_paths = read_lines(Path(root_dir) / 'more_than_one_speakers' / 'TheWellnessTheory.txt')
        self.more_than_one_person_paths = [os.path.basename(line).split(".")[0] for line in self.more_than_one_person_paths]
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        logger.info(f"TheWellnessTheoryListeningDataset: {len(self.video_ids)} videos")
        self.video_ids = [
            video_id for video_id in self.video_ids
            if video_id not in self.more_than_one_person_paths
        ]
        logger.info(f"TheWellnessTheoryListeningDataset: {len(self.video_ids)} videos after filtering")
        ### Filter out videos with low visual quality
        selected_video_ids = set(pd.read_csv(Path(root_dir) / 'csvs_iqa40' / 'TheWellnessTheory_visual_quality.csv')['video_id'].tolist())
        self.video_ids = [video_id for video_id in self.video_ids if video_id in selected_video_ids]
        logger.info(f"Selected {len(self.video_ids)} video ids from {len(selected_video_ids)} selected video ids. {len(self.video_ids) / len(selected_video_ids) * 100:.2f}%")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = False
        if self.use_speaking_scores and item['speak_scores'] is None:
            item['speak_scores'] = -torch.ones_like(item['r_s'][:, :1])
        return item

class RealTalkListeningDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        # self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'RealTalk_listening_preprocessed.h5'
        self.hdf5_path = "/data/zikai/Data/H5_files/hdf5_reaction/RealTalk_listening_preprocessed_reaction_pruned_intensity.h5"
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        
        ### Filter out videos with hyperiqa>=50 and low self-reconstruction quality
        # self.self_reconstruction_quality_path = Path(root_dir) / "self-reconstruction_psnr" / "RealTalk_visual_quality.csv"
        # df_self_reconstruction_quality = pd.read_csv(self.self_reconstruction_quality_path)
        # filtered_self_reconstruction = set(df_self_reconstruction_quality[df_self_reconstruction_quality["PSNR_dB"] > 23]["vid_name"]) # > 23dB
        # filtered_self_reconstruction = {v.replace(".mp4", "") for v in filtered_self_reconstruction}
        # #
        # logger.info(f"RealTalkListeningDataset: {len(self.video_ids)} videos")
        # self.video_ids = [
        #     video_id for video_id in self.video_ids
        #     if video_id in filtered_self_reconstruction
        # ]

        logger.info(f"RealTalkListeningDataset: {len(self.video_ids)} videos after filtering")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = False
        if self.use_speaking_scores and item['speak_scores'] is None:
            logger.info(f"speak_scores is None for RealTalkListening {index}!!!!")
            item['speak_scores'] = -torch.ones_like(item['r_s'][:, :1])
        return item

class InterviewV1TalkingDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'interview_talking_preprocessed.h5'
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        logger.info(f"InterviewV1TalkingDataset: {len(self.video_ids)} videos")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = True
        if self.use_speaking_scores and item['speak_scores'] is None:
            logger.info(f"speak_scores is None for InterviewV1TalkingDataset {index}!!!!")
            item['speak_scores'] = torch.ones_like(item['r_s'][:, :1])
        return item

class InterviewV3TalkingDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'Interview_v3_talking_preprocessed.h5'
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        ### Filter out videos with low visual quality
        selected_video_ids = set(pd.read_csv(Path(root_dir) / 'csvs_iqa50' / 'Interview_v3_talking_visual_quality.csv')['video_id'].tolist())
        self.video_ids = [video_id for video_id in self.video_ids if video_id in selected_video_ids]
        logger.info(f"Selected {len(self.video_ids)} video ids from {len(selected_video_ids)} selected video ids. {len(self.video_ids) / len(selected_video_ids) * 100:.2f}%")
        logger.info(f"InterviewV3TalkingDataset: {len(self.video_ids)} videos")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = True
        if self.use_speaking_scores and item['speak_scores'] is None:
            logger.info(f"speak_scores is None for InterviewV3TalkingDataset {index}!!!!")
            item['speak_scores'] = torch.ones_like(item['r_s'][:, :1])
        return item

class InterviewV1ListeningDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'interview_listening_preprocessed.h5'
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        logger.info(f"InterviewV1ListeningDataset: {len(self.video_ids)} videos")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = False
        if self.use_speaking_scores and item['speak_scores'] is None:
            logger.info(f"speak_scores is None for InterviewV1ListeningDataset {index}!!!!")
            item['speak_scores'] = -torch.ones_like(item['r_s'][:, :1])
        return item

class SeamlessV1ListeningDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'Seamless_v1_listening_preprocessed.h5'
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        logger.info(f"SeamlessV1ListeningDataset: {len(self.video_ids)} videos")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = False
        if self.use_speaking_scores and item['speak_scores'] is None:
            logger.info(f"speak_scores is None for SeamlessV1ListeningDataset {index}!!!!")
            item['speak_scores'] = -torch.ones_like(item['r_s'][:, :1])
        return item

class SeamlessV2ListeningDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'Seamless_v2_listening_preprocessed.h5'
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        ### Filter out videos with hyperiqa>=50 and low self-reconstruction quality
        self.self_reconstruction_quality_path = Path(root_dir) / "self-reconstruction_psnr" / "Seamless_v2_listening_visual_quality.csv"
        df_self_reconstruction_quality = pd.read_csv(self.self_reconstruction_quality_path)
        filtered_self_reconstruction = set(df_self_reconstruction_quality[df_self_reconstruction_quality["PSNR_dB"] > 23]["vid_name"]) # > 23dB
        filtered_self_reconstruction = {v.replace(".mp4", "") for v in filtered_self_reconstruction}
        #
        logger.info(f"SeamlessV2ListeningDataset: {len(self.video_ids)} videos before filtering")
        self.video_ids = [
            video_id for video_id in self.video_ids
            if video_id in filtered_self_reconstruction
        ]
        logger.info(f"SeamlessV2ListeningDataset: {len(self.video_ids)} videos after filtering")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = False
        if self.use_speaking_scores and item['speak_scores'] is None:
            logger.info(f"speak_scores is None for SeamlessV2ListeningDataset {index}!!!!")
            item['speak_scores'] = -torch.ones_like(item['r_s'][:, :1])
        return item

class SeamlessV1TalkingDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'Seamless_v1_talking_preprocessed.h5'
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        logger.info(f"SeamlessV1TalkingDataset: {len(self.video_ids)} videos")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = True
        if self.use_speaking_scores and item['speak_scores'] is None:
            logger.info(f"speak_scores is None for SeamlessV1TalkingDataset {index}!!!!")
            item['speak_scores'] = torch.ones_like(item['r_s'][:, :1])
        return item

class SeamlessV3TalkingDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'Seamless_v3_talking_preprocessed.h5'
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        logger.info(f"SeamlessV3TalkingDataset: {len(self.video_ids)} videos")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = True
        if self.use_speaking_scores and item['speak_scores'] is None:
            logger.info(f"speak_scores is None for SeamlessV3TalkingDataset {index}!!!!")
            item['speak_scores'] = torch.ones_like(item['r_s'][:, :1])
        return item

class SeamlessV3ListeningDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        # self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'Seamless_v3_listening_preprocessed.h5'
        self.hdf5_path = "/data/zikai/Data/H5_files/hdf5_reaction/Seamless_v3_listening_preprocessed_reaction_pruned_intensity.h5"
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        
        # neutral_pose_path = Path(root_dir) / "neutral_pose" / "Seamless_v3_listening" / "v1" / f'latents_{kwargs.get("lia_version")}.pth'
        # if os.path.isfile(neutral_pose_path):
        #     self.neutral_pose = torch.load(neutral_pose_path)
        #     self.video_ids = [video_id for video_id in self.video_ids if self.get_pid_from_video_id(video_id) in self.neutral_pose]
        #     logger.info(f"Loaded neutral pose for SeamlessV3ListeningDataset")
        
        ###
        super().__init__(*args, **kwargs)

    def get_pid_from_video_id(self, video_id):
        return video_id.split("_")[3].replace("A", "")

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = False
        if self.use_speaking_scores and item['speak_scores'] is None:
            logger.info(f"speak_scores is None for SeamlessV3ListeningDataset {index}!!!!")
            item['speak_scores'] = -torch.ones_like(item['r_s'][:, :1])
        pid = self.get_pid_from_video_id(item['metadata']['ref_video_path'])
        if hasattr(self, "neutral_pose"):
            item['r_s_refvid_0'] = self.neutral_pose[pid]
        return item

class SeamlessV3TinyListeningDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'SeamlessV3Tiny_listening_preprocessed.h5'
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        ###
        neutral_pose_path = Path(root_dir) / "neutral_pose" / "Seamless_v3_listening" / "v1" / f'latents_{kwargs.get("lia_version")}.pth'
        if os.path.isfile(neutral_pose_path):
            self.neutral_pose = torch.load(neutral_pose_path)
            self.video_ids = [video_id for video_id in self.video_ids if self.get_pid_from_video_id(video_id) in self.neutral_pose]
            logger.info(f"Loaded neutral pose for SeamlessV3TinyListeningDataset")
        ###
        logger.info(f"SeamlessV3TinyListeningDataset: {len(self.video_ids)} videos")
        super().__init__(*args, **kwargs)

    def get_pid_from_video_id(self, video_id):
        return video_id.split("_")[3].replace("A", "")

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = False
        if self.use_speaking_scores and item['speak_scores'] is None:
            logger.info(f"speak_scores is None for SeamlessV3TinyListeningDataset {index}!!!!")
            item['speak_scores'] = -torch.ones_like(item['r_s'][:, :1])
        pid = self.get_pid_from_video_id(item['metadata']['ref_video_path'])
        if hasattr(self, "neutral_pose"):
            item['r_s_refvid_0'] = self.neutral_pose[pid]
        return item

class SeamlessV3SmallListeningDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'SeamlessV3Small_listening_preprocessed.h5'
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        ###
        neutral_pose_path = Path(root_dir) / "neutral_pose" / "Seamless_v3_listening" / "v1" / f'latents_{kwargs.get("lia_version")}.pth'
        if os.path.isfile(neutral_pose_path):
            self.neutral_pose = torch.load(neutral_pose_path)
            self.video_ids = [video_id for video_id in self.video_ids if self.get_pid_from_video_id(video_id) in self.neutral_pose]
            logger.info(f"Loaded neutral pose for SeamlessV3SmallListeningDataset")
        ###
        logger.info(f"SeamlessV3SmallListeningDataset: {len(self.video_ids)} videos")
        super().__init__(*args, **kwargs)

    def get_pid_from_video_id(self, video_id):
        return video_id.split("_")[3].replace("A", "")

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = False
        if self.use_speaking_scores and item['speak_scores'] is None:
            logger.info(f"speak_scores is None for SeamlessV3SmallListeningDataset {index}!!!!")
            item['speak_scores'] = -torch.ones_like(item['r_s'][:, :1])
        pid = self.get_pid_from_video_id(item['metadata']['ref_video_path'])
        if hasattr(self, "neutral_pose"):
            item['r_s_refvid_0'] = self.neutral_pose[pid]
        return item

class EurieListeningDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'Eurie_listening_preprocessed.h5'
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        ###
        neutral_pose_path = Path(root_dir) / "neutral_pose" / "Eurie" / "v1" / f'latents_{kwargs.get("lia_version")}.pth'
        if os.path.isfile(neutral_pose_path):
            self.neutral_pose = torch.load(neutral_pose_path)
            logger.info(f"Loaded neutral pose for EurieListeningDataset")
        ###
        logger.info(f"EurieListeningDataset: {len(self.video_ids)} videos")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = False
        if hasattr(self, "neutral_pose"):
            item['r_s_refvid_0'] = self.neutral_pose['Eurie4']
        return item

########################################################################################
### Influencers (Talking)
########################################################################################

class PhilHellmuthTalkingDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'PhilHellmuth_talking_preprocessed.h5'
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        logger.info(f"PhilHellmuthTalkingDataset: {len(self.video_ids)} videos")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = True
        return item

class CatherineMccordTalkingDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'CatherineMccord_talking_preprocessed.h5'
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        logger.info(f"CatherineMccordTalkingDataset: {len(self.video_ids)} videos")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = True
        return item

class ChloeTingTalkingDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'ChloeTing_talking_preprocessed.h5'
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        logger.info(f"ChloeTingTalkingDataset: {len(self.video_ids)} videos")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = True
        return item

class ChristiLukasiakTalkingDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'ChristiLukasiak_talking_preprocessed.h5'
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        logger.info(f"ChristiLukasiakTalkingDataset: {len(self.video_ids)} videos")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = True
        return item

class GeorgiaHassaratiTalkingDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'GeorgiaHassarati_talking_preprocessed.h5'
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        logger.info(f"GeorgiaHassaratiTalkingDataset: {len(self.video_ids)} videos")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = True
        return item

class JamieHessTalkingDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'JamieHess_talking_preprocessed.h5'
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        logger.info(f"JamieHessTalkingDataset: {len(self.video_ids)} videos")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = True
        return item

class JessicaWeissTalkingDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'JessicaWeiss_talking_preprocessed.h5'
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        logger.info(f"JessicaWeissTalkingDataset: {len(self.video_ids)} videos")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = True
        return item

class IsabelTimermanTalkingDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'IsabelTimerman_talking_preprocessed.h5'
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        logger.info(f"IsabelTimermanTalkingDataset: {len(self.video_ids)} videos")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = True
        return item


########################################################################################
### Influencers (Listening)
########################################################################################

class PhilHellmuthListeningDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'phil_hellmuth_listening_preprocessed.h5'
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        logger.info(f"PhilHellmuthListeningDataset: {len(self.video_ids)} videos")
        self.neutral_pose = torch.load(f"data/neutral_pose/influencers/v1/latents_{kwargs.get('lia_version')}.pth")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = False
        item['r_s_refvid_0'] = self.neutral_pose['phil_hellmuth']
        return item


class AllNoSmileListeningDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'all_no_smile_listening_preprocessed.h5'
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        logger.info(f"AllNoSmileListeningDataset: {len(self.video_ids)} videos")
        self.neutral_pose = torch.load(f"data/neutral_pose/influencers/v1/latents_{kwargs.get('lia_version')}.pth")
        super().__init__(*args, **kwargs)

    def get_pid_from_video_id(self, video_id):
        for pid in ["catherine_mccord", "christi_lukasiak", "georgia_hassarati", "isabel_timerman", "jamie_hess", "jessica_weiss", "phil_hellmuth", "chloe_ting"]:
            if pid in video_id.lower() or pid.split("_")[0] in video_id.lower():
                return pid
        return None

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = False
        item['r_s_refvid_0'] = self.neutral_pose[self.get_pid_from_video_id(item['metadata']['ref_video_path'])]
        return item

class SeamlessV3SmallToSamAltmanListeningDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'SeamlessV3SmallToSamAltman_listening_preprocessed.h5'
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        logger.info(f"SeamlessV3SmallToSamAltmanListeningDataset: {len(self.video_ids)} videos before filtering")
        visual_quality_json = json.load(open("data/hyperiqa/SeamlessV3SmallToSamAltman_visual_quality.json", "r"))
        selected = {k for k, v in visual_quality_json.items() if v['quality_score'] > 50}
        self.video_ids = [v for v in self.video_ids if v.split(".")[0] in selected]
        logger.info(f"SeamlessV3SmallToSamAltmanListeningDataset: {len(self.video_ids)} videos after filtering")
        super().__init__(*args, **kwargs)

    def get_pid_from_video_id(self, video_id):
        return video_id.split("_")[3].replace("A", "")

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = False
        return item

class EurieSmallToEurieListeningDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        root_dir = kwargs.get("root_dir")
        self.hdf5_path = Path(root_dir) / f'hdf5_{kwargs.get("lia_version")}' / 'SeamlessV3SmallToSamAltman_listening_preprocessed.h5'
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_ids = list(f['r_s'].keys())
        logger.info(f"SeamlessV3SmallToSamAltmanListeningDataset: {len(self.video_ids)} videos before filtering")
        visual_quality_json = json.load(open("data/hyperiqa/SeamlessV3SmallToSamAltman_visual_quality.json", "r"))
        selected = {k for k, v in visual_quality_json.items() if v['quality_score'] > 50}
        self.video_ids = [v for v in self.video_ids if v.split(".")[0] in selected]
        logger.info(f"SeamlessV3SmallToSamAltmanListeningDataset: {len(self.video_ids)} videos after filtering")
        super().__init__(*args, **kwargs)

    def get_pid_from_video_id(self, video_id):
        return video_id.split("_")[3].replace("A", "")

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['is_speaking'] = False
        return item
