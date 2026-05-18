import torch
from .base import BaseDataset
from pathlib import Path
import random
import h5py

class TFHPDataset(BaseDataset):
    def __init__(
        self,
        root_dir,
        window_size=60,
        curr_window_size=50,
        transform=None,
        audio_transform=None,
        audio_sample_rate=16000,
        fps=25,
        split='train',
        train_ratio=0.9,
        random_seed=42,
        wav2vec_model_path='./checkpoints/wav2vec2-base-960h',
        use_speaking_scores=False,
    ):
        """
        Args:
            root_dir (str): Root directory of the dataset
            window_size (int): Number of frames to sample in each window
            transform (callable, optional): Optional transform to be applied on video frames
            audio_transform (callable, optional): Optional transform to be applied on audio
            audio_sample_rate (int): Sample rate for audio
            fps (int): Frames per second of the video
            split (str): 'train' or 'val' to specify which split to use
            train_ratio (float): Ratio of data to use for training (default: 0.9)
            random_seed (int): Random seed for reproducible splits (default: 42)
            wav2vec_model_path (str): Path to the wav2vec model
            use_speaking_scores (bool): Whether to use speaking scores (default: False)
        """
        self.hdf5_path = Path(root_dir) / "tfhp_preprocessed.h5"
        super().__init__(
            root_dir,
            window_size,
            curr_window_size,
            transform,
            audio_transform,
            audio_sample_rate,
            fps,
            split,
            train_ratio,
            random_seed,
            wav2vec_model_path,
            use_speaking_scores,
        )
        self.set_audio_length = self.window_size * self.audio_sample_rate // self.fps  # 38400

    def index_dataset(self):
        """
        Index the TFHP dataset by finding all video subdirectories with sufficient frames.
        TFHP dataset structure expected:
        root_dir/
        │   ├── video1/
        |   |   ├── 000
        |   |   |   ├── chunk_000
        |   |   |   |   ├── 00.jpg
        |   |   |   |   ├── 01.jpg
        |   |   |   |   ├── ...
        |   |   |   |   ├── 59.jpg
        |   |   |   |   ├── audio.wav
        |   |   |   ├── chunk_001
        |   |   |   ├── ...
        |   |   ├── 001
        |   |   |   ├── chunk_000
        |   |   |   |   ├── 00.jpg
        |   |   |   |   ├── 01.jpg
        |   |   |   |   ├── ...
        |   |   |   |   ├── 59.jpg
        |   |   |   |   ├── audio.wav
        |   |   |   ├── ...
        |   |   ├── ...
        |   ├── video2
        |   ├── ...
        """
        all_identities = []
        all_chunks = []
        identities_subdir_num = {}
        identities_chunk_num = {}

        with h5py.File(self.hdf5_path, 'r') as f:
            for sample_id in f['r_s'].keys():
                all_identities.append(sample_id)

                identity_subdirs = f[f'r_s/{sample_id}'].keys()
                identities_subdir_num[sample_id] = len(identity_subdirs)

                for subdir in identity_subdirs:

                    identity_subdir_chunks = f[f'r_s/{sample_id}/{subdir}'].keys()
                    identities_chunk_num[f"{sample_id}/{subdir}"] = len(identity_subdir_chunks)

                    for chunk in identity_subdir_chunks:
                        if f[f'r_s/{sample_id}/{subdir}/{chunk}'][:].shape[0] != 60 or f[f'audio/{sample_id}/{subdir}/{chunk}'][:].shape[1] != 38400:
                            continue
                        # print(f"Audio shape: {f[f'audio/{sample_id}/{subdir}/{chunk}'][:].shape[1]}")
                        all_chunks.append(f"{sample_id}/{subdir}/{chunk}")

        self.identities_subdir_num = identities_subdir_num
        self.identities_chunk_num = identities_chunk_num

        # Create reproducible train/val split
        random.seed(self.random_seed)
        random.shuffle(all_identities)
        random.shuffle(all_chunks)

        split_idx = int(len(all_identities) * self.train_ratio)

        if self.split == 'train':
            self.identities = all_identities[:split_idx]
            self.chunks = [x for x in all_chunks if x.split('/')[0] in self.identities]
        elif self.split == 'val':
            self.identities = all_identities[split_idx:]
            self.chunks = [x for x in all_chunks if x.split('/')[0] in self.identities]
        else:
            raise ValueError(f"Split must be 'train' or 'val', got {self.split}")

        print(f"TFHP: Created {self.split} split with {len(self.identities)} samples "
              f"(total samples: {len(all_identities)})")
        print(f"TFHP: All sample num: {len(self.chunks)}")

    def __getitem__(self, index):
        chunk_path = self.chunks[index]
        sample_id = chunk_path.split('/')[0]
        subdir_id = chunk_path.split('/')[1]
        chunk_name = chunk_path.split('/')[2]
        chunk_id = int(chunk_name.split('_')[1])

        if chunk_id >= self.identities_chunk_num[f"{sample_id}/{subdir_id}"] - 1:
            # chunk_id = 0
            with h5py.File(self.hdf5_path, 'r') as f:
                audio = f[f'audio/{chunk_path}'][:]
                audio = torch.from_numpy(audio).float().squeeze(0)
                r_s = f[f'r_s/{chunk_path}'][:]
                r_s = torch.from_numpy(r_s).float()
        else:
            next_chunk_path = f"{sample_id}/{subdir_id}/chunk_{chunk_id + 1:03d}"
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

        if self.audio_transform:
            audio = self.audio_transform(audio)

        assert audio.shape[0] == int(self.set_audio_length), f"sample_id: {sample_id}, chunk_name: {chunk_name}, audio.shape: {audio.shape}, sr: {self.audio_sample_rate}, window_size: {self.window_size}, fps: {self.fps}"

        item = {
            'r_s': r_s, # [T, 512]
            'audio': audio, # [T*sr/25]
            'metadata': {
                'num_frame': r_s.shape[0],
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

    def __len__(self):
        return len(self.chunks)
