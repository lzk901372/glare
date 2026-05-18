from torch.utils.data import Dataset
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor
import torch
from loguru import logger

class BaseDataset(Dataset):
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
        wav2vec_model_path='./checkpoints/wav2vec2-base-960h',  # Make it configurable
        use_speaking_scores=False,
        lia_path='./checkpoints/float.pth',
        lia_version='highres',
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
            hdf5_window_size (int): Number of frames per chunk in the HDF5 file (default: 60)
            transform (callable, optional): Optional transform to be applied on video frames
            audio_transform (callable, optional): Optional transform to be applied on audio
            audio_sample_rate (int): Sample rate for audio
            fps (int): Frames per second of the video
            split (str): 'train' or 'val' to specify which split to use
            train_ratio (float): Ratio of data to use for training (default: 0.9)
            random_seed (int): Random seed for reproducible splits (default: 42)
            wav2vec_model_path (str): Path to wav2vec2 model checkpoints
            use_speaking_scores (bool): Whether to use speaking scores (default: False)
            lia_path (str): Path to LIA checkpoint (default: './checkpoints/float.pth')
            lia_version (str): Version of LIA (default: 'highres') # 'highres' or 'LIA_X'
            use_alpha_mean (bool): Whether to use alpha mean (default: True)
            use_alpha_std (bool): Whether to use alpha std (default: True)
            use_speed_mean (bool): Whether to use speed mean (default: True)
            use_speed_std (bool): Whether to use speed std (default: True)
            use_accel_mean (bool): Whether to use accel mean (default: True)
            use_accel_std (bool): Whether to use accel std (default: True)
        """
        self.root_dir = Path(root_dir)
        self.window_size = window_size
        self.curr_window_size = curr_window_size
        self.hdf5_window_size = hdf5_window_size
        self.transform = transform
        self.audio_transform = audio_transform
        self.audio_sample_rate = audio_sample_rate
        self.fps = fps
        self.split = split
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.use_speaking_scores = use_speaking_scores
        self.lia_version = lia_version
        self.lia_path = lia_path
        self.min_num_chunks = (self.window_size - 1) // self.hdf5_window_size + 1
        logger.info(f"In BaseDataset: min_num_chunks: {self.min_num_chunks}")
        # Load wav2vec preprocessor with error handling
        try:
            self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
                wav2vec_model_path, 
                local_files_only=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load wav2vec2 model from {wav2vec_model_path}: {e}")

        self.index_dataset()
        if lia_version == 'LIA_X':
            self.lia_Q = torch.load("checkpoints/lia-x.pt", weights_only=True, map_location='cpu')['dec.direction.weight'] #  (1024, 40)
            logger.info(f"In BaseDataset: Loaded LIA checkpoint from checkpoints/lia-x.pt")
        else:
            self.lia_Q = torch.load(self.lia_path, map_location="cpu")['motion_autoencoder.dec.direction.weight'] # (512, 20)
            logger.info(f"In BaseDataset: Loaded LIA checkpoint from {self.lia_path}")

        self.use_alpha_mean = use_alpha_mean
        self.use_alpha_std = use_alpha_std
        self.use_speed_mean = use_speed_mean
        self.use_speed_std = use_speed_std
        self.use_accel_mean = use_accel_mean
        self.use_accel_std = use_accel_std

    def __len__(self):
        pass

    def index_dataset(self):
        pass

    def get_metadata(self, subdir):
        pass

    def __getitem__(self, idx):
        pass