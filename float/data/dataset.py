from torch.utils.data import Dataset
from pathlib import Path
from .base import BaseDataset
from .RAVDESS import RAVDESSDataset
from .HDTF import HDTFDataset
from .VideoDataset import (
                            InterviewV1TalkingDataset,
                            InterviewV1ListeningDataset,
                            InterviewV3TalkingDataset,
                            SeamlessV1ListeningDataset,
                            SeamlessV2ListeningDataset,
                            SeamlessV1TalkingDataset,
                            SeamlessV3TalkingDataset,
                            SeamlessV3ListeningDataset,
                            SeamlessV3TinyListeningDataset,
                            SeamlessV3SmallListeningDataset,
                            SeamlessV3SmallToSamAltmanListeningDataset,
                            EurieListeningDataset,
                            Hallo3Dataset,
                            PsychologyIsListeningDataset,
                            AmplifyMeHubListeningDataset,
                            TheWellnessTheoryListeningDataset,
                            InterviewV1TalkingDataset,
                            RealTalkListeningDataset,
                            PhilHellmuthTalkingDataset, PhilHellmuthListeningDataset,
                            CatherineMccordTalkingDataset, ChloeTingTalkingDataset, ChristiLukasiakTalkingDataset,
                            GeorgiaHassaratiTalkingDataset, IsabelTimermanTalkingDataset, JamieHessTalkingDataset,
                            JessicaWeissTalkingDataset,
                            AllNoSmileListeningDataset)

class CombinedDataset(BaseDataset):
    def __init__(self, datasets):
        self.datasets = datasets

        # Combine subdirs from all datasets for fast access
        self.subdirs = []
        self.dataset_offsets = [0]  # Track which dataset each index belongs to
        for dataset in datasets:
            if hasattr(dataset, 'subdirs'):
                self.subdirs.extend(dataset.subdirs)
            else:
                # Fallback: create dummy entries
                self.subdirs.extend([None] * len(dataset))
            self.dataset_offsets.append(len(self.subdirs))

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        idx_dataset = 0
        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset[idx]
            idx -= len(dataset)
        raise IndexError

def build_dataset(
    root_dir,
    dataset_names,
    transform,
    window_size=60,
    curr_window_size=50,
    split='train',
    audio_transform=None,
    audio_sample_rate=16000,
    fps=25,
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
    Build a combined dataset from multiple datasets with weighted sampling.

    Args:
        root_dir (str): Root directory of the dataset
        dataset_names (str or list): Dataset name(s) to use
        transform (callable): Transform to apply to video frames
        window_size (int): Number of frames to sample in each window
        curr_window_size (int): Number of frames to sample in the current window
        split (str): 'train' or 'val' to specify which split to use
        audio_transform (callable): Transform to apply to audio
        audio_sample_rate (int): Sample rate for audio
        fps (int): Frames per second of the video
        wav2vec_model_path (str): Path to wav2vec2 model checkpoints
        use_speaking_scores (bool): Whether to use speaking scores
        lia_version (str): Version of LIA to use
        lia_path (str): Path to LIA checkpoint
        use_alpha_mean (bool): Whether to use alpha mean
        use_alpha_std (bool): Whether to use alpha std
        use_speed_mean (bool): Whether to use speed mean
        use_speed_std (bool): Whether to use speed std
        use_accel_mean (bool): Whether to use accel mean
        use_accel_std (bool): Whether to use accel std
    """

    # Mapping of dataset names to their corresponding classes
    dataset_classes = {
        "RAVDESS": RAVDESSDataset,
        "HDTF": HDTFDataset,
        "Hallo3": Hallo3Dataset,
        "PsychologyIsListening": PsychologyIsListeningDataset,
        "AmplifyMeHubListening": AmplifyMeHubListeningDataset,
        "TheWellnessTheoryListening": TheWellnessTheoryListeningDataset,
        "InterviewV1Talking": InterviewV1TalkingDataset,
        "InterviewV3Talking": InterviewV3TalkingDataset,
        "InterviewV1Listening": InterviewV1ListeningDataset,
        "SeamlessV1Listening": SeamlessV1ListeningDataset,
        "SeamlessV2Listening": SeamlessV2ListeningDataset,
        "SeamlessV1Talking": SeamlessV1TalkingDataset,
        "SeamlessV3Talking": SeamlessV3TalkingDataset,
        "SeamlessV3Listening": SeamlessV3ListeningDataset,
        "SeamlessV3TinyListening": SeamlessV3TinyListeningDataset,
        "SeamlessV3SmallListening": SeamlessV3SmallListeningDataset,
        "EurieListening": EurieListeningDataset,
        "SeamlessV3SmallToSamAltmanListening": SeamlessV3SmallToSamAltmanListeningDataset,
        "PhilHellmuthTalking": PhilHellmuthTalkingDataset,
        "PhilHellmuthListening": PhilHellmuthListeningDataset,
        "AllNoSmileListening": AllNoSmileListeningDataset,
        "RealTalkListening": RealTalkListeningDataset,
        "CatherineMccordTalking": CatherineMccordTalkingDataset,
        "ChloeTingTalking": ChloeTingTalkingDataset,
        "ChristiLukasiakTalking": ChristiLukasiakTalkingDataset,
        "GeorgiaHassaratiTalking": GeorgiaHassaratiTalkingDataset,
        "IsabelTimermanTalking": IsabelTimermanTalkingDataset,
        "JamieHessTalking": JamieHessTalkingDataset,
        "JessicaWeissTalking": JessicaWeissTalkingDataset,
    }

    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    # Common parameters for all datasets
    common_params = {
        'root_dir': root_dir,
        'transform': transform,
        'window_size': window_size,
        'curr_window_size': curr_window_size,
        'split': split,
        'audio_transform': audio_transform,
        'audio_sample_rate': audio_sample_rate,
        'fps': fps,
        'wav2vec_model_path': wav2vec_model_path,
        'use_speaking_scores': use_speaking_scores,
        'lia_version': lia_version,
        'lia_path': lia_path,
        'use_alpha_mean': use_alpha_mean,
        'use_alpha_std': use_alpha_std,
        'use_speed_mean': use_speed_mean,
        'use_speed_std': use_speed_std,
        'use_accel_mean': use_accel_mean,
        'use_accel_std': use_accel_std,
    }

    datasets = []
    for dataset_name in dataset_names:
        if dataset_name not in dataset_classes:
            raise ValueError(f"Invalid dataset: {dataset_name}")

        dataset_class = dataset_classes[dataset_name]
        datasets.append(dataset_class(**common_params))

    return CombinedDataset(datasets)
