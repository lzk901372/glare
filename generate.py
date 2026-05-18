"""
    Inference Stage 2
"""
from typing import List
import os, torch, random, cv2, torchvision, subprocess, librosa, datetime, tempfile, face_alignment
import numpy as np
import time, math
import albumentations as A
import albumentations.pytorch.transforms as A_pytorch
import soundfile as sf

from tqdm import tqdm
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor

from float.models.FLOAT import FLOAT
from float.options.base_options import BaseOptions
from float.models.intensity_encoder import IntensityEncoder

### Debug
from float.data import RAVDESSDataset
from torch.utils.data import DataLoader
from torchvision import transforms

from tools.detect_failure_case import DetectFailureCase

# Add this import at the top of generate.py
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("Warning: InsightFace not available. Install with: pip install insightface")

class DataProcessor:
    def __init__(self, opt):
        self.opt = opt
        self.fps = opt.fps
        self.sampling_rate = opt.sampling_rate
        self.input_size = opt.input_size

        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

        # wav2vec2 audio preprocessor
        self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(opt.wav2vec_model_path, local_files_only=True)

        # intensity encoder
        self.intensity_encoder = IntensityEncoder(opt)

        # image transform
        self.transform = A.Compose([
                A.Resize(height=opt.input_size, width=opt.input_size, interpolation=cv2.INTER_AREA),
                A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
                A_pytorch.ToTensorV2(),
            ])

    @torch.no_grad()
    def process_img(self, img:np.ndarray) -> np.ndarray:
        mult = 360. / img.shape[0]

        resized_img = cv2.resize(img, dsize=(0, 0), fx = mult, fy = mult, interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
        bboxes = self.fa.face_detector.detect_from_image(resized_img)
        bboxes = [(int(x1 / mult), int(y1 / mult), int(x2 / mult), int(y2 / mult), score) for (x1, y1, x2, y2, score) in bboxes if score > 0.95]
        bboxes = bboxes[0] # Just use first bbox

        bsy = int((bboxes[3] - bboxes[1]) / 2)
        bsx = int((bboxes[2] - bboxes[0]) / 2)
        my  = int((bboxes[1] + bboxes[3]) / 2)
        mx  = int((bboxes[0] + bboxes[2]) / 2)

        bs = int(max(bsy, bsx) * 1.6)
        # bs = int(max(bsy, bsx) * 1.2)  # larger face
        # bs = int(max(bsy, bsx) * 1.8)  # smaller face
        img = cv2.copyMakeBorder(img, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=0)
        my, mx  = my + bs, mx + bs      # BBox center y, bbox center x

        crop_img = img[my - bs:my + bs,mx - bs:mx + bs]
        crop_img = cv2.resize(crop_img, dsize = (self.input_size, self.input_size), interpolation = cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
        return crop_img

    def default_img_loader(self, path) -> np.ndarray:
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def default_aud_loader(self, path: str) -> torch.Tensor:
        speech_array, sampling_rate = librosa.load(path, sr = self.sampling_rate)
        intensity_scores = self.intensity_encoder.encode_waveform(
            waveform=speech_array,
            num_frames=math.ceil(len(speech_array) * self.fps / self.sampling_rate),
            fps=self.fps,
            sampling_rate=sampling_rate,
            device='cpu',
        )
        # print('### speech_array (before)', speech_array)
        # speech_array = np.zeros_like(speech_array)  # idle
        # print('### speech_array (after)', speech_array)
        return self.wav2vec_preprocessor(speech_array, sampling_rate = sampling_rate, return_tensors = 'pt').input_values[0], intensity_scores

    def default_aud_loader_mix_audio_silence(self, path: str, chunk_second: int = 8) -> torch.Tensor:
        """
        Mix audio with the original audio and silence every n-second chunk to check the model's ability to handle silence.
        - If the chunk index is even, replace the chunk with silence.
        - If the chunk index is odd, keep the original chunk.
        """
        speech_array, sampling_rate = librosa.load(path, sr=self.sampling_rate)

        # Number of samples in n seconds
        chunk_size = int(chunk_second * sampling_rate)

        # Create a copy to modify
        modified_array = speech_array.copy()

        # Iterate over chunks
        for i in range(0, len(modified_array), chunk_size):
            chunk_idx = i // chunk_size
            if chunk_idx % 2 == 0:
                # Even index → silence
                modified_array[i : i + chunk_size] = 0.0
            # Odd index → keep original (do nothing)

        return self.wav2vec_preprocessor(
            modified_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_values[0]

    def preprocess(
        self,
        ref_path:str,
        audio_path:str,
        ref_r_s_path:str = None,
        prev_r_s_path:str = None,
        prev_video_path:str = None,
        no_crop:bool = False,
    ) -> dict:
        s = self.default_img_loader(ref_path)
        if not no_crop:
            s = self.process_img(s)
        s = self.transform(image=s)['image'].unsqueeze(0)
        a, intensity_scores = self.default_aud_loader(audio_path)
        a = a.unsqueeze(0)
        intensity_scores = intensity_scores.unsqueeze(0)
        if ref_r_s_path is not None:
            ref_r_s = torch.load(ref_r_s_path)
            if isinstance(ref_r_s, dict):
                ref_r_s = list(ref_r_s.values())[0]
            if len(ref_r_s.shape) == 2:
                ref_r_s = ref_r_s.unsqueeze(0)
            ref_r_s = ref_r_s[:, :self.opt.num_ref_frames]
        else:
            ref_r_s = None

        if prev_r_s_path is not None:
            assert prev_video_path is not None, "prev_video_path is required when prev_r_s_path is provided"
            print(f"prev_r_s_path: {prev_r_s_path}")
            prev_r_s = torch.load(prev_r_s_path)
            if len(prev_r_s.shape) == 2:
                prev_r_s = prev_r_s.unsqueeze(0)
            # prev_r_s: (1, T, 512)
            prev_a, prev_intensity_scores = self.default_aud_loader(prev_video_path)
            prev_a = prev_a.unsqueeze(0)
            prev_intensity_scores = prev_intensity_scores.unsqueeze(0)
            num_frame = min(prev_r_s.shape[1], 60000)
            start_frame = max(0, prev_r_s.shape[1] - num_frame)
            prev_r_s = prev_r_s[:, start_frame:]
            prev_a = prev_a[:, int(start_frame * self.sampling_rate / self.fps):] # (1, T, 512)
            prev_intensity_scores = prev_intensity_scores[:, start_frame:]

            # Pad or trim prev_a to match the length of prev_s
            T = int(num_frame * self.sampling_rate / self.fps)
            if prev_a.shape[1] < T:
                prev_a = torch.cat([prev_a, torch.zeros(1, T - prev_a.shape[1])], dim=1)
            elif prev_a.shape[1] > T:
                prev_a = prev_a[:, :T]

            if prev_intensity_scores.shape[1] < T:
                prev_intensity_scores = torch.cat([prev_intensity_scores, torch.zeros(1, T - prev_intensity_scores.shape[1])], dim=1)
            elif prev_intensity_scores.shape[1] > T:
                prev_intensity_scores = prev_intensity_scores[:, :T]

            # Save padded/trimmed prev_a to temporary file
            prev_a_temp_path = None
            if prev_a is not None:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                    prev_a_temp_path = temp_audio.name
                # Convert tensor to numpy and save as wav file
                prev_a_numpy = prev_a.squeeze(0).cpu().numpy()
                sf.write(prev_a_temp_path, prev_a_numpy, self.sampling_rate)
        else:
            prev_r_s = None
            prev_a = None
            prev_intensity_scores = None
            prev_a_temp_path = None

        print(f"s.shape: {s.shape}, a.shape: {a.shape}")
        print(f"s.min(): {s.min()}, s.max(): {s.max()}")
        print(f"a.min(): {a.min()}, a.max(): {a.max()}")
        print(f"s.mean(): {s.mean()}, s.std(): {s.std()}")
        print(f"a.mean(): {a.mean()}, a.std(): {a.std()}")
        if prev_r_s is not None:
            print(f"prev_r_s.shape: {prev_r_s.shape}")
            print(f"prev_r_s.min(): {prev_r_s.min()}, prev_r_s.max(): {prev_r_s.max()}")
            print(f"prev_r_s.mean(): {prev_r_s.mean()}, prev_r_s.std(): {prev_r_s.std()}")
            print(f"prev_a.shape: {prev_a.shape}")
            print(f"prev_a.min(): {prev_a.min()}, prev_a.max(): {prev_a.max()}")
            print(f"prev_a.mean(): {prev_a.mean()}, prev_a.std(): {prev_a.std()}")
        return {'s': s, 'a': a, 'p': None, 'e': None, 'prev_r_s': prev_r_s, 'prev_a': prev_a, 'ref_r_s': ref_r_s, 'intensity': intensity_scores, 'prev_intensity': prev_intensity_scores}, prev_a_temp_path


class InferenceAgent:
    def __init__(self, opt):
        torch.cuda.empty_cache()
        self.opt = opt
        self.rank = opt.rank

        # Load Model
        self.load_model()
        self.load_weight(opt.ckpt_path, rank=self.rank)
        self.G.to(self.rank)
        self.G.eval()

        # Load Data Processor
        self.data_processor = DataProcessor(opt)

    def load_model(self) -> None:
        self.G = FLOAT(self.opt)

    def load_weight(self, checkpoint_path: str, rank: int) -> None:
        # checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict_orig = torch.load("checkpoints/float.pth")
            state_dict = checkpoint['model_state_dict']
            # print(f"state_dict - state_dict_orig: {set(state_dict.keys()) - set(state_dict_orig.keys())}")
            # print(f"state_dict_orig - state_dict: {set(state_dict_orig.keys()) - set(state_dict.keys())}")
            state_dict.update({k: v for k, v in state_dict_orig.items() if k.startswith("motion_autoencoder.")})
        else:
            state_dict = checkpoint

        with torch.no_grad():
            for model_name, model_param in self.G.named_parameters():
                if model_name in state_dict:
                    model_param.copy_(state_dict[model_name].to(rank))
                elif "wav2vec2" in model_name: pass
                else:
                    print(f"! Warning; {model_name} not found in state_dict.")

        del checkpoint, state_dict

    def save_video(
        self,
        vid_target_recon: torch.Tensor,
        video_path: str,
        audio_path: str,
        prev_video_path: str = None,
        prev_a_temp_path: str = None,
    ) -> str:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_filename = temp_video.name
            vid = vid_target_recon.permute(0, 2, 3, 1)
            vid = vid.detach().clamp(-1, 1).cpu()
            vid = ((vid + 1) / 2 * 255).type('torch.ByteTensor')
            torchvision.io.write_video(temp_filename, vid, fps=self.opt.fps)
            if audio_path is not None:
                if prev_video_path is not None:
                    # Create temporary files for audio processing
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio1:
                        temp_audio1_filename = temp_audio1.name
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio2:
                        temp_audio2_filename = temp_audio2.name

                    with open(os.devnull, 'wb') as f:

                        # Concatenate the two audio files
                        command = "ffmpeg -i {} -i {} -filter_complex '[0:0][1:0]concat=n=2:v=0:a=1[out]' -map '[out]' {} -y".format(
                            prev_a_temp_path, audio_path, temp_audio2_filename)
                        subprocess.call(command, shell=True, stdout=f, stderr=f)

                        # Combine the concatenated audio with the video
                        command = "ffmpeg -i {} -i {} -c:v copy -c:a aac -shortest {} -y".format(
                            temp_filename, temp_audio2_filename, video_path)
                        subprocess.call(command, shell=True, stdout=f, stderr=f)

                else:
                    with open(os.devnull, 'wb') as f:
                        command =  "ffmpeg -i {} -i {} -c:v copy -c:a aac {} -y".format(temp_filename, audio_path, video_path)
                        subprocess.call(command, shell=True, stdout=f, stderr=f)
                if os.path.exists(video_path):
                    os.remove(temp_filename)
            else:
                os.rename(temp_filename, video_path)
            return video_path

    @torch.no_grad()
    def run_inference(
        self,
        res_video_path: str,
        ref_path: str,
        audio_path: str,
        ref_r_s_path: str = None,
        prev_r_s_path: str = None,
        prev_video_path: str = None,
        a_cfg_scale: float    = 2.0,
        e_cfg_scale: float    = 1.0,
        r_cfg_scale: float    = 1.0,
        s_cfg_scale: float    = 1.0,
        ref_cfg_scale: float  = 1.0,
        i_cfg_scale: float    = 1.0,
        emo: str             = 'S2E',
        nfe: int            = 10,
        no_crop: bool         = False,
        seed: int            = 25,
        verbose: bool         = False,
        save_latent: bool     = False,
        detect_failure_case: bool = False
    ) -> str:

        data, prev_a_temp_path = self.data_processor.preprocess(
            ref_path, 
            audio_path, 
            ref_r_s_path = ref_r_s_path, 
            prev_r_s_path = prev_r_s_path, 
            prev_video_path = prev_video_path, 
            no_crop = no_crop,
        )
        if verbose: print(f"> [Done] Preprocess.")

        # inference
        torch.cuda.synchronize()
        start_time = time.time()
        output = self.G.inference(
            data        = data,
            a_cfg_scale = a_cfg_scale,
            r_cfg_scale = r_cfg_scale,
            e_cfg_scale = e_cfg_scale,
            s_cfg_scale = s_cfg_scale,
            ref_cfg_scale = ref_cfg_scale,
            i_cfg_scale = i_cfg_scale,
            emo         = emo,
            nfe         = nfe,
            seed        = seed
        )
        d_hat = output['d_hat']
        if save_latent:
            r_d = output['r_d']
        torch.cuda.synchronize()
        print(f"inference took: {time.time() - start_time} sec", flush=True)

        res_video_path = self.save_video(d_hat, res_video_path, audio_path, prev_video_path, prev_a_temp_path)
        if save_latent:
            latent_path = res_video_path.replace(".mp4", "_latent.pt")
            torch.save(r_d.cpu(), latent_path)
        if verbose:
            print(f"> [Done] result saved at {res_video_path}")
            if save_latent:
                print(f"> [Done] latent saved at {latent_path}")

        # detect failure case
        if detect_failure_case:
            failure_case_detector = DetectFailureCase(self.opt, self.G.motion_autoencoder, data, res_video_path)
            failure_case_detector.detect_failure(output['r_d'])

        return res_video_path


class InferenceOptions(BaseOptions):
    def __init__(self):
        super().__init__()

    def initialize(self, parser):
        super().initialize(parser)
        parser.add_argument("--ref_path",
                default=None, type=str,help='ref')
        parser.add_argument('--aud_path',
                default=None, type=str, help='audio')
        parser.add_argument('--ref_r_s_path',
                default=None, type=str, help='ref r_s')
        parser.add_argument('--prev_r_s_path',
                default=None, type=str, help='prev r_s')
        parser.add_argument('--prev_video_path',
                default=None, type=str, help='prev video')
        parser.add_argument('--emo',
                default=None, type=str, help='emotion', choices=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
        parser.add_argument('--no_crop',
                action = 'store_true', help = 'not using crop')
        parser.add_argument('--res_video_path',
                default=None, type=str, help='res video path')
        parser.add_argument('--ckpt_path',
                default="/home/nvadmin/workspace/taek/float-pytorch/checkpoints/float.pth", type=str, help='checkpoint path')
        parser.add_argument('--res_dir',
                default="./results", type=str, help='result dir')
        parser.add_argument('--save_latent',
                action = 'store_true', help = 'save latent')
        parser.add_argument('--detect_failure_case',
                action = 'store_true', help = 'detect failure case')
        return parser


if __name__ == '__main__':
    opt = InferenceOptions().parse()
    opt.rank, opt.ngpus  = 0,1
    agent = InferenceAgent(opt)
    os.makedirs(opt.res_dir, exist_ok = True)

    # -------------- input -------------
    ref_path         = opt.ref_path
    aud_path         = opt.aud_path
    # ----------------------------------

    if opt.res_video_path is None:
        video_name = os.path.splitext(os.path.basename(ref_path))[0]
        audio_name = os.path.splitext(os.path.basename(aud_path))[0]
        call_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        res_video_path = os.path.join(opt.res_dir, "%s-%s-%s-nfe%s-seed%s-acfg%s-ecfg%s-%s.mp4" \
                                    % (call_time, video_name, audio_name, opt.nfe, opt.seed, opt.a_cfg_scale, opt.e_cfg_scale, opt.emo))
    else:
        res_video_path = opt.res_video_path

    agent.run_inference(
        res_video_path,
        ref_path,
        aud_path,
        ref_r_s_path = opt.ref_r_s_path,
        prev_r_s_path = opt.prev_r_s_path,
        prev_video_path = opt.prev_video_path,
        a_cfg_scale = opt.a_cfg_scale,
        e_cfg_scale = opt.e_cfg_scale,
        r_cfg_scale = opt.r_cfg_scale,
        s_cfg_scale = opt.s_cfg_scale,
        ref_cfg_scale = opt.ref_cfg_scale,
        i_cfg_scale = opt.i_cfg_scale,
        emo         = opt.emo,
        nfe         = opt.nfe,
        no_crop     = opt.no_crop,
        seed        = opt.seed,
        # is_speaking = opt.is_speaking,
        save_latent = opt.save_latent,
        detect_failure_case = opt.detect_failure_case
    )

    ### For debugging
    # transform = transforms.Compose([
    #     transforms.Resize((512, 512)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # ])
    # dataset = RAVDESSDataset(
    #     root_dir='/mnt/wwn-0x5002538f34a0a7de/data/RAVDESS/',
    #     window_size=int(2 * 25 + 10),
    #     transform=transform,
    #     split='train',
    #     audio_transform=None,
    #     audio_sample_rate=16000,
    #     fps=25,
    # )
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # opt.vel_loss_weight = 1.0
    # for idx_batch, batch in enumerate(dataloader):
    #     if idx_batch > 10:
    #         break
    #     frames, audio, metadata = batch['frames'], batch['audio'], batch['metadata']
    #     frames = frames.to(opt.device)
    #     audio = audio.to(opt.device)
    #     loss_dict = agent.G(frames, audio)
    #     loss = loss_dict['loss']
    #     loss_ot = loss_dict['loss_ot']
    #     loss_vel = loss_dict['loss_vel']
    #     print(f"[idx_batch: {idx_batch}] loss: {loss}, loss_ot: {loss_ot}, loss_vel: {loss_vel}")
