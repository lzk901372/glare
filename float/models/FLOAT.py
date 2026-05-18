import torch, math
import torch.nn as nn
import torch.nn.functional as F
import time as time_
import random
from loguru import logger
from torchdiffeq import odeint
from transformers import Wav2Vec2Config
from transformers.modeling_outputs import BaseModelOutput

from float.models.wav2vec2 import Wav2VecModel
from float.models.wav2vec2_ser import Wav2Vec2ForSpeechClassification
from float.models.intensity_encoder import IntensityEncoder

from float.models import BaseModel
from float.models.FMT import FlowMatchingTransformer

######## Main Phase 2 model ########
class FLOAT(BaseModel):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(self.opt.num_prev_frames)
        self.num_ref_frames = int(self.opt.num_ref_frames)
        self.num_total_frames = self.num_prev_frames + self.num_frames_for_clip + self.num_ref_frames

        # motion latent auto-encoder
        if opt.lia_version == "LIA_X":
            from float.models.LIA_X.generator import Generator
            self.motion_autoencoder = Generator(size = opt.input_size, motion_dim = opt.dim_m, scale=2)
            self.motion_autoencoder.load_state_dict(torch.load("checkpoints/lia-x.pt", weights_only=True), strict=True)
            logger.info(f"Rank {opt.rank}: Loaded LIA-X checkpoint from checkpoints/lia-x.pt")
        else:
            from float.models.generator import Generator
            self.motion_autoencoder = Generator(size = opt.input_size, style_dim = opt.dim_w, motion_dim = opt.dim_m)
        self.motion_autoencoder.requires_grad_(False)

        # condition encoders
        self.audio_encoder      = AudioEncoder(opt)
        self.emotion_encoder    = Audio2Emotion(opt)

        # intensity encoder: Use Qwen2-Audio-7B-Instruct model to encode intensity scores into a [T, 1] tensor.
        # Only used during inference when there is no already encoded intensity scores.
        # Then the intensity scores will be forwarded to intensity embedder to get the intensity embeddings.
        # Finally, the intensity embeddings will be integrated into the FMT conditioning.
        # self.intensity_encoder = IntensityEncoder(opt)

        # intensity embedder: Embed a [T, 1] tensor into a [T, dim_i] tensor. Default dim_i = 4
        self.intensity_embedder = nn.Sequential(
            nn.Linear(1, opt.dim_i),
            nn.LayerNorm(opt.dim_i),
            nn.SiLU()
        )

        # Reaction scores process
        self.channel_projection = nn.Sequential(
            nn.Linear(opt.dim_h, opt.dim_reaction),
            nn.LayerNorm(opt.dim_reaction),
        )
        self.reaction_classifier = nn.Sequential(
            nn.Linear(opt.dim_reaction, opt.num_reaction_classes),
            nn.LayerNorm(opt.num_reaction_classes),
            nn.Softmax(dim=-1)
        )
        self.reaction_loss = nn.SmoothL1Loss()

        # FMT; Flow Matching Transformer
        self.fmt = FlowMatchingTransformer(opt)

        # ODE options
        self.odeint_kwargs = {
            'atol': self.opt.ode_atol,
            'rtol': self.opt.ode_rtol,
            'method': self.opt.torchdiffeq_ode_method
        }

        # Logging counters for frame selection ratio
        self.start_from_beginning_count = 0
        self.random_start_count = 0
        self.log_interval = getattr(opt, 'frame_selection_log_interval', 100)  # Log every 100 forward passes

    ######## Motion Encoder - Decoder ########
    @torch.no_grad()
    def encode_image_into_latent(self, x: torch.Tensor) -> list:
        """
        Args:
            x: (B, 3, H, W) : input image
        Returns:
            x_r: (B, 512) : appearance latent
            x_r_lambda: (B, 20) : motion latent
            x_r_feats: list of tensors
        """
        if self.opt.lia_version == "LIA_X":
            x_r, _, x_r_feats = self.motion_autoencoder.enc(x, input_target=None)
            x_r_lambda = self.motion_autoencoder.enc.enc_r2t(x_r)
        else:
            x_r, _, x_r_feats = self.motion_autoencoder.enc(x, input_target=None)
            x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
        return x_r, x_r_lambda, x_r_feats

    @torch.no_grad()
    def encode_identity_into_motion(self, x_r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_r: (B, 512) : appearance latent
        Returns:
            r_x: (B, 512) : motion latent
        """
        if self.opt.lia_version == "LIA_X":
            x_r_lambda = self.motion_autoencoder.enc.enc_r2t(x_r)
            r_x = self.motion_autoencoder.dec.direction(x_r_lambda)
        else:
            x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
            r_x = self.motion_autoencoder.dec.direction(x_r_lambda)
        return r_x

    @torch.no_grad()
    def decode_latent_into_image(self, s_r: torch.Tensor , s_r_feats: list, r_d: torch.Tensor) -> dict:
        T = r_d.shape[1]
        d_hat = []
        for t in range(T):
            s_r_d_t = s_r + r_d[:, t]
            if self.opt.lia_version == "LIA_X":
                img_t = self.motion_autoencoder.dec(s_r_d_t, alpha = None, feats = s_r_feats)
            else:
                img_t, _ = self.motion_autoencoder.dec(s_r_d_t, alpha = None, feats = s_r_feats)
            d_hat.append(img_t)
        d_hat = torch.stack(d_hat, dim=1).squeeze()
        return {'d_hat': d_hat}

    def forward(
        self,
        r_s: torch.Tensor,
        audio: torch.Tensor,
        alpha_dict: dict,
        intensity_scores: torch.Tensor = None,
        reaction_scores: torch.Tensor = None,
        # metadata: dict,
    ) -> dict:
        """
        Args:
            r_s: (B, T, 512) : motion latent
            alpha_dict: motion latent conefficient stats. (B, 20)
            audio: (B, T * 16000) : audio
        Returns:
            dict:
                'loss': loss
                'loss_ot': loss of total variation
                'loss_vel': loss of velocity
        """

        # Scale the motion latent to match the LIA-X's expected scale
        r_s = r_s * self.opt.r_s_scale

        ### Comment out for validation
        # assert self.fmt.training
        # assert self.audio_encoder.training
        # assert self.emotion_encoder.training
        B = audio.shape[0]
        num_frame_orig = r_s.shape[1]
        # assert num_frame_orig >= self.num_frames_for_clip
        assert num_frame_orig >= self.num_total_frames

        # 1. Randomly select start frame number
        start_frame_num = random.randint(0, num_frame_orig - self.num_frames_for_clip - self.num_prev_frames - self.num_ref_frames) + self.num_prev_frames
        prev_audio = audio[:, int((start_frame_num - self.num_prev_frames) / self.opt.fps * self.opt.sampling_rate) : int(start_frame_num / self.opt.fps * self.opt.sampling_rate)]
        prev_r_s = r_s[:, start_frame_num - self.num_prev_frames : start_frame_num]

        if intensity_scores is not None:
            prev_intensity_scores = intensity_scores[:, start_frame_num - self.num_prev_frames : start_frame_num]
        else:
            prev_intensity_scores = torch.zeros(B, self.num_prev_frames, 1, device = self.opt.rank)
        # prev_reaction_scores = reaction_scores[:, start_frame_num - self.num_prev_frames : start_frame_num]

        # r_s: (B, L, 512) motion latent
        # audio: (B, L / 25 * 16000)
        r_s_current = r_s[:, start_frame_num : start_frame_num + self.num_frames_for_clip]
        audio_current = audio[:, int(start_frame_num / self.opt.fps * self.opt.sampling_rate) : int((start_frame_num + self.num_frames_for_clip) / self.opt.fps * self.opt.sampling_rate)]

        if intensity_scores is not None:
            intensity_scores_current = intensity_scores[:, start_frame_num : start_frame_num + self.num_frames_for_clip]
        else:
            intensity_scores_current = torch.zeros(B, self.num_frames_for_clip, 1, device = self.opt.rank)
        if reaction_scores is not None:
            reaction_scores_current = reaction_scores[:, start_frame_num : start_frame_num + self.num_frames_for_clip]
        else:
            reaction_scores_current = torch.zeros(B, self.num_frames_for_clip, 1, device = self.opt.rank)

        ref_r_s = r_s[:, start_frame_num + self.num_frames_for_clip: start_frame_num + self.num_frames_for_clip + self.num_ref_frames]
        assert ref_r_s.shape[1] == self.num_ref_frames, f"ref_r_s: {ref_r_s.shape}, self.num_ref_frames: {self.num_ref_frames}"

        # r_s: (B, L' + L, 512) motion latent
        # audio: (B, (L' + L) / 25 * 16000)
        r_s = torch.cat([prev_r_s, r_s_current], dim=1)
        audio = torch.cat([prev_audio, audio_current], dim=1)

        intensity_scores = torch.cat([prev_intensity_scores, intensity_scores_current], dim=1) # (B, L' + L, 1)
        # reaction_scores = torch.cat([prev_reaction_scores, reaction_scores_current], dim=1) # (B, L' + L, 1)

        # Select r_s_0 from random frame
        # r_s_0: (B, 512)
        r_s_0 = r_s[:, random.randint(0, r_s.shape[1] - 1), :]

        # 2. Encode audio
        # audio: (B, audio_len) whole audio before window selection
        # wa: (B, L' + L, 512)
        # we: (B, 1, 7)
        T = math.ceil(audio.shape[-1] * self.opt.fps / self.opt.sampling_rate)

        wa = self.audio_encoder.inference(audio, seq_len=T)
        we = self.emotion_encoder.predict_emotion(audio).unsqueeze(1)
        intensity_scores = self.intensity_embedder(intensity_scores) # (B, L' + L, dim_i)

        assert start_frame_num >= self.num_prev_frames, f"start_frame_num: {start_frame_num}, self.num_prev_frames: {self.num_prev_frames}"

        prev_x = r_s[:, :self.num_prev_frames, :]                    # (B, L', 512)
        prev_wa = wa[:, :self.num_prev_frames]                       # (B, L', 512)
        prev_intensity_scores = intensity_scores[:, :self.num_prev_frames]

        wa_t = wa[:, -self.num_frames_for_clip:]                     # (B, L, 512)
        intensity_scores_t = intensity_scores[:, -self.num_frames_for_clip:] # (B, L, dim_i)
        null_wa = torch.zeros_like(wa_t)                             # (B, L, 512)
        null_we = torch.zeros_like(we)                               # (B, 1, 7)

        intensity_scores_null = torch.zeros_like(intensity_scores_t)
        prev_intensity_scores_null = torch.zeros_like(prev_intensity_scores)

        # 3. Compute loss
        t = torch.rand(B, device = self.opt.rank)                    # (B,)
        x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device = self.opt.rank)  # (B, L, 512)
        x1_target = r_s[:, -self.num_frames_for_clip:, :]           # (B, L, 512)

        x_t = (1- t.unsqueeze(-1).unsqueeze(-1)) * x0 + t.unsqueeze(-1).unsqueeze(-1) * x1_target  # (B, L, 512)
        dx_t = x1_target - x0                                        # (B, L, 512)

		# 	t: (B,) tensor of diffusion timesteps [0, 1]
		# 	x: (B, L, 512) : tensor of sequence of motion latent
		# 	wa:  (B, L, 512)  / tensor sequence of wa latent
		# 	wr:  (B, 512)     / tensor of reference motion latent (i.e., r -> s)
		# 	we:  (B, 1, 7)    / tensor of emotion latent
        #   alpha_dict: (B, 20)   / tensor of alpha statistics
		# 	prev_x:  (B, L', 512) / previous x for auto-regressive generation
		# 	prev_wa: (B, L', 512) / previous audio for auto-regressive generation

        # print(f"x_t: {x_t.shape}, wa_t: {wa_t.shape}, r_s: {r_s.shape}, we: {we.shape}, prev_x: {prev_x.shape}, prev_wa: {prev_wa.shape}")
        # dx1_pred: (B, L, 512)

        if self.fmt.num_stat_cond > 0:
            if random.random() < 0.9:
                alpha_dict = {k: v.unsqueeze(1) for k, v in alpha_dict.items()}
            else:
                alpha_dict = self.fmt.get_null_alpha_statistics(B)
        
        if self.num_ref_frames > 0:
            if random.random() < 0.1:
                ref_r_s = torch.zeros(B, self.num_ref_frames, self.opt.dim_w, device = self.opt.rank)

        if random.random() < 0.9:
            dx1_pred, hidden_latents = self.fmt.forward(
                t, x_t, wa_t, r_s_0, we, alpha_dict=alpha_dict, prev_x=prev_x, prev_wa=prev_wa, ref_x=ref_r_s, train=True, intensity_scores=intensity_scores_t, prev_intensity_scores=prev_intensity_scores,
                # reaction_scores=reaction_scores
            )
        else:
            dx1_pred, hidden_latents = self.fmt.forward(
                t, x_t, null_wa, r_s_0, null_we, alpha_dict=alpha_dict, prev_x=prev_x, prev_wa=prev_wa, ref_x=ref_r_s, train=True, intensity_scores=intensity_scores_null, prev_intensity_scores=prev_intensity_scores_null,
                # reaction_scores=reaction_scores_null
            )

        # print(f"dx_t: {dx_t.shape}, dx1_pred: {dx1_pred.shape}")
        loss_ot = nn.L1Loss()(dx_t, dx1_pred[:, -self.num_frames_for_clip:, :])
        loss_vel = nn.L1Loss()(dx_t[:, :-1, :] - dx_t[:, 1:, :], dx1_pred[:, -self.num_frames_for_clip:-1, :] - dx1_pred[:, -self.num_frames_for_clip+1:, :])

        # Reaction scores loss
        motion_hidden = hidden_latents
        if self.fmt.num_stat_cond + self.fmt.num_null_cond > 0:
            motion_hidden = motion_hidden[:, :-(self.fmt.num_stat_cond + self.fmt.num_null_cond), :]
        if self.num_ref_frames > 0:
            motion_hidden = motion_hidden[:, self.num_ref_frames:, :]
        
        reaction_scores_pred = self.reaction_classifier(self.channel_projection(motion_hidden))
        reaction_scores_pred_curr = reaction_scores_pred[:, -self.num_frames_for_clip:, :]
        assert reaction_scores_pred_curr.shape == reaction_scores_current.shape, f"reaction_scores_pred_curr: {reaction_scores_pred_curr.shape}, reaction_scores_current: {reaction_scores_current.shape}"
        loss_reaction = self.reaction_loss(reaction_scores_pred_curr, reaction_scores_current)
        # print(f"Loss ot: {loss_ot.item()}, Loss vel: {loss_vel.item()}, Loss reaction: {loss_reaction.item()}")

        loss = loss_ot + self.opt.vel_loss_weight * loss_vel + self.opt.reaction_loss_weight * loss_reaction

        return {
            'loss': loss,
            'loss_ot': loss_ot,
            'loss_vel': loss_vel,
            'loss_reaction': loss_reaction
        }

    ######## Motion Sampling and Inference ########
    @torch.no_grad()
    def sample(
        self,
        data: dict,
        a_cfg_scale: float = 1.0,
        r_cfg_scale: float = 1.0,
        e_cfg_scale: float = 1.0,
        s_cfg_scale: float = 1.0,
        ref_cfg_scale: float = 1.0,
        i_cfg_scale: float = 1.0,
        emo: str = None,
        nfe: int = 10,
        seed: int = None,
        alpha_dict = None
    ) -> torch.Tensor:
        """
        alpha_dict: dict, alpha statistics (alpha_mean, alpha_std, speed_mean, speed_std, accel_mean, accel_std). shape: (dim_m_condition,) or None (null conditioning)
        """

        if self.opt.use_alpha_mean and (alpha_dict is None or "alpha_mean" not in alpha_dict):
            logger.warning("Warning: FMT is trained with alpha_mean conditioning, but alpha_mean is not provided. Using null values.")
            alpha_dict = {} if alpha_dict is None else alpha_dict
            alpha_dict["alpha_mean"] = self.fmt.null_alpha_mean.squeeze()
        if self.opt.use_alpha_std and (alpha_dict is None or "alpha_std" not in alpha_dict):
            logger.warning("Warning: FMT is trained with alpha_std conditioning, but alpha_std is not provided. Using null values.")
            alpha_dict = {} if alpha_dict is None else alpha_dict
            alpha_dict["alpha_std"] = self.fmt.null_alpha_std.squeeze()
        if self.opt.use_speed_mean and (alpha_dict is None or "speed_mean" not in alpha_dict):
            logger.warning("Warning: FMT is trained with speed_mean conditioning, but speed_mean is not provided. Using null values.")
            alpha_dict = {} if alpha_dict is None else alpha_dict
            alpha_dict["speed_mean"] = self.fmt.null_speed_mean.squeeze()
        if self.opt.use_speed_std and (alpha_dict is None or "speed_std" not in alpha_dict):
            logger.warning("Warning: FMT is trained with speed_std conditioning, but speed_std is not provided. Using null values.")
            alpha_dict = {} if alpha_dict is None else alpha_dict
            alpha_dict["speed_std"] = self.fmt.null_speed_std.squeeze()
        if self.opt.use_accel_mean and (alpha_dict is None or "accel_mean" not in alpha_dict):
            logger.warning("Warning: FMT is trained with accel_mean conditioning, but accel_mean is not provided. Using null values.")
            alpha_dict = {} if alpha_dict is None else alpha_dict
            alpha_dict["accel_mean"] = self.fmt.null_accel_mean.squeeze()
        if self.opt.use_accel_std and (alpha_dict is None or "accel_std" not in alpha_dict):
            logger.warning("Warning: FMT is trained with accel_std conditioning, but accel_std is not provided. Using null values.")
            alpha_dict = {} if alpha_dict is None else alpha_dict
            alpha_dict["accel_std"] = self.fmt.null_accel_std.squeeze()

        r_s, a = data['r_s'], data['a']
        B = a.shape[0]


        # Scale the motion latent to match the LIA-X's expected scale
        r_s = r_s * self.opt.r_s_scale

        # make time
        time = torch.linspace(0, 1, self.opt.nfe, device=self.opt.rank)

        # encoding audio first with whole audio
        a = a.to(self.opt.rank)
        T = math.ceil(a.shape[-1] * self.opt.fps / self.opt.sampling_rate)
        wa = self.audio_encoder.inference(a, seq_len=T)
        # wa: (B, T, 512)

        # Encode intensity scores

        if "intensity" in data and data["intensity"] is not None:
            intensity_scores = data["intensity"].to(self.opt.rank)
        elif "intensity_scores" in data and data["intensity_scores"] is not None:
            intensity_scores = data["intensity_scores"].to(self.opt.rank)
        else:
            intensity_scores = torch.zeros(B, T, 1, device=self.opt.rank)
        if intensity_scores.shape[1] < T:
            intensity_scores = F.pad(intensity_scores, (0, 0, 0, T - intensity_scores.shape[1]), mode="replicate")
        elif intensity_scores.shape[1] > T:
            intensity_scores = intensity_scores[:, :T]
        intensity_scores = self.intensity_embedder(intensity_scores) # (B, T, dim_i)

        # encoding emotion first
        emo_idx = self.emotion_encoder.label2id.get(str(emo).lower(), None)
        if emo_idx is None:
            we = self.emotion_encoder.predict_emotion(a).unsqueeze(1)
        else:
            we = F.one_hot(torch.tensor(emo_idx, device = a.device), num_classes = self.opt.dim_e).unsqueeze(0).unsqueeze(0)
        # we: (B, 1, 7)

        if 'prev_r_s' in data and data['prev_r_s'] is not None:
            prev_r_s = data['prev_r_s'].to(self.opt.rank)
            prev_a = data['prev_a'].to(self.opt.rank)
            T_prev = int(prev_a.shape[-1] * self.opt.fps / self.opt.sampling_rate)
            prev_wa = self.audio_encoder.inference(prev_a.to(self.opt.rank), seq_len=T_prev)
            sample = prev_r_s # Keep the original scale of the motion latent
            prev_r_s = prev_r_s * self.opt.r_s_scale

            prev_intensity = data["prev_intensity"].to(self.opt.rank)
            prev_intensity = self.intensity_embedder(prev_intensity) # (B, T, dim_i)
        else:
            sample = torch.zeros(B, 0, self.opt.dim_w, device = self.opt.rank)
            T_prev = 0
        
        if self.num_ref_frames > 0:
            if 'ref_r_s' in data and data['ref_r_s'] is not None:
                ref_r_s = data['ref_r_s'].to(self.opt.rank)
                ref_r_s = ref_r_s * self.opt.r_s_scale
            else:
                ref_r_s = torch.zeros(B, self.num_ref_frames, self.opt.dim_w, device = self.opt.rank)
                logger.warning("Warning: ref_r_s is not provided. Using zeros.")
        else:
            ref_r_s = None
        print(f"T: {T}. T_prev: {T_prev}. self.num_frames_for_clip: {self.num_frames_for_clip}. ref_r_s: {ref_r_s.shape if ref_r_s is not None else None}")
                
        # FMT timing measurement
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        # sampleing chunk by chunk
        for t in range(0, int(math.ceil(T / self.num_frames_for_clip))):
            torch.cuda.synchronize()
            start_time = time_.time()
            start_event.record()
            if self.opt.fix_noise_seed:
                seed = self.opt.seed if seed is None else seed
                g = torch.Generator(self.opt.rank)
                g.manual_seed(seed)
                x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device = self.opt.rank, generator = g)
            else:
                x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device = self.opt.rank)

            if t == 0: # should define the previous
                if T_prev > 0:
                    prev_x_t = prev_r_s[:, -self.num_prev_frames:]
                    prev_wa_t = prev_wa[:, -self.num_prev_frames:]
                    prev_intensity_t = prev_intensity[:, -self.num_prev_frames:]
                else:
                    prev_x_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(self.opt.rank)
                    prev_wa_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_a).to(self.opt.rank)
                    prev_intensity_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_i).to(self.opt.rank)
            else:
                prev_x_t = sample[:, -self.num_prev_frames:]
                prev_wa_t = wa[:, t * self.num_frames_for_clip - self.num_prev_frames : t * self.num_frames_for_clip]
                prev_intensity_t = intensity_scores[:, t * self.num_frames_for_clip - self.num_prev_frames : t * self.num_frames_for_clip]
            
            if prev_x_t.shape[1] != self.num_prev_frames:
                prev_x_t = F.pad(prev_x_t, (0, 0, self.num_prev_frames - prev_x_t.shape[1], 0), mode='constant', value=0)
            if prev_wa_t.shape[1] != self.num_prev_frames:
                prev_wa_t = F.pad(prev_wa_t, (0, 0, self.num_prev_frames - prev_wa_t.shape[1], 0), mode='constant', value=0)
            if prev_intensity_t.shape[1] != self.num_prev_frames:
                prev_intensity_t = F.pad(prev_intensity_t, (0, 0, self.num_prev_frames - prev_intensity_t.shape[1], 0), mode='constant', value=0)
            logger.info(f"{t}-th chunk, prev_x_t: {prev_x_t.shape}, prev_wa_t: {prev_wa_t.shape}, x0: {x0.shape}")

            wa_t = wa[:, t * self.num_frames_for_clip: (t+1)*self.num_frames_for_clip]
            if wa_t.shape[1] < self.num_frames_for_clip: # padding by replicate
                wa_t = F.pad(wa_t, (0, 0, 0, self.num_frames_for_clip - wa_t.shape[1]), mode='replicate')
            
            intensity_t = intensity_scores[:, t * self.num_frames_for_clip: (t+1)*self.num_frames_for_clip]
            if intensity_t.shape[1] < self.num_frames_for_clip: # padding by replicate
                intensity_t = F.pad(intensity_t, (0, 0, 0, self.num_frames_for_clip - intensity_t.shape[1]), mode='replicate')

            def sample_chunk(tt, zt):
                
                out = self.fmt.forward_with_cfv(
                        t             = tt.unsqueeze(0),
                        x             = zt,
                        wa             = wa_t,
                        wr             = r_s,
                        we             = we,
                        prev_x         = prev_x_t,
                        prev_wa        = prev_wa_t,
                        ref_x          = ref_r_s,
                        intensity_scores = intensity_t,
                        prev_intensity_scores = prev_intensity_t,
                        a_cfg_scale = a_cfg_scale,
                        r_cfg_scale = r_cfg_scale,
                        e_cfg_scale = e_cfg_scale,
                        s_cfg_scale = s_cfg_scale,
                        ref_cfg_scale = ref_cfg_scale,
                        i_cfg_scale = i_cfg_scale,
                        alpha_dict = alpha_dict
                        )
                out_current = out[:, self.num_prev_frames:]
                return out_current

            # solve ODE
            trajectory_t = odeint(sample_chunk, x0, time, **self.odeint_kwargs)
            sample_t = trajectory_t[-1]
            sample = torch.cat([sample, sample_t / self.opt.r_s_scale], dim=1) # Scale the motion latent back to the original scale
            logger.info(f"{t}-th chunk, sample_t: {sample_t.shape}, sample: {sample.shape}")
            
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            total_time = time_.time() - start_time
            print(f"{t}-th chunk [FMT] {elapsed_ms:.2f} ms (total: {total_time:.3f}s)", flush=True)
        sample =sample[:, :T + T_prev]
        return sample

    @torch.no_grad()
    def inference(
        self,
        data: dict,
        a_cfg_scale = None,
        r_cfg_scale = None,
        e_cfg_scale = None,
        s_cfg_scale = None,
        ref_cfg_scale = None,
        i_cfg_scale = None,
        emo         = None,
        nfe         = 10,
        seed        = None,
        alpha_dict  = None
    ) -> dict:

        torch.cuda.synchronize()
        start_time = time_.time()
        s, a = data['s'], data['a']
        s_r, r_s_lambda, s_r_feats = self.encode_image_into_latent(s.to(self.opt.rank))
        # print(f"s_r: {s_r.shape}, r_s_lambda: {r_s_lambda.shape}, s_r_feats: {type(s_r_feats)}")
        # s_r: (B, 512)
        # r_s_lambda: (B, 20)
        # s_r_feats: list of tensors
        if 's_r' in data:
            r_s = self.encode_identity_into_motion(s_r)
        else:
            r_s = self.motion_autoencoder.dec.direction(r_s_lambda)
            # r_s: (B, 512)
        data['r_s'] = r_s
        torch.cuda.synchronize()
        print(f"encoding time: {time_.time() - start_time}")
        start_time = time_.time()

        # set conditions
        if a_cfg_scale is None: a_cfg_scale = self.opt.a_cfg_scale
        if r_cfg_scale is None: r_cfg_scale = self.opt.r_cfg_scale
        if e_cfg_scale is None: e_cfg_scale = self.opt.e_cfg_scale
        if s_cfg_scale is None: s_cfg_scale = self.opt.s_cfg_scale
        if ref_cfg_scale is None: ref_cfg_scale = self.opt.ref_cfg_scale
        if i_cfg_scale is None: i_cfg_scale = self.opt.i_cfg_scale

        sample = self.sample(
            data,
            a_cfg_scale = a_cfg_scale,
            r_cfg_scale = r_cfg_scale,
            e_cfg_scale = e_cfg_scale,
            s_cfg_scale = s_cfg_scale,
            ref_cfg_scale = ref_cfg_scale,
            i_cfg_scale = i_cfg_scale,
            emo = emo,
            nfe = nfe,
            seed = seed,
            alpha_dict = alpha_dict
        )
        torch.cuda.synchronize()
        print(f"sampling time: {time_.time() - start_time}")
        start_time = time_.time()
        data_out = self.decode_latent_into_image(s_r = s_r, s_r_feats = s_r_feats, r_d = sample)
        data_out['r_d'] = sample
        torch.cuda.synchronize()
        print(f"decodingtime: {time_.time() - start_time}")
        start_time = time_.time()
        return data_out

    @torch.no_grad()
    def inference_from_latent(
        self,
        data: dict,
        a_cfg_scale = None,
        r_cfg_scale = None,
        e_cfg_scale = None,
        s_cfg_scale = None,
        ref_cfg_scale = None,
        i_cfg_scale = None,
        emo = None,
        nfe = 10,
        seed = None,
        alpha_dict = None,
    ) -> dict:
        """
        Args:
            data: dict
                s_r: (1, 512)
                s_r_feats: list of tensors
                r_s: (1, 512)
        """

        s_r, s_r_feats, r_s = data['s_r'], data['s_r_feats'], data['r_s']

        # set conditions
        if a_cfg_scale is None: a_cfg_scale = self.opt.a_cfg_scale
        if r_cfg_scale is None: r_cfg_scale = self.opt.r_cfg_scale
        if e_cfg_scale is None: e_cfg_scale = self.opt.e_cfg_scale
        if s_cfg_scale is None: s_cfg_scale = self.opt.s_cfg_scale
        if ref_cfg_scale is None: ref_cfg_scale = self.opt.ref_cfg_scale
        if i_cfg_scale is None: i_cfg_scale = self.opt.i_cfg_scale

        sample = self.sample(
            data = data,
            a_cfg_scale = a_cfg_scale,
            r_cfg_scale = r_cfg_scale,
            e_cfg_scale = e_cfg_scale,
            s_cfg_scale = s_cfg_scale,
            ref_cfg_scale = ref_cfg_scale,
            i_cfg_scale = i_cfg_scale,
            emo = emo,
            nfe = nfe,
            seed = seed,
            alpha_dict = alpha_dict
        )
        data_out = self.decode_latent_into_image(s_r = s_r, s_r_feats = s_r_feats, r_d = sample)
        return data_out

################ Condition Encoders ################
class AudioEncoder(BaseModel):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.only_last_features = opt.only_last_features

        self.num_frames_for_clip = int(opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(opt.num_prev_frames)

        self.wav2vec2 = Wav2VecModel.from_pretrained(
            opt.wav2vec_model_path, 
            local_files_only = True,
            output_attentions=False, # 关闭注意力机制
            # attn_implementation="eager"   # 保留加速
            attn_implementation="sdpa"
        )
        self.wav2vec2.eval()
        self.wav2vec2.feature_extractor._freeze_parameters()

        for name, param in self.wav2vec2.named_parameters():
            param.requires_grad = False

        audio_input_dim = 768 if opt.only_last_features else 12 * 768

        self.audio_projection = nn.Sequential(
            nn.Linear(audio_input_dim, opt.dim_a),
            nn.LayerNorm(opt.dim_a),
            nn.SiLU()
        )

    def get_wav2vec2_feature(self, a: torch.Tensor, seq_len:int) -> torch.Tensor:
        with torch.inference_mode():
            with torch.amp.autocast(enabled=False, device_type="cuda"):
            # with torch.no_grad():
                a = self.wav2vec2(a, seq_len=seq_len, output_hidden_states = not self.only_last_features, output_attentions=False)
        # a = a.to(dtype=self.audio_projection[0].weight.dtype)
        if self.only_last_features:
            # a = a.last_hidden_state
            a = a.last_hidden_state.to(dtype=self.audio_projection[0].weight.dtype)
        else:
            # a = torch.stack(a.hidden_states[1:], dim=1).permute(0, 2, 1, 3)
            a = torch.stack(a.hidden_states[1:], dim=1).to(dtype=self.audio_projection[0].weight.dtype).permute(0, 2, 1, 3)
            a = a.reshape(a.shape[0], a.shape[1], -1)
        return a

    def forward(self, a:torch.Tensor, prev_a:torch.Tensor = None) -> torch.Tensor:
        if prev_a is not None:
            a = torch.cat([prev_a, a], dim = 1)
            if a.shape[1] % int( (self.num_frames_for_clip + self.num_prev_frames) * self.opt.sampling_rate / self.opt.fps) != 0:
                a = F.pad(a, (0, int((self.num_frames_for_clip + self.num_prev_frames) * self.opt.sampling_rate / self.opt.fps) - a.shape[1]), mode='replicate')
            a = self.get_wav2vec2_feature(a, seq_len = self.num_frames_for_clip + self.num_prev_frames)
        else:
            if a.shape[1] % int( self.num_frames_for_clip * self.opt.sampling_rate / self.opt.fps) != 0:
                a = F.pad(a, (0, int(self.num_frames_for_clip * self.opt.sampling_rate / self.opt.fps) - a.shape[1]), mode = 'replicate')
            a = self.get_wav2vec2_feature(a, seq_len = self.num_frames_for_clip)

        return self.audio_projection(a) # frame by frame

    @torch.no_grad()
    def inference(self, a: torch.Tensor, seq_len:int) -> torch.Tensor:
        if a.shape[1] % int(seq_len * self.opt.sampling_rate / self.opt.fps) != 0:
            a = F.pad(a, (0, int(seq_len * self.opt.sampling_rate / self.opt.fps) - a.shape[1]), mode = 'replicate')
        a = self.get_wav2vec2_feature(a, seq_len=seq_len)
        return self.audio_projection(a)


class Audio2Emotion(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.wav2vec2_for_emotion = Wav2Vec2ForSpeechClassification.from_pretrained(
            opt.audio2emotion_path, 
            local_files_only=True, 
            low_cpu_mem_usage=False,
            # output_attentions=False, # 关闭注意力机制
            attn_implementation="sdpa"   # 保留加速
        )
        self.wav2vec2_for_emotion.eval()

        # seven labels
        self.id2label = {0: "angry", 1: "disgust", 2: "fear", 3: "happy",
                        4: "neutral", 5: "sad", 6: "surprise"}

        self.label2id = {v: k for k, v in self.id2label.items()}

    @torch.no_grad()
    def predict_emotion(self, a: torch.Tensor, prev_a: torch.Tensor = None) -> torch.Tensor:
        if prev_a is not None:
            a = torch.cat([prev_a, a], dim=1)
        logits = self.wav2vec2_for_emotion.forward(a).logits
        return F.softmax(logits, dim=1)     # scores

#######################################################


# This is only for encoding intensity scores during inference when there is no already encoded intensity scores
# The intensity scores are encoded with input audio using Qwen2-Audio-7B-Instruct model with a fixed prompt
# class IntensityEncoder(nn.Module):
#     def __init__(self, opt):
#         super().__init__()
        
#         self.line_pattern = re.compile(
#             r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*$"
#         )

#         self.prompt = self.get_prompt()
#         self.tokenizer = AutoTokenizer.from_pretrained(opt.qwen2audio_path, local_files_only=True)
    
#     def get_prompt(self) -> str:
#         prompt = """
# You are an expert in speech prosody analysis.

# You will be given an audio segment containing one speaker. Detect only the time
# intervals where the speaker shows a clear and localized prosodic fluctuation. A
# prosodic fluctuation means a noticeable change in vocal delivery, such as increased
# pitch, increased loudness, stronger stress, sharper emphasis, faster or slower
# speaking rate, unusual rhythm, hesitation, excitement, surprise, or other affective
# vocal variation. Focus only on acoustic and prosodic cues, not on the semantic
# meaning of the spoken words.

# For each detected prosodic event, output its start time, end time, and intensity score.
# The intensity score must be between 0 and 1 and should represent the salience of the
# prosodic change:
# - 0.50-0.60: weak but noticeable;
# - 0.60-0.80: clear;
# - 0.80-1.00: strong or highly salient.

# Detection rules:
# - Only output events with intensity score >= 0.50.
# - Do not output uncertain or ambiguous events.
# - Do not split one continuous prosodic fluctuation into many short intervals.
# - Merge neighboring intervals if they are part of the same prosodic event.
# - No interval overlapping.
# - Prefer concise intervals that cover the main fluctuation rather than long segments
# with neutral speech.

# Output format:
# - Each line must be: start_time, end_time, intensity_score. Results must only have two decimal.
# - All numbers must use exactly two decimal places.
# - Use seconds as the time unit.
# - Do not output any explanation, label, markdown, bullet point, or extra text.
# - If no valid event is detected, output exactly:
# NULL

# Analyze the audio now and output only the formatted results.
#         """.strip()
#         return prompt

#     def forward(self, intensity_scores: torch.Tensor) -> torch.Tensor:
#         intensity_scores = self.intensity_embed(intensity_scores)
#         intensity_scores = self.norm(intensity_scores)
#         intensity_scores = self.act(intensity_scores)
#         return intensity_scores
    