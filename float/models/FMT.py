import os, math, torch, random
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from torchdiffeq import odeint
from float.models import BaseModel

from timm.layers import use_fused_attn
from timm.models.vision_transformer import Mlp


def enc_dec_mask(T, S, frame_width = 1, expansion = 2, num_ref_frames = 0):
	mask = torch.ones(T, S)
	for i in range(T):
		mask[i, max(0, (i - expansion) * frame_width):(i + expansion + 1) * frame_width] = 0
		mask[i, :num_ref_frames] = 0
		if i < num_ref_frames:
			mask[i, :] = 0
	return mask == 1


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
	"""
	Sinusoidal position encoding table.
	Args:
		n_position (int): the length of the input sequence
		d_hid (int): the dimension of the hidden state
	"""
	def cal_angle(position, hid_idx):
		return position / (10000 ** (2 * (hid_idx // 2) / d_hid))

	def get_posi_angle_vec(position):
		return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

	sinusoid_table = torch.Tensor([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
	sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
	sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1
	if padding_idx is not None: sinusoid_table[padding_idx] = 0.
	return sinusoid_table


class Attention(nn.Module):
	def __init__(
			self,
			dim: int,
			num_heads: int = 8,
			qkv_bias: bool = False,
			qk_norm: bool = False,
			attn_drop: float = 0.,
			proj_drop: float = 0.,
			norm_layer: nn.Module = nn.LayerNorm,
	) -> None:

		super().__init__()
		assert dim % num_heads == 0, 'dim should be divisible by num_heads'
		self.num_heads = num_heads
		self.head_dim = dim // num_heads
		self.scale = self.head_dim ** -0.5
		self.fused_attn = use_fused_attn()

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
		self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
		q, k, v = qkv.unbind(0)
		q, k = self.q_norm(q), self.k_norm(k)

		if self.fused_attn:
			x = F.scaled_dot_product_attention(
				q, k, v,
				attn_mask = ~mask,
				dropout_p=self.attn_drop.p if self.training else 0.,
			)
		else:
			q = q * self.scale
			attn = q @ k.transpose(-2, -1)
			attn = attn.softmax(dim=-1)
			attn = self.attn_drop(attn)
			x = attn @ v

		x = x.transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x

class TimestepEmbedder(nn.Module):
	"""
	Embeds scalar timesteps into vector representations.
	"""
	def __init__(self, hidden_size, frequency_embedding_size = 256):
		super().__init__()
		self.mlp = nn.Sequential(
			nn.Linear(frequency_embedding_size, hidden_size, bias=True),
			nn.SiLU(),
			nn.Linear(hidden_size, hidden_size, bias=True),
		)
		self.frequency_embedding_size = frequency_embedding_size

	@staticmethod
	def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
		"""
		Create sinusoidal timestep embeddings.
		:param t: a 1-D Tensor of N indices, one per batch element.
				  These may be fractional.
		:param dim: the dimension of the output.
		:param max_period: controls the minimum frequency of the embeddings.
		:return: an (N, D) Tensor of positional embeddings.
		"""
		# https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
		half = dim // 2
		freqs = torch.exp(
			-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
		).to(device=t.device)
		args = t[:, None].float() * freqs[None]
		embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
		if dim % 2:
			embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
		return embedding

	def forward(self, t: torch.Tensor) -> torch.Tensor:
		t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
		t_emb = self.mlp(t_freq)
		return t_emb

class SequenceEmbed(nn.Module):
	def __init__(
			self,
			dim_w,
			dim_h,
			norm_layer=None,
			bias=True,
	):
		super().__init__()

		self.proj = nn.Linear(dim_w, dim_h, bias=bias)
		self.norm = norm_layer(dim_h) if norm_layer else nn.Identity()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.norm(self.proj(x))


class FMTBlock(nn.Module):
	"""
	A FMT block inspried by DiT Block
	"""
	def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs) -> None:
		super().__init__()
		self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
		self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
		self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
		mlp_hidden_dim = int(hidden_size * mlp_ratio)
		approx_gelu = lambda: nn.GELU(approximate="tanh")
		self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
		self.adaLN_modulation = nn.Sequential(
			nn.SiLU(),
			nn.Linear(hidden_size, 6 * hidden_size, bias=True)
		)

	def framewise_modulate(self, x, shift, scale) -> torch.Tensor:
		return x * (1 + scale) + shift

	def forward(self, x, c, mask=None) -> torch.Tensor:
		assert mask is not None
		shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
		x = x + gate_msa * self.attn(self.framewise_modulate(self.norm1(x), shift_msa, scale_msa), mask = mask)
		x = x + gate_mlp * self.mlp(self.framewise_modulate(self.norm2(x), shift_mlp, scale_mlp))
		return x

class Decoder(nn.Module):
	"""
	The final decoder of FlowMatchingTransformer.
	"""
	def __init__(self, hidden_size, dim_w):
		super().__init__()
		self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
		self.adaLN_modulation = nn.Sequential(
			nn.SiLU(),
			nn.Linear(hidden_size, 2 * hidden_size, bias=True)
		)
		self.linear = nn.Linear(hidden_size, dim_w, bias=True)

	def framewise_modulate(self, x, shift, scale) -> torch.Tensor:
		return x * (1 + scale) + shift

	def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
		shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
		x = self.framewise_modulate(self.norm_final(x), shift, scale)
		return self.linear(x)


class FlowMatchingTransformer(BaseModel):
	"""
	Flow Matching Transformer (FMT)
	"""
	def __init__(self, opt) -> None:
		super().__init__()
		self.opt = opt
		self.num_stat_cond = int(self.opt.use_alpha_std) + int(self.opt.use_alpha_mean) + int(self.opt.use_speed_std) + int(self.opt.use_speed_mean) + int(self.opt.use_accel_std) + int(self.opt.use_accel_mean)
		self.num_null_cond = int(self.opt.num_null_cond)

		self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
		self.num_ref_frames = int(opt.num_ref_frames)
		self.num_prev_frames = int(opt.num_prev_frames)
		self.num_total_frames = self.num_prev_frames + self.num_frames_for_clip + self.num_ref_frames

		self.hidden_size = opt.dim_h
		self.mlp_ratio = opt.mlp_ratio
		self.fmt_depth = opt.fmt_depth
		self.num_heads = opt.num_heads

		self.x_embedder = SequenceEmbed(opt.dim_w, self.hidden_size)

		# video time position encoding
		self.pos_embed = nn.Parameter(torch.zeros(1, self.num_total_frames, self.hidden_size), requires_grad=False)

		# flow trajectory time encoding
		self.t_embedder = TimestepEmbedder(self.hidden_size)
		if self.opt.disable_motion_latent_condition:
			# self.c_embedder = nn.Linear(opt.dim_a + opt.dim_e, self.hidden_size)
			self.c_embedder = nn.Linear(opt.dim_a + opt.dim_e + opt.dim_i, self.hidden_size)
		else:
			# self.c_embedder = nn.Linear(opt.dim_w + opt.dim_a + opt.dim_e, self.hidden_size)
			self.c_embedder = nn.Linear(opt.dim_w + opt.dim_a + opt.dim_e + opt.dim_i, self.hidden_size)

		# define FMT blocks
		self.blocks = nn.ModuleList([FMTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio) for _ in range(self.fmt_depth)])
		self.decoder = Decoder(self.hidden_size, self.opt.dim_w)

		if self.opt.use_alpha_mean:
			self.alpha_mean_embed = nn.Linear(opt.dim_m_condition, self.hidden_size)
			self.null_alpha_mean = nn.Parameter(torch.zeros(1, opt.dim_m_condition), requires_grad=True)
		if self.opt.use_alpha_std:
			self.alpha_std_embed = nn.Linear(opt.dim_m_condition, self.hidden_size)
			self.null_alpha_std = nn.Parameter(torch.zeros(1, opt.dim_m_condition), requires_grad=True)
		if self.opt.use_speed_mean:
			self.speed_mean_embed = nn.Linear(opt.dim_m_condition, self.hidden_size)
			self.null_speed_mean = nn.Parameter(torch.zeros(1, opt.dim_m_condition), requires_grad=True)
		if self.opt.use_speed_std:
			self.speed_std_embed = nn.Linear(opt.dim_m_condition, self.hidden_size)
			self.null_speed_std = nn.Parameter(torch.zeros(1, opt.dim_m_condition), requires_grad=True)
		if self.opt.use_accel_mean:
			self.accel_mean_embed = nn.Linear(opt.dim_m_condition, self.hidden_size)
			self.null_accel_mean = nn.Parameter(torch.zeros(1, opt.dim_m_condition), requires_grad=True)
		if self.opt.use_accel_std:
			self.accel_std_embed = nn.Linear(opt.dim_m_condition, self.hidden_size)
			self.null_accel_std = nn.Parameter(torch.zeros(1,opt.dim_m_condition), requires_grad=True)
		if self.opt.num_null_cond > 0:
			self.null_cond_embeds = nn.ModuleList([
				nn.Linear(opt.dim_m_condition, self.hidden_size)
				for _ in range(self.num_null_cond)
			])
			self.null_conds = nn.ParameterList([
				nn.Parameter(torch.zeros(1, opt.dim_m_condition), requires_grad=True)
				for _ in range(self.num_null_cond)
			])


		self.initialize_weights()

		# define alignment mask
		alignment_mask = enc_dec_mask(self.num_total_frames, self.num_total_frames, 1, expansion=opt.attention_window, num_ref_frames=self.num_ref_frames).to(opt.rank)
		self.register_buffer('alignment_mask', alignment_mask)

		if self.num_stat_cond + self.num_null_cond > 0:
			expanded_mask = torch.ones(self.num_total_frames+self.num_stat_cond+self.num_null_cond, self.num_total_frames+self.num_stat_cond+self.num_null_cond, device=self.alignment_mask.device, dtype=self.alignment_mask.dtype)
			expanded_mask[:self.num_total_frames, :self.num_total_frames] = self.alignment_mask
			expanded_mask[self.num_total_frames:, :] = 0  # num_stat_cond + num_null_cond number of rows are all 0
			expanded_mask[:, self.num_total_frames:] = 0  # num_stat_cond + num_null_cond number of columns are all 0
			self.alignment_mask = expanded_mask.bool()

	def initialize_weights(self) -> None:
		def _basic_init(module):
			if isinstance(module, nn.Linear):
				torch.nn.init.xavier_uniform_(module.weight)
				if module.bias is not None:
					nn.init.constant_(module.bias, 0)

		self.apply(_basic_init)

		pos_embed = get_sinusoid_encoding_table(self.num_total_frames, self.hidden_size)
		self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

		w = self.x_embedder.proj.weight.data
		nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
		nn.init.constant_(self.x_embedder.proj.bias, 0)

		# Initialize timestep embedding MLP:
		nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
		nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

		# Zero-out adaLN modulation layers in FMT blocks:
		for block in self.blocks:
			nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
			nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

		if self.opt.use_alpha_mean:
			nn.init.xavier_uniform_(self.alpha_mean_embed.weight)
			nn.init.constant_(self.alpha_mean_embed.bias, 0)
		if self.opt.use_alpha_std:
			nn.init.xavier_uniform_(self.alpha_std_embed.weight)
			nn.init.constant_(self.alpha_std_embed.bias, 0)
		if self.opt.use_speed_mean:
			nn.init.xavier_uniform_(self.speed_mean_embed.weight)
			nn.init.constant_(self.speed_mean_embed.bias, 0)
		if self.opt.use_speed_std:
			nn.init.xavier_uniform_(self.speed_std_embed.weight)
			nn.init.constant_(self.speed_std_embed.bias, 0)
		if self.opt.use_accel_mean:
			nn.init.xavier_uniform_(self.accel_mean_embed.weight)
			nn.init.constant_(self.accel_mean_embed.bias, 0)
		if self.opt.use_accel_std:
			nn.init.xavier_uniform_(self.accel_std_embed.weight)
			nn.init.constant_(self.accel_std_embed.bias, 0)
		if self.opt.num_null_cond > 0:
			for null_cond_embed in self.null_cond_embeds:
				nn.init.xavier_uniform_(null_cond_embed.weight)
				nn.init.constant_(null_cond_embed.bias, 0)

		# Zero-out output layers:
		nn.init.constant_(self.decoder.adaLN_modulation[-1].weight, 0)
		nn.init.constant_(self.decoder.adaLN_modulation[-1].bias, 0)
		nn.init.constant_(self.decoder.linear.weight, 0)
		nn.init.constant_(self.decoder.linear.bias, 0)

	def sequence_embedder(self, sequence, dropout_prob, train=False) -> torch.Tensor:
		if train:
			batch_id_for_drop = torch.where(torch.rand(sequence.shape[0], device=sequence.device) < dropout_prob)
			sequence[batch_id_for_drop] = 0
		return sequence

	def get_null_alpha_statistics(self, B):
		null_alpha_stats = {}
		if self.opt.use_alpha_mean:
			null_alpha_stats['alpha_mean'] = self.null_alpha_mean.unsqueeze(0).repeat(B, 1, 1)
		if self.opt.use_alpha_std:
			null_alpha_stats['alpha_std'] = self.null_alpha_std.unsqueeze(0).repeat(B, 1, 1)
		if self.opt.use_speed_mean:
			null_alpha_stats['speed_mean'] = self.null_speed_mean.unsqueeze(0).repeat(B, 1, 1)
		if self.opt.use_speed_std:
			null_alpha_stats['speed_std'] = self.null_speed_std.unsqueeze(0).repeat(B, 1, 1)
		if self.opt.use_accel_mean:
			null_alpha_stats['accel_mean'] = self.null_accel_mean.unsqueeze(0).repeat(B, 1, 1)
		if self.opt.use_accel_std:
			null_alpha_stats['accel_std'] = self.null_accel_std.unsqueeze(0).repeat(B, 1, 1)
		return null_alpha_stats


	def forward(
		self,
		t,
		x,
		wa,
		wr,
		we,
		alpha_dict,
		prev_x = None,
		prev_wa = None,
		ref_x = None,
		train = True,
		intensity_scores = None,
		prev_intensity_scores = None,
		# reaction_scores = None,
		**kwargs
	) -> torch.Tensor:
		"""
		Forward pass of ConditionalFlowMatchingTransformer.

		Args:

			t: (B,) tensor of diffusion timesteps [0, 1]
			x: (B, L, 512) : tensor of sequence of motion latent

			wa:  (B, L, 512)  / tensor sequence of wa latent
			wr:  (B, 512)     / tensor of reference motion latent (i.e., r -> s)
			we:  (B, 1, 7)    / tensor of emotion latent
			alpha_dict: dict, alpha statistics (alpha_mean, alpha_std, speed_mean, speed_std, accel_mean, accel_std). shape: (B, 1, dim_m_condition)

			prev_x:  (B, L', 512) / previous x for auto-regressive generation
			prev_wa: (B, L', 512) / previous audio for auto-regressive generation

			ref_x:  (B, self.num_ref_frames, 512) / tensor of reference motion latent

		Returns:
			velocity: (B, L, 512) / tensor of velocity of motion latent
		"""

		B = x.shape[0]

		# print(f"Mean of wa: {wa.mean()}, Std of wa: {wa.std()}")
		# print(f"Mean of wr: {wr.mean()}, Std of wr: {wr.std()}")
		# print(f"Mean of we: {we.mean()}, Std of we: {we.std()}")
		# print(f"Mean of alpha_dict: {alpha_dict.mean()}, Std of alpha_dict: {alpha_dict.std()}")
		# print(f"Mean of intensity_scores: {intensity_scores.mean()}, Std of intensity_scores: {intensity_scores.std()}")
		# print("--------------------------------")

		# time encoding
		t = self.t_embedder(t).unsqueeze(1)  	# (N, D)

		# condition encoding
		wa = self.sequence_embedder(wa, dropout_prob = self.opt.audio_dropout_prob, train=train)
		wr = self.sequence_embedder(wr.unsqueeze(1), dropout_prob = self.opt.ref_dropout_prob, train=train)
		we = self.sequence_embedder(we, dropout_prob = self.opt.emotion_dropout_prob, train=train)
		alpha_dict = {k: self.sequence_embedder(v, dropout_prob = self.opt.alpha_dropout_prob, train=train) for k, v in alpha_dict.items()}
		if intensity_scores is not None:
			intensity_scores = self.sequence_embedder(intensity_scores, dropout_prob = self.opt.intensity_dropout_prob, train=train)

		# if intensity_scores is not None:
		# 	intensity_scores = self.sequence_embedder(intensity_scores, dropout_prob = self.opt.intensity_dropout_prob, train=train)
		# if reaction_scores is not None:
		# 	reaction_scores = self.sequence_embedder(reaction_scores, dropout_prob = self.opt.reaction_dropout_prob, train=train)

		# previous condition encoding
		if prev_x is not None:
			prev_x  = self.sequence_embedder(prev_x,  dropout_prob=0.5, train=train)
			prev_wa = self.sequence_embedder(prev_wa, dropout_prob=0.5, train=train)
			prev_intensity_scores = self.sequence_embedder(prev_intensity_scores, dropout_prob=0.5, train=train)

			x = torch.cat([prev_x, x], dim=1)
			wa = torch.cat([prev_wa, wa], dim=1)
			intensity_scores = torch.cat([prev_intensity_scores, intensity_scores], dim=1)

		if self.num_ref_frames > 0:
			assert ref_x is not None, "ref_x is required when num_ref_frames > 0"
			x = torch.cat([ref_x, x], dim=1)
		x = self.x_embedder(x)

		# Add control condition to the input
		additional_x_input = []
		if self.opt.use_alpha_mean:
			additional_x_input.append(self.alpha_mean_embed(alpha_dict['alpha_mean'] * self.opt.alpha_mean_scale)) # (B, 1, dim_m_condition)
		if self.opt.use_alpha_std:
			additional_x_input.append(self.alpha_std_embed(alpha_dict['alpha_std'] * self.opt.alpha_std_scale)) # (B, 1, dim_m_condition)
		if self.opt.use_speed_mean:
			additional_x_input.append(self.speed_mean_embed(alpha_dict['speed_mean'] * self.opt.speed_mean_scale)) # (B, 1, dim_m_condition)
		if self.opt.use_speed_std:
			additional_x_input.append(self.speed_std_embed(alpha_dict['speed_std'] * self.opt.speed_std_scale)) # (B, 1, dim_m_condition)
		if self.opt.use_accel_mean:
			additional_x_input.append(self.accel_mean_embed(alpha_dict['accel_mean'] * self.opt.accel_mean_scale)) # (B, 1, dim_m_condition)
		if self.opt.use_accel_std:
			additional_x_input.append(self.accel_std_embed(alpha_dict['accel_std'] * self.opt.accel_std_scale)) # (B, 1, dim_m_condition)
		if self.opt.num_null_cond > 0:
			additional_x_input.extend([self.null_cond_embeds[i](self.null_conds[i]).unsqueeze(1).repeat(B, 1, 1) for i in range(self.num_null_cond)]) # (B, self.num_null_cond, D)

		x = x + self.pos_embed		# (N, L + L', D), where T = opt.wav2vec_sec * opt.fps, D = dim_w

		# Concatenate alpha std to x after positional encoding
		if self.num_stat_cond + self.num_null_cond > 0:
			additional_x_input = torch.cat(additional_x_input, dim=1)  # (B, self.num_stat_cond, D)
			x = torch.cat([x, additional_x_input], dim=1) # (N, L + L' + self.num_stat_cond + self.num_null_cond, D)

		wr = wr.repeat(1, wa.shape[1], 1)
		we = we.repeat(1, wa.shape[1], 1)

		if self.opt.disable_motion_latent_condition:
			c = torch.cat([wa, we, intensity_scores], dim=-1)
		else:
			# print(f"Shape of wr: {wr.shape}, shape of wa: {wa.shape}, shape of we: {we.shape}, shape of intensity_scores: {intensity_scores.shape}")
			c = torch.cat([wr, wa, we, intensity_scores], dim=-1)
		
		# print(f"Shape of c: {c.shape}")

		# print(f"Mean of x: {x.mean()}, Std of x: {x.std()}")
		# print(f"Mean of c: {c.mean()}, Std of c: {c.std()}")
		# print("--------------------------------")
		
		c = self.c_embedder(c)
		c = t + c

		# print(f"Mean of x: {x.mean()}, Std of x: {x.std()}")
		# print(f"Mean of c: {c.mean()}, Std of c: {c.std()}")
		# print("--------------------------------")

		if self.num_ref_frames > 0:
			zero_c = torch.zeros(c.shape[0], self.num_ref_frames, c.shape[2], device=c.device, dtype=c.dtype) # (B, self.num_ref_frames, D)
			c = torch.cat([zero_c, c], dim=1)  # (B, self.num_ref_frames + L + L', D)

		# Add more dimensions to c for std embedding w/o t
		if self.num_stat_cond + self.num_null_cond > 0:
			# TODO: it could be better to use a learned embedding for std
			zero_c = torch.zeros(c.shape[0], self.num_stat_cond + self.num_null_cond, c.shape[2], device=c.device, dtype=c.dtype) # (B, self.num_stat_cond + self.num_null_cond, D)
			c = torch.cat([c, zero_c], dim=1)  # (B, L + L' + self.num_stat_cond + self.num_null_cond, D), where D = dim_w + dim_a + dim_e

		# print(f"Mean of x: {x.mean()}, Std of x: {x.std()}")
		# print(f"Mean of c: {c.mean()}, Std of c: {c.std()}")
		# print("--------------------------------")

		# Forwarding FMT Blocks
		for block in self.blocks:
			x = block(x, c, self.alignment_mask)  # (N, T, D)
			# print(f"Mean of x: {x.mean()}, Std of x: {x.std()}")
		
		hidden_latents = x

		# Remove std embedding from x and c
		if self.num_stat_cond + self.num_null_cond > 0:
			x = x[:, :-self.num_stat_cond - self.num_null_cond, :]  # remove the last dimension which is std embedding
			c = c[:, :-self.num_stat_cond - self.num_null_cond, :]  # remove the last dimension which is std embedding
		
		if self.num_ref_frames > 0:
			x = x[:, self.num_ref_frames:, :] # remove the first dimension which is reference frames
			c = c[:, self.num_ref_frames:, :] # remove the first dimension which is reference frames

		out = self.decoder(x, c)
		# print(f"Mean of out: {out.mean()}, Std of out: {out.std()}")

		return out, hidden_latents

	@torch.no_grad()
	def forward_with_cfv(
		self,
		t,
		x,
		wa,
		wr,
		we,
		alpha_dict,
		prev_x,
		prev_wa,
		ref_x=None,
		intensity_scores=None,
		prev_intensity_scores=None,
		a_cfg_scale=1.0,
		r_cfg_scale=1.0,
		e_cfg_scale=1.0,
		s_cfg_scale=1.0,
		ref_cfg_scale=1.0,
		i_cfg_scale=1.0,
		**kwargs
	) -> torch.Tensor:
		"""
		alpha_dict: dict, alpha statistics (alpha_mean, alpha_std, speed_mean, speed_std, accel_mean, accel_std). shape: (dim_m_condition,)
		"""
		if alpha_dict is not None:
			alpha_dict = {k: v.unsqueeze(0) for k, v in alpha_dict.items()} # (20,) -> (1, 20)

		if a_cfg_scale != 1.0 or r_cfg_scale != 1.0 or e_cfg_scale != 1.0 or s_cfg_scale != 1.0 or ref_cfg_scale != 1.0 or i_cfg_scale != 1.0:
			null_wa = torch.zeros_like(wa)
			null_we = torch.zeros_like(we)
			null_wr = torch.zeros_like(wr)
			null_intensity_scores = torch.zeros_like(intensity_scores)

			if self.num_ref_frames > 0:
				null_ref_x = torch.zeros_like(ref_x)
			else:
				null_ref_x = None

			alpha_dict_cat = {}
			if False:
				# logger.info("use_audio_condition and num_stat_cond == 0")
				audio_cat 	= torch.cat([wa, null_wa], dim=0) 			# concat along batch
				ref_cat  	= torch.cat([wr, wr], dim=0)				# concat along batch
				emotion_cat = torch.cat([we, we], dim=0)		# concat along batch
				x 			= torch.cat([x, x], dim=0)					# concat along batch
				if self.num_ref_frames > 0:
					ref_x_cat   = torch.cat([ref_x, ref_x], dim=0)		# concat along batch
				else:
					ref_x_cat = None

				prev_x_cat  = torch.cat([prev_x, prev_x], dim=0)
				prev_wa_cat = torch.cat([prev_wa, prev_wa], dim=0)

				model_output = self.forward(t, x, audio_cat, ref_cat, emotion_cat, alpha_dict_cat, prev_x_cat, prev_wa_cat, ref_x=ref_x_cat, train=False)
				all_cond, uncond = torch.chunk(model_output, chunks=2, dim=0)

				# Classifier-free vector field (cfv) incremental manner
				return uncond + a_cfg_scale * (all_cond - uncond)
			elif False:
				# logger.info("use_audio_condition and num_stat_cond == 0")
				audio_cat 	= torch.cat([wa, null_wa, wa], dim=0) 			# concat along batch
				ref_cat  	= torch.cat([wr, wr, wr], dim=0)				# concat along batch
				emotion_cat = torch.cat([we, we, we], dim=0)		# concat along batch
				x 			= torch.cat([x, x, x], dim=0)					# concat along batch
				if self.num_ref_frames > 0:
					ref_x_cat   = torch.cat([ref_x, null_ref_x, null_ref_x], dim=0)		# concat along batch
				else:
					ref_x_cat = None

				prev_x_cat  = torch.cat([prev_x, prev_x, prev_x], dim=0)
				prev_wa_cat = torch.cat([prev_wa, prev_wa, prev_wa], dim=0)

				model_output = self.forward(t, x, audio_cat, ref_cat, emotion_cat, alpha_dict_cat, prev_x_cat, prev_wa_cat, ref_x=ref_x_cat, train=False)
				all_cond, uncond, audio_uncond_ref = torch.chunk(model_output, chunks=3, dim=0)

				# Classifier-free vector field (cfv) incremental manner
				return uncond + a_cfg_scale * (audio_uncond_ref - uncond) + ref_cfg_scale * (all_cond - audio_uncond_ref)
			
			elif self.num_stat_cond == 0:
				# logger.info("use_audio_condition and num_stat_cond == 0")
				# audio_cat 	= torch.cat([wa, null_wa, wa], dim=0) 			# concat along batch
				# ref_cat  	= torch.cat([wr, wr, wr], dim=0)				# concat along batch
				# emotion_cat = torch.cat([we, null_we, null_we], dim=0)		# concat along batch
				# x 			= torch.cat([x, x, x], dim=0)					# concat along batch


				audio_cat 	= torch.cat([wa, null_wa, wa, wa], dim=0) 			# concat along batch
				ref_cat  	= torch.cat([wr, wr, wr, wr], dim=0)				# concat along batch
				emotion_cat = torch.cat([we, null_we, null_we, we], dim=0)		# concat along batch
				x 			= torch.cat([x, x, x, x], dim=0)					# concat along batch
				intensity_scores_cat = torch.cat([intensity_scores, null_intensity_scores, null_intensity_scores, intensity_scores], dim=0)	# concat along batch

				if self.num_ref_frames > 0:
					ref_x_cat   = torch.cat([ref_x, ref_x, ref_x, ref_x], dim=0)		# concat along batch
				else:
					ref_x_cat = None

				prev_x_cat  = torch.cat([prev_x, prev_x, prev_x, prev_x], dim=0)
				prev_wa_cat = torch.cat([prev_wa, prev_wa, prev_wa, prev_wa], dim=0)
				prev_intensity_scores_cat = torch.cat([
					prev_intensity_scores, 
					torch.zeros_like(prev_intensity_scores), 
					torch.zeros_like(prev_intensity_scores), 
					torch.zeros_like(prev_intensity_scores), 
				], dim=0) # (4, B, 1)

				model_output, _ = self.forward(t, x, audio_cat, ref_cat, emotion_cat, alpha_dict_cat, prev_x_cat, prev_wa_cat, ref_x=ref_x_cat, train=False, intensity_scores=intensity_scores_cat, prev_intensity_scores=prev_intensity_scores_cat)
				all_cond, uncond, audio_only, audio_emotion_no_intensity = torch.chunk(
					model_output, 
					chunks=4, 
					dim=0
				)

				# Classifier-free vector field (cfv) incremental manner
				return uncond \
					+ a_cfg_scale * (audio_only - uncond) \
					+ e_cfg_scale * (audio_emotion_no_intensity - audio_only) \
					+ i_cfg_scale * (all_cond - audio_emotion_no_intensity)
			
			else:
				# logger.info("use_audio_condition and num_stat_cond > 0")
				audio_cat 	= torch.cat([wa, null_wa, wa, wa, wa], dim=0) 			# concat along batch
				ref_cat  	= torch.cat([wr, wr, wr, wr, wr], dim=0)				# concat along batch
				emotion_cat = torch.cat([we, null_we, null_we, we, we], dim=0)		# concat along batch
				x 			= torch.cat([x, x, x, x, x], dim=0)					# concat along batch
				ref_x_cat   = torch.cat([ref_x, ref_x, ref_x, ref_x, ref_x], dim=0)	# concat along batch
				intensity_scores_cat = torch.cat([intensity_scores, null_intensity_scores, null_intensity_scores, null_intensity_scores, intensity_scores], dim=0)	# concat along batch

				prev_x_cat  = torch.cat([prev_x, prev_x, prev_x, prev_x], dim=0)
				prev_wa_cat = torch.cat([prev_wa, prev_wa, prev_wa, prev_wa], dim=0)
				prev_intensity_scores_cat = torch.cat([
					prev_intensity_scores, 
					torch.zeros_like(prev_intensity_scores), 
					torch.zeros_like(prev_intensity_scores), 
					torch.zeros_like(prev_intensity_scores), 
					prev_intensity_scores
				], dim=0) # (5, B, 1)

				if self.opt.use_alpha_mean:
					alpha_dict_cat['alpha_mean'] = torch.stack([self.null_alpha_mean, self.null_alpha_mean, self.null_alpha_mean, self.null_alpha_mean, alpha_dict['alpha_mean']], dim=0) # (5, 1, 20)
				if self.opt.use_alpha_std:
					alpha_dict_cat['alpha_std'] = torch.stack([self.null_alpha_std, self.null_alpha_std, self.null_alpha_std, self.null_alpha_std, alpha_dict['alpha_std']], dim=0) # (5, 1, 20)
				if self.opt.use_speed_mean:
					alpha_dict_cat['speed_mean'] = torch.stack([self.null_speed_mean, self.null_speed_mean, self.null_speed_mean, self.null_speed_mean, alpha_dict['speed_mean']], dim=0) # (5, 1, 20)
				if self.opt.use_speed_std:
					alpha_dict_cat['speed_std'] = torch.stack([self.null_speed_std, self.null_speed_std, self.null_speed_std, self.null_speed_std, alpha_dict['speed_std']], dim=0) # (5, 1, 20)
				if self.opt.use_accel_mean:
					alpha_dict_cat['accel_mean'] = torch.stack([self.null_accel_mean, self.null_accel_mean, self.null_accel_mean, self.null_accel_mean, alpha_dict['accel_mean']], dim=0) # (5, 1, 20)
				if self.opt.use_accel_std:
					alpha_dict_cat['accel_std'] = torch.stack([self.null_accel_std, self.null_accel_std, self.null_accel_std, self.null_accel_std, alpha_dict['accel_std']], dim=0) # (5, 1, 20)

				model_output, _ = self.forward(t, x, audio_cat, ref_cat, emotion_cat, alpha_dict_cat, prev_x_cat, prev_wa_cat, ref_x=ref_x_cat, train=False, intensity_scores=intensity_scores_cat, prev_intensity_scores=prev_intensity_scores_cat)
				all_cond, uncond, audio_only, audio_emotion_no_intensity, stat_cond = torch.chunk(model_output, chunks=5, dim=0)

				# Classifier-free vector field (cfv) incremental manner
				return uncond \
					+ a_cfg_scale * (audio_only - uncond) \
					+ e_cfg_scale * (audio_emotion_no_intensity - audio_only) \
					+ i_cfg_scale * (all_cond - audio_emotion_no_intensity) \
					+ (1 - t) * s_cfg_scale * (stat_cond - all_cond)
		else:
			logger.info("no cfg")
			return self.forward(t, x, wa, wr, we, alpha_dict, prev_x, prev_wa, ref_x=ref_x, train = False, intensity_scores=intensity_scores, prev_intensity_scores=prev_intensity_scores)
