import re
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration


LINE_PATTERN = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*$"
)


class IntensityEncoder(nn.Module):
    """Encode waveform into frame-level intensity scores with Qwen2-Audio."""

    def __init__(self, opt, cuda=True):
        super().__init__()
        self.opt = opt

        self.model_path = getattr(opt, "qwen2audio_path", None)
        if self.model_path is None:
            raise ValueError("`opt.qwen2audio_path` is required for IntensityEncoder.")

        self.default_sampling_rate = int(getattr(opt, "sampling_rate", 16000))
        self.default_fps = float(getattr(opt, "fps", 25.0))
        self.max_new_tokens = int(getattr(opt, "intensity_max_new_tokens", 256))
        self.min_intensity_score = float(getattr(opt, "intensity_min_score", 0.5))

        self.processor = AutoProcessor.from_pretrained(
            self.model_path, 
            local_files_only=True, 
            trust_remote_code=True,
            # output_attentions=False, # 关闭注意力机制
            attn_implementation="sdpa"   # 保留加速
        )
        if cuda and torch.cuda.is_available():
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                # device_map="auto",
                local_files_only=True,
                trust_remote_code=True,
                # output_attentions=False, # 关闭注意力机制
                attn_implementation="sdpa"   # 保留加速
            ).to("cuda")
            self.input_device = "cuda"
        else:
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                local_files_only=True,
                trust_remote_code=True,
            ).to("cpu")
            self.input_device = "cpu"
        self.model.eval()

        self.prompt = self.get_prompt()
        self.prompt_text = self._build_prompt_text(self.prompt)

    def get_prompt(self) -> str:
        return """
You are an expert in speech prosody analysis.

You will be given an audio segment containing one speaker. Detect only the time
intervals where the speaker shows a clear and localized prosodic fluctuation. A
prosodic fluctuation means a noticeable change in vocal delivery, such as increased
pitch, increased loudness, stronger stress, sharper emphasis, faster or slower
speaking rate, unusual rhythm, hesitation, excitement, surprise, or other affective
vocal variation. Focus only on acoustic and prosodic cues, not on the semantic
meaning of the spoken words.

For each detected prosodic event, output its start time, end time, and intensity score.
The intensity score must be between 0 and 1 and should represent the salience of the
prosodic change:
- 0.50-0.60: weak but noticeable;
- 0.60-0.80: clear;
- 0.80-1.00: strong or highly salient.

Detection rules:
- Only output events with intensity score >= 0.50.
- Do not output uncertain or ambiguous events.
- Do not split one continuous prosodic fluctuation into many short intervals.
- Merge neighboring intervals if they are part of the same prosodic event.
- No interval overlapping.
- Prefer concise intervals that cover the main fluctuation rather than long segments
with neutral speech.

Output format:
- Each line must be: start_time, end_time, intensity_score. Results must only have two decimal.
- All numbers must use exactly two decimal places.
- Use seconds as the time unit.
- Do not output any explanation, label, markdown, bullet point, or extra text.
- If no valid event is detected, output exactly:
NULL

Analyze the audio now and output only the formatted results.
        """.strip()

    def _build_prompt_text(self, prompt: str) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": "chunk.wav"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

    def _to_numpy_waveform(self, waveform: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().float().cpu().numpy()
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            if waveform.shape[0] == 1:
                waveform = waveform[0]
            elif waveform.shape[1] == 1:
                waveform = waveform[:, 0]
            else:
                waveform = waveform.mean(axis=0)
        if waveform.ndim != 1:
            raise ValueError(f"Expected 1D waveform, but got shape={waveform.shape}.")
        return waveform

    def normalize_result(self, text: str, max_time: float) -> str:
        raw = text.strip()
        if not raw:
            return "NULL"

        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        normalized_lines = []
        for ln in lines:
            if ln.upper() == "NULL":
                continue
            m = LINE_PATTERN.match(ln)
            if not m:
                continue
            start = float(m.group(1))
            end = float(m.group(2))
            score = float(m.group(3))
            if end <= start:
                continue
            if start < 0.0 or end > max_time + 1e-6:
                continue
            if score < self.min_intensity_score:
                continue
            score = min(max(score, 0.0), 1.0)
            normalized_lines.append(f"{start:.2f}, {end:.2f}, {score:.2f}")

        if not normalized_lines:
            return "NULL"
        return "\n".join(normalized_lines)

    @staticmethod
    def _time_to_start_frame_1_based(start_time: float, fps: float) -> int:
        return int(np.floor(start_time * fps)) + 1

    @staticmethod
    def _time_to_end_frame_1_based(end_time: float, fps: float) -> int:
        return int(np.ceil(end_time * fps))

    def text_to_frame_vector(
        self,
        text: str,
        num_frames: int,
        fps: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        if num_frames <= 0:
            raise ValueError(f"`num_frames` must be positive, got {num_frames}.")

        fps = float(self.default_fps if fps is None else fps)
        scores = np.zeros((num_frames,), dtype=np.float32)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        for line in lines:
            if line.upper() == "NULL":
                continue
            match = LINE_PATTERN.match(line)
            if not match:
                continue
            start_time = float(match.group(1))
            end_time = float(match.group(2))
            intensity = float(match.group(3))
            if end_time <= start_time:
                continue

            start_frame = self._time_to_start_frame_1_based(start_time, fps)
            end_frame = self._time_to_end_frame_1_based(end_time, fps)
            start_idx = max(0, start_frame - 1)
            end_idx = min(num_frames - 1, end_frame - 1)
            if end_idx < start_idx:
                continue
            scores[start_idx : end_idx + 1] = intensity

        out = torch.from_numpy(scores).unsqueeze(-1)
        if device is not None:
            out = out.to(device)
        return out

    @torch.inference_mode()
    def infer_text(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        sampling_rate: Optional[int] = None,
    ) -> str:
        sampling_rate = int(
            self.default_sampling_rate if sampling_rate is None else sampling_rate
        )
        waveform_np = self._to_numpy_waveform(waveform)
        if waveform_np.size == 0:
            return "NULL"

        inputs = self.processor(
            text=[self.prompt_text],
            audio=[waveform_np],
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        ).to(self.input_device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        generated_ids = generated_ids[:, inputs["input_ids"].size(1) :]
        decoded = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        max_time = float(waveform_np.shape[0]) / float(sampling_rate)
        return self.normalize_result(decoded, max_time=max_time)

    @torch.inference_mode()
    def encode_waveform(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        num_frames: int,
        fps: Optional[float] = None,
        sampling_rate: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        text = self.infer_text(waveform=waveform, sampling_rate=sampling_rate)
        return self.text_to_frame_vector(text=text, num_frames=num_frames, fps=fps, device=device)

    def forward(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        num_frames: int,
        fps: Optional[float] = None,
        sampling_rate: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        return self.encode_waveform(
            waveform=waveform,
            num_frames=num_frames,
            fps=fps,
            sampling_rate=sampling_rate,
            device=device,
        )
