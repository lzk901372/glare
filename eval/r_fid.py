"""R-FID: one FID over the full dataset's real/generated reaction-frame pools."""

from __future__ import annotations

import contextlib
import math
import os
import sys
from pathlib import Path

import numpy as np

from common import (
    EvaluationError, emit_result, index_files, iter_annotation_pairs,
    make_parser, require_same_keys, run_cli,
)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
FEATURE_DIM = 2048


class FeatureMoments:
    """Stable float64 pooled mean/covariance without storing every feature.

    M2 is the sum of centered outer products. Batch merging gives the same
    sample covariance (ddof=1) as numpy.cov(all_features, rowvar=False).
    """

    def __init__(self, dimension: int):
        self.count = 0
        self.mean = np.zeros(dimension, dtype=np.float64)
        self.m2 = np.zeros((dimension, dimension), dtype=np.float64)

    def update(self, features):
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.size or not np.isfinite(values).all():
            raise EvaluationError("Invalid or non-finite Inception feature batch")
        n = len(values)
        if not n:
            return
        batch_mean = values.mean(axis=0)
        centered = values - batch_mean
        delta = batch_mean - self.mean
        combined = self.count + n
        self.m2 += centered.T @ centered
        self.m2 += np.outer(delta, delta) * (self.count * n / combined)
        self.mean += delta * (n / combined)
        self.count = combined

    def statistics(self):
        if self.count < 2:
            raise EvaluationError("FID covariance requires at least two frames per pool")
        covariance = self.m2 / (self.count - 1)
        return self.mean, (covariance + covariance.T) * 0.5


def frechet_distance(real: FeatureMoments, generated: FeatureMoments) -> float:
    from scipy.linalg import sqrtm

    real_mean, real_cov = real.statistics()
    gen_mean, gen_cov = generated.statistics()
    # Eq. (21), using the same Schur matrix-square-root operation as pytorch-fid.
    # Call sqrtm WITHOUT the removed disp argument, so SciPy >=1.18 also works.
    # The published pytorch-fid 0.3.0 distance helper still passes disp=False.
    with contextlib.redirect_stdout(sys.stderr):
        covariance_root = sqrtm(real_cov @ gen_cov)
    if not np.isfinite(covariance_root).all():
        # Do not silently change the covariance or regularize the metric.
        raise EvaluationError("FID covariance square root is non-finite")
    if np.iscomplexobj(covariance_root):
        # Consistent with pytorch-fid's tolerance for numerical imaginary parts.
        if not np.allclose(np.diag(covariance_root).imag, 0, atol=1e-3):
            raise EvaluationError("FID covariance square root has a significant imaginary component")
        covariance_root = covariance_root.real
    mean_distance = float(np.sum((real_mean - gen_mean) ** 2))
    value = float(mean_distance + np.trace(real_cov) + np.trace(gen_cov) - 2 * np.trace(covariance_root))
    if not math.isfinite(value):
        raise EvaluationError("FID numerical computation produced a non-finite value")
    # A mathematically nonnegative distance can be slightly negative after
    # scipy.sqrtm on rank-deficient covariance products. Clip ONLY roundoff;
    # substantial negative values indicate failure and must not be hidden.
    scale = float(mean_distance + np.trace(real_cov) + np.trace(gen_cov))
    tolerance = 1e-6 * max(1.0, scale)
    if value < -tolerance:
        raise EvaluationError(f"FID numerical computation produced a negative value: {value}")
    if value < 0:
        print(f"R-FID: clipped numerical roundoff {value:.6g} to zero.", file=sys.stderr)
        value = 0.0
    return value


def collect_statistics(records, side, model, device, batch_size, expected_fps):
    """Decode in order, count every frame, and select each reaction frame once.

    Real frames use REAL events; generated frames use GENERATED events.
    Unmatched events are included. There is no temporal matching, equal-count
    sampling, class balancing, frame subsampling, extra face crop, or per-clip FID.
    """
    import cv2

    if model is not None:
        import torch
    moments = FeatureMoments(FEATURE_DIM) if model is not None else None
    selected_count = 0
    batch = []

    def flush():
        if not batch:
            return
        # OpenCV BGR was converted to RGB below. Original decoded pixels are
        # converted to [0,1]. pytorch-fid itself does the 299x299 bilinear resize
        # and [-1,1] normalization, exactly once. Do not use ImageNet mean/std.
        tensor = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2)
        tensor = tensor.to(device=device, dtype=torch.float32).div_(255.0)
        with torch.inference_mode():
            features = model(tensor)[0].flatten(1).cpu().numpy()
        moments.update(features)
        batch.clear()

    for number, record in enumerate(records, 1):
        key, annotation, path = record[0], record[side][0], record[side][1]
        if number == 1 or number % 100 == 0 or number == len(records):
            print(f"R-FID {'GT' if side == 1 else 'generated'}: {number}/{len(records)} ({key})", file=sys.stderr)
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise EvaluationError(f"Cannot open video: {path}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or not math.isclose(fps, expected_fps, rel_tol=0, abs_tol=1e-3):
                raise EvaluationError(f"{path}: expected {expected_fps} FPS, got {fps}")
            # Decode to EOF, including neutral tails: metadata alone may be wrong.
            frame_index, event_index = 0, 0
            events = annotation.events
            while True:
                success, bgr = capture.read()
                if not success:
                    break
                if frame_index >= annotation.frame_count:
                    raise EvaluationError(f"{path}: video has more frames than its CSV")
                while event_index < len(events) and events[event_index].end <= frame_index:
                    event_index += 1
                active = event_index < len(events) and events[event_index].start <= frame_index
                if active:
                    selected_count += 1
                    if model is not None:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        if batch and batch[0].shape != rgb.shape:
                            flush()
                        batch.append(rgb)
                        if len(batch) >= batch_size:
                            flush()
                frame_index += 1
            if frame_index != annotation.frame_count:
                raise EvaluationError(
                    f"{path}: decoded {frame_index} frames but CSV contains {annotation.frame_count}; "
                    "video may be truncated or unreadable"
                )
        finally:
            capture.release()
    flush()
    return moments, selected_count


def main():
    parser = make_parser(__doc__)
    parser.add_argument("--gt-video-dir", type=Path, required=True)
    parser.add_argument("--gen-video-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fps", type=float, default=25.0, help="Expected constant FPS of ALL videos (paper: 25)")
    parser.add_argument("--torch-home", type=Path, default=Path(__file__).resolve().parent / ".cache" / "torch")
    args = parser.parse_args()
    if args.batch_size < 1 or not math.isfinite(args.fps) or args.fps <= 0:
        raise EvaluationError("batch-size and fps must be positive")

    gt_videos = index_files(args.gt_video_dir, VIDEO_EXTENSIONS)
    gen_videos = index_files(args.gen_video_dir, VIDEO_EXTENSIONS)
    records = []
    for key, gt, gen in iter_annotation_pairs(args.gt_csv_dir, args.gen_csv_dir):
        records.append((key, (gt, gt_videos.get(key)), (gen, gen_videos.get(key))))
    csv_keys = {record[0]: None for record in records}
    require_same_keys(csv_keys, gt_videos, "CSV/GT videos")
    require_same_keys(csv_keys, gen_videos, "CSV/generated videos")
    expected_counts = [sum(e.end - e.start for record in records for e in record[side][0].events) for side in (1, 2)]

    try:
        import cv2  # noqa: F401 - check the decoding dependency before evaluation
    except ImportError as exc:
        raise EvaluationError("R-FID requires opencv-python; install eval/requirements.txt") from exc
    model, device = None, None
    if min(expected_counts) >= 2:
        try:
            import torch
            from pytorch_fid.inception import InceptionV3
        except ImportError as exc:
            raise EvaluationError("R-FID requires torch, torchvision and pytorch-fid; install eval/requirements.txt") from exc
        os.environ["TORCH_HOME"] = str(args.torch_home.resolve())
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise EvaluationError("CUDA requested but unavailable; explicitly use --device cpu if desired")
        print("R-FID: loading FID Inception-v3 (2048-d pool3); missing weights are downloaded to --torch-home.", file=sys.stderr)
        with contextlib.redirect_stdout(sys.stderr):
            model = InceptionV3(
                [InceptionV3.BLOCK_INDEX_BY_DIM[FEATURE_DIM]],
                resize_input=True, normalize_input=True, requires_grad=False, use_fid_inception=True,
            ).to(device).eval()

    real, real_count = collect_statistics(records, 1, model, device, args.batch_size, args.fps)
    generated, gen_count = collect_statistics(records, 2, model, device, args.batch_size, args.fps)
    if [real_count, gen_count] != expected_counts:
        raise EvaluationError("Decoded reaction-frame counts do not match CSV intervals")
    print(f"R-FID reaction frames: GT={real_count}, generated={gen_count}", file=sys.stderr)
    if min(real_count, gen_count) < 2:
        print("R-FID is null: at least two reaction frames are required in EACH dataset-level pool.", file=sys.stderr)
        emit_result("R-FID", None)
        return
    emit_result("R-FID", frechet_distance(real, generated))


if __name__ == "__main__":
    run_cli(main)
