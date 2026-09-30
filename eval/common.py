"""Shared input validation and event matching for GLARE Appendix D."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


REACTIONS = ("smiling", "laughing", "frowning", "surprised", "nodding", "head_shaking")
MATCH_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5


class EvaluationError(ValueError):
    """Invalid input or an evaluation that cannot be carried out reliably."""


@dataclass(frozen=True)
class Event:
    reaction: str
    start: int
    end: int

    def __post_init__(self):
        if self.reaction not in REACTIONS or self.start < 0 or self.end <= self.start:
            raise EvaluationError(f"Invalid half-open event: {self}")


@dataclass(frozen=True)
class Annotation:
    frame_count: int
    events: tuple[Event, ...]


@dataclass(frozen=True)
class Match:
    ground_truth: Event
    prediction: Event
    tiou: float


def index_files(root: Path, extensions: set[str]) -> dict[str, Path]:
    """Pair by relative path without extension; never silently intersect sets."""
    root = Path(root)
    if not root.is_dir():
        raise EvaluationError(f"Input directory does not exist: {root}")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            key = path.relative_to(root).with_suffix("").as_posix()
            if key in files:
                raise EvaluationError(f"Duplicate sample key {key!r}: {files[key]} and {path}")
            files[key] = path
    if not files:
        raise EvaluationError(f"No {sorted(extensions)} files found in {root}")
    return files


def require_same_keys(reference: dict, other: dict, description: str) -> None:
    missing = sorted(reference.keys() - other.keys())
    extra = sorted(other.keys() - reference.keys())
    if missing or extra:
        raise EvaluationError(
            f"{description}: sample sets differ; missing={missing[:10]}, extra={extra[:10]}"
        )


def read_score_rows(path: Path) -> Iterator[tuple[int, str | None, float]]:
    """Read dense, zero-based frame rows; validate before any thresholding."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        expected = {"frame_index", *REACTIONS}
        if header is None or len(header) != len(expected) or set(header) != expected:
            raise EvaluationError(f"{path}: expected exactly these columns: {sorted(expected)}")
        columns = {name: header.index(name) for name in header}
        count = 0
        for row in reader:
            where = f"{path}: CSV line {reader.line_num}"
            if len(row) != len(header):
                raise EvaluationError(f"{where}: expected {len(header)} fields, got {len(row)}")
            frame_text = row[columns["frame_index"]].strip()
            if not frame_text.isascii() or not frame_text.isdecimal():
                raise EvaluationError(f"{where}: frame_index must be a nonnegative integer")
            frame = int(frame_text)
            if frame != count:
                raise EvaluationError(f"{where}: expected frame_index {count}, got {frame}")
            nonzero = []
            for reaction in REACTIONS:
                try:
                    value = float(row[columns[reaction]])
                except ValueError as exc:
                    raise EvaluationError(f"{where}: invalid {reaction} confidence") from exc
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise EvaluationError(f"{where}: {reaction} confidence must be finite in [0, 1]")
                if value > 0:
                    nonzero.append((reaction, value))
            if len(nonzero) > 1:
                raise EvaluationError(f"{where}: more than one nonzero reaction confidence")
            reaction, confidence = nonzero[0] if nonzero else (None, 0.0)
            yield frame, reaction, confidence
            count += 1
        if count == 0:
            raise EvaluationError(f"{path}: CSV contains no frames")


def temporal_iou(prediction: Event, ground_truth: Event) -> float:
    intersection = max(0, min(prediction.end, ground_truth.end) - max(prediction.start, ground_truth.start))
    enclosing_length = max(prediction.end, ground_truth.end) - min(prediction.start, ground_truth.start)
    return intersection / enclosing_length


def read_annotation(path: Path) -> Annotation:
    """Convert already-postprocessed detector CSVs into half-open events.

    USER-CONFIRMED INPUT CONTRACT: these CSVs have ALREADY undergone detector
    smoothing, short-event filtering, and gap merging. Do NOT repeat any of
    those operations here. Keep only confidence STRICTLY GREATER THAN 0.5;
    join consecutive frames of the same class into [first_frame, last_frame+1).
    A confidence of exactly 0.5 is neutral. Even one neutral frame splits events.
    """
    events = []
    active_class, start, frame_count = None, 0, 0
    for frame, reaction, confidence in read_score_rows(path):
        label = reaction if confidence > CONFIDENCE_THRESHOLD else None
        if label != active_class:
            if active_class is not None:
                events.append(Event(active_class, start, frame))
            active_class, start = label, frame
        frame_count = frame + 1
    if active_class is not None:
        events.append(Event(active_class, start, frame_count))
    return Annotation(frame_count, tuple(events))


def iter_annotation_pairs(gt_dir: Path, gen_dir: Path) -> Iterator[tuple[str, Annotation, Annotation]]:
    ground_truth = index_files(gt_dir, {".csv"})
    generated = index_files(gen_dir, {".csv"})
    require_same_keys(ground_truth, generated, "GT/generated CSVs")
    for key in sorted(ground_truth):
        gt, gen = read_annotation(ground_truth[key]), read_annotation(generated[key])
        if gt.frame_count != gen.frame_count:
            raise EvaluationError(
                f"{key}: paired CSV lengths differ: GT={gt.frame_count}, generated={gen.frame_count}"
            )
        yield key, gt, gen


def match_events(ground_truth: tuple[Event, ...], predictions: tuple[Event, ...]) -> list[Match]:
    """Appendix D.2: class-aware, descending-tIoU greedy one-to-one matching.

    This function is called separately for each clip, so events never match
    across videos. Equal tIoU values are ordered by GT then prediction time.
    """
    matches = []
    for reaction in REACTIONS:
        real = sorted((e for e in ground_truth if e.reaction == reaction), key=lambda e: (e.start, e.end))
        generated = sorted((e for e in predictions if e.reaction == reaction), key=lambda e: (e.start, e.end))
        candidates = []
        for gi, gt in enumerate(real):
            for pi, pred in enumerate(generated):
                if pred.start >= gt.end:
                    break
                if pred.end <= gt.start:
                    continue
                iou = temporal_iou(pred, gt)
                if iou >= MATCH_THRESHOLD:
                    candidates.append((-iou, gi, pi))
        used_real, used_generated = set(), set()
        for negative_iou, gi, pi in sorted(candidates):
            if gi not in used_real and pi not in used_generated:
                used_real.add(gi)
                used_generated.add(pi)
                matches.append(Match(real[gi], generated[pi], -negative_iou))
    return matches


def make_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--gt-csv-dir", type=Path, required=True)
    parser.add_argument("--gen-csv-dir", type=Path, required=True)
    return parser


def emit_result(name: str, value: float | None) -> None:
    """stdout is one JSON object, diagnostics belong to stderr."""
    if value is not None and not math.isfinite(value):
        raise EvaluationError(f"{name}: non-finite result")
    print(json.dumps({name: value}, allow_nan=False))


def run_cli(main) -> None:
    try:
        main()
    except (EvaluationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
