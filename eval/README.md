# GLARE Reaction Evaluation

Evaluate generated listener videos against paired ground-truth videos using four reaction metrics.

## Scripts

| Script | Metric | What it measures |
| --- | --- | --- |
| `r_f1.py` | R-F1 ↑ | Event detection F1: pool counts across the dataset per class, then average over six classes. Absent classes score 0. |
| `r_tiou.py` | R-tIoU ↑ | Mean temporal IoU over all matched events in the dataset. |
| `r_atd.py` | R-ATD ↓ | Mean asymmetric onset, offset and duration penalties over matched events, normalized by GT duration and **reported ×100**. |
| `r_fid.py` | R-FID ↓ | FID over all reaction frames pooled across videos, using each side's own reaction intervals, including unmatched events. |

Event matching is one-to-one within each video and class, with **tIoU ≥ 0.5**. R-ATD's ×100 scale is an implementation reporting convention.

`common.py` shares validation and matching; `run_eval.sh` runs all four scripts concurrently. The first three metrics use CSVs only; R-FID also uses videos.

## Inputs

Prepare four directories containing the same samples, paired by **relative path without the extension**:

```text
data/
  gt_videos/clip_001.mp4
  gen_videos/clip_001.mp4
  gt_csv/clip_001.csv
  gen_csv/clip_001.csv
```

Paired videos must have aligned frames, matching frame counts and durations, and a common constant frame rate (**25 FPS** by default; configurable with `--fps`).

Each CSV must contain exactly these columns, with one row per video frame, including neutral frames:

```csv
frame_index,smiling,laughing,frowning,surprised,nodding,head_shaking
0,0,0,0,0,0,0
1,0.65,0,0,0,0,0
2,0.75,0,0,0,0,0
3,0,0,0,0,0,0
```

- Frame indices start at **0** and increase consecutively. CSV row counts must match the videos.
- Confidence values are finite numbers in **[0, 1]**, with at most one nonzero class per frame.
- CSVs must **already include detector smoothing, short-event filtering and gap merging**; these operations are not repeated.
- Only scores **>0.5** are retained. Consecutive same-class frames form `[first_frame, last_frame + 1)`; neutral frames split events. The example yields `smiling [1, 3)`.

Missing pairs, extra samples, invalid CSVs or video/CSV alignment errors stop evaluation.

## Run

From the project root, using Python 3.10 or later:

```bash
python -m pip install -r eval/requirements.txt

bash eval/run_eval.sh \
  --gt-video-dir data/gt_videos \
  --gen-video-dir data/gen_videos \
  --gt-csv-dir data/gt_csv \
  --gen-csv-dir data/gen_csv \
  --device cuda:0 \
  --output eval/results.json
```

Use `--device cpu` without a GPU. R-FID uses pytorch-fid's FID-specific Inception-v3 weights, downloaded on first use and cached under `eval/.cache/torch/`.

## Output

The terminal and output JSON contain **one dataset-level value per metric**. Example values below are illustrative:

```json
{
  "R-F1": 0.6,
  "R-tIoU": 0.7,
  "R-ATD": 57.0,
  "R-FID": 15.0
}
```

Higher R-F1 and R-tIoU are better; lower R-ATD and R-FID are better. With no matched events, R-tIoU is `0` and R-ATD is `null`. R-FID is `null` if either reaction-frame pool contains fewer than two frames. Diagnostics go to stderr. If any script fails, no new combined result is written.
