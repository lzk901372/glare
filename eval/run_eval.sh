#!/usr/bin/env bash
# Launch all four standalone metrics concurrently; publish only a complete result.
set -euo pipefail

# Git Bash launched directly from PowerShell may lack its own Unix utilities.
if [[ "${OSTYPE:-}" == msys* || "${OSTYPE:-}" == cygwin* ]]; then
    export PATH="/usr/bin:$PATH"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"
GT_CSV=""
GEN_CSV=""
GT_VIDEO=""
GEN_VIDEO=""
DEVICE="cpu"
BATCH_SIZE="32"
FPS="25"
TORCH_HOME_DIR="$SCRIPT_DIR/.cache/torch"
OUTPUT="$SCRIPT_DIR/results.json"

usage() {
    cat <<'HELP'
Usage:
  bash eval/run_eval.sh \
    --gt-csv-dir PATH --gen-csv-dir PATH \
    --gt-video-dir PATH --gen-video-dir PATH [options]

Options:
  --device cpu|cuda:0     PyTorch device (default: cpu)
  --batch-size N          Inception batch size (default: 32)
  --fps NUMBER           Required constant video FPS (default: 25)
  --torch-home PATH      Inception cache (default: eval/.cache/torch)
  --output PATH          Final JSON file (default: eval/results.json)
  --python EXECUTABLE    Python >=3.10 interpreter (default: $PYTHON or python)
  --help                 Show this help

All four metrics run concurrently. stdout is a single JSON object containing
R-F1, R-tIoU, R-ATD (formula x100), and R-FID. Diagnostics go to stderr.
Each input directory must contain exactly the same relative sample stems.
HELP
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --gt-csv-dir|--gen-csv-dir|--gt-video-dir|--gen-video-dir|--device|--batch-size|--fps|--torch-home|--output|--python)
            if [[ $# -lt 2 || -z "$2" ]]; then
                printf 'Missing value for %s\n' "$1" >&2
                exit 2
            fi
            case "$1" in
                --gt-csv-dir) GT_CSV="$2" ;;
                --gen-csv-dir) GEN_CSV="$2" ;;
                --gt-video-dir) GT_VIDEO="$2" ;;
                --gen-video-dir) GEN_VIDEO="$2" ;;
                --device) DEVICE="$2" ;;
                --batch-size) BATCH_SIZE="$2" ;;
                --fps) FPS="$2" ;;
                --torch-home) TORCH_HOME_DIR="$2" ;;
                --output) OUTPUT="$2" ;;
                --python) PYTHON_BIN="$2" ;;
            esac
            shift 2
            ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$GT_CSV" || -z "$GEN_CSV" || -z "$GT_VIDEO" || -z "$GEN_VIDEO" ]]; then
    usage >&2
    exit 2
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/glare-eval.XXXXXXXX")"
PIDS=()
cleanup() {
    # Remove only the four known temporary files; no recursive deletion.
    rm -f -- "$WORK/r_f1.json" "$WORK/r_tiou.json" "$WORK/r_atd.json" "$WORK/r_fid.json"
    rmdir -- "$WORK"
}
stop_children() {
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit 130
}
trap cleanup EXIT
trap stop_children INT TERM

CSV_ARGS=(--gt-csv-dir "$GT_CSV" --gen-csv-dir "$GEN_CSV")
for metric in r_f1 r_tiou r_atd; do
    "$PYTHON_BIN" "$SCRIPT_DIR/$metric.py" "${CSV_ARGS[@]}" > "$WORK/$metric.json" &
    PIDS+=("$!")
done
"$PYTHON_BIN" "$SCRIPT_DIR/r_fid.py" "${CSV_ARGS[@]}" \
    --gt-video-dir "$GT_VIDEO" --gen-video-dir "$GEN_VIDEO" \
    --device "$DEVICE" --batch-size "$BATCH_SIZE" --fps "$FPS" \
    --torch-home "$TORCH_HOME_DIR" > "$WORK/r_fid.json" &
PIDS+=("$!")

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED=1
    fi
done
if [[ "$FAILED" -ne 0 ]]; then
    printf 'Evaluation failed; no combined result was written. Any previous output is unchanged.\n' >&2
    exit 1
fi

"$PYTHON_BIN" - "$WORK" "$OUTPUT" <<'PY'
import json
import math
import os
import sys
import tempfile
from pathlib import Path

work, output = map(Path, sys.argv[1:])
result = {}
for script, name in (("r_f1", "R-F1"), ("r_tiou", "R-tIoU"), ("r_atd", "R-ATD"), ("r_fid", "R-FID")):
    data = json.loads((work / (script + ".json")).read_text(encoding="utf-8"))
    if set(data) != {name}:
        raise ValueError(f"Invalid output from {script}: {data}")
    value = data[name]
    if value is None:
        if name not in {"R-ATD", "R-FID"}:
            raise ValueError(f"{name} cannot be null")
    elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid result from {script}: {value}")
    result[name] = value

payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
output.parent.mkdir(parents=True, exist_ok=True)
temporary = None
try:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, prefix=".glare-eval-", suffix=".json", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    os.replace(temporary, output)
finally:
    if temporary is not None and temporary.exists():
        temporary.unlink()
print(payload, end="")
print(f"Saved evaluation results to {output.resolve()}", file=sys.stderr)
PY
