#!/usr/bin/env bash
set -euo pipefail
# Deterministic 4 FPS x 16 seconds = 64 RGB 384x384 PNG frames.
# This is a suggested V-JEPA 2.1 sampling recipe, NOT a universal model requirement.
# The exact checkpoint loader must do its required normalization and tensor packing.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="${1:-$SCRIPT_DIR/lift_video_output/lift_jitter_cctv.mp4}"
OUTPUT="${2:-$SCRIPT_DIR/lift_video_output/vjepa_frames}"
SAMPLE_FPS="${SAMPLE_FPS:-4}"
MODEL_SIZE="${MODEL_SIZE:-384}"
CROP_SIZE="${CROP_SIZE:-720}"
CROP_X="${CROP_X:--1}"
CROP_Y="${CROP_Y:--1}"
if ! command -v ffmpeg >/dev/null 2>&1; then echo 'ffmpeg required' >&2; exit 1; fi
if [[ ! -f "$INPUT" ]]; then echo "No video found: $INPUT" >&2; exit 1; fi
mkdir -p "$OUTPUT"
rm -f "$OUTPUT"/clip_*.png
# Default center square crop; override CROP_X/Y/SIZE to include both door panels.
if [[ "$CROP_X" == -1 ]]; then CROP_X="(iw-$CROP_SIZE)/2"; fi
if [[ "$CROP_Y" == -1 ]]; then CROP_Y="(ih-$CROP_SIZE)/2"; fi
ffmpeg -hide_banner -loglevel error -y -i "$INPUT" \
  -vf "fps=$SAMPLE_FPS,crop=$CROP_SIZE:$CROP_SIZE:$CROP_X:$CROP_Y,scale=$MODEL_SIZE:$MODEL_SIZE:flags=lanczos,format=rgb24" \
  -start_number 0 "$OUTPUT/clip_%04d.png"
echo "V-JEPA candidate input frames: $OUTPUT ($(find "$OUTPUT" -name 'clip_*.png' | wc -l | tr -d ' ') frames)"
echo 'Warning: verify chosen checkpoint frame count, crop, normalization and dense-token output.'

# Align V-JEPA candidate frames with synthetic source ground truth.
GT_FILE="$(dirname "$INPUT")/ground_truth.json"
if [[ -f "$GT_FILE" ]]; then
  python3 "$SCRIPT_DIR/build_vjepa_manifest.py" \
    --ground-truth "$GT_FILE" --frames-dir "$OUTPUT" --sample-fps "$SAMPLE_FPS"
else
  echo "Ground truth not found at $GT_FILE; sample labels skipped." >&2
fi
