#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${OUTPUT_ROOT:-$ROOT/lift_dataset}"
FPS="${FPS:-24}"; DURATION="${DURATION:-16}"
WIDTH="${WIDTH:-1280}"; HEIGHT="${HEIGHT:-720}"
SAMPLES="${SAMPLES:-16}"; SEED="${SEED:-71}"
TREE_SHADOW="${TREE_SHADOW:-0.85}"; FLICKER="${FLICKER:-0.55}"
RENDER_DEVICE="${RENDER_DEVICE:-auto}"; PREPARE_VJEPA="${PREPARE_VJEPA:-1}"
KEEP_FRAMES="${KEEP_FRAMES:-0}"
# SCENARIOS=(
#  healthy_clean healthy_tree_shadow healthy_ball_crossing healthy_ball_reopen
#  healthy_balloon_partial healthy_external_flicker
#  anomaly_door_desync anomaly_interior_light_flicker anomaly_desync_tree_shadow
#  anomaly_desync_ball_crossing anomaly_interior_flicker_balloon_partial
#  stress_balloon_full_occlusion
# )
SCENARIOS=(
 healthy_clean
)
if [[ "${1:-}" == '--list' ]]; then printf '%s\n' "${SCENARIOS[@]}"; exit 0; fi
if [[ "${1:-}" == '--help' ]]; then
 echo 'Usage: bash render_dataset_mac.sh [all|scenario-name|--list|--help]'
 echo 'Environment: FPS DURATION WIDTH HEIGHT SAMPLES SEED TREE_SHADOW FLICKER RENDER_DEVICE PREPARE_VJEPA KEEP_FRAMES OUTPUT_ROOT'
 exit 0
fi
if command -v blender >/dev/null 2>&1; then BLENDER="$(command -v blender)"
elif [[ -x /Applications/Blender.app/Contents/MacOS/Blender ]]; then BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
else echo 'Install Blender: brew install --cask blender' >&2; exit 1; fi
command -v ffmpeg >/dev/null || { echo 'Install FFmpeg: brew install ffmpeg' >&2; exit 1; }
command -v python3 >/dev/null || { echo 'python3 required' >&2; exit 1; }
mkdir -p "$DEST"
CHOSEN="${1:-all}"
if [[ "$CHOSEN" == all ]]; then TODO=("${SCENARIOS[@]}")
else
  found=0
  for s in "${SCENARIOS[@]}"; do [[ "$s" == "$CHOSEN" ]] && found=1; done
  [[ "$found" == 1 ]] || { echo "Unknown scenario $CHOSEN; use --list" >&2; exit 2; }
  TODO=("$CHOSEN")
fi
for case_name in "${TODO[@]}"; do
  if [[ "$case_name" == healthy_* ]]; then LABEL=healthy
  elif [[ "$case_name" == anomaly_* ]]; then LABEL=anomaly
  else LABEL=stress; fi
  name="${case_name}.mp4"
  d="$DEST/$LABEL/$case_name"
  mkdir -p "$d/frames"
  echo "=== Rendering $case_name [$LABEL] ==="
  "$BLENDER" -b -t 0 --python "$ROOT/lift_scene.py" -- \
    --scenario "$case_name" --video-file "$name" --output "$d/frames" \
    --fps "$FPS" --seconds "$DURATION" --width "$WIDTH" --height "$HEIGHT" \
    --seed "$SEED" --tree-shadow "$TREE_SHADOW" --flicker "$FLICKER"
  SCENE="$d/frames/lift_scene.blend"
  [[ -s "$SCENE" ]] || { echo "Scene generation failed $SCENE" >&2; exit 1; }
  # Metal acceleration and CPU fallback. Render full video using one Blender process.
  render() {
    "$BLENDER" -b "$SCENE" -t 0 --python "$ROOT/configure_render.py" -- \
      --device "$1" --samples "$SAMPLES" --render
  }
  if [[ "$RENDER_DEVICE" == cpu ]]; then render cpu
  elif ! render "$RENDER_DEVICE"; then
    echo 'Metal/auto rendering failed, retrying on CPU...' >&2
    rm -f "$d"/frames/frame_*.png
    render cpu
  fi
  ffmpeg -hide_banner -loglevel error -y -framerate "$FPS" -start_number 1 \
    -i "$d/frames/frame_%04d.png" -c:v libx264 -pix_fmt yuv420p \
    -crf 19 -movflags +faststart "$d/$name"
  if [[ "$PREPARE_VJEPA" == 1 ]]; then
    bash "$ROOT/prepare_vjepa_frames.sh" "$d/$name" "$d/vjepa_frames"
  fi
  if [[ "$KEEP_FRAMES" != 1 ]]; then rm -f "$d"/frames/frame_*.png; fi
  echo "Created $d/$name"
done
python3 "$ROOT/build_dataset_index.py" --root "$DEST"
echo "Index: $DEST/dataset_index.csv and $DEST/dataset_index.json"
