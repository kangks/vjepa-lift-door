# Lift-door V-JEPA 2.1 NN benchmark – Mac Apple Silicon

This package generates **12 independent labelled videos** from a consistent scene and camera. Eleven are healthy/anomaly benchmark candidates; the twelfth is an explicit **stress/unknown-visibility** test. Door panels disappear behind wall pockets when opened. The scene includes procedural street-side tree shadows, exterior lighting, rubber ball bouncing, safe obstruction-triggered reopening, and balloon occlusion.

## Install

```bash
brew install --cask blender
brew install ffmpeg
```

Blender 4.x and 5.x are intended targets. This bundle was syntax-checked but **not rendered on a Mac**. Validate one clip first.

## Run a smoke test

```bash
cd lift_scene
bash render_dataset_mac.sh --list
WIDTH=640 HEIGHT=360 FPS=12 SAMPLES=8 PREPARE_VJEPA=0 RENDER_DEVICE=cpu \
  bash render_dataset_mac.sh healthy_clean
```

## Generate all benchmark videos

```bash
WIDTH=1280 HEIGHT=720 FPS=24 DURATION=16 SAMPLES=16 RENDER_DEVICE=metal \
  PREPARE_VJEPA=1 bash render_dataset_mac.sh all
```

**Rendering 12 x 384 frames with Cycles may be computationally expensive**. Start with smoke tests and switch to batch overnight as appropriate. Set `KEEP_FRAMES=1` to retain the original rendered PNGs (very large). Default removes them after encoding. `OUTPUT_ROOT=/path/...` chooses a destination. To add a new scenario, extend `SCENARIOS` in the scene and shell scripts.

## Scenario table

| Scenario | Video label | Injected fault | Nuisance / exception |
|---|---|---|---|
| healthy_clean | healthy | none | none |
| healthy_tree_shadow | healthy | none | swaying leaves |
| healthy_ball_crossing | healthy | none | ball bouncing across pavement |
| healthy_ball_reopen | healthy | none | ball entering / safety reopen |
| healthy_balloon_partial | healthy | none | near-camera balloon, partial view loss |
| healthy_external_flicker | healthy | none | exterior streetlamp flicker |
| anomaly_door_desync | anomaly | panel desynchronization | none |
| anomaly_interior_light_flicker | anomaly | lift cabin light flicker | none |
| anomaly_desync_tree_shadow | anomaly | panel desynchronization | tree shadows |
| anomaly_desync_ball_crossing | anomaly | panel desynchronization | bouncing ball |
| anomaly_interior_flicker_balloon_partial | anomaly | cabin light flicker | partial balloon occlusion |
| stress_balloon_full_occlusion | stress | none | full view obstruction; do not treat as confidently healthy |

## Output

```
lift_dataset/
  healthy/healthy_clean/healthy_clean.mp4
  healthy/healthy_clean/ground_truth.json
  healthy/healthy_clean/ground_truth_frames.csv
  healthy/healthy_clean/vjepa_frames/clip_0000.png ...
  healthy/healthy_clean/vjepa_frames/vjepa_sample_manifest.json
  anomaly/anomaly_door_desync/...
  stress/stress_balloon_full_occlusion/...
  dataset_index.csv
  dataset_index.json
```

* `video_health` is `healthy`, `anomaly`, or `stress`.
* `anomaly` is true for **either** the door-sync fault or interior-light fault; `mechanical_anomaly` is true only for door desync.
* The tree shadows, external light flicker, ball crossings, obstruction-triggered **safe** reopening, and partial camera occlusion are **healthy disturbances**, not equipment faults.
* `observable=false` is used during full occlusion; exclude those evaluation windows from the equipment fault confusion matrix.
* Each `ground_truth.json` contains frame labels and on/off intervals. These are injected procedural labels, **not observed failures**.
* V-JEPA 2.1 crop is a *candidate input*, not the final normalized model tensor. The checkpoint's exact preprocessing must be used.
* The crop defaults to centered 720x720 -> 384x384. Check it retains both door edges, especially with balloon shots. Adjust `CROP_X`, `CROP_Y`, and `CROP_SIZE` if needed.
* A 16-second clip at 4 FPS yields 64 frames. 64 frames is an experimental choice, **not mandatory for every checkpoint**. At 4 FPS short flashes may be missed; consider 8 FPS / shorter windows for electrical flicker.
* To avoid train-test leakage, use independent seeds and differing animated nuisance trajectories for your *healthy memory bank* versus held-out tests. A single clip per scenario is a functional smoke test, not a statistically meaningful benchmark.
* A pooled V-JEPA embedding may focus on camera-level novelty. Compare it to localized dense tokens around the doors and cabin lights, plus conventional CV baselines.

## Window evaluator

```bash
python3 evaluate_windows.py \
  --manifest lift_dataset/anomaly/anomaly_door_desync/vjepa_frames/vjepa_sample_manifest.json \
  --scores detector_window_scores.csv --threshold 0.7
```

Input CSV needs `window_start_sample,window_end_sample_exclusive,anomaly_score`. Choose the threshold using an independent healthy validation set; do **not** fit it on anomaly test labels.
