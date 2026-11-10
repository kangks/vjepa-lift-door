# DESIGN (Track B): fresh-seed, twin-free replication of pixel vs V-JEPA 2.1 patch-kNN on the lift-door scene, plus a cycle-alignment arm

Written 2026-11-08 (UTC), AFTER the pilot (which only measured timings and checked the pipeline against the
seed-71 reference; pilot clips are excluded from every analysis) and BEFORE any analysis clip is rendered.
Its sha256 is in `DESIGN.sha256`; `score.py` refuses to score if the hash does not match. Nothing is tuned
after TEST is seen; every arm and every failed bar is reported. All data are synthetic Blender renders of ONE
procedural scene and ONE camera; nothing here speaks to real CCTV.

## Question

Q1 (primary). On fresh, independently seeded renders with no twin and no shared background of any TEST clip
in the bank, is the phase-aligned grey patch-kNN detector (pixel_patchKNN) non-inferior to frozen V-JEPA 2.1
dense patch-kNN (D_patchKNN) in fault-event recall, at an operating point fixed on independent healthy CAL
footage with a false-alarm budget of 1 alert-episode per camera-hour?

Q2 (alignment arm). Does pixel's standing relative to D depend on the bank being phase-aligned with the
deployed footage? Bank cycles are time-shifted (0.25, 0.5, 1.0 s) or door-speed-scaled (x0.95, x1.05 cycle
period) relative to CAL/TEST.

Q3 (comparison with the 6-clip benchmark). Fresh-seed pooled per-sample AUROC for D vs pixel (earlier,
seed 71, n = 1 per scenario: 0.811 vs 0.796).

## Data (generator, seed families, splits)

- Generator: `gen/lift_scene_jitter.py`, a copy of `workshop/lift-door/lift_scene/lift_scene.py`
  (sha256 of the original recorded in `inputs_sha256.txt`) with jitter arguments added. With all jitter = 0
  it reproduces the original ground truth exactly (checked on the Mac for anomaly_desync_ball_crossing and
  healthy_ball_reopen, seed 71: frame labels and event intervals identical, and identical to the shipped dataset).
- Scenarios: the 11 benchmark scenarios (6 healthy, 5 anomaly). The stress scenario is skipped.
- Seed family f (28 families, ids 1..28): its own leaf layout (`--seed 1000+f`, the "background") and tree-wind
  time offset U(0, 20) s. Per clip (family x scenario, independent draws, sha256-seeded from
  master seed 20261109): door cycle period 8 s x (1 + U(-0.03, 0.03)) (R6 speed +-3%); door start offset
  U(-0.05, 0.05) s (R6 start +-50 ms); right-leaf timing lag U(-0.02, 0.02) s (R6 interleaf +-20 ms; healthy,
  not labelled); desync onset shift U(-0.25, 0.25) s (cycle units); interior-flicker burst shift U(-0.5, 0.5) s;
  exterior-flicker shift U(-0.3, 0.3) s; ball start shift U(-0.4, 0.4) s (U(-0.15, 0.15) s for the safe-reopen
  ball, which is tied to the door reopen); ball lane U(-0.2, 0.2) m; balloon start shift U(-0.4, 0.4) s and
  vertical offset U(-0.08, 0.08); Cycles sampling seed and sensor-noise seed per clip. Fault severities are
  unchanged (desync 0.18 m peak lag, 1 nominal second per cycle; cabin flicker bursts 1.2 s, two per clip).
- Splits by family, random permutation (seed fixed in `make_jobs.py`): BANK 6 families, CAL 6, TEST 16.
  - BANK: the 5 bank scenario types of the frozen suite (clean, tree_shadow, ball_crossing, balloon_partial,
    external_flicker) -> 30 clips, 60 door cycles. healthy_ball_reopen is never in the bank (as in the suite).
  - CAL: the 6 healthy scenarios -> 36 clips (0.16 camera-hours).
  - TEST: all 11 scenarios -> 176 clips (96 healthy, 80 fault clips, 160 fault events), 16 families per scenario
    (each healthy scenario additionally has 6 CAL families, and each bank scenario 6 BANK families).
  - No TEST family, leaf layout or clip is in BANK or CAL (asserted in `score.py`); no twin of any TEST clip exists
    outside TEST.
- Rendering (decision from the pilot; see ATTEMPTS.md): Blender 5.2.3 LTS Linux (sha256-verified tarball),
  Cycles on the L4 GPU (OptiX, CUDA fallback), 16 samples (as `render_dataset_mac.sh`), default denoiser
  (OpenImageDenoise) run on the GPU, persistent data on,
  1280x720 camera, but only (a) the frames the V-JEPA pipeline uses: frame 1 + 6k (t = k/4 s, the frames
  ffmpeg `fps=4` selects from 24 fps) and (b) the pixels it uses: render border = the centre 720x720 crop
  (`prepare_vjepa_frames.sh` default), then lanczos to 384x384 RGB (same filter). 72 samples (18 s) are rendered;
  CAL/TEST use samples 0..63 (the 16 s protocol); the extra 2 s are only for bank time shifts.
- Sensor/codec (R6 `media.post` recipe): + N(0, 2) per pixel in 8-bit units (sigma 2/255), clip, libx264
  CRF 23 yuv420p -g 32 -bf 0 at 4 fps, decode to RGB24. Detectors see only the decoded frames.
- Deviation from the original seed-71 pipeline (declared): 24 fps CRF 19 encode -> 4 fps sampling is replaced by
  direct 4 fps rendering of the cropped region and the R6 noise + CRF 23 path; the V-JEPA input contract is
  unchanged (64 RGB 384x384 frames at 4 fps, centre crop, lanczos). Pilot validation numbers are in ATTEMPTS.md.

## Embedding

Frozen V-JEPA 2.1 ViT-L EMA encoder (`shared/vitl_ema_encoder_only.pt`, sha256 checked on every instance
against COMMON_BRIEF), loaded with the frozen `encoder.py` (`build_encoder('vitl', ckpt, img_size=384,
num_frames=16)`), 4 non-overlapping 16-frame windows -> [32, 24, 24, 1024] tokens, stored float16 exactly like the
suite token cache. Only deviation from `lift_vjepa_suite.tokens()`: device = CUDA (the frozen function would pick
CPU on EC2). Bank clips are also embedded from sample offsets 1, 2 and 4 (alignment arm).

## Detectors (frozen code, imported read-only with python -B)

`lift_vjepa_suite.py` (sha256 in `inputs_sha256.txt`): `map_for`, `grey_patches`, `unit`. Detectors:
D_patchKNN, G_pooled (V-JEPA tokens), pixel_patchKNN, illum_pixel_patchKNN (grey patch means). Cycle bank
as in suite v2: each 16 s clip = two 8 s halves (16 token steps); the bank = all 60 BANK half-cycles.
- Envelope tau(t, block) = max over leave-one-FAMILY-out bank maps (bank half-cycle vs the half-cycles of the
  other 5 bank families) + 1e-4 (suite v2 recipe, LOO unit widened from clip to family because a family shares a
  background).
- Token score s(t) = max over blocks of map / tau; persistent score p(t) = max(min(s(t-1), s(t)), min(s(t), s(t+1)))
  (equivalent to the v2 ">= 2 consecutive token steps" rule, finding-1 step 0b); per sample = token value repeated x2.

## Threshold rule (CAL only)

Alert at sample k iff p(k) > lambda. lambda per detector (and per bank variant) = the lowest value such that the
number of CAL alert-episodes (maximal runs of alert samples) <= floor(1 per camera-hour x CAL hours). With
0.16 CAL hours the allowance is 0 episodes, i.e. lambda = max p over CAL. Primary CAL set = all 6 healthy
scenarios (CAL6). Sensitivity: CAL5 (without healthy_ball_reopen, mirroring the suite where reopen was a
held-out healthy disturbance). No TEST data are used to set anything.

## Metrics and estimands

- Fault events: each door_desync and interior_light_flicker interval in TEST (2 per fault clip; 160 total).
  An event is detected if any alert sample lies in [start - 0.5 s, end + 0.5 s) (suite rule).
- False-alert episode: an alert episode with no sample within 0.5 s of a real fault (suite rule, episode-level),
  on any TEST clip; rate per TEST camera-hour (176 x 16 s = 0.782 h).
- PRIMARY: event recall at the CAL6 lambda, bank aligned (A0). Estimand: recall(pixel_patchKNN) - recall(D_patchKNN).
  Non-inferiority margin -0.05 (declared before data): pixel is "non-inferior" iff the lower end of the 95%
  bootstrap interval of the difference is > -0.05. A non-significant difference is NOT called equivalence.
- Secondaries (Holm across all of them, alpha 0.05, two-sided bootstrap p):
  S1 recall G_pooled - D; S2 recall illum_pixel - D; S3 pooled6 per-sample AUROC (score s) pixel - D;
  S4 TEST false-alert episodes per hour pixel - D; AL1..AL5 recall pixel - D under each misaligned bank
  (phase 0.25 s, 0.5 s, 1.0 s; speed x0.95, x1.05), each with lambda recalibrated on CAL against that bank.
  Non-inferiority status at -0.05 is also reported for every recall contrast.
- Descriptive (no test): per-detector recall, FA/h, per-scenario hits and FA episodes, median onset delay,
  CAL5 sensitivity, event-level AUROC (event peak p vs healthy-clip max p), pooled6 AUROC of s and p,
  pooled AUROC over all TEST clips, and the drop from A0 in each alignment variant.
- Q3: pooled6 = TEST fault clips + TEST healthy_ball_reopen, positives = samples within +-0.5 s of a fault,
  negatives = other observable samples, score s (finding-1 definition), reported next to 0.811 vs 0.796.

## Inference

Family-level cluster bootstrap, B = 2000, seed 20261109: CAL families and TEST families resampled with
replacement jointly in each draw, lambda recomputed from the resampled CAL; BANK fixed (conditional on the bank).
Percentile 95% intervals. Intervals from 16 TEST families are indicative of family-to-family variation in this
generator only.

## Stop rules and money

Track cap US$15; stop launching at US$12 (80%). Up to 4 g6.xlarge workers for render/embed (each self-terminating,
wall-clock cap), then one CPU instance for scoring. If the projected cost would exceed the cap, the number of TEST
families is reduced first (to no fewer than 8), then CAL, and this is written down. A worker failure is retried at
most once; clips that cannot be produced are listed and the analysis runs only if every planned clip exists
(otherwise STOP and report). Any deviation is appended to this folder's ATTEMPTS.md, never edited into this file.
