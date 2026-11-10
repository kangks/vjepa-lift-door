# Track A DESIGN: event-level, fixed-false-alarm-budget re-analysis of R6 and the 6-clip lift-door benchmark

Written 2026-11-08 ~05:10 UTC, BEFORE any scoring in this folder. sha256 of this file goes into `DESIGN.sha256`;
`run_track_a.py` refuses to score if the hash does not match. Nothing is tuned after the run; every arm, every
failed gate and every attempted run is reported. All data are synthetic Blender renders; no real footage is tested.

Scope: re-analysis of EXISTING per-clip maps and cached tokens only. No rendering, no encoder runs, no model
downloads. Compute on one EC2 CPU instance (us-east-1, ml-sandbox, tags project=vjepa-arxiv-followup, track=a),
self-terminating. Track cap US$2 (stop launching at US$1.60).

## Question

Under one event-level protocol with a threshold set only from healthy CAL footage at a fixed false-alarm budget,
is the model-free grey-patch nearest-neighbour detector non-inferior to frozen V-JEPA 2.1 dense nearest-neighbour
detection of door faults, on (1) the pre-registered lift-lobby R6 study and (2) the 6-clip lift-door v2 benchmark
(original bank and twin-out bank)?

## Inputs (read-only; hashes recomputed and logged by the run)

R6 (`workshop/lift-lobby-water-leak/output/ec2/lift-lobby-r6-g3c2/r6/primary/`):
- `cycles.json` sha256 b8a782f09d4142a9eca8cdcd55a0917905cd07b9a0cc3eb7ef51506cd7bf4685 (configs, splits, R6 decisions, G scores, taus)
- `thresholds.json` sha256 416790b9f4dd70efd3b1a9bc69dcb38aae1e4f758cb4f18038d337d48f5aeccb
- `<id>_{D,pixel,illum_pixel,flow_residual}.npz` (`blocks` [16,23,23]; 245 clips x 4 arms; sha256 of the sorted
  per-file sha256 list = 00b5576395cccfa0e6445a745dbda33e1dab0252d6a07f4d6a5fcb9e25d537c7)
- Interval code (frozen, imported unmodified): `scripts/r6/motion.py` sha256 8456a6630cd15541cd8868a05281c190849bb36c1fba4021a9844267e5804854,
  `scripts/sync_screen/motion.py` sha256 3c158e93963182504dd33db4ba8675af55226902653a31fa1510c05d910d9c54
  (its unused `scripts.sync_screen.common` import is satisfied by a stub module; `geometry()` intervals do not use it).
- Map-producing code (provenance only, not re-run): `scripts/r6/cv_maps.py` 2c79d466..., `scripts/r6/readouts.py` a6174b2f...,
  `scripts/sync_screen/readouts.py` b7347254..., `scripts/r6/score.py` d456c095...; G3 identity source_sha 1ff15670... (L4, fp32, fps 8).

Lift-door (6-clip v2 benchmark, seed 71):
- Frozen suite: `.claude/worktrees/vjepa-talk-draft/workshop/lift-door/eval_suite/lift_vjepa_suite.py`
  sha256 45e0e712e0759e5bfb858e16ba11d24433419311dd8b51bfef8d82de951208cb and `lift_vjepa_suite_v2.py`
  sha256 d7f04e56981443527c3e45b7ea49dd5f566308e697551ab4972cf103fc7706fc; `results_v2/summary.json` bb876097...
- Token caches `results_v1/token_cache/<name>_<sha256(frames)[:16]>.npy` for the 11 non-stress clips (abort on any miss).
- Frames and labels `workshop/lift-door/lift_scene/lift_dataset/*/*/{vjepa_frames,ground_truth.json}` (stress clip excluded).
- Finding 1 (`research/arxiv_gap/finding1_pixel_matches_vjepa_8nov/`): `run_finding1.py` sha256 e5e349f2... imported read-only
  for `fd_map`, `auroc`, `avg_precision`; `results.json` sha256 b29acd54... used only for the reproduction gate.

## Detectors

R6 (maps exist for these only; names are R6-native):
- `D`: V-JEPA 2.1 dense token cosine distance to the single coherent nearest BANK clip (minimax over clips of the
  2x2 block-mean map max), i.e. a D_clipNN-type readout. R6 has NO per-patch kNN (D_patchKNN) map.
- `pixel`: grey absolute residual vs the coherent nearest BANK clip (pixel_clipNN-type), per-tubelet patch means.
- `illum_pixel`: same after per-frame least-squares gain/offset fit.
- `flow_residual`: within-tubelet Farneback flow vector-difference magnitude vs the coherent nearest BANK clip.
- `G`: whole-clip pooled V-JEPA vector distance (clip-level scalar only; no time axis, so no persistence and no onset).
- Missing in R6 (reported, not reconstructed): D_patchKNN, pixel_patchKNN, FD_median, per-token G.

Lift-door (maps from the frozen `v1.map_for`, v2 cycle bank of 5 bank clips x 2 cycles, same procedure as Finding 1):
- `D_patchKNN` (primary V-JEPA arm), `D_clipNN` (bridge to R6 `D`), `G_pooled`, `pixel_patchKNN` (primary pixel arm),
  `illum_pixel_patchKNN`, `FD_median` (Finding 1 `fd_map`).
- `pixel_clipNN` (new bridge arm, defined here before data, mirrors R6 `pixel`): candidate maps `v1.blocks(|test - b|)` on
  `v1.grey_patches(frames, False)` for each bank cycle b; choose the cycle with the smallest map max (first on ties).
- No flow-residual map exists for the lift-door set; not computed.

## Per-token scores and persistence (one protocol)

- Token step: R6 0.25 s (16 steps per 4 s clip, fps 8 recipe); lift-door 0.5 s (32 steps per 16 s clip).
- PRIMARY score ("global"): s(t) = max over the 23x23 blocks of the raw map at token step t.
- SECONDARY score ("envelope"): s(t) = max over blocks of map(t,b) / tau_env(t,b); lift-door tau_env = v2 LOO envelope
  (max over LOO bank maps at the same cycle time and block + 1e-4); R6 tau_env = max over the stratum's CAL maps at the
  same t and block + 1e-4. Reported descriptively only.
- Persistence: an alert needs >= k consecutive token steps with s(t) > lambda. PRIMARY k = 2 token steps (the frozen
  detector rule; 1.0 s lift-door, 0.5 s R6). SECONDARY (R6 only, descriptive): k = 4 steps (1.0 s).
  Persistent score q(t) = max over the length-k windows containing t of the window minimum; alerting steps are
  exactly those with q(t) > lambda (equivalent to `v2.persist` for k = 2; the run checks this).
- Alert episode: a maximal run of alerting token steps within one clip (lift-door test clips: the 32-step clip; CAL units:
  one cycle / one clip, runs do not cross units).

## Threshold rule (CAL only, fixed false-alarm budget)

- Budget B = 1 alert episode per camera-hour. CAL duration H (hours); allowed episodes A = floor(B x H).
- lambda = the smallest value such that the number of CAL alert episodes at lambda is <= A (computed exactly from the
  sorted CAL q values; alert is strict `>`).
- R6 CAL = the 25 CAL clips (close 20 clips = 80 s, reopen 5 clips = 20 s), scored against the full BANK by R6;
  lambda per mode stratum (close, reopen), as R6 froze taus per stratum. For `G`: lambda = max CAL G score of the stratum.
- Lift-door CAL = the 10 LOO bank cycles (each bank cycle vs the cycles of the other 4 bank clips; 80 s); twin-out CAL =
  the 8 LOO cycles of the 4 remaining bank clips (each vs the other 3 clips; 64 s). Deviation from the DOSSIER (v)
  outline: lift-door CAL is not disjoint from the bank (LOO); there is no independent healthy lift-door footage.
- Stated before data: every CAL here is far shorter than 1 h, so A = 0 and the rule reduces to "zero CAL alert episodes".
  TEST healthy footage (R6 160 s; lift-door 96 s) cannot verify a 1/h rate; observed TEST episode rates are reported raw.

## Events and labels

- R6 events: one per TEST defect clip (`clean_defect` L1-L4: 25/25/15/15 = 80 independent seed families; `defect_reflection`
  L1-L3: 15/15/10 = 40 matched variants). Event interval = `motion.geometry(c, 0, c['faulty_side'])['intervals']['defect_active']`
  (first brake -> delayed closure). Token step i spans [0.25 i, 0.25 (i+1)) s.
- Lift-door events: the 10 equipment-fault intervals (door_desync, interior_light_flicker; 2 per fault clip in 5 fault clips).
  Scored clips: 5 fault clips + healthy_ball_reopen (stress clip excluded), as Finding 1.
- Event window = [start - 0.5 s, end + 0.5 s]. An event is detected if any alerting token step overlaps its window
  (lift-door: via frozen `v1.evaluate` on alert samples = token alerts repeated x2).
- False alert episode: an episode with no overlap with any fault window (lift-door: on observable samples; all scored samples
  are observable).
- Onset delay of a detected event: (end time of the k-th step of the first alerting run overlapping the window) - event start.
  Retrospective, excludes processing latency, may be negative within the -0.5 s tolerance.

## Primary estimand and decision (pre-declared)

- PRIMARY: R6 clean-defect event recall (80 events) at the CAL-budget lambda (global score, k = 2), and the paired
  difference Delta = recall(pixel) - recall(D). Family-level cluster bootstrap, 2000 resamples, numpy default_rng(20261109),
  stratified by family class (defect level 1-4 / healthy / reopen) as in R6 `family_samples`; all matched variants of a
  family move together; CAL and lambda are fixed (not resampled). 95% percentile interval.
- Non-inferiority margin -0.05 (declared before data). Readout: lower 95% bound > -0.05 -> "pixel non-inferior to D at
  margin 0.05 (R6)"; also lower bound > 0 -> "pixel superior". Upper bound < -0.05 -> "pixel inferior". Otherwise "inconclusive".
  A non-significant difference is never called equivalence.
- Same estimand on the lift-door set (10 events; pixel_patchKNN - D_patchKNN, and bridge pixel_clipNN - D_clipNN) for the
  original and twin-out banks, with a clip-level cluster bootstrap over the 6 scored clips (2000, same seed; resamples with no
  event skipped and counted). n = 1 clip per scenario: labelled INDICATIVE ONLY, no decision bar.

## Secondary (R6, pixel - D, Holm-adjusted across these 5 contrasts)

- S1 event-level AUROC: positives = per-event peak of q over the event window (80 clean events); negatives = q(t) of every
  token step of the 40 healthy TEST clips (640 steps). Mann-Whitney with mid-ranks.
- S2 event-level AP on the same units (step-wise, sklearn-equivalent).
- S3 onset delay: median over events detected by both arms of (delay_pixel - delay_D).
- S4 hard-negative false alerts: mean false-alert episodes per clip over reflection_only (40), exposure_only (10) and
  reopen TEST (10) clips.
- S5 defect_reflection event recall (40 matched variants).
- Two-sided bootstrap p = min(1, 2 min((#Delta* <= 0 + 1)/(B+1), (#Delta* >= 0 + 1)/(B+1))) (R6 `sign_p` form), then Holm.
- Descriptive (no p-values): every other arm and arm difference vs D, per-level recall with Clopper-Pearson 95%, healthy TEST
  episode counts and raw per-hour rates, false alerts on defect clips, the envelope and k = 4 sensitivities, and for the
  lift-door set: event recall, event AUROC/AP (positives = 10 event peaks; negatives = q of non-window observable token steps
  of the 6 scored clips), onset, false-alert episodes split into "overlaps a disturbance interval +-0.5 s" vs other.

## Gates (STOP the affected dataset on failure; report)

- R0: for every R6 row and map arm, blocks.max() equals the stored clip score within 1e-6 relative, and
  (blocks.max() > R6 tau) equals the stored R6 alert. G alert from stored score > tau equals stored alert.
- R1: recomputed defect_active / dwell intervals reproduce the stored D `dwell_timing.onset_error_s`
  (first support step start - dwell start) within 1e-6 s on every row where it is defined.
- L0: lift-door per-token scores for D_patchKNN, D_clipNN, G_pooled, pixel_patchKNN, illum_pixel_patchKNN, FD_median
  reproduce Finding 1 `per_sample_scores` (`s` envelope and `s_global`) within max(2e-6 absolute, 1e-5 relative), for both
  banks and all 6 clips (Finding 1 stored 6 decimals; float32 maps). [Amended 05:11 UTC before any scoring: the first
  hashed version (930fd991...) said 2e-6 absolute only.]
- L1: v2.persist(s > lambda) equals (q > lambda) for k = 2 on every scored series (checked at the primary lambda).

## Stop rule and runtime

- Gate failure -> no analysis for that dataset. Analysis wall cap 45 min (`timeout`), instance hard cap 60 min, then
  `shutdown -h now` (instance-initiated shutdown = terminate). One analysis run; code bugs found during the run may be fixed
  and re-run only if no TEST result was printed before the fix, and each attempt is logged in ATTEMPTS.md.

## Outputs (this folder)

`run_track_a.py`, `bootstrap.sh`, `launch.sh`, `results.json`, `tables.md`, `run.log`, `RESULTS.md`, `ATTEMPTS.md`, `STATUS.md`,
`OUTPUTS.sha256`.

## Amendment 1 (2026-11-08 ~05:25 UTC, after run a1, BEFORE any lift-door protocol result was computed)

Run a1: R6 gates R0/R1 PASS and R6 analysis completed (unchanged by this amendment). Lift-door gate L0 FAILED
(16 of 72 series outside tolerance; max abs error 0.135, mostly illum_pixel_patchKNN and twin-out D arms). The run stopped
before any lift-door threshold, event or false-alert result was computed. Likely cause: float32 einsum/BLAS differences
between the Mac (where Finding 1 ran) and the Linux instance, amplified by envelope denominators near 1e-4 on
near-deterministic renders. This is not established; the a2 run logs per-series diagnostics.

Gate L0 is replaced by L0' for run a2 (lift-door only; R6 is re-run unchanged and must reproduce a1 exactly):
- L0'(a), the bar: decision-level reproduction of frozen `results_v2/summary.json` (the gate Finding 1 Step 0 used) for
  D_patchKNN, D_clipNN, G_pooled, pixel_patchKNN, illum_pixel_patchKNN x {envelope, global} x the 6 scored clips on the
  original bank, with alerts = v2.persist(map > tau) as in v2: event hits, false_alert_samples and alert_times_s identical.
  Failure -> STOP lift-door.
- L0'(b), diagnostic only (no bar): max abs and relative difference vs Finding 1 `per_sample_scores` per detector, bank,
  clip and score type (envelope/global), reported in results.json.
- Lift-door results from a2 are labelled "after gate amendment 1" in RESULTS.md.
