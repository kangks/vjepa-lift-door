# Track C design: robustness controls for finding 2 (V-JEPA dense patch-kNN floods under mild capture changes)

Written 2026-11-08 ~05:15Z, BEFORE any Track C encoding or scoring. sha256 stored in `DESIGN.sha256`; `run_c.py`
refuses to run if the hash differs. Nothing below is tuned after data are seen; every arm is reported.
Compute: EC2 only (us-east-1, profile ml-sandbox, g6.xlarge, NVIDIA L4), cap US$8, tags project=vjepa-arxiv-followup,
track=c. Synthetic Blender renders only (the existing 6-clip benchmark, seed 71). No new renders, no real footage,
no downloads of datasets, no Bedrock, no Cosmos.

## Question

Finding 2 (`finding2_brittleness_8nov/`): with a clean bank and the frozen max-LOO+1e-4 envelope, frozen V-JEPA 2.1
dense patch-kNN floods (FA up to 300/300) under sensor noise sigma >= 1 and JPEG q <= 90, while pixel patch means
do not. Its evidence had one noise seed (shared by all test clips), one bank noise realisation per clip, no CIs, and
no recipe controls. Track C asks, per perturbation family and level, for each of D_patchKNN, pixel_patchKNN and
illum_pixel_patchKNN:
1. Is the flood stable across noise seeds (with test noise independent of bank noise)?
2. Is it a THRESHOLD problem (a matched healthy bank and/or a softer quantile threshold restores the operating
   point) or a FEATURE problem (it does not restore, or ranking degrades even with matched calibration)?
3. Do simple pre-filters (spatial blur before encoding; causal token temporal averaging) remove it?
4. Is it V-JEPA-specific (do the pixel arms show it at the same levels)?
5. Mechanism (label-free): how far do tokens move under each perturbation compared with healthy margins?

## Frozen pieces (read-only; sha256 checked at run time)

- `lift_vjepa_suite.py` 45e0e712e0759e5bfb858e16ba11d24433419311dd8b51bfef8d82de951208cb (v1: load_frames,
  sample_labels, unit, blocks, grey_patches, map_for, evaluate, FROZEN_DESIGN bank)
- `lift_vjepa_suite_v2.py` d7f04e56981443527c3e45b7ea49dd5f566308e697551ab4972cf103fc7706fc (cycles, persist n=2)
- `results_v2/summary.json` bb876097593b912d5d6b8ad21ed54c28580e54a78a5d05a0d00105a41c75f716
- `scripts/vjepa/encoder.py` 87ccc263e7764ad42f2c9ee1f0483d4b614cdfd6bdcaa2cf36a758fe5dc53356 (+ reference/vjepa2
  at git 204698b45b3712590f06245fbfba32d3be539812)
- R8 `run_robustness.py` 1fdd865b5183f5c06f28df6b36a9f04c6a4dc6f9513e7fcab2a99e59dfcbaa01 (perturb, tau_for, score,
  DETS, TESTS, PERTS); R8 `results.json` f50aa919...; finding-2 `results.json` fdaded7c... (comparison only).
- Checkpoint `shared/vitl_ema_encoder_only.pt` sha256 06248f29136a5d26c30e466d717d8845c25341e220f500468c0c162a3317cf20.
- Bank = v1.FROZEN_DESIGN healthy_bank (5 clips, 10 cycles). Tests = R8 TESTS (5 anomaly clips + healthy_ball_reopen;
  10 fault events). Frames = the existing `vjepa_frames/clip_*.png` (64 x 384 x 384 RGB) via v1.load_frames.

## Encoding on CUDA (declared deviation from the Mac pipeline)

v1.tokens only knows MPS/CPU, so `run_c.py` calls the same frozen `encoder.py` functions in the same way
(build_encoder("vitl", ckpt, img_size=384, num_frames=16); dense_tokens per 16-frame window; concatenate to
[32,24,24,1024]; store fp16; reload as float32 and unit-normalise) with device="cuda",
torch.backends.cuda.matmul.allow_tf32 = False and torch.backends.cudnn.allow_tf32 = False.
PRIMARY pipeline: ALL tokens (clean bank, clean tests, perturbed) are CUDA-encoded, so device drift never enters a
contrast. The MPS clean cache (results_v1/token_cache, 11 files) is used only for gates and as a device-drift readout.
The R8 1.1 GiB perturbed MPS cache is NOT reused (device consistency); everything is recomputed.

## Gates (STOP and report if any fails)

- G0 (numpy, MPS tokens): scoring the 3 detectors x 6 tests from the MPS clean cache reproduces
  results_v2/summary.json exactly (FA, alert times, event hits) -- same rule as finding 2 G0.
- G0f: sha256(frames.tobytes())[:16] of each of the 11 clips equals the MPS cache filename key.
- G1 (device): CUDA clean tokens vs MPS cache: median per-token cosine >= 0.999 for every clip (else STOP: broken
  encoder setup). Min/median cosine and the CUDA-baseline decisions vs MPS-baseline decisions are REPORTED, not gated.
- G2 (determinism): healthy_ball_reopen clean re-encoded twice on CUDA: byte-identical fp16, or min cosine >= 0.99999
  with identical decisions (reported which).
- Env pins (dry run): python, torch, CUDA, driver, GPU name, numpy, PIL, timm, einops recorded. Disk floor 6 GiB.
- CUDA baseline (reference for all rules below): FA0_d, faults0_d, AUROC0_d = clean CUDA bank + clean CUDA tests,
  frozen envelope, persistence; AUROC as in finding 2 E1.

## Perturbations (applied to the 64 x 384 x 384 uint8 frames; R8 arithmetic: float32, rint, clip 0-255, uint8)

Test clips (6), raw pipeline:
- Deterministic ladders: gain {0.8, 0.9, 0.95, 1.05, 1.1, 1.25}; shift {1, 2, 4, 8} px right and down
  (np.pad edge, as finding 2 E3); JPEG quality {90, 70, 50, 30} (PIL defaults). 14 levels.
- R8 P5 exact (noise sigma 6, rng(71) fresh per clip, i.e. the same pattern on every clip): kept for linkage.
- Noise ladder sigma {1, 2, 3, 6} x 5 TEST seeds s in {1001..1005}: clip i (index in TESTS order) uses
  np.random.default_rng([s, i]).normal(0, sigma, frames.shape) (float32); the same standard-normal draw is
  scaled across sigma (common random numbers). Test noise is independent across clips and seeds.
- P1-P5 of R8 are covered: gain 0.8, gain 1.25, shift 8, JPEG 30 are ladder rungs; P5 by the R8-exact arm.

Matched banks (bank clips perturbed, healthy only; never test clips):
- Noise sigma {1,2,3,6}: realisations r in {0,1,2,3}; bank clip j (FROZEN bank order) uses
  np.random.default_rng([5000 + r, j]) -- independent of all test seeds. Bank sizes: R1 = r0 only (10 cycles),
  R4 = r0..r3 (40 cycles).
- All 14 deterministic levels (gain, shift, JPEG): bank clips perturbed identically, R1 only (deterministic, so a
  larger bank is not possible).
- LOO for a bank cycle excludes every cycle (all realisations) of its own source clip; for R1 this is exactly
  rr.tau_for (asserted on the clean bank).

## Readout controls

- Threshold rules, all calibrated ONLY on bank LOO maps (never on test), each + 1e-4:
  ENV (frozen; per cycle-time and 2x2 block max), BLKQ99 / BLKQ100 (per block, 99th percentile / max over all cycle
  times and LOO cycles; numpy percentile, linear), GLBQ9999 / GLBQ100 (one scalar: 99.99th percentile / max over all
  LOO map values; GLBQ100 = the v2 'global' policy). Primary = ENV; the others are secondary.
- Blur pre-filter (pipelines blur1, blur2): frames -> perturbation -> spatial Gaussian blur sigma_b in {1.0, 2.0} px
  (separable, radius ceil(3 sigma_b), np.pad reflect, per channel, float32 -> rint/clip uint8) -> encoder and pixel
  features. Bank and tests both blurred; clean blurred bank, frozen-recipe ENV tau. Conditions: clean, noise
  sigma {1,3,6} x 5 test seeds, JPEG {90,50,30}.
- Token temporal averaging (no new encodes; raw tokens): causal moving average over w in {2,4} token steps
  (mean of u_{max(0,t-w+1)..t}, then unit-normalise; pixel/illum features averaged the same way without
  normalisation), applied identically to bank and tests; clean bank, ENV tau. Conditions: clean, noise
  sigma {1,2,3,6} x 5 seeds, JPEG {90,70,50,30}. Note the causal lag can cost fault hits; reported.

## Metrics

Per detector x bank-set x rule x condition: faults found (of 10), summed FA (ceiling 300; v1.evaluate, persistence),
pooled AUROC (finding-2 E1 definition: per-step score = max over blocks of map / ENV tau of THAT bank-set, no
persistence; positives = near-fault samples, negatives = other observable samples; Mann-Whitney, ties 0.5).
Noise conditions: FA and faults are averaged over the 5 seeds (sum over 6 clips per seed); AUROC is pooled over
seeds x clips.
95% bootstrap CIs (percentile, B = 1000, numpy rng 20261108): resample the 6 clips with replacement and, where seeds
exist, the 5 seeds with replacement independently; statistic recomputed on the resample (summed FA scaled to 6
clips). INDICATIVE ONLY: 6 clips, 5 seeds, one scene, correlated samples.

Mechanism (label-free, iv): for every test condition with an encode, per clip and pooled: mean and median token
cosine distance 1 - cos(u_clean, u_pert) at matched (t,h,w) (u_clean = the same pipeline's clean CUDA tokens), and
ratios to (a) the median healthy nearest-neighbour distance (LOO over the clean bank, per patch before blocks, as
finding 2 E4) and (b) median ENV tau; fraction of blocks with blocks(1-cos) > ENV tau. Pixel/illum: the same on
|g_pert - g_clean|. Device drift: the same statistic for CUDA vs MPS clean tokens.

## Pre-declared decision rules

FA_lim_d = max(1.5 x FA0_d, FA0_d + 10) for the pipeline's own baseline. A (detector, level) is DEGRADED if its
clean-bank ENV mean FA > FA_lim_d. For each DEGRADED (detector, family in {noise, jpeg, gain, shift}, level):
- THRESHOLD problem: the matched bank (R4 for noise, R1 otherwise) with ENV gives FA upper CI <= FA_lim_d AND mean
  faults >= faults0_d - 1 AND AUROC >= AUROC0_d - 0.05. "THRESHOLD (secondary rule)" if only a secondary rule on the
  matched bank meets the same three conditions.
- FEATURE problem: for EVERY matched-bank rule (ENV, BLKQ99, BLKQ100, GLBQ9999, GLBQ100) the FA lower CI > FA_lim_d,
  OR the matched-bank AUROC upper CI < AUROC0_d - 0.05.
- otherwise MIXED / UNRESOLVED.
- Supporting mechanism label (not part of the decision): "drift exceeds healthy margin" if pooled median token
  drift / median healthy-NN distance >= 1.
Gain/shift/JPEG have no seeds; their CIs are over clips only and their decisions use R1 matched banks.
Pre-filter rescue: a pre-filter RESCUES a DEGRADED raw (detector, level) if, under that pre-filter, FA upper CI <=
FA_lim of the pre-filtered (clean) baseline AND the pre-filtered perturbed condition's mean faults >= faults0_d
(raw) - 1.
V-JEPA-specific at a level: D_patchKNN DEGRADED while neither pixel arm is DEGRADED at that level.
Breaking point per ladder and detector: first rung where clean-bank ENV mean FA > 150 (as finding 2 E3).
Tail-sampling readout: R1 vs R4 matched-bank FA for noise (descriptive).

## Stages, cost and stop rules

Pilot (one instance): gates, 11 clean CUDA encodes, determinism re-encode, ONE perturbation (noise sigma 1, seed
1001, 6 test clips) encoded and scored, plus 5 bank encodes (noise sigma 1, r0) so that a score-stage SMOKE run
(1 seed, sigma 1 only; a code-path test, its numbers are labelled smoke and are not results) exercises every scoring
path; seconds per clip measured; projected cost reported before scaling.
Full run (one instance, self-terminating; second instance only if the pilot projects > 2 h): 609 encodes
(11 clean + 84 deterministic + 6 R8-P5 + 120 noise + 150 matched-bank [80 noise + 70 deterministic] + 2 x 119 blur),
then scoring. The full run re-encodes the clean clips on a fresh instance (cross-instance determinism readout).
Hard limits: stage timeouts; whole-instance shutdown watchdog; progress to S3 every <= 5 min; stop launching at
80% of US$8. No ladder is extended after seeing data. Every level, arm and failed gate is reported. If the full run
cannot finish within the cap, the remaining arms are listed as not run.

## Known limits (stated up front)

Synthetic renders, one scene, one render seed, 6 test clips, 10 faults, perturbations on pre-processed 384 x 384
frames (not a camera pipeline), i.i.d. Gaussian noise and still-image JPEG only. CIs are indicative. Non-significant
is not equivalence. CUDA numerics differ from the Mac MPS numbers of finding 2; Track C numbers are compared within
Track C, and against finding 2 only descriptively.
