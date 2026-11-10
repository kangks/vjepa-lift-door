# Track B results: fresh-seed, twin-free lift-door replication + cycle-alignment arm (8 Nov 2026)

All data are synthetic Blender renders of one procedural scene and one camera. Nothing here is about real CCTV.
Pre-registration: `DESIGN.md` (sha256 `6db864ab...`, hash-gated in `score.py`), plus `AMENDMENT_1_20261108.md`
(two bug fixes: worker argument format, scorer download filter; neither changes the design, and neither touched a
produced clip or a computed result). Full tables: `tables_autogen.md`. Numbers: `results.json` (sha256 `64b4abec...`).
Every run, including the failures: `ATTEMPTS.md`.

## What ran

- 302 clips rendered and embedded on EC2 (g6.xlarge, L4). 28 seed families: BANK 6 (5 bank scenario types,
  30 clips, 60 half-cycles, plus time-shifted and x0.95 / x1.05 speed variants), CAL 6 (36 healthy clips,
  0.16 camera-hours) and TEST 16 (176 clips, all 11 scenarios, **160 fault events**: 96 door-desync, 64 cabin-flicker).
  Jitter as in R6 (cycle period +-3%, start +-50 ms, interleaf +-20 ms); fault and disturbance timing jitter;
  per-family leaf layout and wind phase. Sensor noise sigma 2/255 and CRF 23. No TEST family, background or
  twin appears in BANK or CAL (asserted in the scorer).
- Frozen detectors from `lift_vjepa_suite.py`: D_patchKNN, G_pooled, pixel_patchKNN, illum_pixel_patchKNN. The
  envelope is computed leave-one-family-out on BANK. lambda is set on CAL only (1 alert-episode per camera-hour,
  which allows 0 CAL episodes at 0.16 h). Alerts need >= 2 consecutive token steps.
- Family cluster bootstrap, 2000 draws, CAL and TEST resampled jointly. Holm correction over 9 secondaries.

## Primary result (aligned bank, CAL6 operating point)

| Detector | Event recall (hits / 160) | 95% CI | TEST false-alert episodes (per TEST h) |
|---|---|---|---|
| D_patchKNN (V-JEPA 2.1 dense) | **0.000 (0/160)** | [0.000, 0.000] | 3 (3.8/h), all on healthy_ball_reopen |
| G_pooled (V-JEPA pooled) | 0.000 (0/160) | [0.000, 0.231] | 0 |
| pixel_patchKNN | **0.394 (63/160)** | [0.369, 0.419] | 0 |
| illum_pixel_patchKNN | 0.144 (23/160) | [0.113, 0.181] | 5 (6.4/h) |

The pre-registered primary contrast is recall(pixel) - recall(D) = **+0.394, 95% CI [+0.369, +0.419]**. The lower
bound is above the -0.05 margin, so pixel is **non-inferior** (and better) by the pre-registered rule.

**What this actually means. Read it before quoting the line above.** The primary bar is met only because D detects
nothing at this operating point. Pixel's hits are almost all cabin-flicker events (62/64). Pixel found **1/96 door-desync
events**; D found 0/96. Per scenario (hits/events): flicker 30/32 and flicker+balloon 32/32 for pixel, 0/32 for D;
desync, desync+tree and desync+ball 0/32, 0/32 and 1/32 for pixel, 0/32 each for D. **Both detectors fail the door-fault
task on fresh seeds at a strict false-alarm budget.**

## Comparison with the 6-clip benchmark (Q3)

Pooled per-sample AUROC (finding-1 definition: score `s`; TEST fault clips + TEST healthy_ball_reopen):

| | D_patchKNN | pixel_patchKNN | pixel - D |
|---|---|---|---|
| Earlier: 6-clip set, seed 71, n = 1 per scenario, twin in bank | 0.811 | 0.796 | -0.015 (tie) |
| **Fresh seeds, twin-free, 16 TEST families** | **0.708** | **0.741** | **+0.033, CI [+0.022, +0.041], Holm p < 0.001** |

Both detectors fall by about 0.06-0.10 AUROC. The ranking flips to pixel ahead by a small but consistent margin.
Event-level AUROC (event peak `p` vs the healthy-clip maximum): D 0.434 (below chance), pixel 0.624.

## Secondary contrasts (Holm over 9)

| Contrast | Point | 95% CI | Holm p |
|---|---|---|---|
| S1 recall G - D | +0.000 | [0.000, +0.231] | 1.00 |
| S2 recall illum - D | +0.144 | [+0.113, +0.181] | < 0.001 |
| S3 pooled6 AUROC pixel - D | +0.033 | [+0.022, +0.041] | < 0.001 |
| S4 FA/h pixel - D | -3.8 | [-14.1, +7.7] | 1.00 |
| AL phase +0.25 s, recall pixel - D | +0.194 | [+0.181, +0.206] | < 0.001 |
| AL phase +0.50 s | +0.000 | [0, 0] | 1.00 |
| AL phase +1.00 s | +0.013 | [0, +0.031] | 1.00 |
| AL speed x0.95 | +0.000 | [-0.019, 0] | 1.00 |
| AL speed x1.05 | +0.244 | [+0.206, +0.300] | < 0.001 |

## Cycle-alignment arm (Q2)

Bank cycles are shifted or speed-scaled relative to CAL/TEST, and lambda is recalibrated on CAL against each bank.

| Bank variant | Recall D | Recall pixel | Pooled6 AUROC D | Pooled6 AUROC pixel |
|---|---|---|---|---|
| aligned | 0.000 | 0.394 | 0.708 | 0.741 |
| phase +0.25 s | 0.000 | 0.194 | 0.625 | 0.682 |
| phase +0.50 s | 0.000 | 0.000 | 0.595 | 0.649 |
| phase +1.00 s | 0.000 | 0.013 | 0.520 | 0.546 |
| speed x0.95 | 0.000 | 0.000 | 0.731 | 0.790 |
| speed x1.05 | 0.000 | 0.244 | 0.749 | 0.808 |

- Phase error degrades both detectors by similar amounts. At +1.0 s, AUROC falls by 0.19 for D and 0.20 for pixel, and
  both end near chance.
- Pixel's recall advantage disappears from +0.5 s.
- Because D's recall is already 0 when aligned, the data cannot show that pixel depends on alignment *more* than D does.
  The hypothesis from DOSSIER (iii) is **not supported and not refuted**.
- Speed-scaled banks did not lower AUROC; it went slightly up for both detectors. This is unexplained and post hoc.

## Failed bars and diagnostics

Diagnostics are post hoc and descriptive only: no new test, no tuning.

- **Failed: D_patchKNN detects 0/160 events at the CAL budget, in every bank variant (CAL6).** With CAL5
  (ball_reopen excluded) it is still 0/160 aligned, and 4/160 at speed x1.05.
- **Why the budget kills recall.** lambda is the CAL maximum of the envelope-ratio score. Healthy disturbances
  that are novel relative to the bank push that ratio far above the fault peaks:
  - healthy_ball_reopen: CAL max 14.3x for D and 44x for pixel;
  - healthy_tree_shadow with an unseen wind phase: 8.7x for D and 23x for pixel.
  - Door-desync peaks sit at about 1x the envelope for both detectors (median 1.06 for D, 0.66 for pixel), the same
    level as healthy_clean (CAL max 1.39 for D, 1.01 for pixel). So even with a clean-only calibration set, neither
    detector could detect desync without false alarms.
  - Our reading is that the R6-scale timing jitter in the bank (+-3% speed, +-50 ms) makes door positions at a given
    phase vary about as much as the 0.165 m desync lag. Per-patch nearest-neighbour then explains each leaf from a
    different bank cycle.
  - We checked on one clip that the fault is rendered: door asymmetry rises during the fault window. The frame
    difference to a jittered healthy clip is of a similar size.
- Cabin flicker is a large global luminance event: pixel ratio about 190x, D about 2.8x. Pixel catches it and D,
  at this budget, does not.

## Limitations (honest list)

- Everything is synthetic: one scene, one camera, one generator. Fault severities are those of the original
  generator; no severity ladder.
- The EC2 render pipeline is not pixel-identical to seed 71: Metal vs OptiX, GPU denoiser, direct 4 fps cropped
  rendering. On a zero-jitter re-render, V-JEPA tokens have cosine 0.88 to the cached reference tokens; the added
  sensor noise alone gives 0.835. Flicker faults are probably more visible here than in the seed-71 clip (ATTEMPTS.md).
  So the 0.811/0.796 comparison is a replication under a slightly different render domain, not a re-measurement.
- The operating point is strict. With 0.16 CAL hours, the 1 per hour budget means zero CAL alerts. A longer CAL set
  would allow a few episodes and a lower lambda. CAL6 contains ball_reopen, which the bank never contains. That
  choice is pre-registered; CAL5 is reported.
- The score is the frozen suite's per-block ratio to the envelope (max over blocks of map/tau). It is dominated by
  low-variance blocks. A different score (e.g. global tau, or a z-score) could rank differently; not tested here.
- The bootstrap is conditional on one fixed BANK of 6 families. 16 TEST families give indicative intervals only.
  "Non-inferior" is the pre-registered decision rule. It does not say that either detector works.
- Not tested: other encoders, trained heads, real footage, the IPAD-style phase-aware memory.

## Plain-language verdict for the Gap-1 paper

Once twin leakage and n = 1 are removed (fresh, independently jittered seeds, no shared background), the 6-clip
result does not hold up for either detector:
- at a false-alarm budget set on independent healthy footage, frozen V-JEPA 2.1 dense patch-kNN detects 0 of 160 fault events;
- the grey-pixel baseline detects 63 of 160, nearly all of them light-flicker events;
- neither detector catches door desync (pixel 1 of 96, V-JEPA 0 of 96);
- threshold-free, pixel now ranks samples slightly better than V-JEPA (0.741 vs 0.708; the 6-clip set gave 0.796 vs 0.811);
- a phase error of 0.5-1 s erodes both detectors.

What this adds to the audit paper: a pre-registered negative replication. The earlier "10/10 events" on the
6-clip set depended on a near-deterministic bank that contained each test clip's twin. "Pixel matches V-JEPA" survives
only in the weak sense that pixel is not worse; neither readout is a working door-fault detector under realistic
timing variation in this generator. It should be reported as a robustness/leakage finding, not as evidence that
either method works.
