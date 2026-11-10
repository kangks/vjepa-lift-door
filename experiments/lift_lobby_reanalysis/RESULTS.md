# Track A RESULTS: event-level, CAL-budget re-analysis of R6 and the 6-clip lift-door benchmark

Design: `DESIGN.md` (final sha256 23409c28e89b4d487cd0f19fb15b49db9951d008e16e38b35abad03649e085bf, including Amendment 1).
Run a2 on EC2 c7i.2xlarge i-03aa414686b8e8c8b (TERMINATED). `results.json` sha256 951ea4dc65f6d807d3c851d25274eff57348750a25302f88c5f74fe873bd6a49.
Full tables: `tables.md`. Every run: `ATTEMPTS.md`. All data are synthetic Blender renders. Real footage was not tested.

## Protocol in one paragraph

Each token step gets a score: the maximum of the detector map over the 23x23 blocks ("global", primary). An alert needs at
least 2 consecutive token steps above lambda. Lambda comes from healthy CAL footage only, at a budget of 1 alert episode
per camera-hour. Both CALs are far shorter than an hour (R6: 100 s; lift-door LOO bank: 80 s / 64 s), so the budget allows
0 episodes and the rule becomes "zero CAL alert episodes". This was stated before the data. An event is detected if an
alerting step overlaps its interval +-0.5 s. A false alert is an episode with no overlap with any fault window.

## R6 (pre-registered study, 80 independent defect families): PRIMARY

Gates R0 (stored map maxima and R6 decisions reproduced, 0 mismatches) and R1 (interval code reproduces stored D onset
errors, 107 rows) both PASSED. a1 and a2 gave identical R6 results.

**Primary: clean-defect event recall, pixel 80/80 (1.000, CP95 0.955-1.000) vs D 69/80 (0.863, CP95 0.767-0.929);
Delta = +0.1375, 95% family-cluster bootstrap CI [+0.075, +0.200] (B = 2000, 130 TEST families).
Pre-declared readout: pixel non-inferior at margin -0.05, and also superior (lower bound > 0).**
D misses are concentrated at L1 (15/25; L2 24/25, L3/L4 all found). Pixel finds 25/25 at every level.

Secondary, pixel minus D (Holm across 5):

| Contrast | point | 95% CI | p Holm |
|---|---|---|---|
| S1 event AUROC (80 events vs 640 healthy steps) | +0.010 (1.000 vs 0.990) | [+0.004, +0.017] | 0.005 |
| S2 event AP | +0.048 (1.000 vs 0.952) | [+0.023, +0.076] | 0.005 |
| S3 onset delay, median paired diff (69 events found by both) | -0.75 s (pixel median 0.30 s vs D 1.01 s from first brake) | [-0.75, -0.50] | 0.005 |
| S4 hard-negative false episodes per clip (60 clips) | -0.37 (pixel 45/60 vs D 67/60) | [-0.50, -0.24] | 0.005 |
| S5 defect+reflection recall (40) | 0.000 (40/40 vs 40/40) | [0, 0] | 1.0 |

Bootstrap p-values bottom out at 1/(B+1) ~ 0.001 before Holm. The S3 interval is quantised by the 0.25 s token step.

False alerts at the primary operating point (episodes; clips flagged out of clips):

| Arm | healthy TEST (40 clips, 160 s) | reflection-only (40) | exposure-only (10) | reopen (10) | false episodes on clean defect clips |
|---|---|---|---|---|---|
| D | 4 ep, 4/40 flagged (~90 /h) | 67 ep, 40/40 | 0 | 0 | 6 |
| pixel | 2 ep, 2/40 (~45 /h) | 40 ep, 40/40 | 2 ep, 2/10 | 3 ep, 3/10 | 0 |
| illum_pixel | 2, 2/40 | 40, 40/40 | 2, 2/10 | 2, 2/10 | 0 |
| flow_residual | 0, 0/40 | 14, 14/40 | 0 | 7, 7/10 | 0 |
| G (clip-level) | 1, 1/40 | 40/40 | 10/10 | 4/10 | 0 |

Other arms, clean recall (descriptive): illum_pixel 80/80; flow_residual 68/80 (L4 only 5/15); G 16/80.

Sensitivities (descriptive only):
- k = 4 steps (1.0 s persistence): D 68/80, pixel 80/80, illum 80/80, flow 73/80; pixel healthy false episodes 0/40, D 2.
- Envelope score (per-position tau from 20 close / 5 reopen CAL clips): every map arm finds 80/80, but every arm also flags
  all 40 healthy TEST clips. With CALs this small the envelope is unusable, so it gives no evidence either way.

## Lift-door 6-clip benchmark (n = 1 clip per scenario; INDICATIVE ONLY, after gate Amendment 1)

The original gate L0 (numeric match to Finding 1 per-sample scores) FAILED in run a1 on 16/72 series. The largest gap was
0.135 abs at a value of 760 (relative 1.8e-4; worst relative 2.5e-3). Likely cause: Mac-vs-Linux float32 BLAS. The run
stopped before any lift-door result. Amendment 1 changed the bar to decision-level reproduction of the frozen
`results_v2/summary.json`: in a2 that gate passed on all 60 detector x policy x clip decisions, and L1 (persistence
equivalence) passed.

**Primary (global score, CAL budget): every detector finds 0/10 fault events, on both the original and the twin-out bank.**
A single CAL-derived threshold over the raw map maximum sits above every fault peak. The pre-declared NI readout
"pixel non-inferior" (Delta 0.0, CI [0, 0]) is degenerate (0 vs 0) and carries no information. Threshold-free event AUROC
(10 event peaks vs non-fault steps): original pixel_patchKNN 0.953 vs D_patchKNN 0.928 (clip-bootstrap diff [+0.001, +0.064]).
Twin-out: 0.782 vs 0.771.

Secondary envelope score (the frozen v2 policy; lambda from LOO CAL):

| Bank | Detector | events | false episodes (96 s) | of which overlap a disturbance | onset median s |
|---|---|---|---|---|---|
| original | D_patchKNN | 10/10 | 9 | 5 | 1.04 |
| original | D_clipNN | 10/10 | 12 | 6 | 0.08 |
| original | pixel_patchKNN | 10/10 | 1 | 1 | 1.08 |
| original | pixel_clipNN | 10/10 | 1 | 1 | 1.08 |
| original | illum_pixel_patchKNN | 10/10 | 1 | 1 | 1.08 |
| original | FD_median | 10/10 | 6 | 6 | 1.04 |
| original | G_pooled | 0/10 | 1 | 1 | - |
| twin-out | D_patchKNN | 10/10 | 7 | 1 | 0.50 |
| twin-out | pixel_patchKNN | 9/10 | 3 | 3 | 1.00 |
| twin-out | pixel_clipNN / illum / FD_median | 10/10 each | 3 / 4 / 3 | 2 / 4 / 3 | 1.0 |
| twin-out | G_pooled | 3/10 | 5 | 5 | 0.0 |

## Failed bars and limitations

- Lift-door gate L0 failed as written (a1). The lift-door results are from the amended gate (decision-level). That
  amendment was made after R6 results had been printed, but before any lift-door result was computed.
- **The 1 episode/camera-hour budget is nominal only.** With 80-100 s of CAL it reduces to "zero CAL episodes". On TEST
  healthy footage the observed rates are ~45/h (pixel) and ~90/h (D) from 2 and 4 episodes in 160 s. CP95 on flagged-clip
  rate: pixel 0.006-0.169, D 0.028-0.237. Neither arm meets 1/h on TEST, and 160 s cannot verify such a rate anyway.
- Reflection-only clips are flagged 40/40 by D, pixel, illum and G, so S5 (defect+reflection recall 40/40 everywhere) is
  confounded by the reflection itself and is uninformative.
- R6 `D` and `pixel` are coherent nearest-clip (minimax) readouts, not per-patch kNN. R6 has no D_patchKNN, FD_median or
  per-token G. The R6 primary contrast is therefore pixel_clipNN-type vs D_clipNN-type. On the lift-door set the
  patch-kNN and clip-NN pairs behave alike under the envelope score.
- The lift-door CAL is the leave-one-out bank, not independent footage. Lift-door n = 1 per scenario, with twin leakage in
  the original bank. Its intervals are indicative only.
- Onset is retrospective, measured from the first brake (R6) or the event start (lift-door). It excludes processing latency.
- CAL and lambda were held fixed in the bootstrap, so the intervals do not include threshold uncertainty.

## Plain-language verdict for the Gap-1 paper

Under one event-level protocol with a CAL-only threshold, R6's pre-registered contrast holds and gets stronger.
Model-free pixel nearest-clip detection found 80/80 independent defects against 69/80 for V-JEPA 2.1 dense. It was
non-inferior and in fact superior (+0.14, CI +0.075 to +0.20). It alerted about 0.75 s earlier, with fewer
hard-negative episodes overall. However, pixel did alert on exposure steps and safe reopens, where D did not.
This is a single comparable R6 row for the audit's "pixel baselines" claim.

The 6-clip lift-door set adds no weight to the operating-point story. With a single global CAL threshold every detector,
V-JEPA included, finds nothing. Detection there depends entirely on the per-position envelope fitted to the bank. Under that
envelope, pixel ties D on events (10/10; twin-out 9/10 vs 10/10) and has fewer false episodes (1 vs 9). That is the
same picture as Finding 1, with the same n = 1 caveat.

Two honest qualifiers belong in the paper:
- The "fixed false-alarm budget" is nominal. Neither arm meets 1/h on TEST, and TEST is too short to verify it.
- All evidence is rendered.
