# Ported from .claude/worktrees/vjepa-talk-draft/workshop/lift-door/eval_suite/lift_vjepa_suite.py (sha256 45e0e712e0759e5b) on 2026-11-06; logic unchanged.
# Ported from .claude/worktrees/vjepa-talk-draft/workshop/lift-door/eval_suite/lift_vjepa_suite_v2.py (sha256 d7f04e5698144352) on 2026-11-06; logic unchanged.
import hashlib
import json

FROZEN_DESIGN = {
    "frozen_at": "2026-11-06, before any result, by the brainstorm partner (w1Z:pF)",
    "sample_fps": 4, "frames": 64, "crop": "centre 720x720 -> 384x384 (prepare_vjepa_frames.sh recipe)",
    "vjepa": "V-JEPA 2.1 ViT-L ema_encoder, windows of 16 frames, stride 16 (no overlap), "
             "dense tokens [time, 24, 24, 1024]; one token time = 2 samples = 0.5 s",
    "healthy_bank": ["healthy_clean", "healthy_tree_shadow", "healthy_ball_crossing",
                     "healthy_balloon_partial", "healthy_external_flicker"],
    "held_out_healthy": ["healthy_ball_reopen"],
    "equipment_anomaly_label": "ground-truth 'anomaly' (door desync OR interior light flicker); "
                               "ball, balloon, tree shadow, exterior flicker are healthy disturbances",
    "detectors": {
        "D_clipNN": "V-JEPA dense, nearest healthy CLIP (R6 minimax), same time, same patch",
        "D_patchKNN": "V-JEPA dense, per-patch nearest over ALL bank clips, same time (k=1)",
        "D_patchKNN_t1": "V-JEPA dense, per-patch nearest over bank clips and time t-1..t+1 (k=1)",
        "G_pooled": "V-JEPA mean-pooled vector per token time, nearest bank clip",
        "pixel_patchKNN": "grey patch mean abs difference, per-patch nearest, same time",
        "illum_pixel_patchKNN": "same after per-frame mean/std normalisation",
        "CV_rule_cabin_flicker": "FIXED RULE: cabin ROI (x150-270, y90-230 of 384) mean luminance "
                                 "drops >10% below the median of the previous 8 samples (2 s). "
                                 "No door-sync rule exists (rule set written before data)",
    },
    "threshold_policies": {
        "global": "tau = max over all leave-one-out bank maps + 1e-4 (no healthy bank sample alerts)",
        "envelope": "tau(t, block) = max over leave-one-out bank maps at that time and block + 1e-4",
    },
    "primary": "D_patchKNN with envelope policy (declared before results; all others are reported too)",
    "event_hit_rule": "an event counts as detected if any alert sample lies inside its interval +-0.5 s",
    "false_alert_rule": "alert samples farther than 0.5 s from every equipment-anomaly interval",
}

FROZEN_V2 = {
    "frozen_at": "2026-11-06 ~02:30 +08, before scoring interior flicker, combination and stress clips",
    "bank_unit": "healthy door cycles: 5 bank clips x 2 cycles of 8 s (16 token times / 32 samples)",
    "loo": "bank cycle scored against bank cycles of the OTHER 4 clips",
    "persistence": ">= 2 consecutive token steps (1.0 s) over threshold",
    "policies": {"global": "max over all LOO maps + 1e-4",
                 "envelope": "max over LOO maps at the same cycle time and block + 1e-4"},
    "primary": "D_patchKNN, envelope policy, with persistence",
    "cv_rule_A_door": "door ROI x100-290, y75-300; asymmetry = mean |left half - mirrored right half|; "
                      "alert if > 1.1 x healthy-bank max at the same cycle sample for 2 consecutive samples",
    "cv_rule_B_light": "v1 cabin luminance dip > 10% below the median of the previous 8 samples",
    "v1_reference": "v1 frozen design and results kept unchanged in results_v1/",
}

BANK = FROZEN_DESIGN["healthy_bank"]
REFERENCE = "healthy_clean"


def design_sha256() -> str:
    payload = json.dumps({"v1": FROZEN_DESIGN, "v2": FROZEN_V2}, indent=1, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
