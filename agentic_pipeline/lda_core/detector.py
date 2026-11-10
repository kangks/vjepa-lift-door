# Ported from .claude/worktrees/vjepa-talk-draft/workshop/lift-door/eval_suite/lift_vjepa_suite.py (sha256 45e0e712e0759e5b) on 2026-11-06; logic unchanged.
# Ported from .claude/worktrees/vjepa-talk-draft/workshop/lift-door/eval_suite/lift_vjepa_suite_v2.py (sha256 d7f04e5698144352) on 2026-11-06; logic unchanged.
from __future__ import annotations

import numpy as np


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def blocks(d: np.ndarray) -> np.ndarray:
    """[T,24,24] -> overlapping 2x2 block means [T,23,23]."""
    return 0.25 * (d[:, :-1, :-1] + d[:, 1:, :-1] + d[:, :-1, 1:] + d[:, 1:, 1:])


def patch_knn_map(test: np.ndarray, bank: list[np.ndarray]) -> np.ndarray:
    dists = [1.0 - np.einsum("thwd,thwd->thw", test, b, optimize=True) for b in bank]
    return blocks(np.min(dists, axis=0))


def cycles(x: np.ndarray) -> list[np.ndarray]:
    return [x[:16], x[16:32]]


def persist(flags: np.ndarray, n: int = 2) -> np.ndarray:
    out = np.zeros_like(flags, dtype=bool)
    run = 0
    for i, flag in enumerate(flags):
        run = run + 1 if flag else 0
        if run >= n:
            out[i - n + 1:i + 1] = True
    return out


def build_envelope(bank_tokens: dict[str, np.ndarray]) -> np.ndarray:
    """Leave-one-clip-out max envelope across cycle-aligned block maps."""
    names = list(bank_tokens)
    loo = []
    for clip in names:
        for cycle in range(2):
            others = [cycles(bank_tokens[other])[cycle_idx] for other in names if other != clip
                      for cycle_idx in range(2)]
            loo.append(patch_knn_map(cycles(bank_tokens[clip])[cycle], others))
    return (np.max(loo, axis=0) + 1e-4).astype(np.float32)


def detect(
    test_tokens_unit: np.ndarray,
    bank_tokens: dict[str, np.ndarray],
    envelope: np.ndarray,
    exclude: list[str] | None = None,
) -> dict:
    """Run the frozen D_patchKNN cycle-bank detector and return its result."""
    excluded = sorted(set(exclude or []))
    test = np.asarray(test_tokens_unit, dtype=np.float32)
    names = [name for name in bank_tokens if name not in excluded]
    if not names:
        raise ValueError("detector requires at least one non-excluded bank clip")
    bank_cycles = [cycle for name in names for cycle in cycles(bank_tokens[name])]
    tau = np.concatenate([envelope, envelope], axis=0)
    maps = np.concatenate([patch_knn_map(cyc, bank_cycles) for cyc in cycles(test)], axis=0)
    excess = (maps - tau).astype(np.float32)
    flags = persist((excess > 0).reshape(32, -1).any(axis=1))
    alert_samples = np.repeat(flags, 2)
    alert_times = [float(t) for t in np.arange(64, dtype=np.float32)[alert_samples] / 4.0]
    ti, y, x = np.unravel_index(int(np.argmax(excess)), excess.shape)
    peak_time = float(ti * 0.5)
    alert = bool(alert_times)
    return {
        "alert": alert,
        "alert_times_s": alert_times,
        "peak_time_s": peak_time,
        "peak_block_xy": [int(x), int(y)],
        "max_score_over_tau": float((maps / tau).max()),
        "window_s": [max(0.0, min(peak_time - 2.0, 12.0)), max(0.0, min(peak_time - 2.0, 12.0)) + 4.0]
        if alert else None,
        "bank_mode": "loo" if excluded else "full",
        "excluded_bank_clips": excluded,
        "excess": excess,
    }


def sample_labels(gt: dict) -> dict:
    """Per 4 fps sample labels, matching the frozen v1 manifest sampler."""
    frames = gt["frames"]
    idx = [min(len(frames) - 1, int(round(k * gt["fps"] / 4))) for k in range(64)]
    return {
        "anomaly": np.array([bool(frames[i]["anomaly"]) for i in idx]),
        "observable": np.array([bool(frames[i].get("observable", True)) for i in idx]),
        "events": gt["event_intervals"],
    }


def evaluate(alert_samples: np.ndarray, lab: dict) -> dict:
    times = np.arange(64) / 4.0
    anomaly_iv = [e for e in lab["events"] if e["type"] in ("door_desync", "interior_light_flicker")]
    hits = []
    for event in lab["events"]:
        inside = (times >= event["start_sec"] - 0.5) & (times < event["end_sec_exclusive"] + 0.5)
        hits.append({
            "type": event["type"], "start": event["start_sec"], "end": event["end_sec_exclusive"],
            "alerted": bool(alert_samples[inside].any()),
        })
    near = np.zeros(64, bool)
    for event in anomaly_iv:
        near |= (times >= event["start_sec"] - 0.5) & (times < event["end_sec_exclusive"] + 0.5)
    false_alerts = alert_samples & ~near & lab["observable"]
    return {
        "events": hits,
        "false_alert_samples": int(false_alerts.sum()),
        "false_alert_times_s": [float(t) for t in times[false_alerts]],
        "alert_times_s": [float(t) for t in times[alert_samples]],
    }
