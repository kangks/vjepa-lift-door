# Ported from .claude/worktrees/vjepa-talk-draft/workshop/lift-door/eval_suite/reasoner_opus_v2.py (sha256 b2406cf7eba08f0f) on 2026-11-06; logic unchanged.
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .reasoner import SITE_CONTEXT


def _png(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def overlay(frame: np.ndarray, excess_tmax: np.ndarray) -> np.ndarray:
    full = np.zeros((24, 24), np.float32)
    full[:23, :23] = np.maximum(excess_tmax, 0)
    x = full / max(float(full.max()), 1e-9)
    rgb = np.stack([np.clip(3 * x, 0, 1), np.clip(3 * x - 1, 0, 1), np.clip(3 * x - 2, 0, 1)], -1)
    heat = Image.fromarray((rgb * 255).astype(np.uint8)).resize((384, 384), Image.NEAREST)
    return np.asarray(Image.blend(Image.fromarray(frame), heat, 0.5))


def build_evidence(
    frames_cur: np.ndarray,
    frames_ref: np.ndarray,
    excess: np.ndarray,
    peak_time_s: float,
) -> dict:
    t0 = max(0.0, min(float(peak_time_s) - 2.0, 12.0))
    ks = list(range(int(t0 * 4), int(t0 * 4) + 16))
    win_tokens = slice(ks[0] // 2, ks[-1] // 2 + 1)
    peak_k = ks[int(np.argmax([excess[k // 2].max() for k in ks]))]
    map_rgb = overlay(frames_cur[peak_k], excess[win_tokens].max(0))
    diffs = []
    for k in ks[::4]:
        diff = np.clip(
            4 * np.abs(frames_cur[k].astype(np.int16) - frames_ref[k].astype(np.int16)), 0, 255
        ).astype(np.uint8)
        diffs.append((k, _png(diff)))
    return {
        "map_png": _png(map_rgb),
        "diffs": diffs,
        "ks": ks,
        "peak_k": peak_k,
        "window_s": [t0, t0 + 4.0],
    }
