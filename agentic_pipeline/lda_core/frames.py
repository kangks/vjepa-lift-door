# Ported from .claude/worktrees/vjepa-talk-draft/workshop/lift-door/eval_suite/lift_vjepa_suite.py (sha256 45e0e712e0759e5b) on 2026-11-06; logic unchanged.
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .design import FROZEN_DESIGN


def decode_mp4(path: str | Path) -> np.ndarray:
    """Decode the offline ffmpeg recipe to the first 64 RGB frames."""
    path = Path(path)
    vf = "fps=4,crop=720:720:(iw-720)/2:(ih-720)/2,scale=384:384:flags=lanczos,format=rgb24"
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    arr = np.frombuffer(raw, np.uint8).reshape(-1, 384, 384, 3)
    if len(arr) < FROZEN_DESIGN["frames"]:
        raise ValueError(f"{path.name}: only {len(arr)} decoded frames")
    return arr[:64].copy()


def load_png_frames(case_dir: str | Path) -> np.ndarray:
    pngs = sorted((Path(case_dir) / "vjepa_frames").glob("clip_*.png"))
    if len(pngs) < FROZEN_DESIGN["frames"]:
        raise ValueError(f"{Path(case_dir).name}: expected 64 vjepa_frames PNGs, found {len(pngs)}")
    return np.stack([np.asarray(Image.open(p).convert("RGB")) for p in pngs[:64]])


def frames_sha16(frames: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(frames).tobytes()).hexdigest()[:16]
