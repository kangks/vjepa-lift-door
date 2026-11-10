# Ported from workshop/lift-lobby-water-leak/scripts/vjepa/encoder.py (sha256 87ccc263e7764ad4) on 2026-11-06; logic unchanged.
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ.get("LDA_VJEPA_REPO", Path(__file__).resolve().parents[1] / "vendor" / "vjepa2"))
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ARCH = {"vitb": "vit_base", "vitl": "vit_large"}
PATCH, TUBELET = 16, 2


def build_encoder(arch: str, ckpt: Path, img_size: int = 384, num_frames: int = 64):
    sys.path.insert(0, str(REPO))
    from app.vjepa_2_1.models import vision_transformer as vit_encoder

    enc = vit_encoder.__dict__[ARCH[arch]](
        patch_size=PATCH, img_size=(img_size, img_size), num_frames=num_frames,
        tubelet_size=TUBELET, use_sdpa=True, use_SiLU=False, wide_SiLU=True,
        uniform_power=False, use_rope=True, img_temporal_dim_size=1, interpolate_rope=True,
    )
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "ema_encoder" in sd:
        sd = sd["ema_encoder"]
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
    enc.load_state_dict(sd, strict=True)
    return enc.eval()


def to_tensor(frames_uint8: np.ndarray) -> torch.Tensor:
    """[T,H,W,3] uint8 RGB -> [1,3,T,H,W] float32, ImageNet-normalised."""
    x = (frames_uint8.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x).permute(3, 0, 1, 2).unsqueeze(0).contiguous()


@torch.inference_mode()
def dense_tokens(enc, frames_uint8: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Return tokens as [T/2, H/16, W/16, D] float32."""
    t, h, w, _ = frames_uint8.shape
    out = enc(to_tensor(frames_uint8).to(device))
    return out[0].float().cpu().numpy().reshape(t // TUBELET, h // PATCH, w // PATCH, -1)


def encode_clip(enc, frames64: np.ndarray) -> np.ndarray:
    """Encode a 64-frame clip as four non-overlapping 16-frame windows."""
    return np.concatenate([dense_tokens(enc, frames64[s:s + 16], device="cpu") for s in range(0, 64, 16)], axis=0)
