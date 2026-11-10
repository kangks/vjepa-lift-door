from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="encoder smoke requires torch; local STEP_02R venv omits torch")

from lda_core.encoder import build_encoder, dense_tokens
from lda_core.frames import frames_sha16, load_png_frames
from lda_core.detector import unit


@pytest.mark.slow
def test_encoder_smoke_matches_cached_mps_tokens(request):
    try:
        checkpoint = os.environ["LDA_CKPT_PATH"]
    except KeyError:
        checkpoint = None
    ckpt = Path(checkpoint) if checkpoint else None
    if ckpt is None or not ckpt.is_file():
        message = "LDA_CKPT_PATH not set"
        if os.environ.get("LDA_REQUIRE_CKPT") == "1":
            pytest.fail(f"{message} or checkpoint file is missing: {checkpoint!r}")
        pytest.skip(message)

    cache_value = os.environ.get("LDA_MPS_TOKEN_CACHE")
    if not cache_value:
        pytest.skip("LDA_MPS_TOKEN_CACHE not set")
    cache_path = Path(cache_value)
    if not cache_path.is_file():
        pytest.skip(f"LDA_MPS_TOKEN_CACHE not found: {cache_path}")

    cases = request.getfixturevalue("cases")
    case = cases["healthy_clean"]
    frames64 = load_png_frames(case)
    expected_cache_name = f"healthy_clean_{frames_sha16(frames64)}.npy"
    if cache_path.name != expected_cache_name:
        pytest.skip(
            f"LDA_MPS_TOKEN_CACHE does not match healthy_clean frames: expected {expected_cache_name}"
        )
    reference = np.load(cache_path).astype(np.float32)[:8]
    enc = build_encoder("vitl", ckpt, img_size=384, num_frames=16).to("cpu")
    start = time.perf_counter()
    actual = dense_tokens(enc, frames64[:16], device="cpu")
    seconds = time.perf_counter() - start
    assert actual.shape == (8, 24, 24, 1024)
    cosine = (unit(actual) * unit(reference)).sum(axis=-1)
    mean_cosine = float(cosine.mean())
    min_cosine = float(cosine.min())
    max_abs_diff = float(np.abs(actual - reference).max())
    print(
        f"ENCODER_SMOKE shape={actual.shape} mean_cosine={mean_cosine:.9f} "
        f"min_cosine={min_cosine:.9f} max_abs_token_diff={max_abs_diff:.9f} "
        f"cpu_seconds_per_window={seconds:.3f}"
    )
    if 0.99 <= mean_cosine <= 0.999:
        print("DRIFT WARNING: mean cosine is 0.99-0.999; review window drift risk before STEP_03.")
    assert mean_cosine >= 0.99, f"mean cosine {mean_cosine:.6f} is below 0.99"
