import json

import numpy as np
from PIL import Image

from lda_core.design import BANK
from lda_core.detector import build_envelope, detect, unit
from lda_core.evidence import build_evidence
from lda_core.frames import frames_sha16, load_png_frames


def _tokens(name, frames, cache_dir):
    return unit(np.load(cache_dir / f"{name}_{frames_sha16(frames)}.npy").astype(np.float32))


def _image(data):
    import io
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def test_reasoner_v2_map_and_diffs_are_pixel_exact(eval_dir, cases, reasoner_rows):
    cache = eval_dir / "results_v1/token_cache"
    frames = {name: load_png_frames(cases[name]) for name in (*BANK, *reasoner_rows)}
    tokens = {name: _tokens(name, frames[name], cache) for name in frames}
    bank = {name: tokens[name] for name in BANK}
    envelope = build_envelope(bank)
    reference = frames["healthy_clean"]
    evidence_dir = eval_dir / "results_v2/reasoner_v2_evidence"
    for name in ("anomaly_door_desync", "anomaly_interior_light_flicker", "healthy_ball_reopen"):
        detector = detect(tokens[name], bank, envelope)
        evidence = build_evidence(frames[name], reference, detector["excess"], detector["peak_time_s"])
        expected_map = np.asarray(Image.open(evidence_dir / f"{name}_map.png").convert("RGB"))
        actual_map = _image(evidence["map_png"])
        assert np.array_equal(actual_map, expected_map), (
            name, "map max abs diff", int(np.abs(actual_map.astype(np.int16) - expected_map.astype(np.int16)).max())
        )
        for k, png in evidence["diffs"]:
            diff_name = f"{name}_diff_t{k / 4:.2f}.png"
            expected = np.asarray(Image.open(evidence_dir / diff_name).convert("RGB"))
            actual = _image(png)
            assert np.array_equal(actual, expected), (
                name, diff_name,
                "max abs diff", int(np.abs(actual.astype(np.int16) - expected.astype(np.int16)).max()),
            )
