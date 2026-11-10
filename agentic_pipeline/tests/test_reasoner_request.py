import base64
import importlib
import io
import json
import sys

import numpy as np
from PIL import Image

from lda_core.design import BANK
from lda_core.detector import build_envelope, detect, unit
from lda_core.evidence import build_evidence
from lda_core.frames import frames_sha16, load_png_frames
from lda_core.reasoner import MODEL, SCHEMA, build_request, fingerprints


def _tokens(name, frames, cache_dir):
    return unit(np.load(cache_dir / f"{name}_{frames_sha16(frames)}.npy").astype(np.float32))


def _original_reasoner_module(eval_dir):
    sys.path.insert(0, str(eval_dir))
    return importlib.import_module("reasoner_opus_v2")


def _original_text_blocks(reasoner_v2, frames_ref, evidence, window):
    t0, t1 = window
    ks = list(range(int(t0 * 4), int(t0 * 4) + 16))
    peak_k = evidence["peak_k"]
    ctx = {
        "detector": "V-JEPA 2.1 dense embeddings + nearest-neighbour novelty detector",
        "flagged_window_s": [t0, t0 + 4.0],
        "site_context": reasoner_v2.SITE_CONTEXT,
        "inputs": "CUR_xx flagged clip (4 fps); REF_xx normal cycle same times; MAP detector map; "
                  "DIFF_xx amplified |CUR-REF|",
    }
    text = [json.dumps(ctx), f"MAP (detector anomaly map, max over window, on frame t={peak_k / 4:.2f}s)"]
    text.extend(f"DIFF_{i:02d} t={k / 4:.2f}s" for i, k in enumerate(ks[::4]))
    text.extend(f"REF_{i:02d} t={k / 4:.2f}s" for i, k in enumerate(ks[::2]))
    text.extend(f"CUR_{i:02d} t={k / 4:.2f}s" for i, k in enumerate(ks))
    return [item.encode("utf-8") for item in text]


def test_fingerprints_and_29_image_blind_request(eval_dir, cases, reasoner_rows):
    original_reasoner = _original_reasoner_module(eval_dir)
    expected_fingerprints = {
        "system_sha256": "deb36ff95475b2b1a021f4d8675fb121925713f5016ea6455d556cfa6cf22e3d",
        "schema_sha256": "f9df55d80e6ef444643e89ce6302e1c07d6b21b2aee8395d7aae25b0f28563fe",
        "site_context_sha256": "d74ce540188297ea4cd3fe82434821b8b38aed7f3e5e5713364b58d2f7c626a0",
    }
    assert fingerprints() == expected_fingerprints
    cache = eval_dir / "results_v1/token_cache"
    names = (*BANK, "anomaly_door_desync", "anomaly_interior_light_flicker", "healthy_ball_reopen")
    frames = {name: load_png_frames(cases[name]) for name in names}
    tokens = {name: _tokens(name, frames[name], cache) for name in names}
    bank = {name: tokens[name] for name in BANK}
    envelope = build_envelope(bank)
    for name in ("anomaly_door_desync", "anomaly_interior_light_flicker", "healthy_ball_reopen"):
        detector = detect(tokens[name], bank, envelope)
        window = reasoner_rows[name]["window_s"]
        assert detector["window_s"] == window
        evidence = build_evidence(frames[name], frames["healthy_clean"], detector["excess"], detector["peak_time_s"])
        payload, image_count, frame_ids = build_request(frames[name], frames["healthy_clean"], evidence, window)
        assert image_count == 29
        assert len(frame_ids) == 16
        assert payload["model"] == MODEL == "anthropic.claude-opus-5-5"
        assert payload["system"] == original_reasoner.SYSTEM
        assert payload["max_tokens"] == 4000
        assert payload["output_config"]["effort"] == "high"
        assert payload["output_config"]["format"]["schema"] == SCHEMA
        content = payload["messages"][0]["content"]
        text_blocks = [item["text"].encode("utf-8") for item in content if item["type"] == "text"]
        expected_text = _original_text_blocks(original_reasoner, frames["healthy_clean"], evidence, window)
        assert text_blocks == expected_text

        run_id = f"run-test-{name}-0000"
        clip_key = f"incoming/{run_id}/clip.mp4"
        forbidden = [*cases.keys(), run_id, clip_key, "ground_truth", "anomaly_", "healthy_", "stress_",
                     "door_desync", "interior_light_flicker", "ball_reopen", "ball_crossing", "balloon",
                     "tree_shadow", "external_flicker"]
        joined = b"\n".join(text_blocks).lower()
        for secret_string in forbidden:
            assert secret_string.lower().encode("utf-8") not in joined, (name, secret_string)

        for item in content:
            if item["type"] != "image":
                continue
            encoded = item["source"]["data"]
            png = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(png))
            assert image.format == "PNG"
            assert image.info == {}, (name, image.info)
