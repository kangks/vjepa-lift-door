from __future__ import annotations

import hashlib

import numpy as np

from lda_core.design import BANK
from lda_core.detector import unit
from lda_core.frames import frames_sha16, load_png_frames
from services.knn import handler as knn


def _cached_tokens(name, frames, eval_dir):
    path = eval_dir / "results_v1/token_cache" / f"{name}_{frames_sha16(frames)}.npy"
    return unit(np.load(path).astype(np.float32))


def _mp4_sha(case):
    return hashlib.sha256((case / f"{case.name}.mp4").read_bytes()).hexdigest()


def test_file_storage_build_bank_then_detect_acceptance_cases(tmp_path, monkeypatch, eval_dir, cases):
    storage_root = tmp_path / "store"
    monkeypatch.setenv("LDA_STORAGE", f"file:{storage_root}")
    monkeypatch.setenv("LDA_OTEL_MODE", "file")
    monkeypatch.setenv("LDA_OTEL_FILE_DIR", str(tmp_path / "spans"))
    knn._BANK_CACHE.clear()
    from lda_core.storage import FileStorage
    storage = FileStorage(storage_root)

    clips = {}
    for name in BANK:
        frames = load_png_frames(cases[name])
        tokens = _cached_tokens(name, frames, eval_dir)
        frames_key = f"test-inputs/{name}/frames.npy"
        tokens_key = f"test-inputs/{name}/tokens.npy"
        storage.put_npy(frames_key, frames)
        storage.put_npy(tokens_key, tokens.astype(np.float16))
        clips[name] = {
            "frames_key": frames_key,
            "tokens_key": tokens_key,
            "mp4_sha256": _mp4_sha(cases[name]),
        }
    bank_result = knn.handler({
        "mode": "build_bank", "bank_build_id": "step02-test", "clips": clips,
        "ckpt_sha256": "offline-smoke",
    })
    assert bank_result["bank_prefix"] == "bank/v2/step02-test"
    assert storage.exists("bank/CURRENT.json")

    for name in ("anomaly_door_desync", "healthy_clean"):
        frames = load_png_frames(cases[name])
        tokens = _cached_tokens(name, frames, eval_dir)
        frames_key = f"runs/test-{name}/frames.npy"
        tokens_key = f"runs/test-{name}/tokens.npy"
        storage.put_npy(frames_key, frames)
        storage.put_npy(tokens_key, tokens.astype(np.float16))
        response = knn.handler({
            "mode": "detect", "run_id": f"test-{name}",
            "frames_key": frames_key, "tokens_key": tokens_key,
            "mp4_sha256": _mp4_sha(cases[name]),
        })
        assert storage.exists(f"runs/test-{name}/anomaly_map.npy")
        if name == "anomaly_door_desync":
            assert response["alert"] is True
            assert response["window_s"] == [3.0, 7.0]
            assert response["bank_mode"] == "full"
            assert len(response["evidence_keys"]["diffs"]) == 4
            assert storage.exists(response["evidence_keys"]["map"])
        else:
            assert response["alert"] is False
            assert response["bank_mode"] == "loo"
            assert response["excluded_bank_clips"] == ["healthy_clean"]
            assert response["evidence_keys"] == {}
