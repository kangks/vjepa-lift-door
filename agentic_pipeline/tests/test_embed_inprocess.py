from __future__ import annotations

import importlib
import logging
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lda_core.storage import FileStorage
from scripts.local_seed import main as local_seed_main


AGENTIC_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = AGENTIC_ROOT.parent / "lift_scene" / "lift_dataset"
CHECKPOINT_SHA256 = "06248f29136a5d26c30e466d717d8845c25341e220f500468c0c162a3317cf20"
REQUEST_BODY = {
    "run_id": "bank-diag-healthy_clean",
    "clip_key": "dataset/lift_dataset/healthy/healthy_clean/healthy_clean.mp4",
}


def _checkpoint_path() -> Path:
    try:
        configured = os.environ["LDA_CKPT_PATH"]
    except KeyError:
        configured = None
    path = Path(configured) if configured else None
    if path is None or not path.is_file():
        message = "LDA_CKPT_PATH not set"
        if os.environ.get("LDA_REQUIRE_CKPT") == "1":
            pytest.fail(f"{message} or checkpoint file is missing: {configured!r}")
        pytest.skip(message)
    return path


def _seed_only_healthy_clean(tmp_path: Path, store: Path) -> None:
    source = tmp_path / "data" / "healthy" / "healthy_clean"
    source.mkdir(parents=True)
    test_clip = os.environ.get("LDA_TEST_CLIP")
    if test_clip:
        shutil.copyfile(test_clip, source / "healthy_clean.mp4")
        (source / "ground_truth.json").write_text('{"frames":[]}\n')
    else:
        original = DATASET_ROOT / "healthy" / "healthy_clean"
        shutil.copyfile(original / "healthy_clean.mp4", source / "healthy_clean.mp4")
        shutil.copyfile(original / "ground_truth.json", source / "ground_truth.json")
    local_seed_main(["--data", str(source.parent.parent), "--store", str(store)])


@pytest.fixture(scope="module")
def embed_inprocess_context(tmp_path_factory):
    traceback_path = Path("/tmp/lda_03R3_traceback.txt")
    tmp_path = tmp_path_factory.mktemp("embed-inprocess")
    store_a = tmp_path / "store-a"
    store_b = tmp_path / "store-b"
    checkpoint_path = _checkpoint_path()
    environment = {
        "LDA_STORAGE": f"file:{store_a}",
        "LDA_CKPT_PATH": str(checkpoint_path),
        "LDA_CKPT_SHA256": CHECKPOINT_SHA256,
        "LDA_OTEL_MODE": "file",
        "LDA_OTEL_FILE_DIR": str(tmp_path / "otel"),
        "LDA_CKPT_DIR": str(tmp_path / "empty-checkpoints"),
    }
    previous_environment = {key: os.environ.get(key) for key in environment}
    try:
        os.environ.update(environment)
        _seed_only_healthy_clean(tmp_path / "seed-a", store_a)
        _seed_only_healthy_clean(tmp_path / "seed-b", store_b)

        embed_app = importlib.import_module("services.embed.app")
        with TestClient(embed_app.app, raise_server_exceptions=False) as client:
            health = None
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                health = client.get("/health")
                if health.status_code == 200:
                    break
                time.sleep(0.25)
            assert health is not None and health.status_code == 200, (
                None if health is None else health.text
            )
            assert health.json()["ckpt_sha256"] == CHECKPOINT_SHA256
            yield client, embed_app, store_a, store_b, health
    except Exception:
        traceback_path.parent.mkdir(parents=True, exist_ok=True)
        traceback_path.write_text(traceback.format_exc())
        raise
    finally:
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.mark.slow
def test_embed_inprocess_writable_store(embed_inprocess_context):
    client, embed_app, store_a, _store_b, health = embed_inprocess_context
    embed_app.STORAGE = FileStorage(store_a)
    assert health.status_code == 200
    response = client.post("/embed", json=REQUEST_BODY)
    assert response.status_code == 200, response.text
    result = response.json()
    assert set(result) == {
        "run_id", "frames_key", "tokens_key", "mp4_sha256", "frames_sha16", "seconds",
    }
    assert result["run_id"] == REQUEST_BODY["run_id"]
    assert result["frames_sha16"] == "d70281b04d333e66"
    frames = np.load(store_a / result["frames_key"], allow_pickle=False)
    tokens = np.load(store_a / result["tokens_key"], allow_pickle=False)
    assert frames.shape == (64, 384, 384, 3)
    assert frames.dtype == np.uint8
    assert tokens.shape == (32, 24, 24, 1024)
    assert tokens.dtype == np.float16


@pytest.mark.perm
@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores chmod; covered by the in-image non-root run")
def test_embed_inprocess_permission_error(embed_inprocess_context, caplog):
    client, embed_app, _store_a, store_b, _health = embed_inprocess_context
    embed_app.STORAGE = FileStorage(store_b)
    runs_b = store_b / "runs"
    runs_b.chmod(0o555)
    try:
        with caplog.at_level(logging.ERROR):
            response = client.post("/embed", json=REQUEST_BODY)
        assert response.status_code == 500, response.text
        assert response.json()["error"] == "PermissionError"
        assert "PermissionError" in caplog.text, caplog.text
    finally:
        runs_b.chmod(0o755)
