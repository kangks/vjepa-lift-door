from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

import pytest

from lda_core.storage import FileStorage
from scripts import local_run_event
from scripts.local_seed import seed_dataset
from services.coordinator import main as coordinator


class _Enqueued(Exception):
    pass


def _preserve_umask():
    original = os.umask(0)
    os.umask(original)
    return original


def _write_one_clip(data_dir: Path) -> None:
    case = data_dir / "healthy" / "ownership_clip"
    case.mkdir(parents=True)
    (case / "ownership_clip.mp4").write_bytes(b"fake mp4")
    (case / "ground_truth.json").write_text('{"frames":[]}\n')


def _prepare_claim_failure(store_dir: Path, run_id: str):
    store_dir.mkdir(mode=0o777)
    os.chmod(store_dir, 0o777)
    for relative in (
        "queue", "queue/dead", "deadletter", "meta", "runs", "incoming", "tickets",
    ):
        directory = store_dir / relative
        directory.mkdir(parents=True, mode=0o777)
        os.chmod(directory, 0o777)
    run_dir = store_dir / "runs" / run_id
    run_dir.mkdir(mode=0o777)
    queue_file = store_dir / "queue" / f"{run_id}.json"
    queue_file.write_text(json.dumps(local_run_event._event(
        f"incoming/{run_id}/clip.mp4", "s3",
    )))
    return FileStorage(store_dir), queue_file, run_dir


def _assert_two_attempts_deadletter(store, queue_file, run_id, monkeypatch, capsys):
    sleeps = []
    monkeypatch.setattr(coordinator.time, "sleep", lambda seconds: sleeps.append(seconds))
    attempts = {}

    coordinator._process_file_queue_item(queue_file, store, attempts)
    first_log = capsys.readouterr().out.strip().splitlines()
    assert len(first_log) == 1
    assert json.loads(first_log[0])["attempt"] == 1
    assert "PermissionError" in first_log[0]
    assert queue_file.is_file()
    assert not (store.root / "deadletter" / f"{run_id}.json").exists()
    assert sleeps == [2]

    coordinator._process_file_queue_item(queue_file, store, attempts)
    second_log = capsys.readouterr().out.strip().splitlines()
    assert len(second_log) == 1
    assert json.loads(second_log[0])["attempt"] == 2
    assert "PermissionError" in second_log[0]
    deadletter = json.loads((store.root / "deadletter" / f"{run_id}.json").read_text())
    assert deadletter["run_id"] == run_id
    assert deadletter["attempts"] == 2
    assert "PermissionError" in deadletter["error"]
    assert deadletter["ts"]
    assert not queue_file.exists()
    assert (store.root / "queue" / "dead" / queue_file.name).is_file()
    assert attempts == {}


def test_local_run_event_keeps_scenario_metadata_out_of_service_runs(tmp_path, monkeypatch):
    original_umask = _preserve_umask()
    data_dir = tmp_path / "data"
    store = tmp_path / "store"
    _write_one_clip(data_dir)
    monkeypatch.setenv("LDA_CKPT_DIR", str(tmp_path / "empty-checkpoints"))
    monkeypatch.setenv("LDA_CKPT_SHA256", "")
    run_id = "run-ownership-test"

    def stop_after_enqueue(_seconds):
        assert (store / "queue" / f"{run_id}.json").is_file()
        raise _Enqueued()

    monkeypatch.setattr(local_run_event.time, "sleep", stop_after_enqueue)
    try:
        seed_dataset(data_dir, store)
        with pytest.raises(_Enqueued):
            local_run_event.run_event(
                store, scenario="ownership_clip", run_id=run_id, timeout=10,
            )
    finally:
        os.umask(original_umask)

    assert not (store / "runs" / run_id).exists()
    metadata = json.loads((store / "meta" / f"{run_id}.json").read_text())
    assert metadata == {"scenario": "ownership_clip", "split": "healthy"}
    assert (store / "incoming" / run_id / "clip.mp4").read_bytes() == b"fake mp4"
    for relative in (
        "queue", "queue/dead", "deadletter", "meta", "runs", "incoming", "tickets",
    ):
        assert stat.S_IMODE((store / relative).stat().st_mode) == 0o777
    assert stat.S_IMODE((store.parent / "otel").stat().st_mode) == 0o777


def test_file_queue_retries_claim_permission_error_then_deadletters(
    tmp_path, monkeypatch, capsys,
):
    original_umask = _preserve_umask()
    run_id = "run-claim-permission-test"
    store, queue_file, _run_dir = _prepare_claim_failure(tmp_path / "store", run_id)
    original_put = FileStorage.put_bytes_if_absent

    def deny_claim(self, key, data, content_type="application/octet-stream"):
        if key == f"runs/{run_id}/claim.json":
            raise PermissionError("simulated claim write failure")
        return original_put(self, key, data, content_type)

    monkeypatch.setattr(FileStorage, "put_bytes_if_absent", deny_claim)
    try:
        _assert_two_attempts_deadletter(store, queue_file, run_id, monkeypatch, capsys)
    finally:
        os.umask(original_umask)


@pytest.mark.perm
@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores chmod; covered by the in-image non-root run")
def test_file_queue_real_claim_permission_failure_deadletters(tmp_path, monkeypatch, capsys):
    original_umask = _preserve_umask()
    run_id = "run-real-claim-permission-test"
    store, queue_file, run_dir = _prepare_claim_failure(tmp_path / "store", run_id)
    run_dir.chmod(0o555)
    try:
        with pytest.raises(PermissionError):
            store.put_bytes_if_absent(f"runs/{run_id}/claim.json", b"claim")
        _assert_two_attempts_deadletter(store, queue_file, run_id, monkeypatch, capsys)
    finally:
        run_dir.chmod(0o777)
        os.umask(original_umask)


def test_local_run_event_fails_fast_when_deadletter_appears(tmp_path, monkeypatch):
    original_umask = _preserve_umask()
    data_dir = tmp_path / "data"
    store = tmp_path / "store"
    _write_one_clip(data_dir)
    monkeypatch.setenv("LDA_CKPT_DIR", str(tmp_path / "empty-checkpoints"))
    monkeypatch.setenv("LDA_CKPT_SHA256", "")
    seed_dataset(data_dir, store)
    run_id = "run-deadletter-fast-fail"
    deadletter = store / "deadletter" / f"{run_id}.json"
    sleeps = []

    def publish_deadletter(seconds):
        sleeps.append(seconds)
        deadletter.write_text(json.dumps({
            "run_id": run_id,
            "error": "PermissionError",
            "attempts": 2,
            "ts": "2026-11-06T00:00:00+00:00",
        }))

    monkeypatch.setattr(local_run_event.time, "sleep", publish_deadletter)
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="dead-lettered run"):
            local_run_event.run_event(
                store, scenario="ownership_clip", run_id=run_id, timeout=900,
            )
    finally:
        os.umask(original_umask)
    assert time.monotonic() - started < 10
    assert sleeps == [1]
