from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path


def _mkdir_world_writable(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o777)


def _event(key: str, fmt: str) -> dict:
    if fmt == "eventbridge":
        return {
            "version": "0",
            "source": "aws.s3",
            "detail-type": "Object Created",
            "detail": {
                "bucket": {"name": "local"},
                "object": {"key": key},
            },
        }
    return {
        "Records": [{
            "eventSource": "aws:s3",
            "eventName": "ObjectCreated:Put",
            "s3": {
                "bucket": {"name": "local"},
                "object": {"key": urllib.parse.quote_plus(key)},
            },
        }],
    }


def _write_json(path: Path, value: dict) -> None:
    _mkdir_world_writable(path.parent)
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n")


def run_event(
    store_dir: Path,
    *,
    scenario: str | None = None,
    timeout: int = 900,
    fmt: str = "s3",
    replay_run_id: str | None = None,
    expect_duplicate: bool = False,
    run_id: str | None = None,
) -> dict:
    os.umask(0o000)
    store_dir = store_dir.expanduser().resolve()
    runs = store_dir / "runs"
    meta_dir = store_dir / "meta"
    queue_dir = store_dir / "queue"
    if replay_run_id:
        if not expect_duplicate:
            raise ValueError("--replay-run-id requires --expect-duplicate")
        previous_path = runs / replay_run_id / "result.json"
        previous = json.loads(previous_path.read_text())
        clip_key = previous["clip_key"]
        info_path = meta_dir / f"{replay_run_id}.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        scenario = info.get("scenario", replay_run_id)
        run_id = replay_run_id
        before_sha = hashlib.sha256(previous_path.read_bytes()).hexdigest()
        queue_path = queue_dir / f"replay-{replay_run_id}-{secrets.token_hex(2)}.json"
    else:
        if not scenario:
            raise ValueError("--scenario is required unless replaying a run")
        matches = list((store_dir / "dataset" / "lift_dataset").glob(f"*/{scenario}/{scenario}.mp4"))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one seeded clip for {scenario}, found {len(matches)}")
        source = matches[0]
        split = source.parent.parent.name
        run_id = run_id or (
            "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + f"-{scenario}-{secrets.token_hex(2)}"
        )
        clip_key = f"incoming/{run_id}/clip.mp4"
        target = store_dir / clip_key
        _mkdir_world_writable(target.parent)
        shutil.copyfile(source, target)
        _write_json(meta_dir / f"{run_id}.json", {"scenario": scenario, "split": split})
        before_sha = None
        queue_path = queue_dir / f"{run_id}.json"
    deadletter_path = store_dir / "deadletter" / f"{run_id}.json"

    if expect_duplicate and queue_path.exists():
        raise FileExistsError(queue_path)
    _write_json(queue_path, _event(clip_key, fmt))
    result_path = runs / run_id / "result.json"
    deadline = time.monotonic() + timeout
    if expect_duplicate:
        while time.monotonic() < deadline and queue_path.exists():
            if deadletter_path.exists():
                deadletter = json.loads(deadletter_path.read_text())
                raise RuntimeError(
                    f"coordinator dead-lettered run {run_id}: {deadletter.get('error')}"
                )
            time.sleep(1)
        if queue_path.exists():
            raise TimeoutError(f"duplicate queue item was not acknowledged: {queue_path}")
        if deadletter_path.exists():
            deadletter = json.loads(deadletter_path.read_text())
            raise RuntimeError(
                f"coordinator dead-lettered run {run_id}: {deadletter.get('error')}"
            )
        after_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
        if before_sha != after_sha:
            raise RuntimeError("duplicate delivery modified the stored result")
        return {"run_id": run_id, "decision": "DUPLICATE_SKIPPED", "trace_id": previous.get("trace_id")}

    while time.monotonic() < deadline:
        if deadletter_path.exists():
            deadletter = json.loads(deadletter_path.read_text())
            raise RuntimeError(
                f"coordinator dead-lettered run {run_id} after "
                f"{deadletter.get('attempts')} attempts: {deadletter.get('error')}"
            )
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result.get("error"):
                raise RuntimeError(f"coordinator run failed: {result['error']}")
            return result
        time.sleep(1)
    raise TimeoutError(f"timed out waiting for {result_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Queue one local S3-style or EventBridge event.")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--scenario")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--format", choices=("s3", "eventbridge"), default="s3")
    parser.add_argument("--replay-run-id")
    parser.add_argument("--expect-duplicate", action="store_true")
    args = parser.parse_args(argv)
    result = run_event(
        args.store,
        scenario=args.scenario,
        timeout=args.timeout,
        fmt=args.format,
        replay_run_id=args.replay_run_id,
        expect_duplicate=args.expect_duplicate,
    )
    print(f"run_id {result.get('run_id')}")
    print(f"decision {result.get('decision')}")
    print(f"trace_id {result.get('trace_id')}")
    if result.get("timings"):
        print(f"timings {json.dumps(result['timings'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
