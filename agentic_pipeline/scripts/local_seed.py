from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


CHECKPOINT_NAME = "vitl_ema_encoder_only.pt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_dataset(data_dir: Path, store_dir: Path) -> dict:
    data_dir = data_dir.expanduser().resolve()
    store_dir = store_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {data_dir}")
    previous_umask = os.umask(0o000)
    try:
        store_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
        os.chmod(store_dir, 0o777)
        for relative in (
            "queue", "queue/dead", "deadletter", "meta", "runs", "incoming", "tickets",
        ):
            directory = store_dir / relative
            directory.mkdir(parents=True, exist_ok=True, mode=0o777)
            os.chmod(directory, 0o777)
        otel_dir = store_dir.parent / "otel"
        otel_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
        os.chmod(otel_dir, 0o777)
    finally:
        os.umask(previous_umask)

    copies = []
    for ground_truth in sorted(data_dir.glob("*/*/ground_truth.json")):
        case_dir = ground_truth.parent
        split = case_dir.parent.name
        scenario = case_dir.name
        video = case_dir / f"{scenario}.mp4"
        if not video.is_file():
            raise FileNotFoundError(f"mp4 is missing for {scenario}: {video}")
        destination = store_dir / "dataset" / "lift_dataset" / split / scenario
        destination.mkdir(parents=True, exist_ok=True)
        target_video = destination / video.name
        target_truth = destination / "ground_truth.json"
        shutil.copyfile(video, target_video)
        shutil.copyfile(ground_truth, target_truth)
        copies.append({
            "scenario": scenario,
            "split": split,
            "mp4_key": target_video.relative_to(store_dir).as_posix(),
            "ground_truth_key": target_truth.relative_to(store_dir).as_posix(),
            "mp4_sha256": _sha256(target_video),
            "ground_truth_sha256": _sha256(target_truth),
        })
    if not copies:
        raise ValueError(f"no split/scenario/ground_truth.json files found under {data_dir}")

    ckpt_dir = Path(os.environ.get("LDA_CKPT_DIR", "./local_runs/ckpt")).expanduser()
    checkpoint = ckpt_dir / CHECKPOINT_NAME
    expected_sha = os.environ.get("LDA_CKPT_SHA256", "")
    if checkpoint.is_file():
        checkpoint_sha = _sha256(checkpoint)
        if expected_sha and expected_sha != checkpoint_sha:
            raise ValueError(
                f"checkpoint sha256 mismatch: env has {expected_sha}, file has {checkpoint_sha}"
            )
        expected_sha = checkpoint_sha

    manifest_path = store_dir / "models" / "vjepa21" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "sha256": expected_sha,
        "checkpoint": CHECKPOINT_NAME,
        "dataset": copies,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"scenarios": copies, "manifest": manifest_path.as_posix(), "ckpt_sha256": expected_sha}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Seed a local file store with lift-door clips.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    args = parser.parse_args(argv)
    result = seed_dataset(args.data, args.store)
    print(json.dumps({
        "scenario_count": len(result["scenarios"]),
        "scenarios": [row["scenario"] for row in result["scenarios"]],
        "manifest": result["manifest"],
        "ckpt_sha256": result["ckpt_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
