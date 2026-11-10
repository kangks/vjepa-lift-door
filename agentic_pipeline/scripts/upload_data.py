from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lda_core.output_utils import read_stack_outputs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_bucket(path: Path) -> str:
    outputs = read_stack_outputs(path, "LiftDoorAgenticBuild")
    return outputs.get("BucketName") or outputs.get("bucket")


def _upload_if_changed(client, bucket: str, key: str, path: Path, content_type: str) -> dict:
    digest = sha256(path)
    try:
        head = client.head_object(Bucket=bucket, Key=key)
        if head.get("Metadata", {}).get("sha256") == digest:
            return {"key": key, "sha256": digest, "action": "skip"}
    except client.exceptions.NoSuchKey:
        pass
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if code not in {"404", "NoSuchKey", "NotFound"}:
            raise
    client.upload_file(
        str(path), bucket, key,
        ExtraArgs={"Metadata": {"sha256": digest}, "ContentType": content_type},
    )
    return {"key": key, "sha256": digest, "action": "upload"}


def build_upload_list(data_dir: Path, ckpt: Path, license_path: Path):
    files = []
    mp4s = []
    for ground_truth_path in sorted(data_dir.glob("*/*/ground_truth.json")):
        case = ground_truth_path.parent
        split = case.parent.name
        name = case.name
        video = case / f"{name}.mp4"
        if not video.exists():
            raise FileNotFoundError(video)
        video_key = f"dataset/lift_dataset/{split}/{name}/{name}.mp4"
        gt_key = f"dataset/lift_dataset/{split}/{name}/ground_truth.json"
        files.extend([
            (video_key, video, "video/mp4"),
            (gt_key, ground_truth_path, "application/json"),
        ])
        mp4s.append({
            "scenario": name,
            "split": split,
            "key": video_key,
            "mp4_sha256": sha256(video),
            "ground_truth_sha256": sha256(ground_truth_path),
        })
    if len(mp4s) != 12:
        raise ValueError(f"expected all 12 dataset scenarios, found {len(mp4s)}")
    files.extend([
        ("models/vjepa21/vitl_ema_encoder_only.pt", ckpt, "application/octet-stream"),
        ("models/vjepa21/LICENSE.vjepa2.txt", license_path, "text/plain"),
    ])
    checkpoint_sha = sha256(ckpt)
    manifest = {
        "sha256": checkpoint_sha,
        "bytes": ckpt.stat().st_size,
        "source_path": str(ckpt),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    dataset_index = {"scenarios": mp4s}
    return files, manifest, dataset_index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--license", dest="license_path", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    bucket = _profile_bucket(root / "outputs" / f"{args.profile}-build.json")
    if not bucket:
        raise ValueError("build outputs do not contain BucketName")
    files, manifest, index = build_upload_list(args.data, args.ckpt, args.license_path)
    if args.dry_run:
        for key, path, _ in files:
            print(f"{key}\t{sha256(path)}\t{path.stat().st_size}")
        print(f"models/vjepa21/manifest.json\t{manifest['sha256']}")
        print(f"dataset/lift_dataset/dataset_index.json\t{len(index['scenarios'])} scenarios")
        return 0

    import boto3
    client = boto3.Session(profile_name=args.profile, region_name="us-east-1").client("s3")
    rows = [_upload_if_changed(client, bucket, key, path, content_type)
            for key, path, content_type in files]
    client.put_object(
        Bucket=bucket, Key="models/vjepa21/manifest.json",
        Body=json.dumps(manifest, indent=2).encode(), ContentType="application/json",
        Metadata={"sha256": manifest["sha256"]},
    )
    index_bytes = json.dumps(index, indent=2).encode()
    client.put_object(
        Bucket=bucket, Key="dataset/lift_dataset/dataset_index.json",
        Body=index_bytes, ContentType="application/json",
        Metadata={"sha256": hashlib.sha256(index_bytes).hexdigest()},
    )
    print(json.dumps({"bucket": bucket, "files": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
