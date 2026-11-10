from __future__ import annotations

import argparse
import datetime as dt
import json
import secrets
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lda_core.output_utils import read_stack_outputs


def _stack_outputs(path: Path, stack_name: str = "LiftDoorAgentic") -> dict:
    return read_stack_outputs(path, stack_name)


def _build_outputs(path: Path) -> dict:
    return read_stack_outputs(path, "LiftDoorAgenticBuild")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--batch-id")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    data = args.data or root.parent / "lift_scene/lift_dataset"
    scenario = next(data.glob(f"*/{args.scenario}/ground_truth.json"), None)
    if scenario is None:
        raise FileNotFoundError(f"unknown scenario: {args.scenario}")
    case = scenario.parent
    split = case.parent.name
    source_key = f"dataset/lift_dataset/{split}/{case.name}/{case.name}.mp4"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_id = f"run-{stamp}-{args.scenario}-{secrets.token_hex(2)}"
    target_key = f"incoming/{run_id}/clip.mp4"

    build = _build_outputs(root / "outputs" / f"{args.profile}-build.json")
    bucket = build.get("BucketName") or build.get("bucket")
    if not bucket:
        raise ValueError("build outputs do not contain BucketName")
    import boto3
    client = boto3.Session(profile_name=args.profile, region_name="us-east-1").client("s3")
    client.copy_object(
        Bucket=bucket, Key=target_key,
        CopySource={"Bucket": bucket, "Key": source_key},
        MetadataDirective="REPLACE",
        Metadata={"scenario": args.scenario, "batch_id": args.batch_id or ""},
        ContentType="video/mp4",
    )
    deadline = time.monotonic() + args.timeout
    result_key = f"runs/{run_id}/result.json"
    while time.monotonic() < deadline:
        try:
            result = json.loads(client.get_object(Bucket=bucket, Key=result_key)["Body"].read())
            row = {"run_id": run_id, "decision": result.get("decision"),
                   "trace_id": result.get("trace_id"), "timings": result.get("timings")}
            print(json.dumps(row))
            evidence = Path(__file__).resolve().parents[4] / ".claude/worktrees/vjepa-talk-draft/agentic_plan/evidence" / args.profile
            evidence.mkdir(parents=True, exist_ok=True)
            with (evidence / "runs.jsonl").open("a") as output:
                output.write(json.dumps(row) + "\n")
            return 0
        except client.exceptions.NoSuchKey:
            pass
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code not in {"404", "NoSuchKey", "NotFound"}:
                raise
        time.sleep(5)
    raise TimeoutError(f"timed out waiting for s3://{bucket}/{result_key}")


if __name__ == "__main__":
    raise SystemExit(main())
