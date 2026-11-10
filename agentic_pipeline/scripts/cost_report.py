from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lda_core.output_utils import read_stack_outputs

PRICE_IN = 4.4e-6
PRICE_OUT = 22.0e-6
FARGATE_CPU_HOUR = 0.03238
FARGATE_GB_HOUR = 0.00356
LAMBDA_GB_SECOND_ARM = 0.0000133334


def estimate(result: dict) -> dict:
    reasoner = result.get("reasoner") or {}
    usage = reasoner.get("usage") or {}
    bedrock = usage.get("input_tokens", 0) * PRICE_IN + usage.get("output_tokens", 0) * PRICE_OUT
    timings = result.get("timings") or {}
    embed_seconds = sum(timings.get("encode_windows", [])) + timings.get("decode", 0)
    coordinator_seconds = timings.get("total", 0)
    fargate = embed_seconds / 3600 * (16 * FARGATE_CPU_HOUR + 32 * FARGATE_GB_HOUR)
    fargate += coordinator_seconds / 3600 * (0.5 * FARGATE_CPU_HOUR + FARGATE_GB_HOUR)
    lambda_gb_seconds = result.get("lambda_gb_seconds", 0)
    lambda_cost = lambda_gb_seconds * LAMBDA_GB_SECOND_ARM
    return {
        "run_id": result.get("run_id"),
        "bedrock_usd": bedrock,
        "fargate_usd_est": fargate,
        "lambda_usd_est": lambda_cost,
        "total_usd_est": bedrock + fargate + lambda_cost,
        "basis": "estimated with frozen offline Opus rates and per-service timings",
    }


def _stack_outputs(path: Path, stack_name: str) -> dict:
    return read_stack_outputs(path, stack_name)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    build_outputs = _stack_outputs(
        root / "outputs" / f"{args.profile}-build.json", "LiftDoorAgenticBuild",
    )
    bucket = build_outputs.get("BucketName") or build_outputs.get("bucket")
    if not bucket:
        raise ValueError("build outputs do not contain BucketName")

    main_outputs = _stack_outputs(root / "outputs" / f"{args.profile}.json", "LiftDoorAgentic")
    region = main_outputs.get("Region", "us-east-1")

    import boto3
    client = boto3.Session(
        profile_name=args.profile, region_name=region,
    ).client("s3")
    result = json.loads(client.get_object(
        Bucket=bucket, Key=f"runs/{args.run_id}/result.json",
    )["Body"].read())
    print(json.dumps(estimate(result), indent=2))


if __name__ == "__main__":
    main()
