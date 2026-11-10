from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def profile_region(profile: str) -> str:
    result = subprocess.run(
        ["aws", "configure", "get", "region", "--profile", profile],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "us-east-1"


def output_path(root: Path, profile: str, stack_name: str) -> Path:
    suffix = "-build" if stack_name == "LiftDoorAgenticBuild" else ""
    return root / "outputs" / f"{profile}{suffix}.json"


def refresh(profile: str, stack_name: str, root: Path | None = None) -> Path:
    root = root or Path(__file__).resolve().parents[1]
    region = profile_region(profile)
    response = subprocess.run(
        [
            "aws", "cloudformation", "describe-stacks",
            "--stack-name", stack_name,
            "--profile", profile,
            "--region", region,
            "--output", "json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    document = json.loads(response.stdout)
    stacks = document.get("Stacks", [])
    if len(stacks) != 1:
        raise ValueError(f"expected exactly one stack named {stack_name}, found {len(stacks)}")
    values = {
        item["OutputKey"]: item["OutputValue"]
        for item in stacks[0].get("Outputs", [])
    }
    path = output_path(root, profile, stack_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({stack_name: values}, indent=2) + "\n")
    print(f"OUTPUT={path}")
    print(f"REGION={region}")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Refresh a nested CDK outputs file from CloudFormation.")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--stack", required=True)
    args = parser.parse_args(argv)
    refresh(args.profile, args.stack)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
