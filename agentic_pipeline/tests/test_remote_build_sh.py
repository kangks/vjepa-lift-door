from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = SOURCE_ROOT / "scripts/remote_build.sh"
pytestmark = pytest.mark.skipif(
    not SOURCE_SCRIPT.is_file(),
    reason="remote_build.sh is intentionally excluded from the remote image bundle",
)


def _sandbox(tmp_path: Path, *, bucket="test-bucket", region_in_output="ignored-region"):
    root = tmp_path / "agentic"
    scripts = root / "scripts"
    outputs = root / "outputs"
    dist = root / "dist"
    fakebin = tmp_path / "bin"
    for path in (scripts, outputs, dist, fakebin):
        path.mkdir(parents=True, exist_ok=True)
    script = scripts / "remote_build.sh"
    script.write_bytes(SOURCE_SCRIPT.read_bytes())
    zip_path = dist / "src.zip"
    zip_path.write_bytes(b"source archive fixture")
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    (dist / "src.zip.sha256").write_text(f"{digest}  src.zip\n")
    outputs_doc = {
        "LiftDoorAgenticBuild": {
            "BucketName": bucket,
            "ImageBuildProject": "lda-image-build",
            "E2eTestProject": "lda-e2e-test",
            "Region": region_in_output,
        }
    }
    (outputs / "ml-sandbox-build.json").write_text(json.dumps(outputs_doc))

    jq = fakebin / "jq"
    jq.write_text(r"""#!/usr/bin/env python3
import json
import sys
args=sys.argv[1:]
expression=args[-2]
path=args[-1]
value=json.load(open(path))
parts=[part for part in expression.replace('// empty','').strip().lstrip('.').split('.') if part]
for part in parts:
    if not isinstance(value,dict):
        value=None
        break
    value=value.get(part)
print('' if value is None else value)
""")
    jq.chmod(0o755)
    aws_log = tmp_path / "aws.calls"
    aws = fakebin / "aws"
    aws.write_text("""#!/bin/sh
set -eu
echo "$*" >> "$AWS_CALL_LOG"
case "$*" in
  "configure get region --profile ml-sandbox") echo "us-east-1" ;;
  *) echo "unexpected AWS call in dry-run: $*" >&2; exit 99 ;;
esac
""")
    aws.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fakebin}:{env.get('PATH', '')}"
    env["AWS_CALL_LOG"] = str(aws_log)
    return script, env, aws_log, digest


def test_remote_build_dry_run_reads_nested_outputs_and_never_starts_build(tmp_path):
    script, env, aws_log, digest = _sandbox(tmp_path)
    result = subprocess.run(
        ["bash", str(script), "ml-sandbox", "images", "--dry-run"],
        cwd=script.parents[1], env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert f"BUCKET=test-bucket" in result.stdout
    assert "REGION=us-east-1" in result.stdout
    assert "S3_ENDPOINT=https://s3.us-east-1.amazonaws.com" in result.stdout
    assert "PROJECT=lda-image-build" in result.stdout
    assert f"TAG=src-{digest[:12]}" in result.stdout
    assert "aws codebuild start-build" in result.stdout
    calls = aws_log.read_text().splitlines()
    assert calls == ["configure get region --profile ml-sandbox"]


def test_remote_build_dry_run_fails_clearly_if_nested_bucket_is_empty(tmp_path):
    script, env, aws_log, _ = _sandbox(tmp_path, bucket="")
    result = subprocess.run(
        ["bash", str(script), "ml-sandbox", "images", "--dry-run"],
        cwd=script.parents[1], env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "LiftDoorAgenticBuild.BucketName" in result.stderr
    assert aws_log.read_text().splitlines() == ["configure get region --profile ml-sandbox"]


def test_remote_build_chain_dry_run_plans_images_then_e2e_without_starting_build(tmp_path):
    script, env, aws_log, digest = _sandbox(tmp_path)
    result = subprocess.run(
        ["bash", str(script), "ml-sandbox", "chain", "--dry-run"],
        cwd=script.parents[1], env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("DRY_RUN=true") == 2
    assert result.stdout.count(f"TAG=src-{digest[:12]}") == 2
    assert "PROJECT=lda-image-build" in result.stdout
    assert "PROJECT=lda-e2e-test" in result.stdout
    assert result.stdout.index("PROJECT=lda-image-build") < result.stdout.index(
        "PROJECT=lda-e2e-test"
    )
    assert aws_log.read_text().splitlines() == [
        "configure get region --profile ml-sandbox",
        "configure get region --profile ml-sandbox",
    ]
