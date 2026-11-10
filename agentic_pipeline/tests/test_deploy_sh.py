from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = SOURCE_ROOT / "scripts/deploy.sh"
pytestmark = pytest.mark.skipif(
    not SOURCE_SCRIPT.is_file(),
    reason="deploy.sh is intentionally excluded from the remote image bundle",
)


def _sandbox(tmp_path: Path):
    root = tmp_path / "agentic"
    scripts = root / "scripts"
    infra = root / "infra"
    fakebin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    infra.mkdir()
    fakebin.mkdir()
    (scripts / "deploy.sh").write_bytes(SOURCE_SCRIPT.read_bytes())
    (infra / "accounts.json").write_text(json.dumps({"test-profile": "123456789012"}))
    cdk = fakebin / "cdk"
    cdk.write_text("""#!/bin/sh
set -eu
case "$1" in
  synth)
    echo "fake synth"
    ;;
  diff)
    variant=${FAKE_DIFF_VARIANT:-base}
    echo "Stack LiftDoorAgentic" >&2
    echo "[+] AWS::ECR::Repository Repo" >&2
    echo "│ RepositoryName: lda-test" >&2
    if [ "$variant" = changed ]; then
      echo "[~] AWS::ECR::Repository Repo: immutable -> mutable" >&2
    else
      echo "[~] AWS::ECR::Repository Repo: immutable" >&2
    fi
    echo "Resource diff payload" >&2
    echo "✨ Number of stacks with differences: 1" >&2
    ;;
  deploy)
    echo "FAKE_DEPLOY $*" >> "$FAKE_CDK_LOG"
    ;;
  *)
    echo "unexpected cdk command: $*" >&2
    exit 97
    ;;
esac
""")
    cdk.chmod(0o755)
    aws = fakebin / "aws"
    aws.write_text("""#!/bin/sh
set -eu
case "$*" in
  "sts get-caller-identity --profile test-profile --query Account --output text")
    echo 123456789012
    ;;
  *)
    echo "unexpected aws command: $*" >&2
    exit 98
    ;;
esac
""")
    aws.chmod(0o755)
    log = tmp_path / "cdk.log"
    env = os.environ.copy()
    env["PATH"] = f"{fakebin}:{env.get('PATH', '')}"
    env["FAKE_CDK_LOG"] = str(log)
    return root, scripts / "deploy.sh", env, log


def _main_diff_args(script: Path, action: str, diff_path: Path):
    args = ["bash", str(script), "test-profile", action, "--stacks", "main"]
    if action == "--diff-only":
        args += ["--diff-out", str(diff_path)]
    else:
        args += ["--reviewed-diff", str(diff_path)]
    args += [
        "-c", "imageTag=src-abc123",
        "-c", "imageDigests=/tmp/images.json",
    ]
    return args


def test_diff_only_captures_stderr_to_required_diff_file(tmp_path):
    _, script, env, _ = _sandbox(tmp_path)
    target = tmp_path / "diff.txt"
    result = subprocess.run(
        _main_diff_args(script, "--diff-only", target),
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    captured = target.read_text()
    assert "[+] AWS::ECR::Repository" in captured
    assert "[~] AWS::ECR::Repository" in captured
    assert "│ RepositoryName" in captured
    assert "DIFF_SHA256=" in result.stdout


def test_deploy_uses_exact_step05c_context_form_when_diff_matches(tmp_path):
    _, script, env, cdk_log = _sandbox(tmp_path)
    reviewed = tmp_path / "reviewed.txt"
    captured = subprocess.run(
        _main_diff_args(script, "--diff-only", reviewed),
        env=env, capture_output=True, text=True,
    )
    assert captured.returncode == 0, captured.stderr

    result = subprocess.run(
        _main_diff_args(script, "--deploy", reviewed),
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "REVIEWED_DIFF_SHA256=" in result.stdout
    assert "ACTUAL_DIFF_SHA256=" in result.stdout
    assert "FAKE_DEPLOY" in cdk_log.read_text()


def test_deploy_aborts_on_changed_diff_before_fake_deploy(tmp_path):
    _, script, env, cdk_log = _sandbox(tmp_path)
    reviewed = tmp_path / "reviewed.txt"
    captured = subprocess.run(
        _main_diff_args(script, "--diff-only", reviewed),
        env=env, capture_output=True, text=True,
    )
    assert captured.returncode == 0, captured.stderr

    changed_env = {**env, "FAKE_DIFF_VARIANT": "changed"}
    result = subprocess.run(
        _main_diff_args(script, "--deploy", reviewed),
        env=changed_env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "reviewed diff changed" in result.stderr
    assert not cdk_log.exists()


def test_deploy_aborts_on_empty_reviewed_diff(tmp_path):
    _, script, env, cdk_log = _sandbox(tmp_path)
    reviewed = tmp_path / "empty.txt"
    reviewed.write_text("")
    result = subprocess.run(
        _main_diff_args(script, "--deploy", reviewed),
        env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "fewer than 3 lines" in result.stderr
    assert not cdk_log.exists()


def test_deploy_does_not_normalize_away_changed_resource_marker(tmp_path):
    _, script, env, cdk_log = _sandbox(tmp_path)
    reviewed = tmp_path / "reviewed.txt"
    captured = subprocess.run(
        _main_diff_args(script, "--diff-only", reviewed),
        env=env, capture_output=True, text=True,
    )
    assert captured.returncode == 0, captured.stderr

    changed_env = {**env, "FAKE_DIFF_VARIANT": "changed"}
    result = subprocess.run(
        _main_diff_args(script, "--deploy", reviewed),
        env=changed_env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "reviewed diff changed" in result.stderr
    assert not cdk_log.exists()
