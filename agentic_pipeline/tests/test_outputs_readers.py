from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BUILD = {
    "BucketName": "nested-build-bucket",
    "ImageBuildProject": "lda-image-build",
    "E2eTestProject": "lda-e2e-test",
}
MAIN = {
    "ClusterName": "nested-cluster",
    "CoordinatorServiceName": "nested-coordinator",
    "CoordinatorTaskDefinitionArn": "arn:task",
    "PublicSubnetIds": ["subnet-a", "subnet-b"],
    "CoordinatorSecurityGroupId": "sg-coordinator",
    "EmbedServiceName": "nested-embed",
}


def _outputs(tmp_path: Path) -> tuple[Path, Path]:
    build_path = tmp_path / "ml-sandbox-build.json"
    main_path = tmp_path / "ml-sandbox.json"
    build_path.write_text(json.dumps({"LiftDoorAgenticBuild": BUILD}))
    main_path.write_text(json.dumps({"LiftDoorAgentic": MAIN}))
    return build_path, main_path


def _module(name: str):
    if not (ROOT / "scripts" / f"{name}.py").is_file():
        pytest.skip(f"scripts/{name}.py is intentionally excluded from the image bundle")
    return importlib.import_module(f"scripts.{name}")


def _run_output_cli(path: Path, stack: str, *keys: str) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-m", "lda_core.output_utils", str(path), stack, *keys],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return result.stdout.rstrip("\n").split("\t")


def test_upload_data_reads_nested_build_outputs(tmp_path):
    build_path, _ = _outputs(tmp_path)
    upload_data = _module("upload_data")
    assert upload_data._profile_bucket(build_path) == BUILD["BucketName"]


def test_run_event_reads_nested_main_and_build_outputs(tmp_path):
    build_path, main_path = _outputs(tmp_path)
    run_event = _module("run_event")
    assert run_event._stack_outputs(main_path)["ClusterName"] == MAIN["ClusterName"]
    assert run_event._build_outputs(build_path)["BucketName"] == BUILD["BucketName"]


def test_export_trace_reads_nested_build_outputs(tmp_path):
    build_path, _ = _outputs(tmp_path)
    export_trace = _module("export_trace")
    assert export_trace._outputs(build_path, "LiftDoorAgenticBuild")["BucketName"] == BUILD["BucketName"]


def test_cost_report_reads_nested_build_and_main_outputs(tmp_path):
    build_path, main_path = _outputs(tmp_path)
    cost_report = _module("cost_report")
    assert cost_report._stack_outputs(
        build_path, "LiftDoorAgenticBuild",
    )["BucketName"] == BUILD["BucketName"]
    assert cost_report._stack_outputs(main_path, "LiftDoorAgentic")["ClusterName"] == MAIN["ClusterName"]


def test_cloud_build_bank_reads_nested_build_and_main_outputs():
    path = ROOT / "scripts/cloud_build_bank.sh"
    if not path.is_file():
        pytest.skip("scripts/cloud_build_bank.sh is intentionally excluded from the image bundle")
    source = path.read_text()
    assert 'read_stack_outputs(sys.argv[1], "LiftDoorAgentic")' in source
    assert 'read_stack_outputs(sys.argv[1], "LiftDoorAgenticBuild")' in source


def test_demo_up_reads_nested_main_outputs(tmp_path):
    path = ROOT / "scripts/demo_up.sh"
    if not path.is_file():
        pytest.skip("scripts/demo_up.sh is intentionally excluded from the image bundle")
    assert 'read_stack_outputs(sys.argv[1], "LiftDoorAgentic")' in path.read_text()
    _, main_path = _outputs(tmp_path)
    assert _run_output_cli(
        main_path, "LiftDoorAgentic", "ClusterName", "EmbedServiceName",
        "CoordinatorServiceName",
    ) == ["nested-cluster", "nested-embed", "nested-coordinator"]


def test_demo_down_reads_nested_main_outputs(tmp_path):
    path = ROOT / "scripts/demo_down.sh"
    if not path.is_file():
        pytest.skip("scripts/demo_down.sh is intentionally excluded from the image bundle")
    assert 'read_stack_outputs(sys.argv[1], "LiftDoorAgentic")' in path.read_text()
    _, main_path = _outputs(tmp_path)
    assert _run_output_cli(
        main_path, "LiftDoorAgentic", "ClusterName", "EmbedServiceName",
        "CoordinatorServiceName",
    ) == ["nested-cluster", "nested-embed", "nested-coordinator"]


def test_destroy_reads_nested_build_outputs(tmp_path):
    path = ROOT / "scripts/destroy.sh"
    if not path.is_file():
        pytest.skip("scripts/destroy.sh is intentionally excluded from the image bundle")
    assert 'read_stack_outputs(sys.argv[1], "LiftDoorAgenticBuild")' in path.read_text()
    build_path, _ = _outputs(tmp_path)
    assert _run_output_cli(build_path, "LiftDoorAgenticBuild", "BucketName") == [
        "nested-build-bucket",
    ]


def test_refresh_outputs_writes_exact_nested_stack_map(tmp_path, monkeypatch):
    path = ROOT / "scripts/refresh_outputs.py"
    if not path.is_file():
        pytest.skip("scripts/refresh_outputs.py is intentionally excluded from the image bundle")
    refresh_outputs = importlib.import_module("scripts.refresh_outputs")

    def fake_run(command, *, capture_output, text, check):
        if command[:3] == ["aws", "configure", "get"]:
            return SimpleNamespace(returncode=0, stdout="us-east-1\n", stderr="")
        assert command[:3] == ["aws", "cloudformation", "describe-stacks"]
        response = {
            "Stacks": [{
                "StackName": "LiftDoorAgenticBuild",
                "Outputs": [
                    {"OutputKey": "BucketName", "OutputValue": "live-bucket"},
                    {"OutputKey": "ImageBuildProject", "OutputValue": "lda-image-build"},
                ],
            }],
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(response), stderr="")

    monkeypatch.setattr(refresh_outputs.subprocess, "run", fake_run)
    output = refresh_outputs.refresh("ml-sandbox", "LiftDoorAgenticBuild", root=tmp_path)
    assert json.loads(output.read_text()) == {
        "LiftDoorAgenticBuild": {
            "BucketName": "live-bucket",
            "ImageBuildProject": "lda-image-build",
        },
    }
