import re
from pathlib import Path

import yaml


def test_codebuild_buildspecs_parse_and_enforce_remote_build_guards():
    root = Path(__file__).resolve().parents[1]
    image_path = root / "buildspec-images.yml"
    e2e_path = root / "buildspec-e2e.yml"
    image_text = image_path.read_text()
    e2e_text = e2e_path.read_text()
    embed_dockerfile = (root / "services/embed/Dockerfile").read_text()
    image = yaml.safe_load(image_text)
    e2e = yaml.safe_load(e2e_text)
    assert image["version"] == 0.2
    assert e2e["version"] == 0.2
    assert "06248f29136a5d26c30e466d717d8845c25341e220f500468c0c162a3317cf20" in image_text
    assert "torch==2.14.1" in embed_dockerfile
    assert "torchvision==0.29.2" in embed_dockerfile
    assert "docker push" in image_text
    assert "CODEBUILD_BUILD_SUCCEEDING" in image_text
    assert "docker inspect" in image_text
    assert "ecr describe-images" not in image_text.lower()
    assert "AWS_ACCESS_KEY" not in image_text + e2e_text
    assert "compose_env_check.txt" in e2e_text
    assert "AWS_EC2_METADATA_DISABLED" in e2e_text
    assert "docker-compose.codebuild.yml" in e2e_text
    assert "internal" in e2e_text.lower()
    assert "local_build_bank.py" in e2e_text
    assert "http://embed:8080" in e2e_text
    assert "http://knn:8080/2015-03-31/functions/function/invocations" in e2e_text
    assert "docker_stats.txt" in e2e_text
    assert "timings.txt" in e2e_text
    assert "up -d --wait --wait-timeout 900" in e2e_text
    assert "compose_logs.txt" in e2e_text
    assert "e2e_images.txt" in e2e_text
    assert "--ckpt-sha" in e2e_text
    assert "chmod -R a+rwX /tmp/lda/local_runs" in e2e_text
    trap_lines = [
        line.strip() for line in e2e_text.splitlines()
        if line.strip().startswith("trap ")
    ]
    assert trap_lines == ["trap on_exit EXIT"]
    assert "logs --no-color --timestamps" in e2e_text
    assert "--timeout 900" in e2e_text
    assert "store_perms.txt" in e2e_text
    assert "upload_deadletter" in e2e_text
    run_event_positions = [
        match.start() for match in re.finditer(r"scripts/local_run_event\.py", e2e_text)
    ]
    chmod_positions = [
        match.start() for match in re.finditer(
            r"chmod -R a\+rwX /tmp/lda/local_runs", e2e_text
        )
    ]
    assert len(run_event_positions) == 2
    for position in run_event_positions:
        prior_chmods = [chmod for chmod in chmod_positions if chmod < position]
        assert prior_chmods
        assert max(prior_chmods) > max(
            [run for run in run_event_positions if run < position] or [-1]
        )
    assert "-m pip freeze" in image_text
    for service in ("embed", "knn", "ticket", "coordinator"):
        assert f'"{service}:$REPO_{service.upper()}"' in image_text
    assert '"/tmp/pip_freeze_${service}.txt"' in image_text
    for service in ("pip_freeze_embed", "pip_freeze_knn", "pip_freeze_ticket", "pip_freeze_coordinator"):
        assert service in image_text
    assert "dpkg -l ffmpeg libtiff6" in image_text


def test_inimage_heavy_tests_use_pinned_pytest_and_pipefail_before_push_guard():
    root = Path(__file__).resolve().parents[1]
    image_path = root / "buildspec-images.yml"
    image_text = image_path.read_text()
    image = yaml.safe_load(image_text)
    build_blocks = image["phases"]["build"]["commands"]
    embed_block = next(block for block in build_blocks if "pytest_inimage_embed.txt" in block)
    coordinator_block = next(
        block for block in build_blocks if "pytest_inimage_coordinator.txt" in block
    )
    requirements = (root / "requirements-test.txt").read_text().splitlines()
    pytest_pin = next(line.strip() for line in requirements if line.lower().startswith("pytest"))
    assert re.fullmatch(r"pytest==[A-Za-z0-9_.+-]+", pytest_pin)
    assert "P_PYTEST=$(grep -i '^pytest' requirements-test.txt)" in embed_block
    assert "TEST_ROOT=/workspace/sydney_physical_ai_october2026/workshop/lift-door/agentic" in embed_block
    assert f"pip install --user -q $P_PYTEST" in embed_block
    assert "-e LDA_REQUIRE_CKPT=1" in embed_block
    assert "-m 'slow or perm' $TEST_ROOT/tests/test_embed_inprocess.py $TEST_ROOT/tests/test_encoder_smoke.py" in embed_block
    assert "-e LDA_REQUIRE_CKPT=1" in coordinator_block
    assert "-m perm $TEST_ROOT/tests/test_run_dir_ownership.py" in coordinator_block

    for block, run_prefix, output in (
        (embed_block, "docker run --rm -v /tmp/ckpt:/ckpt:ro", "pytest_inimage_embed.txt"),
        (coordinator_block, 'docker run --rm -v "$PWD:$TEST_ROOT:ro"', "pytest_inimage_coordinator.txt"),
    ):
        command_position = block.index(run_prefix)
        assert block.index("set -euo pipefail") < command_position
        assert f"| tee /tmp/{output}" in block
        command_end = block.index(f"| tee /tmp/{output}", command_position)
        docker_run_args = block[command_position:command_end].split(' -c "', 1)[0]
        assert "--user" not in docker_run_args

    push_guard = image_text.index('test "$CODEBUILD_BUILD_SUCCEEDING" = "1"')
    assert image_text.index("pytest_inimage_embed.txt") < push_guard
    assert image_text.index("pytest_inimage_coordinator.txt") < push_guard


def test_e2e_parity_errors_and_json_artifacts_are_captured():
    root = Path(__file__).resolve().parents[1]
    e2e_text = (root / "buildspec-e2e.yml").read_text()
    e2e = yaml.safe_load(e2e_text)
    build_block = "\n".join(e2e["phases"]["build"]["commands"])

    assert "2> /tmp/lda/evidence/local_parity.err" in build_block
    assert "cat /tmp/lda/evidence/local_parity.err" in build_block
    assert "parity_report.py failed" in build_block

    capture_start = build_block.index("capture_run_artifacts()")
    exit_start = build_block.index("on_exit() {")
    exit_end = build_block.index("\n}", exit_start)
    capture_function = build_block[capture_start:exit_start]
    exit_function = build_block[exit_start:exit_end]
    assert capture_start < exit_start
    assert "capture_run_artifacts || true" in exit_function
    assert exit_function.index("capture_run_artifacts") < exit_function.index(
        "docker compose", exit_function.index("capture_run_artifacts")
    )
    for name in ("detector.json", "result.json", "reasoner_request_meta.json"):
        assert name in capture_function
    assert '"$store"/meta/*.json' in capture_function
    assert "bank/CURRENT.json" in capture_function
    assert "manifest.json" in capture_function
    assert "*.npy" not in capture_function
    assert "*.png" not in capture_function
    helper_start = build_block.index("copy_trace_otlp() {")
    helper_end = build_block.index("\n}", helper_start)
    helper_block = build_block[helper_start:helper_end]
    trap_position = build_block.index("trap on_exit EXIT")
    assert helper_start < trap_position
    assert "copy_trace_otlp || true" in exit_function
    assert exit_function.index("stop -t 30") < exit_function.index("copy_trace_otlp || true")
    assert "trace_dir=/tmp/lda/local_runs/otel" in helper_block
    assert "trace_dest=/tmp/lda/evidence/trace_otlp" in helper_block
    assert 'for span_file in "$trace_dir"/*.jsonl' in helper_block
    assert 'cp "$span_file" "$trace_dest/"' in helper_block


def test_e2e_stops_services_before_trace_copy_and_checks():
    root = Path(__file__).resolve().parents[1]
    e2e = yaml.safe_load((root / "buildspec-e2e.yml").read_text())
    build_block = "\n".join(e2e["phases"]["build"]["commands"])
    exit_start = build_block.index("on_exit() {")
    exit_end = build_block.index("\n}", exit_start)
    exit_block = build_block[exit_start:exit_end]

    scenario_loop = build_block.index("for scenario in")
    duplicate_replay = build_block.index("--replay-run-id")
    explicit_stop = build_block.index("stop -t 30", duplicate_replay)
    trace_copy = build_block.index("copy_trace_otlp", explicit_stop)
    trace_checks = build_block.index(
        "for run_id in $(awk '/^run_id/{print $2}' /tmp/lda/evidence/runs.txt); do",
        explicit_stop,
    )
    assert scenario_loop < duplicate_replay < explicit_stop < trace_copy < trace_checks
    assert "--settle-seconds 5" in build_block[trace_checks:]

    exit_stop = exit_block.index("stop -t 30")
    exit_logs = exit_block.index("logs --no-color --timestamps")
    exit_copy = exit_block.index("copy_trace_otlp || true")
    exit_down = exit_block.index(" down ")
    assert exit_stop < exit_logs < exit_copy < exit_down
