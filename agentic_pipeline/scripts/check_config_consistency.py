#!/usr/bin/env python3
"""Check that runtime settings agree across code, containers, compose, and CDK."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA = "06248f29136a5d26c30e466d717d8845c25341e220f500468c0c162a3317cf20"
CHECKPOINT_NAME = "vitl_ema_encoder_only.pt"
checks: list[tuple[str, str, str, str]] = []


def record(name: str, files: str, passed: bool | None, detail: str = "") -> None:
    status = "PASS" if passed is True else "FAIL" if passed is False else "SKIPPED"
    checks.append((name, files, status, detail))


def text(relative: str) -> str:
    return (ROOT / relative).read_text()


def add_compose_environment(config: dict, names: set[str]) -> None:
    for spec in config.get("services", {}).values():
        env = spec.get("environment", {})
        if isinstance(env, dict):
            names.update(str(key) for key in env)
        else:
            names.update(str(row).split("=", 1)[0] for row in env)


def compose_mount_targets(spec: dict) -> set[str]:
    targets = set()
    for volume in spec.get("volumes", []):
        parts = str(volume).split(":")
        if len(parts) >= 2:
            targets.add(parts[-2] if parts[-1] in {"ro", "rw"} else parts[-1])
    return targets


def main_stack_embed_env(source: str) -> dict[str, str]:
    start = source.index('self.embed_container = self.embed_task.add_container(')
    block = source[start:source.index('logging=ecs.LogDrivers.aws_logs(', start)]
    return {
        key: value.strip().strip('"')
        for key, value in re.findall(
            r'"(LDA_[A-Z0-9_]+)"\s*:\s*("[^"]*"|[^,\n]+)', block
        )
    }


def main_stack_container_env_blocks(source: str) -> dict[str, str]:
    markers = {
        "embed": "self.embed_container = self.embed_task.add_container(",
        "coordinator": "self.coordinator_container = self.coordinator_task.add_container(",
        "oneoff": "self.oneoff_container = self.oneoff_task.add_container(",
    }
    shared_environment_start = source.index("coordinator_environment = {")
    shared_environment_end = source.index(
        "self.coordinator_container = self.coordinator_task.add_container(",
        shared_environment_start,
    )
    shared_environment = source[shared_environment_start:shared_environment_end]
    blocks = {}
    for service, marker in markers.items():
        start = source.index(marker)
        logging_marker = (
            "logging=ecs.GenericLogDriver("
            if service == "oneoff"
            else "logging=ecs.LogDrivers.aws_logs("
        )
        end = source.index(logging_marker, start)
        blocks[service] = source[start:end]
        if service in {"coordinator", "oneoff"} and "environment=coordinator_environment" in blocks[service]:
            blocks[service] = shared_environment + blocks[service]
    return blocks


def env_reads(source: str) -> set[str]:
    names: set[str] = set()
    patterns = (
        re.compile(
            r'os\.environ(?:\.get)?\(\s*[\'"](LDA_[A-Z0-9_]+)[\'"]'
        ),
        re.compile(
            r'os\.environ\[\s*[\'"](LDA_[A-Z0-9_]+)[\'"]\s*\]'
        ),
    )
    for pattern in patterns:
        names.update(pattern.findall(source))
    return names


def py_files(folder: str) -> list[Path]:
    return sorted((ROOT / folder).rglob("*.py"))


def run_checks() -> int:
    embed_dockerfile = text("services/embed/Dockerfile")
    coordinator_dockerfile = text("services/coordinator/Dockerfile")
    image_text = text("buildspec-images.yml")
    e2e_text = text("buildspec-e2e.yml")
    compose = yaml.safe_load(text("local/docker-compose.yml"))
    compose_override = yaml.safe_load(text("local/docker-compose.codebuild.yml"))
    embed_compose = compose["services"]["embed"]
    embed_env = embed_compose.get("environment", {})
    embed_targets = compose_mount_targets(embed_compose)
    cdk_path = ROOT / "infra/lda_infra/main_stack.py"
    cdk_exists = cdk_path.is_file()
    cdk_source = cdk_path.read_text() if cdk_exists else ""
    embed_stack_env = main_stack_embed_env(cdk_source) if cdk_exists else {}
    container_env_blocks = main_stack_container_env_blocks(cdk_source) if cdk_exists else {}

    agent_observability_services = [
        service for service, block in container_env_blocks.items()
        if service != "oneoff" and '"AGENT_OBSERVABILITY_ENABLED"' in block
    ]
    otlp_log_services = []
    otlp_log_headers_ok = True
    for service, block in container_env_blocks.items():
        if service == "oneoff":
            continue
        exporter = re.search(r'"OTEL_LOGS_EXPORTER"\s*:\s*"([^"]+)"', block)
        if exporter and exporter.group(1) == "otlp":
            otlp_log_services.append(service)
            otlp_log_headers_ok &= (
                '"OTEL_EXPORTER_OTLP_LOGS_HEADERS"' in block
                and "x-aws-log-group=" in block
            )
    record(
        "CDK telemetry exporter scoping",
        "infra/lda_infra/main_stack.py",
        (
            agent_observability_services == ["coordinator"]
            and "coordinator" in otlp_log_services
            and otlp_log_headers_ok
        ) if cdk_exists else None,
        (
            f"AGENT_OBSERVABILITY_ENABLED services={agent_observability_services}; "
            f"OTLP log services={otlp_log_services} all have x-aws-log-group headers"
        ) if cdk_exists else "no infra/ in bundle; enforced on the working tree",
    )
    record(
        "CDK one-off task environment reuse",
        "infra/lda_infra/main_stack.py",
        (
            "coordinator_environment = {" in cdk_source
            and "environment=coordinator_environment" in container_env_blocks["coordinator"]
            and "environment=coordinator_environment" in container_env_blocks["oneoff"]
        ) if cdk_exists else None,
        "OneoffContainer uses the same coordinator environment definition"
        if cdk_exists else "no infra/ in bundle; enforced on the working tree",
    )
    trace_endpoint_ok = cdk_exists and all(
        '"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"' in container_env_blocks.get(service, "")
        and 'f"https://xray.{region}.amazonaws.com/v1/traces"' in container_env_blocks.get(service, "")
        for service in ("embed", "coordinator", "oneoff")
    )
    record(
        "CDK ADOT trace endpoint uses stack region",
        "infra/lda_infra/main_stack.py",
        trace_endpoint_ok if cdk_exists else None,
        "embed, coordinator, and one-off task envs use the X-Ray endpoint from region"
        if cdk_exists else "no infra/ in bundle; enforced on the working tree",
    )
    coordinator_entrypoint = text("services/coordinator/entrypoint.sh")
    embed_entrypoint = text("services/embed/entrypoint.sh")
    entrypoints_honor_args = all(
        'exec opentelemetry-instrument "$@"' in source and 'exec "$@"' in source
        for source in (coordinator_entrypoint, embed_entrypoint)
    )
    record(
        "container entrypoints honor command overrides",
        "services/coordinator/entrypoint.sh; services/embed/entrypoint.sh",
        entrypoints_honor_args,
        "both AWS and file modes exec supplied arguments",
    )
    telemetry_source = text("lda_core/telemetry.py")
    aws_exporter_import = (
        "from amazon.opentelemetry.distro.exporter.otlp.aws.traces."
        "otlp_aws_span_exporter import"
    )
    no_silent_exporter_fallback = (
        aws_exporter_import in telemetry_source
        and 'raise RuntimeError("ADOT OTLPAwsSpanExporter unavailable")' in telemetry_source
        and "from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter"
        not in telemetry_source
    )
    record(
        "aws mode requires signed ADOT exporter",
        "lda_core/telemetry.py",
        no_silent_exporter_fallback,
        "aws mode uses OTLPAwsSpanExporter and raises on ImportError without unsigned fallback",
    )

    record(
        "checkpoint path values",
        "embed Dockerfile; compose; image buildspec",
        f"LDA_CKPT_PATH=/ckpt/{CHECKPOINT_NAME}" in embed_dockerfile
        and embed_env.get("LDA_CKPT_PATH") == f"/ckpt/{CHECKPOINT_NAME}"
        and len(re.findall(
            rf"-e LDA_CKPT_PATH=/ckpt/{re.escape(CHECKPOINT_NAME)}\b", image_text
        )) >= 2,
        (
            f"image={f'/ckpt/{CHECKPOINT_NAME}' in embed_dockerfile}; "
            f"compose={embed_env.get('LDA_CKPT_PATH')}"
        ),
    )
    record(
        "CDK embed checkpoint environment",
        "infra/lda_infra/main_stack.py",
        (
            embed_stack_env.get("LDA_CKPT_PATH") == f"/tmp/model/{CHECKPOINT_NAME}"
            and embed_stack_env.get("LDA_CKPT_S3_KEY")
            == f"models/vjepa21/{CHECKPOINT_NAME}"
        ) if cdk_exists else None,
        "LDA_CKPT_PATH is under /tmp and the S3 key is configured"
        if cdk_exists else
        "no infra/ in bundle; enforced on the working tree by gate G2",
    )
    record(
        "checkpoint mounts and checksum",
        "compose; buildspec-images; buildspec-e2e",
        "/ckpt" in embed_targets
        and any(str(volume).endswith(":/ckpt:ro") for volume in embed_compose.get("volumes", []))
        and len(re.findall(r"-v /tmp/ckpt:/ckpt:ro", image_text)) >= 2
        and f"LDA_CKPT_SHA256: {EXPECTED_SHA}" in text("local/docker-compose.yml")
        and image_text.count(f"LDA_CKPT_SHA256={EXPECTED_SHA}") >= 2
        and e2e_text.count(f"LDA_CKPT_SHA256={EXPECTED_SHA}") >= 1
        and "/ckpt:ro" in text("local/docker-compose.yml")
        and "export LDA_CKPT_DIR=/tmp/lda/ckpt" in e2e_text
        and f"models/vjepa21/manifest.json" in text("services/embed/app.py"),
        "local/buildspec paths mount /ckpt read-only; buildspec and compose use the expected SHA; ECS resolves SHA from manifest",
    )
    record(
        "checkpoint fail-fast contract",
        "services/embed/app.py",
        'raise RuntimeError("LDA_CKPT_PATH is required")' in text("services/embed/app.py")
        and "LDA_CKPT_S3_KEY is required to download it" in text("services/embed/app.py")
        and "LDA_REQUIRE_CKPT" in text("tests/test_embed_inprocess.py")
        and "LDA_REQUIRE_CKPT" in text("tests/test_encoder_smoke.py"),
        "path required; absent file requires an S3 key; strict in-image mode fails instead of skipping",
    )

    storage_services = []
    for service, spec in compose["services"].items():
        env = spec.get("environment", {})
        if isinstance(env, dict) and str(env.get("LDA_STORAGE", "")).startswith("file:/store"):
            storage_services.append(service)
            if "/store" not in compose_mount_targets(spec):
                record("file store mounts", f"compose service {service}", False, "missing /store mount")
    record(
        "file store mounts",
        "compose services; buildspec-e2e",
        all("/store" in compose_mount_targets(compose["services"][service])
            for service in storage_services)
        and "-v /tmp/lda/local_runs/store:/store" in e2e_text
        and "chmod -R a+rwX /tmp/lda/local_runs" in e2e_text,
        f"file:/store services={','.join(storage_services)}; host store is mounted and chmodded",
    )

    run_event_positions = [
        match.start() for match in re.finditer(r"scripts/local_run_event\.py", e2e_text)
    ]
    chmod_positions = [
        match.start() for match in re.finditer(r"chmod -R a\+rwX /tmp/lda/local_runs", e2e_text)
    ]
    chmod_before_each = len(run_event_positions) == 2 and all(
        any(chmod > max(
            [position for position in run_event_positions if position < event_position]
            or [-1]
        ) and chmod < event_position for chmod in chmod_positions)
        for event_position in run_event_positions
    )
    record(
        "store permissions before queue calls",
        "buildspec-e2e",
        chmod_before_each
        and "chmod -R a+rwX /tmp/lda/local_runs" in e2e_text[:e2e_text.index("up -d --wait")],
        "a+rwX is applied before compose up and separately before both run_event calls",
    )

    host_run_mkdirs = []
    for path in py_files("scripts"):
        if path.name == "check_config_consistency.py":
            continue
        source = path.read_text()
        for line_no, line in enumerate(source.splitlines(), 1):
            if ".mkdir(" in line and re.search(r"(?:runs\s*/|/runs/|['\"]runs['\"]\s*/)", line):
                host_run_mkdirs.append(f"{path.relative_to(ROOT)}:{line_no}")
    record(
        "host does not create run result directories",
        "scripts/*.py",
        not host_run_mkdirs,
        ", ".join(host_run_mkdirs) or "no mkdir calls target runs/<run_id>",
    )

    uid_ok = all(
        "USER appuser" in dockerfile
        and "useradd --create-home --uid 10001 appuser" in dockerfile
        for dockerfile in (embed_dockerfile, coordinator_dockerfile)
    )
    root_override = re.search(
        r"(?:--user\s+(?:0|root)\b|^\s*user:\s*[\"']?(?:0|root)\b)",
        image_text + e2e_text + text("local/docker-compose.yml")
        + text("local/docker-compose.codebuild.yml") + cdk_source,
        re.MULTILINE,
    )
    record(
        "container uid and root overrides",
        "embed/coordinator Dockerfiles; buildspecs; compose",
        uid_ok and root_override is None,
        "embed and coordinator use appuser uid 10001; no configured root override",
    )
    record(
        "CDK task definition user overrides",
        "infra/lda_infra/main_stack.py",
        not re.search(r"(?:--user\s+(?:0|root)\b|user\s*=\s*[\"']?(?:0|root)\b)", cdk_source)
        if cdk_exists else None,
        "no root user override in task definitions"
        if cdk_exists else
        "no infra/ in bundle; enforced on the working tree by gate G2",
    )

    read_sources = [path.read_text() for folder in ("services", "lda_core") for path in py_files(folder)]
    read_names = set().union(*(env_reads(source) for source in read_sources))
    compose_names: set[str] = set()
    add_compose_environment(compose, compose_names)
    add_compose_environment(compose_override, compose_names)
    buildspec_names = set(re.findall(r"(?:-e|export)\s+(LDA_[A-Z0-9_]+)(?:=|\s)", image_text + e2e_text))
    dockerfile_names = set(re.findall(r"\b(LDA_[A-Z0-9_]+)=", embed_dockerfile + coordinator_dockerfile))
    cdk_names = set(re.findall(r'"(LDA_[A-Z0-9_]+)"\s*:', cdk_source))
    cdk_names.update(re.findall(r'add_environment\(\s*"(LDA_[A-Z0-9_]+)"', cdk_source))
    configured_names = compose_names | buildspec_names | dockerfile_names | cdk_names
    optional_readers = {
        "LDA_VJEPA_REPO": "The vendored encoder repository is the default; this override is only for local diagnostics.",
        "LDA_EMBED_IMAGE": "Optional provenance metadata attached to a locally built bank; runtime image identity is not required.",
        "LDA_BUCKET": "AWS storage mode receives this from ECS/Lambda definitions in infra/; gate G2 verifies it on the working tree.",
        "LDA_CKPT_S3_KEY": "ECS downloads a missing checkpoint from this key; infra/ is absent from the bundle and gate G2 verifies it on the working tree.",
        "LDA_QUEUE_URL": "The AWS SQS queue URL is set by the coordinator task definition; the bundle runs the file queue.",
        "LDA_KNN_FUNCTION": "The AWS Lambda function name is set by the coordinator task definition; compose uses LDA_KNN_URL.",
        "LDA_TICKET_FUNCTION": "The AWS Lambda function name is set by the coordinator task definition; compose uses LDA_TICKET_URL.",
        "LDA_TICKET_TABLE": "The DynamoDB table name is set by ECS/Lambda definitions in infra/; gate G2 verifies it on the working tree.",
        "LDA_LEASE_PARAMETER": "The reaper's SSM parameter name is set by its CDK function environment; gate G2 verifies it on the working tree.",
        "LDA_SERVICES": "The reaper's service list is set by its CDK function environment; gate G2 verifies it on the working tree.",
    }
    missing_names = sorted(read_names - configured_names - set(optional_readers))
    record(
        "runtime LDA environment coverage",
        "services; lda_core; compose; buildspecs; CDK",
        not missing_names,
        "all runtime reads are configured or documented optional"
        + (f"; missing={missing_names}" if missing_names else ""),
    )
    warnings = sorted(configured_names - read_names)
    for name in warnings:
        record("WARNING set but not read", "configuration", True, name)
    for name, reason in sorted(optional_readers.items()):
        record("OPTIONAL LDA environment", "services/lda_core", True, f"{name}: {reason}")

    source_files = [
        path for folder in ("tests", "services", "lda_core") for path in py_files(folder)
    ]
    forbidden = (
        "Path" + ".home(",
        "expand" + "user(",
        "~/" + ".cache/",
    )
    home_hits: list[str] = []
    parent_hits: list[str] = []
    parent_pattern = re.compile(r"(?:\.parents|_parents)\s*\[\s*(\d+)\s*\]")
    for path in source_files:
        source = path.read_text()
        for token in forbidden:
            if token in source:
                home_hits.append(f"{path.relative_to(ROOT)} contains {token}")
        for match in parent_pattern.finditer(source):
            index = int(match.group(1))
            if index < 2:
                continue
            is_guarded_root = (
                path == ROOT / "tests/conftest.py"
                and index == 4
                and re.search(r"len\(_parents\)\s*>\s*4\s*else\s*None", source)
            )
            if not is_guarded_root:
                parent_hits.append(f"{path.relative_to(ROOT)} index {index}")
    record(
        "runtime and test paths avoid home defaults",
        "tests; services; lda_core",
        not home_hits,
        ", ".join(home_hits) or "no home, expanduser, or ~/.cache path defaults",
    )
    record(
        "test repository roots are depth-safe",
        "tests",
        not parent_hits,
        ", ".join(parent_hits) or "only the guarded conftest parent lookup remains",
    )

    event_tree = yaml.safe_load(e2e_text)
    event_commands = "\n".join(event_tree["phases"]["build"]["commands"])
    event_traps = [
        line.strip() for line in e2e_text.splitlines() if line.strip().startswith("trap ")
    ]
    image_build = yaml.safe_load(image_text)
    image_commands = "\n".join(image_build["phases"]["build"]["commands"])
    pytest_pipelines = [
        (block, output)
        for block, output in (
            (image_commands, "/tmp/pytest_inimage_embed.txt"),
            (image_commands, "/tmp/pytest_inimage_coordinator.txt"),
        )
    ]
    pipefail_ok = all(
        "set -euo pipefail" in block and f"| tee {output}" in block
        for block, output in pytest_pipelines
    )
    record(
        "e2e waits, trap, and in-image pipefail",
        "buildspec-e2e; buildspec-images",
        "up -d --wait --wait-timeout 900" in event_commands
        and event_traps == ["trap on_exit EXIT"]
        and pipefail_ok,
        "compose wait is bounded; exactly one on_exit trap; in-image pytest pipelines fail on pipe errors",
    )

    errors = [row for row in checks if row[2] == "FAIL"]
    print("| Check | Files | Result |")
    print("|---|---|---|")
    for name, files, status, detail in checks:
        if name.startswith("WARNING"):
            status = "WARNING"
        elif name.startswith("OPTIONAL"):
            status = "OPTIONAL"
        print(f"| {name} | {files} | {status}: {detail} |")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(run_checks())
