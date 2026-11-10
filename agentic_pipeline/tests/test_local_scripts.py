from __future__ import annotations

import json
import shutil
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests
import yaml

from lda_core.design import BANK
from scripts import local_build_bank
from scripts.check_trace import check_trace
from scripts.local_seed import seed_dataset
from scripts.parity_report import EXPECTED_SUMMARY_SHA256, build_report


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "fixtures" / "results_v2_summary.json"
RUN_ID = "run-20261106T120000-synthetic-a1b2"


def _span(name, span_id, parent, *, service="lift-door-coordinator", alert=None):
    attributes = {"lift.run_id": RUN_ID}
    if alert is not None:
        attributes["lift.alert"] = alert
    return {
        "traceId": "0123456789abcdef0123456789abcdef",
        "spanId": span_id,
        "parentSpanId": parent,
        "name": name,
        "attributes": attributes,
        "resource": {"attributes": {"service.name": service}},
    }


def test_local_seed_copies_one_clip_and_writes_models_manifest(tmp_path):
    data = tmp_path / "data"
    case = data / "healthy" / "fake_clip"
    case.mkdir(parents=True)
    (case / "fake_clip.mp4").write_bytes(b"fake mp4")
    (case / "ground_truth.json").write_text('{"frames":[]}\n')

    store = tmp_path / "store"
    result = seed_dataset(data, store)
    seeded = store / "dataset/lift_dataset/healthy/fake_clip"
    manifest = json.loads((store / "models/vjepa21/manifest.json").read_text())

    assert len(result["scenarios"]) == 1
    assert (seeded / "fake_clip.mp4").read_bytes() == b"fake mp4"
    assert json.loads((seeded / "ground_truth.json").read_text()) == {"frames": []}
    assert manifest["sha256"] == ""
    assert manifest["dataset"][0]["scenario"] == "fake_clip"


def test_check_trace_passes_five_spans_and_rejects_orphan_and_second_trace(tmp_path):
    spans = [
        _span("invoke_agent lift-door-coordinator", "root", None),
        _span("execute_tool embed", "embed-client", "root"),
        _span("embed", "embed-server", "embed-client", service="lda-embed"),
        _span("execute_tool detect", "knn-client", "root"),
        _span("detect", "knn-server", "knn-client", service="lda-knn", alert=False),
    ]
    otel_dir = tmp_path / "otel"
    otel_dir.mkdir()
    path = otel_dir / "synthetic.jsonl"
    path.write_text("\n".join(json.dumps(span) for span in spans) + "\n")
    checks = check_trace(otel_dir, None, RUN_ID)
    assert all(passed for _, passed, _ in checks), checks

    broken = [
        *spans,
        {
            **_span("orphaned", "orphan", "lost-parent", alert=False),
            "traceId": "fedcba9876543210fedcba9876543210",
        },
    ]
    path.write_text("\n".join(json.dumps(span) for span in broken) + "\n")
    broken_checks = check_trace(otel_dir, None, RUN_ID)
    assert not all(passed for _, passed, _ in broken_checks)
    assert not next(passed for name, passed, _ in broken_checks if name.startswith("3a"))
    assert not next(passed for name, passed, _ in broken_checks if name.startswith("3d parent"))


def test_parity_report_compares_fixture_and_rejects_modified_summary(tmp_path):
    reference = json.loads(SUMMARY.read_text())["detectors"]["D_patchKNN"]["envelope"][
        "anomaly_door_desync"
    ]
    store = tmp_path / "store"
    dataset_dir = store / "dataset/lift_dataset/healthy/anomaly_door_desync"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "ground_truth.json").write_text(json.dumps({
        "fps": 4,
        "frames": [{"anomaly": False, "observable": True} for _ in range(64)],
        "event_intervals": [
            {
                "type": event["type"],
                "start_sec": event["start"],
                "end_sec_exclusive": event["end"],
            }
            for event in reference["events"]
        ],
    }))
    run_dir = store / "runs" / RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / "scenario.json").write_text(json.dumps({"scenario": "anomaly_door_desync"}))
    (run_dir / "result.json").write_text(json.dumps({"run_id": RUN_ID, "clip_key": "incoming/clip.mp4"}))
    t0 = max(0.0, min(reference["peak_time_s"] - 2.0, 12.0))
    (run_dir / "detector.json").write_text(json.dumps({
        **reference,
        "alert": bool(reference["alert_times_s"]),
        "window_s": [t0, t0 + 4],
        "bank_mode": "full",
        "excluded_bank_clips": [],
    }))

    report = build_report(store, SUMMARY)
    assert f"summary sha256 `{EXPECTED_SUMMARY_SHA256}`" in report
    assert "anomaly_door_desync" in report
    assert "20.12" in report or "18.8344" in report
    assert "door_desync=HIT" in report

    corrupt = tmp_path / "wrong-summary.json"
    shutil.copyfile(SUMMARY, corrupt)
    corrupt.write_text(corrupt.read_text() + "\n")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        build_report(store, corrupt)


def test_compose_yaml_merge_is_internal_and_has_no_aws_credentials():
    base = yaml.safe_load((ROOT / "local/docker-compose.yml").read_text())
    override = yaml.safe_load((ROOT / "local/docker-compose.codebuild.yml").read_text())
    merged = {
        **base,
        "services": {
            name: {**base.get("services", {}).get(name, {}), **override.get("services", {}).get(name, {})}
            for name in set(base.get("services", {})) | set(override.get("services", {}))
        },
        "networks": {**base.get("networks", {}), **override.get("networks", {})},
    }

    assert set(merged["services"]) == {"embed", "knn", "ticket", "coordinator"}
    allowed_aws = {"AWS_EC2_METADATA_DISABLED"}
    for service, spec in merged["services"].items():
        environment = spec.get("environment", {})
        names = list(environment) if isinstance(environment, dict) else [
            str(entry).split("=", 1)[0] for entry in environment
        ]
        assert not [
            key for key in names
            if key.startswith("AWS_") and key not in allowed_aws
        ], service
        assert "ports" not in override.get("services", {}).get(service, {}), service
        assert "lda" in spec.get("networks", []), service
    assert merged["networks"]["lda"]["internal"] is True

    embed_healthcheck = merged["services"]["embed"]["healthcheck"]
    healthcheck_command = " ".join(embed_healthcheck["test"])
    assert "LDA_CKPT_SHA256" in healthcheck_command
    assert "ckpt_sha256" in healthcheck_command
    assert embed_healthcheck["interval"] == "10s"
    assert embed_healthcheck["timeout"] == "6s"
    assert embed_healthcheck["retries"] == 6
    assert embed_healthcheck["start_period"] == "600s"


def _make_bank_store(store: Path) -> None:
    for name in BANK:
        clip_dir = store / "dataset" / "lift_dataset" / "healthy" / name
        clip_dir.mkdir(parents=True, exist_ok=True)
        (clip_dir / f"{name}.mp4").write_bytes(b"fake mp4")
    manifest = store / "models" / "vjepa21" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"sha256": "expected-checkpoint-sha"}))


def _health_server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, server.server_address[1]


def test_local_build_bank_waits_for_embed_and_checks_knn_tcp(tmp_path, monkeypatch):
    expected_sha = "expected-checkpoint-sha"
    health_events = []

    class HealthHandler(BaseHTTPRequestHandler):
        request_count = 0

        def do_GET(self):
            type(self).request_count += 1
            code = 503 if type(self).request_count == 1 else 200
            health_events.append(str(code))
            body = (
                b'{"ckpt_sha256":"expected-checkpoint-sha"}'
                if code == 200
                else b'{"status":"starting"}'
            )
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]

    _make_bank_store(tmp_path / "store")
    server_holder = []

    def start_server_after_initial_refusal():
        time.sleep(0.05)
        server = ThreadingHTTPServer(("127.0.0.1", port), HealthHandler)
        server_holder.append(server)
        server.serve_forever()

    server_thread = threading.Thread(target=start_server_after_initial_refusal, daemon=True)
    server_thread.start()
    events = []
    attempts = []
    real_get = requests.get

    def recording_get(url, **kwargs):
        try:
            response = real_get(url, **kwargs)
        except requests.ConnectionError:
            attempts.append("refused")
            events.append("refused")
            raise
        attempts.append(str(response.status_code))
        events.append(str(response.status_code))
        return response

    monkeypatch.setattr(local_build_bank.requests, "get", recording_get)
    posted = []

    def fake_post(url, payload, timeout=900):
        events.append("post")
        posted.append((url, payload))
        if url.endswith("/embed"):
            return {
                "frames_key": "frames",
                "tokens_key": "tokens",
                "mp4_sha256": "0" * 64,
            }
        return {"built": True}

    monkeypatch.setattr(local_build_bank, "_post_json", fake_post)
    try:
        result = local_build_bank.build_bank(
            tmp_path / "store",
            embed_url=f"http://127.0.0.1:{port}",
            knn_url=f"http://127.0.0.1:{port}/knn",
            build_id="readiness-test",
            ready_timeout=2,
            poll_interval=0.01,
        )
    finally:
        if server_holder:
            server_holder[0].shutdown()
            server_holder[0].server_close()
        server_thread.join(timeout=1)

    assert result == {"built": True}
    assert attempts[0] == "refused"
    assert "503" in attempts
    assert attempts[-1] == "200"
    assert health_events == ["503", "200"]
    assert len(posted) == len(BANK) + 1
    assert events.index("post") > max(
        index for index, status in enumerate(events) if status == "200"
    )
    assert posted[-1][1]["mode"] == "build_bank"
    assert posted[-1][1]["ckpt_sha256"] == expected_sha


def test_local_build_bank_times_out_when_embed_checkpoint_sha_is_wrong(tmp_path, monkeypatch):
    class WrongShaHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"ckpt_sha256":"wrong-checkpoint-sha"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    server, thread, port = _health_server(WrongShaHandler)
    posted = []
    monkeypatch.setattr(
        local_build_bank,
        "_post_json",
        lambda *args, **kwargs: posted.append((args, kwargs)),
    )
    _make_bank_store(tmp_path / "store")
    try:
        with pytest.raises(TimeoutError, match="embed readiness timed out"):
            local_build_bank.build_bank(
                tmp_path / "store",
                embed_url=f"http://127.0.0.1:{port}",
                knn_url=f"http://127.0.0.1:{port}/knn",
                ready_timeout=0.05,
                poll_interval=0.005,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
    assert posted == []
