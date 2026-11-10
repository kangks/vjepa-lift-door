from __future__ import annotations

import json
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExportResult, SpanExporter, SimpleSpanProcessor
from opentelemetry.trace import SpanKind

from services.coordinator import main as coordinator

ROOT = Path(__file__).resolve().parents[1]


class _CollectingExporter(SpanExporter):
    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        return None


class _Storage:
    def __init__(self):
        self.objects = {}

    def put_bytes_if_absent(self, key, value, content_type):
        if key in self.objects:
            return False
        self.objects[key] = value
        return True

    def put_bytes(self, key, value, content_type):
        self.objects[key] = value


def _record(run_id: str) -> dict:
    return {
        "s3": {
            "bucket": {"name": "test-bucket"},
            "object": {"key": f"incoming/{run_id}/clip.mp4"},
        }
    }


def test_coordinator_two_deliveries_two_traces(monkeypatch):
    embed_source = (ROOT / "services/embed/app.py").read_text()
    response_keys = (
        "run_id",
        "frames_key",
        "tokens_key",
        "mp4_sha256",
        "frames_sha16",
        "seconds",
    )
    for key in response_keys:
        assert f'"{key}":' in embed_source

    provider = TracerProvider()
    exporter = _CollectingExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    coordinator_tracer = provider.get_tracer("lift-door-coordinator")
    sqs_tracer = provider.get_tracer("boto3sqs")
    monkeypatch.setattr(coordinator, "TRACER", coordinator_tracer)
    monkeypatch.setattr(coordinator, "flush", lambda: None)

    storage = _Storage()

    def embed_call(run_id, clip_key, headers):
        return {
            "run_id": run_id,
            "frames_key": f"runs/{run_id}/frames.npy",
            "tokens_key": f"runs/{run_id}/tokens.npy",
            "mp4_sha256": "a" * 64,
            "frames_sha16": "0" * 16,
            "seconds": {},
        }

    def detect_call(event):
        return {
            "alert": False,
            "alert_times_s": [],
            "peak_time_s": None,
            "peak_block_xy": None,
            "max_score_over_tau": 0.0,
            "window_s": None,
            "bank_mode": "full",
            "bank_prefix": "bank/v2/test",
        }

    with sqs_tracer.start_as_current_span("process prior SQS message", kind=SpanKind.CONSUMER) as active:
        first = coordinator._process_record(
            _record("run-test-first-0001"),
            storage=storage,
            embed_call=embed_call,
            detect_call=detect_call,
        )
        assert trace.get_current_span() is active
        second = coordinator._process_record(
            _record("run-test-second-0002"),
            storage=storage,
            embed_call=embed_call,
            detect_call=detect_call,
        )
        assert trace.get_current_span() is active

    roots = [
        span for span in exporter.spans
        if span.name == "invoke_agent lift-door-coordinator"
    ]
    assert len(roots) == 2
    assert len({span.context.trace_id for span in roots}) == 2
    assert all(span.parent is None for span in roots)
    assert {
        span.attributes.get("lift.run_id") for span in roots
    } == {first["run_id"], second["run_id"]}
    assert json.loads(storage.objects[f"runs/{first['run_id']}/result.json"])["trace_id"]
    assert json.loads(storage.objects[f"runs/{second['run_id']}/result.json"])["trace_id"]
