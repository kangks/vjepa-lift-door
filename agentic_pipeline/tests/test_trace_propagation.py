from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_trace_parentage_across_embed_detect_ticket_and_adot_server(tmp_path: Path) -> None:
    spans_dir = tmp_path / "spans"
    script = r'''
import importlib
import json
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from opentelemetry import baggage, context, trace
from opentelemetry.trace import SpanKind

import lda_core.telemetry as telemetry
from lda_core.telemetry import init, inject_traceparent, set_genai_attrs, start_server_span

run_id = "run-test-trace-propagation"
coordinator = importlib.import_module("services.coordinator.main")
knn = importlib.import_module("services.knn.handler")
ticket = importlib.import_module("services.ticket.handler")
embed_tracer = init("lda-embed")

knn._detect = lambda event, storage: {
    "alert": True,
    "max_score_over_tau": 0.5,
    "window_s": [3.0, 7.0],
    "bank_mode": "loo",
}

app = FastAPI()

@app.post("/embed")
async def embed(payload: dict, request: Request):
    with start_server_span(embed_tracer, "embed", dict(request.headers)) as span:
        set_genai_attrs(span, **{"lift.run_id": payload["run_id"]})
    return {"ok": True}

client = TestClient(app)
baggage_context = baggage.set_baggage("lift.run_id", run_id)
baggage_token = context.attach(baggage_context)
with coordinator.TRACER.start_as_current_span(
    "invoke_agent lift-door-coordinator", kind=SpanKind.SERVER
) as root:
    root_span_id = f"{root.get_span_context().span_id:016x}"
    set_genai_attrs(root, **{"lift.run_id": run_id, "lift.clip_key": "incoming/test/clip.mp4"})
    with coordinator.TRACER.start_as_current_span("execute_tool embed", kind=SpanKind.CLIENT):
        embed_headers = inject_traceparent()
        assert "traceparent" in embed_headers and "baggage" in embed_headers
        response = client.post(
            "/embed",
            json={"run_id": run_id, "clip_key": "incoming/test/clip.mp4"},
            headers=embed_headers,
        )
        assert response.status_code == 200, response.text

    with coordinator.TRACER.start_as_current_span("execute_tool detect", kind=SpanKind.CLIENT):
        detect_result = knn.handler({
            "mode": "detect",
            "run_id": run_id,
            "_otel": inject_traceparent(),
        })
        assert detect_result["alert"] is True

    with coordinator.TRACER.start_as_current_span(
        "chat anthropic.claude-opus-5-5"
    ) as reasoner_span:
        reasoner_span.set_attribute("lift.run_id", run_id)

    with coordinator.TRACER.start_as_current_span("execute_tool create_ticket", kind=SpanKind.CLIENT):
        ticket_event = {
            "run_id": run_id,
            "clip_key": "incoming/test/clip.mp4",
            "classification": "UNCERTAIN",
            "window_s": None,
            "recommended_action": "HUMAN_REVIEW",
            "trace_id": telemetry.trace_id_hex(root),
            "_otel": inject_traceparent(),
        }
        ticket_result = ticket.handler(ticket_event)
        assert ticket_result["ticket_id"]
    set_genai_attrs(root, **{"lift.decision": "TICKET"})
context.detach(baggage_token)

# Simulate ADOT's active FastAPI SERVER span. The helper should attach the
# explicit embed span underneath it when both carry the same valid trace ID.
with coordinator.TRACER.start_as_current_span("auto_client", kind=SpanKind.CLIENT):
    carrier = inject_traceparent()
    upper_carrier = {key.upper(): value for key, value in carrier.items()}
    extracted = telemetry.server_context_from_headers(upper_carrier)
    assert trace.get_current_span(extracted).get_span_context().is_valid
    with coordinator.TRACER.start_as_current_span(
        "adot_fastapi_server", context=extracted, kind=SpanKind.SERVER
    ) as auto_server:
        auto_server_id = f"{auto_server.get_span_context().span_id:016x}"
        with start_server_span(coordinator.TRACER, "embed_auto_child", carrier) as child:
            assert child.get_span_context().trace_id == auto_server.get_span_context().trace_id

for provider in tuple(telemetry._providers.values()):
    provider.force_flush(timeout_millis=5000)

coordinator_rows = [
    json.loads(line)
    for line in (Path(os.environ["LDA_OTEL_FILE_DIR"]) / "lift-door-coordinator.jsonl")
    .read_text().splitlines()
]
child_row = next(row for row in coordinator_rows if row["name"] == "embed_auto_child")
assert child_row["parentSpanId"] == auto_server_id, child_row
reasoner_row = next(
    row for row in coordinator_rows if row["name"] == "chat anthropic.claude-opus-5-5"
    and row["attributes"].get("lift.run_id") == run_id
)
assert reasoner_row["parentSpanId"] == root_span_id, reasoner_row
'''
    env = os.environ.copy()
    env.update({
        "LDA_OTEL_MODE": "file",
        "LDA_OTEL_FILE_DIR": str(spans_dir),
        "LDA_STORAGE": f"file:{tmp_path / 'store'}",
        "LDA_TICKET_STORE": f"file:{tmp_path / 'tickets'}",
    })
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    checked = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_trace.py"),
            "--otel-dir",
            str(spans_dir),
            "--run-id",
            "run-test-trace-propagation",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = checked.stdout + checked.stderr
    assert checked.returncode == 0, output
    assert "| 3d parent links resolve | PASS | orphans=[] |" in output
    assert "| 3c reasoner branch matches alert | PASS | alert=True reasoner_spans=1 |" in output
    for hop in ("embed", "detect"):
        assert f"| 3b {hop} client and downstream span | PASS |" in output
        assert "propagated_parent=True" in output

    spans = [
        json.loads(line)
        for path in spans_dir.glob("*.jsonl")
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    ticket_client = next(
        span for span in spans
        if span["name"] == "execute_tool create_ticket"
        and span["attributes"].get("lift.run_id") == "run-test-trace-propagation"
    )
    ticket_server = next(
        span for span in spans
        if span["name"] == "create_ticket"
        and span["resource"]["attributes"].get("service.name") == "lda-ticket"
        and span["attributes"].get("lift.run_id") == "run-test-trace-propagation"
    )
    assert ticket_server["parentSpanId"] == ticket_client["spanId"], (
        "ticket propagated_parent=False", ticket_client, ticket_server,
    )


def test_embed_service_uses_explicit_request_context_extraction() -> None:
    embed_source = (ROOT / "services/embed/app.py").read_text()
    knn_source = (ROOT / "services/knn/handler.py").read_text()
    ticket_source = (ROOT / "services/ticket/handler.py").read_text()
    coordinator_source = (ROOT / "services/coordinator/main.py").read_text()
    assert "start_server_span(TRACER, \"embed\", dict(request.headers))" in embed_source
    assert "start_server_span(" in knn_source and 'event.get("_otel")' in knn_source
    assert 'start_server_span(TRACER, "create_ticket", event.get("_otel"))' in ticket_source
    for hop, next_hop in (
        ('with _tool_span("embed")', 'with _tool_span("detect")'),
        ('with _tool_span("detect")', 'result["detector"]'),
        ('with _tool_span("create_ticket")', 'ticket_result ='),
    ):
        start = coordinator_source.index(hop)
        end = coordinator_source.index(next_hop, start + len(hop))
        assert "inject_traceparent()" in coordinator_source[start:end], hop
    ticket_span = coordinator_source.index('with _tool_span("create_ticket")')
    assert coordinator_source.index(
        'ticket_event["_otel"] = inject_traceparent()', ticket_span,
    ) < coordinator_source.index("ticket_result =", ticket_span)
