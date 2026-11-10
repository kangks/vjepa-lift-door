import json
import os
import subprocess
import sys


def test_file_exporter_propagates_context_and_truncates_attributes(tmp_path):
    output_dir = tmp_path / "spans"
    script = r'''
import os
from opentelemetry import baggage, context
from opentelemetry.trace import SpanKind
from lda_core.telemetry import extract_context, flush, init_lambda, inject_traceparent, set_genai_attrs
tracer = init_lambda("telemetry-smoke")
bag = baggage.set_baggage("lift.run_id", "run-test-telemetry")
token = context.attach(bag)
with tracer.start_as_current_span("root", kind=SpanKind.SERVER) as root:
    set_genai_attrs(root, **{"lift.long": "x" * 5000})
    carrier = inject_traceparent()
    extracted = extract_context(carrier)
    with tracer.start_as_current_span("child", context=extracted):
        pass
context.detach(token)
flush()
'''
    env = os.environ.copy()
    env["LDA_OTEL_MODE"] = "file"
    env["LDA_OTEL_FILE_DIR"] = str(output_dir)
    subprocess.run([sys.executable, "-c", script], check=True, env=env, capture_output=True, text=True)
    rows = [json.loads(line) for line in (output_dir / "telemetry-smoke.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    root = next(row for row in rows if row["name"] == "root")
    child = next(row for row in rows if row["name"] == "child")
    assert root["attributes"]["lift.run_id"] == "run-test-telemetry"
    assert len(root["attributes"]["lift.long"]) == 4096
    assert root["traceId"] == child["traceId"]
    assert root["spanId"] == child["parentSpanId"]
    assert root["resource"]["attributes"]["service.name"] == "telemetry-smoke"
