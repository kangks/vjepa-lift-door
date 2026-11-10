from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_file_mode_exports_last_ending_server_spans_and_checks_two_requests(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "spans"
    script = r'''
import json
import sys
from pathlib import Path

from opentelemetry import baggage, context
from opentelemetry.trace import SpanKind

from lda_core.telemetry import (
    extract_context, init, init_lambda, inject_traceparent, start_server_span,
)

output_dir = Path(sys.argv[1])
coordinator = init_lambda("lift-door-coordinator")
embed = init("lda-embed")
knn = init_lambda("lda-knn")

for run_id in ("run-span-flush-1", "run-span-flush-2"):
    baggage_ctx = baggage.set_baggage("lift.run_id", run_id)
    token = context.attach(baggage_ctx)
    with coordinator.start_as_current_span(
        "invoke_agent lift-door-coordinator", kind=SpanKind.SERVER,
    ) as root:
        root.set_attribute("lift.run_id", run_id)
        root.set_attribute("lift.alert", False)
        root.set_attribute("lift.decision", "NO_ALERT_LOG_ONLY")

        with coordinator.start_as_current_span(
            "execute_tool detect", kind=SpanKind.CLIENT,
        ) as detect_client:
            detect_client.set_attribute("lift.run_id", run_id)
            carrier = inject_traceparent()
            with knn.start_as_current_span(
                "detect", context=extract_context(carrier), kind=SpanKind.SERVER,
            ) as detect_server:
                detect_server.set_attribute("lift.run_id", run_id)

        with coordinator.start_as_current_span(
            "execute_tool embed", kind=SpanKind.CLIENT,
        ) as embed_client:
            embed_client.set_attribute("lift.run_id", run_id)
            carrier = inject_traceparent()
            with start_server_span(embed, "embed", carrier) as server:
                server.set_attribute("lift.run_id", run_id)
                with embed.start_as_current_span("embed.decode_clip"):
                    pass
                with embed.start_as_current_span("embed.vjepa_encode_window"):
                    pass

            # SimpleSpanProcessor must have exported this last-ending server span
            # synchronously. Do not call flush() before this assertion.
            rows = [
                json.loads(line)
                for line in (output_dir / "lda-embed.jsonl").read_text().splitlines()
            ]
            assert any(
                row["name"] == "embed" and row["attributes"].get("lift.run_id") == run_id
                for row in rows
            )
    context.detach(token)
'''
    env = os.environ.copy()
    env["LDA_OTEL_MODE"] = "file"
    env["LDA_OTEL_FILE_DIR"] = str(output_dir)
    subprocess.run(
        [sys.executable, "-c", script, str(output_dir)],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    for run_id in ("run-span-flush-1", "run-span-flush-2"):
        result = subprocess.run(
            [
                sys.executable, str(root / "scripts/check_trace.py"),
                "--otel-dir", str(output_dir), "--run-id", run_id,
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**env, "PYTHONPATH": str(root)},
        )
        assert "3d parent links resolve | PASS | orphans=[]" in result.stdout
        assert "3b embed client and downstream span | PASS" in result.stdout
