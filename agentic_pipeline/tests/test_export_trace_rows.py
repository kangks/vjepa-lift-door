from __future__ import annotations

import json

import pytest

from scripts.check_trace import _xray_segments, check_spans
from scripts.export_trace import QUERY_FIELDS, logs_insights_row_to_segment


@pytest.fixture
def step07_rows_47():
    """Real-shaped field/value rows based on STEP_07's 47-span trace structure."""
    run_id = "run-test-step07-rows"
    trace_id = "0123456789abcdef0123456789abcdef"
    root_id = "0000000000000001"

    def row(span_id, parent_id, name, service, kind, attributes=None, index=0):
        values = {
            "traceId": trace_id,
            "spanId": span_id,
            "parentSpanId": parent_id or "",
            "name": name,
            "kind": kind,
            "resource.attributes.service.name": service,
            "attributes.lift.run_id": run_id,
            "attributes.lift.clip_key": "incoming/run-test-step07-rows/clip.mp4",
            "attributes.lift.alert": "",
            "attributes.lift.max_score_over_tau": "",
            "attributes.lift.window_s": "",
            "attributes.lift.decision": "",
            "attributes.lift.classification": "",
            "attributes.lift.ticket_id": "",
            "attributes.gen_ai.operation.name": "",
            "attributes.gen_ai.agent.name": "",
            "attributes.gen_ai.tool.name": "",
            "attributes.gen_ai.provider.name": "",
            "attributes.gen_ai.system": "",
            "attributes.gen_ai.request.model": "",
            "attributes.gen_ai.usage.input_tokens": "",
            "attributes.gen_ai.usage.output_tokens": "",
            "attributes.openinference.span.kind": "",
            "startTimeUnixNano": str(1_700_000_000_000_000_000 + index * 10),
            "endTimeUnixNano": str(1_700_000_000_000_000_005 + index * 10),
        }
        values.update(attributes or {})
        return [
            {"field": field, "value": str(values.get(field, ""))}
            for field in QUERY_FIELDS
        ]

    rows = [
        row(root_id, None, "invoke_agent lift-door-coordinator", "lda-coordinator", "SERVER", {
            "attributes.gen_ai.operation.name": "invoke_agent",
            "attributes.gen_ai.agent.name": "lift-door-coordinator",
            "attributes.openinference.span.kind": "AGENT",
            "attributes.lift.decision": "TICKET",
        }, 0),
        row("0000000000000002", root_id, "execute_tool embed", "lda-coordinator", "CLIENT", {
            "attributes.gen_ai.operation.name": "execute_tool",
            "attributes.gen_ai.tool.name": "embed",
            "attributes.openinference.span.kind": "TOOL",
        }, 1),
        row("0000000000000003", "0000000000000002", "embed", "lda-embed", "SERVER", {}, 2),
        row("0000000000000004", root_id, "execute_tool detect", "lda-coordinator", "CLIENT", {
            "attributes.gen_ai.operation.name": "execute_tool",
            "attributes.gen_ai.tool.name": "detect",
            "attributes.openinference.span.kind": "TOOL",
        }, 3),
        row("0000000000000005", "0000000000000004", "detect", "lda-knn", "SERVER", {
            "attributes.lift.alert": "true",
            "attributes.lift.max_score_over_tau": "4.25",
            "attributes.lift.window_s": "[11.0, 15.0]",
        }, 4),
        row("0000000000000006", root_id, "chat anthropic.claude-opus-5-5", "lda-coordinator", "INTERNAL", {
            "attributes.gen_ai.operation.name": "chat",
            "attributes.gen_ai.provider.name": "aws.bedrock",
            "attributes.gen_ai.request.model": "anthropic.claude-opus-5-5",
            "attributes.gen_ai.usage.input_tokens": "7318",
            "attributes.gen_ai.usage.output_tokens": "1220",
            "attributes.openinference.span.kind": "LLM",
            "attributes.lift.classification": "SUSPECTED_MALFUNCTION",
        }, 5),
        row("0000000000000007", root_id, "execute_tool create_ticket", "lda-coordinator", "CLIENT", {
            "attributes.gen_ai.operation.name": "execute_tool",
            "attributes.gen_ai.tool.name": "create_ticket",
            "attributes.openinference.span.kind": "TOOL",
        }, 6),
        row("0000000000000008", "0000000000000007", "create_ticket", "lda-ticket", "SERVER", {
            "attributes.lift.ticket_id": "MOCK-step07",
        }, 7),
    ]
    rows.extend(
        row(
            f"{index:016x}", root_id, f"support span {index}", "lda-coordinator",
            "INTERNAL", {}, index,
        )
        for index in range(9, 48)
    )
    assert len(rows) == 47
    return rows, run_id


def test_export_trace_rows(step07_rows_47):
    rows, run_id = step07_rows_47
    segments = [logs_insights_row_to_segment(row) for row in rows]
    assert len(segments) == 47
    assert segments[0]["origin"] == "lda-coordinator"
    assert segments[0]["annotations"]["gen_ai.agent.name"] == "lift-door-coordinator"
    assert segments[4]["annotations"]["lift.alert"] is True
    assert segments[4]["annotations"]["lift.window_s"] == [11.0, 15.0]
    assert segments[5]["annotations"]["gen_ai.usage.input_tokens"] == 7318
    assert json.loads(json.dumps(segments[7]))["annotations"]["lift.ticket_id"] == "MOCK-step07"

    checked = check_spans(_xray_segments(segments), run_id)
    assert all(passed for _, passed, _ in checked), checked
