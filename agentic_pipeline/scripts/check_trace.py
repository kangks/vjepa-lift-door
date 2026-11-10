from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


ROOT_SPAN = "invoke_agent lift-door-coordinator"


def _span_from_otel(value: dict) -> dict:
    return {
        "trace_id": value.get("traceId") or value.get("trace_id"),
        "span_id": value.get("spanId") or value.get("span_id"),
        "parent_id": value.get("parentSpanId") or value.get("parent_span_id"),
        "name": value.get("name", ""),
        "attributes": value.get("attributes") or {},
        "resource": value.get("resource") or {},
    }


def _xray_segments(value) -> list[dict]:
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict):
        rows = value.get("Traces", value.get("traces", [value]))
    else:
        raise ValueError("trace JSON must be an object or array")
    output = []

    def add_segment(raw):
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            return
        annotations = dict(raw.get("annotations") or {})
        metadata = raw.get("metadata") or {}
        for section in metadata.values():
            if isinstance(section, dict):
                annotations.update(section)
        output.append({
            "trace_id": raw.get("trace_id") or raw.get("traceId"),
            "span_id": raw.get("id") or raw.get("span_id"),
            "parent_id": raw.get("parent_id") or raw.get("parentSpanId"),
            "name": raw.get("name", ""),
            "attributes": annotations,
            "resource": {"attributes": {"service.name": raw.get("origin") or raw.get("name", "")}},
        })
        for child in raw.get("subsegments", []):
            add_segment(child)

    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("Segments"), list):
            for segment in row["Segments"]:
                if isinstance(segment, dict):
                    add_segment(segment.get("Document", segment))
                else:
                    add_segment(segment)
        elif isinstance(row, dict):
            add_segment(row.get("Document", row))
    return output


def load_jsonl(otel_dir: Path) -> list[dict]:
    spans = []
    for path in sorted(otel_dir.glob("*.jsonl")):
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                spans.append(_span_from_otel(value))
    return spans


def check_spans(spans: list[dict], run_id: str) -> list[tuple[str, bool, str]]:
    matching = [
        span for span in spans
        if str(span.get("attributes", {}).get("lift.run_id", "")) == run_id
    ]
    rows: list[tuple[str, bool, str]] = []
    trace_ids = {span.get("trace_id") for span in matching if span.get("trace_id")}
    rows.append(("3a one traceId for run", len(matching) > 0 and len(trace_ids) == 1,
                 f"spans={len(matching)} traceIds={len(trace_ids)}"))
    if not matching:
        return rows

    trace_spans = [
        span for span in spans
        if span.get("trace_id") in trace_ids and span.get("span_id")
    ]
    spans_by_id = {span["span_id"]: span for span in trace_spans}
    ids = set(spans_by_id)
    roots = [span for span in matching if span.get("name") == ROOT_SPAN]
    rows.append(("3b coordinator root present", len(roots) == 1, f"count={len(roots)}"))
    root = roots[0] if roots else None
    rows.append(("3d root has no parent", root is not None and not root.get("parent_id"),
                 f"parent={root.get('parent_id') if root else None}"))
    orphans = [
        span for span in matching
        if span.get("name") != ROOT_SPAN and span.get("parent_id") not in ids
    ]
    rows.append(("3d parent links resolve", not orphans,
                 f"orphans={[span.get('name') for span in orphans]}"))

    def descends_from(span, ancestor) -> bool:
        parent_id = span.get("parent_id")
        visited = set()
        while parent_id and parent_id not in visited:
            if parent_id == ancestor.get("span_id"):
                return True
            visited.add(parent_id)
            parent = spans_by_id.get(parent_id)
            if parent is None:
                return False
            parent_id = parent.get("parent_id")
        return False

    def service(span):
        return (span.get("resource", {}).get("attributes", {}).get("service.name")
                or span.get("resource", {}).get("service.name"))

    for tool, server_name, server_service in (
        ("embed", "embed", "lda-embed"),
        ("detect", "detect", "lda-knn"),
    ):
        clients = [span for span in matching if span.get("name") == f"execute_tool {tool}"]
        servers = [span for span in matching if span.get("name") == server_name and service(span) == server_service]
        parented = bool(clients and servers and any(
            descends_from(server, client)
            for client in clients for server in servers
        ))
        rows.append((f"3b {tool} client and downstream span", bool(clients and servers and parented),
                     f"clients={len(clients)} downstream={len(servers)} propagated_parent={parented}"))

    alert_values = [
        span["attributes"]["lift.alert"]
        for span in matching if "lift.alert" in span.get("attributes", {})
    ]
    alert = any(value is True for value in alert_values)
    reasoner = [
        span for span in matching
        if span.get("name", "").startswith("chat anthropic.claude-opus-5-5")
        or span.get("name") == "reasoner.dry_run"
    ]
    rows.append(("3c reasoner branch matches alert", bool(reasoner) if alert else not reasoner,
                 f"alert={alert} reasoner_spans={len(reasoner)}"))

    ticket_clients = [
        span for span in matching if span.get("name") == "execute_tool create_ticket"
    ]
    ticket_servers = [
        span for span in matching
        if span.get("name") in {"create_ticket", "handler"}
        and service(span) in {"lda-ticket", "aws.lambda"}
    ]
    decision = root.get("attributes", {}).get("lift.decision") if root else None
    ticket_expected = decision == "TICKET"
    has_ticket = bool(ticket_clients and ticket_servers)
    ticket_parented = bool(ticket_clients and ticket_servers and any(
        descends_from(server, client)
        for client in ticket_clients for server in ticket_servers
    ))
    rows.append(("3c ticket branch matches decision",
                 (has_ticket and ticket_parented)
                 if ticket_expected else not ticket_clients and not ticket_servers,
                 f"decision={decision} ticket_client={len(ticket_clients)} "
                 f"ticket_server={len(ticket_servers)} propagated_parent={ticket_parented}"))
    return rows


def check_trace(otel_dir: Path | None, trace_json: Path | None, run_id: str):
    if bool(otel_dir) == bool(trace_json):
        raise ValueError("provide exactly one of --otel-dir or --trace-json")
    spans = load_jsonl(otel_dir) if otel_dir else _xray_segments(json.loads(trace_json.read_text()))
    return check_spans(spans, run_id)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check one local or exported trace against ACCEPTANCE §3.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--otel-dir", type=Path)
    source.add_argument("--trace-json", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--settle-seconds", type=float, default=0)
    args = parser.parse_args(argv)
    if args.settle_seconds < 0:
        parser.error("--settle-seconds must be non-negative")
    rows = check_trace(args.otel_dir, args.trace_json, args.run_id)
    if args.settle_seconds and any(
        check == "3d parent links resolve" and not passed
        for check, passed, _ in rows
    ):
        time.sleep(args.settle_seconds)
        rows = check_trace(args.otel_dir, args.trace_json, args.run_id)
    print("| Check | Result | Detail |")
    print("|---|---|---|")
    for check, passed, detail in rows:
        print(f"| {check} | {'PASS' if passed else 'FAIL'} | {detail} |")
    return 0 if all(passed for _, passed, _ in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
