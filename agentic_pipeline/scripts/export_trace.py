from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lda_core.output_utils import read_stack_outputs


ROOT_SPAN = "invoke_agent lift-door-coordinator"
POLL_SECONDS = 30
POLL_BUDGET_SECONDS = 600
QUERY_FIELDS = (
    "traceId",
    "spanId",
    "parentSpanId",
    "name",
    "kind",
    "resource.attributes.service.name",
    "attributes.lift.run_id",
    "attributes.lift.clip_key",
    "attributes.lift.alert",
    "attributes.lift.max_score_over_tau",
    "attributes.lift.window_s",
    "attributes.lift.decision",
    "attributes.lift.classification",
    "attributes.lift.ticket_id",
    "attributes.gen_ai.operation.name",
    "attributes.gen_ai.agent.name",
    "attributes.gen_ai.tool.name",
    "attributes.gen_ai.provider.name",
    "attributes.gen_ai.system",
    "attributes.gen_ai.request.model",
    "attributes.gen_ai.usage.input_tokens",
    "attributes.gen_ai.usage.output_tokens",
    "attributes.openinference.span.kind",
    "startTimeUnixNano",
    "endTimeUnixNano",
)


def _outputs(path: Path, stack_name: str):
    return read_stack_outputs(path, stack_name)


def _attribute_value(name: str, value):
    if value is None or value == "":
        return None
    if name == "lift.alert":
        if isinstance(value, bool):
            return value
        lowered = str(value).strip().lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
    if name in {"gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens"}:
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if name in {"lift.max_score_over_tau", "startTimeUnixNano", "endTimeUnixNano"}:
        try:
            return float(value) if "." in str(value) else int(value)
        except (TypeError, ValueError):
            return value
    if name == "lift.window_s":
        try:
            return json.loads(value) if isinstance(value, str) else value
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def logs_insights_row_to_segment(row: list[dict]) -> dict:
    """Convert a Logs Insights field/value row to check_trace's X-Ray segment shape."""
    if isinstance(row, dict):
        fields = dict(row)
    else:
        fields = {
            cell["field"]: cell.get("value")
            for cell in row
            if isinstance(cell, dict) and cell.get("field")
        }

    trace_id = fields.get("traceId")
    span_id = fields.get("spanId")
    if not trace_id or not span_id:
        raise ValueError("Logs Insights span row is missing traceId or spanId")

    attributes = {}
    for field_name in QUERY_FIELDS:
        if not field_name.startswith("attributes."):
            continue
        key = field_name[len("attributes."):]
        value = _attribute_value(key, fields.get(field_name))
        if value is not None:
            attributes[key] = value

    segment = {
        "trace_id": str(trace_id),
        "id": str(span_id),
        "parent_id": fields.get("parentSpanId") or None,
        "name": fields.get("name") or "",
        "kind": fields.get("kind"),
        "origin": fields.get("resource.attributes.service.name") or "",
        "annotations": attributes,
    }
    for field_name, output_name in (
        ("startTimeUnixNano", "start_time_unix_nano"),
        ("endTimeUnixNano", "end_time_unix_nano"),
    ):
        if fields.get(field_name) not in (None, ""):
            segment[output_name] = _attribute_value(field_name, fields[field_name])
    return segment


def _query(run_id: str, trace_id: str) -> str:
    return (
        "fields " + ", ".join(QUERY_FIELDS)
        + f'\n| filter attributes.lift.run_id = "{run_id}" or traceId = "{trace_id}"'
        + "\n| sort startTimeUnixNano asc | limit 200"
    )


def _expected_services(result: dict) -> set[str]:
    expected = {"lda-coordinator", "lda-embed", "lda-knn"}
    if (result.get("decision") or {}).get("decision") == "TICKET":
        expected.add("lda-ticket")
    return expected


def _coverage(segments: list[dict], run_id: str, expected_services: set[str]) -> tuple[bool, set[str]]:
    matching = [
        segment for segment in segments
        if segment.get("annotations", {}).get("lift.run_id") == run_id
    ]
    services = {segment.get("origin") for segment in matching if segment.get("origin")}
    has_root = any(
        segment.get("name") == ROOT_SPAN
        and segment.get("origin") == "lda-coordinator"
        for segment in matching
    )
    return has_root and expected_services <= services, services


def _query_results(logs, query: str, profile: str, deadline: float):
    now = int(time.time())
    query_id = logs.start_query(
        logGroupName="aws/spans",
        startTime=now - 3600,
        endTime=now + 60,
        queryString=query,
    )["queryId"]
    query_deadline = min(deadline, time.monotonic() + POLL_SECONDS - 5)
    while time.monotonic() < query_deadline:
        response = logs.get_query_results(queryId=query_id)
        status = response["status"]
        if status == "Complete":
            return response.get("results", [])
        if status in {"Failed", "Cancelled", "Timeout"}:
            raise RuntimeError(f"Logs Insights query status: {status}")
        time.sleep(min(2, max(0, query_deadline - time.monotonic())))
    return []


def _evidence_dir(profile: str) -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / ".claude/worktrees/vjepa-talk-draft/agentic_plan/evidence"
        / profile
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    build_out = _outputs(
        root / "outputs" / f"{args.profile}-build.json", "LiftDoorAgenticBuild",
    )
    bucket = build_out.get("BucketName") or build_out.get("bucket")
    if not bucket:
        raise ValueError("build outputs are missing BucketName")

    import boto3
    session = boto3.Session(profile_name=args.profile, region_name="us-east-1")
    s3 = session.client("s3")
    logs = session.client("logs")
    xray = session.client("xray")
    result = json.loads(s3.get_object(
        Bucket=bucket, Key=f"runs/{args.run_id}/result.json",
    )["Body"].read())
    trace_id = result["trace_id"]
    expected_services = _expected_services(result)
    query = _query(args.run_id, trace_id)
    print(f"Logs Insights query:\n{query}", flush=True)

    deadline = time.monotonic() + POLL_BUDGET_SECONDS
    attempt = 0
    segments: list[dict] = []
    services: set[str] = set()
    complete = False
    while time.monotonic() < deadline:
        attempt += 1
        attempt_started = time.monotonic()
        rows = _query_results(logs, query, args.profile, deadline)
        segments = [logs_insights_row_to_segment(row) for row in rows]
        complete, services = _coverage(segments, args.run_id, expected_services)
        missing = sorted(expected_services - services)
        print(
            f"span poll {attempt}: rows={len(rows)} root="
            f"{any(row.get('name') == ROOT_SPAN for row in segments)} "
            f"services={sorted(services)} missing={missing}",
            flush=True,
        )
        if complete:
            break
        delay = min(
            max(0, POLL_SECONDS - (time.monotonic() - attempt_started)),
            max(0, deadline - time.monotonic()),
        )
        if delay:
            time.sleep(delay)

    evidence = _evidence_dir(args.profile)
    evidence.mkdir(parents=True, exist_ok=True)
    span_path = evidence / f"trace_{args.run_id}.json"
    span_path.write_text(json.dumps(segments, indent=2))
    if not complete:
        raise TimeoutError(
            f"trace {trace_id} missing root/downstream services after "
            f"{POLL_BUDGET_SECONDS}s; services={sorted(services)}"
        )

    xray_id = f"1-{trace_id[:8]}-{trace_id[8:]}"
    traces = xray.batch_get_traces(TraceIds=[xray_id]).get("Traces", [])
    xray_path = evidence / f"xray_{args.run_id}.json"
    xray_path.write_text(json.dumps(traces, indent=2))
    if not traces:
        raise RuntimeError(f"X-Ray returned no trace for {xray_id}")

    checker = root / "scripts/check_trace.py"
    subprocess.run(
        [
            sys.executable, str(checker), "--trace-json", str(span_path),
            "--run-id", args.run_id,
        ],
        check=True,
    )
    print(json.dumps({
        "run_id": args.run_id,
        "trace_id": trace_id,
        "xray_trace_id": xray_id,
        "services": sorted(services),
        "span_export": str(span_path),
        "xray_export": str(xray_path),
    }, indent=2))


if __name__ == "__main__":
    main()
