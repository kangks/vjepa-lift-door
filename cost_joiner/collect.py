"""Read-only collection of every input the report needs, written to out/raw/*.json.

AWS calls made here are Get*/List*/Describe*/BatchGet* and Pricing only:
  pricing:GetProducts, s3:ListObjectsV2, s3:GetObject (batch/ manifests only),
  cloudwatch:GetMetricStatistics, lambda:ListFunctions, xray:BatchGetTraces,
  logs:DescribeLogStreams, logs:GetLogEvents (Lambda REPORT lines only),
  states:ListExecutions, states:GetExecutionHistory, ssm:GetParameter (Attribute token,
  in memory only), ce:GetCostAndUsage (opt-in, US$0.01 per request).
No writes to AWS, no Bedrock calls, nothing that starts or stops anything.

Secrets: the Attribute token stays in a local variable. Only non-secret JWT
timestamps (iat/exp) are recorded, to explain a 401.
"""

from __future__ import annotations

import base64
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ACCOUNT_ID = "<ACCOUNT_ID>"
REGION = "us-east-1"
BUCKET = "<DATA_BUCKET>"
CLUSTER = "lda"
SERVICES = ("lda-embed", "lda-coordinator")
STATE_MACHINE_ARN = f"arn:aws:states:{REGION}:{ACCOUNT_ID}:stateMachine:lda-batch-pass"
QUEUES = ("lda-door-events", "lda-door-events-dlq")
LOG_GROUPS = ("aws/spans", "/lda/agent/coordinator", "/lda/ecs/coordinator", "/lda/ecs/embed", "/lda/lambda/knn",
              "/lda/lambda/ticket", "/lda/lambda/reaper", "/aws/lambda/lda-reaper", "/aws/lambda/lda-batch-step",
              "/aws/lambda/lda-batch-watchdog", "/aws/lambda/lda-batch-deadman",
              "/aws/vendedlogs/states/lda-batch-pass")
LAMBDA_REPORT_GROUPS = ("/lda/lambda/knn", "/lda/lambda/ticket", "/aws/lambda/lda-batch-step",
                        "/aws/lambda/lda-batch-watchdog", "/aws/lambda/lda-batch-deadman", "/lda/lambda/reaper")
SCHEDULE_GROUPS = ("lda-batch", "lda-batch-deadman", "lda-batch-test")
EVENT_RULES = ("lda-incoming-object-created", "lda-idle-reaper")
DDB_TABLE = "lda-mock-tickets"


def _utc(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=_utc))


def _session(profile: str):
    import boto3

    return boto3.Session(profile_name=profile, region_name=REGION)


# -- pricing ------------------------------------------------------------------------------------

def collect_pricing(sess) -> dict:
    from .rates import PRICE_QUERIES

    pr = sess.client("pricing", region_name="us-east-1")
    rates, problems = {}, []
    for key, (service, usagetype, begin) in PRICE_QUERIES.items():
        resp = pr.get_products(ServiceCode=service, FormatVersion="aws_v1", MaxResults=20, Filters=[
            {"Type": "TERM_MATCH", "Field": "regionCode", "Value": REGION},
            {"Type": "TERM_MATCH", "Field": "usagetype", "Value": usagetype}])
        found = []
        for raw in resp["PriceList"]:
            prod = json.loads(raw)
            for term in prod["terms"].get("OnDemand", {}).values():
                for dim in term["priceDimensions"].values():
                    if dim.get("beginRange") == begin:
                        found.append({"usd": float(dim["pricePerUnit"]["USD"]), "unit": dim["unit"],
                                      "description": dim["description"], "usagetype": usagetype,
                                      "service": service, "sku": prod["product"].get("sku"),
                                      "operation": prod["product"]["attributes"].get("operation", ""),
                                      "publication_date": prod.get("publicationDate", "")})
        if service == "AmazonCloudWatch" and key.startswith("logs_ingest"):
            found = [f for f in found if f["operation"] == "PutLogEvents"] or found
        if len(found) != 1:
            problems.append(f"{key}: {len(found)} matching price dimensions for {service}/{usagetype}/begin={begin}")
            if not found:
                continue
        rates[key] = found[0]
    return {"source": "AWS Pricing API pricing:GetProducts us-east-1 (read-only)",
            "fetched_at": datetime.now(timezone.utc).isoformat(), "rates": rates, "problems": problems}


# -- run records ----------------------------------------------------------------------------------

def collect_runs(runs_dir: Path) -> list[dict]:
    rows = []
    for d in sorted(p for p in runs_dir.iterdir() if (p / "result.json").is_file()):
        r = json.loads((d / "result.json").read_text())
        rs = r.get("reasoner") or {}
        det = r.get("detector") or {}
        rows.append({
            "run_id": r["run_id"], "clip_key": r.get("clip_key"), "error": r.get("error"),
            "trace_id": r.get("trace_id"), "ticket_id": r.get("ticket_id"),
            "decision": (r.get("decision") or {}).get("decision"),
            "timings": r.get("timings") or {},
            "reasoner": {"usage": rs.get("usage"), "cost_usd": rs.get("cost_usd"), "latency_s": rs.get("latency_s"),
                         "image_count": rs.get("image_count"), "error": rs.get("error")},
            "model": "anthropic.claude-opus-5-5" if rs.get("usage") else None,
            "detector": {"alert": det.get("alert"), "evidence_diffs": len((det.get("evidence_keys") or {}).get("diffs", [])),
                         "bank_prefix": det.get("bank_prefix")},
            "source_file": str(d / "result.json"),
        })
    return rows


# -- S3 per-run listing -------------------------------------------------------------------------

def collect_s3(sess, run_ids: list[str]) -> dict:
    s3 = sess.client("s3")
    out = {}
    pag = s3.get_paginator("list_objects_v2")
    for rid in run_ids:
        objs = []
        for prefix in (f"runs/{rid}/", f"incoming/{rid}/"):
            for page in pag.paginate(Bucket=BUCKET, Prefix=prefix):
                for o in page.get("Contents", []):
                    objs.append({"key": o["Key"], "size": o["Size"], "last_modified": _utc(o["LastModified"]),
                                 "storage_class": o.get("StorageClass")})
        out[rid] = objs
    # Bucket-wide totals by top-level prefix, for the storage reconciliation line.
    totals: dict[str, dict] = {}
    for page in pag.paginate(Bucket=BUCKET):
        for o in page.get("Contents", []):
            top = o["Key"].split("/", 1)[0]
            t = totals.setdefault(top, {"objects": 0, "bytes": 0})
            t["objects"] += 1
            t["bytes"] += o["Size"]
    return {"bucket": BUCKET, "listed_at": datetime.now(timezone.utc).isoformat(), "runs": out,
            "bucket_totals_by_prefix": totals}


# -- batch driver artefacts -----------------------------------------------------------------------

def collect_batch(sess) -> dict:
    s3 = sess.client("s3")
    pag = s3.get_paginator("list_objects_v2")
    keys = []
    for page in pag.paginate(Bucket=BUCKET, Prefix="batch/"):
        for o in page.get("Contents", []):
            if o["Key"].endswith(("/manifest.json", "/ecs_tasks.json")) and "/source/" not in o["Key"]:
                keys.append(o["Key"])
    out = {}
    for k in sorted(keys):
        out[k] = json.loads(s3.get_object(Bucket=BUCKET, Key=k)["Body"].read())
    return out


# -- CloudWatch metrics -----------------------------------------------------------------------

def _stat(cw, ns, name, dims, start, end, period, stats):
    pts = []
    t = start
    step = timedelta(seconds=period * 1400)
    while t < end:
        t2 = min(end, t + step)
        r = cw.get_metric_statistics(Namespace=ns, MetricName=name, Dimensions=dims, StartTime=t, EndTime=t2,
                                     Period=period, Statistics=stats)
        pts += r["Datapoints"]
        t = t2
    return [{k: (_utc(v) if k == "Timestamp" else v) for k, v in p.items() if k != "Unit"}
            for p in sorted(pts, key=lambda p: p["Timestamp"])]


def collect_metrics(sess, days: list[str]) -> dict:
    cw = sess.client("cloudwatch")
    start = datetime.fromisoformat(days[0]).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(days[-1]).replace(tzinfo=timezone.utc) + timedelta(days=1)
    day = 86400
    m: dict = {"window": [start.isoformat(), end.isoformat()], "ecs_live_task_count_1m": {}, "daily": {}}
    for svc in SERVICES:
        m["ecs_live_task_count_1m"][svc] = _stat(cw, "AWS/ECS", "LiveTaskCount",
                                                 [{"Name": "ClusterName", "Value": CLUSTER},
                                                  {"Name": "ServiceName", "Value": svc}], start, end, 60, ["Maximum"])
    lam = sess.client("lambda")
    fns = []
    for page in lam.get_paginator("list_functions").paginate():
        fns += [f["FunctionName"] for f in page["Functions"] if f["FunctionName"].startswith("lda-")]
    d = m["daily"]
    for fn in sorted(fns):
        for name in ("Invocations", "Duration", "Errors", "Throttles"):
            d[f"lambda/{fn}/{name}"] = _stat(cw, "AWS/Lambda", name, [{"Name": "FunctionName", "Value": fn}],
                                              start, end, day, ["Sum", "SampleCount"])
    for q in QUEUES:
        for name in ("NumberOfMessagesSent", "NumberOfMessagesReceived", "NumberOfMessagesDeleted",
                     "NumberOfEmptyReceives"):
            d[f"sqs/{q}/{name}"] = _stat(cw, "AWS/SQS", name, [{"Name": "QueueName", "Value": q}], start, end, day,
                                         ["Sum"])
    for g in LOG_GROUPS:
        d[f"logs/{g}/IncomingBytes"] = _stat(cw, "AWS/Logs", "IncomingBytes", [{"Name": "LogGroupName", "Value": g}],
                                             start, end, day, ["Sum"])
    for name in ("ExecutionsStarted", "ExecutionsSucceeded", "ExecutionsFailed"):
        d[f"states/lda-batch-pass/{name}"] = _stat(cw, "AWS/States", name,
                                                    [{"Name": "StateMachineArn", "Value": STATE_MACHINE_ARN}],
                                                    start, end, day, ["Sum"])
    for g in SCHEDULE_GROUPS:
        d[f"scheduler/{g}/InvocationAttemptCount"] = _stat(cw, "AWS/Scheduler", "InvocationAttemptCount",
                                                            [{"Name": "ScheduleGroup", "Value": g}], start, end, day,
                                                            ["Sum"])
    for rule in EVENT_RULES:
        for name in ("MatchedEvents", "Invocations"):
            d[f"events/{rule}/{name}"] = _stat(cw, "AWS/Events", name, [{"Name": "RuleName", "Value": rule}],
                                               start, end, day, ["Sum"])
    for name in ("ConsumedWriteCapacityUnits", "ConsumedReadCapacityUnits"):
        d[f"dynamodb/{DDB_TABLE}/{name}"] = _stat(cw, "AWS/DynamoDB", name,
                                                   [{"Name": "TableName", "Value": DDB_TABLE}], start, end, day,
                                                   ["Sum"])
        d[f"dynamodb/{DDB_TABLE}/by_run_id/{name}"] = _stat(
            cw, "AWS/DynamoDB", name, [{"Name": "TableName", "Value": DDB_TABLE},
                                       {"Name": "GlobalSecondaryIndexName", "Value": "by_run_id"}],
            start, end, day, ["Sum"])
    return m


def collect_lambda_config(sess) -> dict:
    lam = sess.client("lambda")
    out = {}
    for page in lam.get_paginator("list_functions").paginate():
        for f in page["Functions"]:
            if f["FunctionName"].startswith("lda-"):
                out[f["FunctionName"]] = {"memory_mb": f["MemorySize"], "arch": (f.get("Architectures") or ["x86_64"])[0],
                                          "ephemeral_mb": (f.get("EphemeralStorage") or {}).get("Size", 512),
                                          "package_type": f.get("PackageType")}
    return out


# -- Lambda REPORT lines (exact billed duration per request id) ---------------------------

_REPORT = re.compile(r"REPORT RequestId: (?P<rid>\S+)\s+Duration: (?P<dur>[\d.]+) ms\s+Billed Duration: (?P<billed>\d+) ms"
                     r"\s+Memory Size: (?P<mem>\d+) MB\s+Max Memory Used: (?P<maxmem>\d+) MB"
                     r"(?:\s+Init Duration: (?P<init>[\d.]+) ms)?")


def collect_lambda_reports(sess, start: datetime, end: datetime) -> dict:
    logs = sess.client("logs")
    out: dict = {}
    for group in LAMBDA_REPORT_GROUPS:
        rows = []
        try:
            streams = []
            for page in logs.get_paginator("describe_log_streams").paginate(logGroupName=group):
                streams += page["logStreams"]
        except logs.exceptions.ResourceNotFoundException:
            out[group] = {"error": "log group not found"}
            continue
        for st in streams:
            last = st.get("lastEventTimestamp") or st.get("lastIngestionTime") or 0
            if last and last < start.timestamp() * 1000:
                continue
            token = None
            while True:
                kw = dict(logGroupName=group, logStreamName=st["logStreamName"], startFromHead=True,
                          startTime=int(start.timestamp() * 1000), endTime=int(end.timestamp() * 1000))
                if token:
                    kw["nextToken"] = token
                resp = logs.get_log_events(**kw)
                for ev in resp["events"]:
                    mm = _REPORT.search(ev["message"])
                    if mm:  # only the platform REPORT line is kept; no application log content
                        rows.append({"request_id": mm["rid"], "duration_ms": float(mm["dur"]),
                                     "billed_ms": int(mm["billed"]), "memory_mb": int(mm["mem"]),
                                     "init_ms": float(mm["init"]) if mm["init"] else None,
                                     "ts": datetime.fromtimestamp(ev["timestamp"] / 1000, timezone.utc).isoformat()})
                nt = resp.get("nextForwardToken")
                if not resp["events"] or nt == token:
                    break
                token = nt
        out[group] = {"reports": sorted(rows, key=lambda r: r["ts"])}
    return out


# -- X-Ray per-run trace summaries ---------------------------------------------------------

def _walk(doc, parent_service, acc):
    acc["span_count"] += 1
    name = doc.get("name")
    ann = doc.get("annotations") or {}
    aws = doc.get("aws") or {}
    dur = (doc.get("end_time") or 0) - (doc.get("start_time") or 0)
    if name == "S3" or ann.get("aws.local.service") == "AWS::S3":
        rid = aws.get("request_id") or doc.get("id")
        op = ann.get("aws.local.operation") or aws.get("operation") or "unknown"
        acc["s3"][rid] = {"op": op, "caller": parent_service}
    elif name == "Lambda" and aws.get("lambda.function.name"):
        acc["lambda_calls"][aws.get("request_id") or doc.get("id")] = {
            "function": aws["lambda.function.name"], "round_trip_s": dur}
    elif name in ("lda-coordinator", "lda-embed", "lda-knn", "lda-ticket") and not doc.get("inferred"):
        key = name if name != "lda-embed" or ann.get("span.name") == "embed" else "lda-embed-other"
        acc["services"].setdefault(key, []).append(dur)
        parent_service = name
    elif name and name.startswith("bedrock-mantle"):
        acc["bedrock_s"] = max(acc.get("bedrock_s") or 0, dur)
    elif name == "knn.load_bank":
        acc["knn_load_bank_s"] = dur
    for sub in doc.get("subsegments") or []:
        _walk(sub, parent_service, acc)


def collect_xray(sess, runs: list[dict]) -> dict:
    xr = sess.client("xray")
    ids = {}
    for r in runs:
        t = r.get("trace_id")
        if t and len(t) == 32:
            ids[f"1-{t[:8]}-{t[8:]}"] = r["run_id"]
    out, unprocessed = {}, []
    keys = list(ids)
    for i in range(0, len(keys), 5):
        resp = xr.batch_get_traces(TraceIds=keys[i:i + 5])
        unprocessed += resp.get("UnprocessedTraceIds", [])
        for tr in resp.get("Traces", []):
            acc = {"span_count": 0, "s3": {}, "lambda_calls": {}, "services": {}, "bedrock_s": None,
                   "knn_load_bank_s": None}
            seg_bytes = 0
            for seg in tr["Segments"]:
                seg_bytes += len(seg["Document"])
                doc = json.loads(seg["Document"])
                caller = doc.get("name") if doc.get("name", "").startswith("lda-") else None
                if doc.get("inferred") and doc.get("name") in ("S3", "Lambda"):
                    caller = "inferred"
                _walk(doc, caller, acc)
            s3_ops: dict[str, int] = {}
            for v in acc["s3"].values():
                s3_ops[v["op"]] = s3_ops.get(v["op"], 0) + 1
            out[ids[tr["Id"]]] = {
                "trace_id": tr["Id"], "duration_s": tr.get("Duration"), "segment_count": len(tr["Segments"]),
                "span_count": acc["span_count"], "segment_document_bytes": seg_bytes,
                "s3_ops": s3_ops, "lambda_calls": acc["lambda_calls"],
                "service_durations_s": acc["services"], "bedrock_s": acc["bedrock_s"],
                "knn_load_bank_s": acc["knn_load_bank_s"]}
    return {"traces": out, "unprocessed": unprocessed,
            "missing_runs": sorted(set(ids.values()) - set(out))}


# -- Step Functions -----------------------------------------------------------------------

def collect_sfn(sess) -> dict:
    sf = sess.client("stepfunctions")
    execs = []
    for page in sf.get_paginator("list_executions").paginate(stateMachineArn=STATE_MACHINE_ARN):
        execs += page["executions"]
    out = []
    for e in execs:
        events = []
        for page in sf.get_paginator("get_execution_history").paginate(executionArn=e["executionArn"]):
            events += page["events"]
        counts: dict[str, int] = {}
        pass_input = None
        for ev in events:
            counts[ev["type"]] = counts.get(ev["type"], 0) + 1
            if ev["type"] == "ExecutionStarted":
                try:
                    pass_input = json.loads(ev["executionStartedEventDetails"]["input"])
                except Exception:  # noqa: BLE001
                    pass_input = None
        out.append({"name": e["name"], "status": e["status"], "start": _utc(e["startDate"]),
                    "stop": _utc(e.get("stopDate")), "event_counts": counts,
                    # Standard workflows bill per state transition; each *StateEntered event is one.
                    "state_transitions": sum(v for k, v in counts.items() if k.endswith("StateEntered")),
                    "batch_id": (pass_input or {}).get("batch_id"), "pass_id": (pass_input or {}).get("pass_id")})
    return {"state_machine": STATE_MACHINE_ARN, "executions": out}


# -- Cost Explorer (opt-in) ----------------------------------------------------------------

def collect_cost_explorer(sess, start: str, end: str) -> dict:
    ce = sess.client("ce", region_name="us-east-1")
    calls = 0
    result = {}
    for name, kw in {
        "by_service_and_project_tag": dict(GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"},
                                                    {"Type": "TAG", "Key": "Project"}]),
        "lda_tag_by_usage_type": dict(Filter={"Tags": {"Key": "Project", "Values": ["lift-door-agentic"]}},
                                      GroupBy=[{"Type": "DIMENSION", "Key": "USAGE_TYPE"}]),
    }.items():
        pages = []
        args = dict(TimePeriod={"Start": start, "End": end}, Granularity="DAILY",
                    Metrics=["UnblendedCost", "UsageQuantity"], **kw)
        while True:
            r = ce.get_cost_and_usage(**args)
            calls += 1
            pages += r["ResultsByTime"]
            if not r.get("NextPageToken"):
                break
            args["NextPageToken"] = r["NextPageToken"]
        result[name] = pages
    return {"fetched_at": datetime.now(timezone.utc).isoformat(), "requests": calls,
            "request_cost_usd": round(calls * 0.01, 2), "results": result}


# -- Attribute ---------------------------------------------------------------------------

def collect_attribute(profile: str, days: list[str]) -> dict:
    from .attribute_client import AttributeApiClient, token_from_ssm

    out: dict = {"days": {}, "fetched_at": datetime.now(timezone.utc).isoformat()}
    try:
        token = token_from_ssm(profile)
    except Exception as exc:  # noqa: BLE001
        out["status"] = f"NO_TOKEN: {type(exc).__name__}"
        return out
    try:
        claims = json.loads(base64.urlsafe_b64decode(token.split(".")[1] + "=" * (-len(token.split(".")[1]) % 4)))
        out["token_iat"] = datetime.fromtimestamp(claims["iat"], timezone.utc).isoformat() if "iat" in claims else None
        out["token_exp"] = datetime.fromtimestamp(claims["exp"], timezone.utc).isoformat() if "exp" in claims else None
    except Exception:  # noqa: BLE001
        out["token_exp"] = None
    client = AttributeApiClient(token)
    del token
    status, body = client._get("/api/v1/health")
    out["health_status"] = status
    if status == 401:
        exp = out.get("token_exp")
        expired = exp and datetime.fromisoformat(exp) < datetime.now(timezone.utc)
        out["status"] = ("UNAUTHORIZED: REST API JWT expired at " + exp) if expired else "UNAUTHORIZED"
        return out
    from .attribute_client import IndeterminateJobError, JobFailedError

    for d in days:
        day: dict = {}
        for ep in ("aicost", "resources", "workloads"):
            try:
                r = client.get_daily(ep, d, allow_partial=True)
                rows = [x for x in r.rows if ACCOUNT_ID in str(x.get("account", ""))]
                day[ep] = {"status": "OK", "job_id": r.job_id, "reason": r.reason, "org_rows": len(r.rows),
                           "account_rows": rows, "allow_partial": True, "fetched_at": r.fetched_at}
            except (IndeterminateJobError, JobFailedError, RuntimeError) as exc:
                # Includes the known reproducible /api/v1/workloads/daily vendor-side failure
                # (attributes/app/cost_joiner/attribute_costs.py): reported, never zero-filled.
                day[ep] = {"status": f"UNAVAILABLE: {type(exc).__name__}: {str(exc)[:300]}"}
        out["days"][d] = day
    out["status"] = "OK"
    return out


# -- orchestration -------------------------------------------------------------------------

def collect_all(out_dir: Path, runs_dir: Path, profile: str, days: list[str], *, cost_explorer: bool,
                attribute: bool) -> list[str]:
    raw = out_dir / "raw"
    sess = _session(profile)
    acct = sess.client("sts").get_caller_identity()["Account"]
    if acct != ACCOUNT_ID:
        raise RuntimeError(f"profile {profile} resolves to account {acct}, expected {ACCOUNT_ID}")
    notes = []
    runs = collect_runs(runs_dir)
    _dump(raw / "runs.json", runs)
    _dump(raw / "pricing.json", collect_pricing(sess))
    _dump(raw / "s3.json", collect_s3(sess, [r["run_id"] for r in runs]))
    _dump(raw / "batch.json", collect_batch(sess))
    _dump(raw / "metrics.json", collect_metrics(sess, days))
    _dump(raw / "lambda_config.json", collect_lambda_config(sess))
    start = datetime.fromisoformat(days[0]).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(days[-1]).replace(tzinfo=timezone.utc) + timedelta(days=1)
    _dump(raw / "lambda_reports.json", collect_lambda_reports(sess, start, end))
    _dump(raw / "xray.json", collect_xray(sess, runs))
    _dump(raw / "sfn.json", collect_sfn(sess))
    if cost_explorer:
        _dump(raw / "cost_explorer.json",
              collect_cost_explorer(sess, days[0], (date.fromisoformat(days[-1]) + timedelta(days=1)).isoformat()))
        notes.append("cost explorer queried")
    if attribute:
        _dump(raw / "attribute.json", collect_attribute(profile, days))
    _dump(raw / "collect_meta.json", {"collected_at": datetime.now(timezone.utc).isoformat(), "profile": profile,
                                      "account": acct, "days": days, "runs_dir": str(runs_dir),
                                      "cost_explorer": cost_explorer, "attribute": attribute})
    return notes
