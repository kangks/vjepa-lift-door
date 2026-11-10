"""Offline join: out/raw/*.json -> per-invocation cost rows, unallocated lines, reconciliation.

Every money cell carries one source tag:
  ATTRIBUTE_APPORTIONED  Attribute daily workload cost, apportioned by lda busy-seconds
  APP_METERED            lda's own per-call token counts x the app's rate constant
  LIST_RATE_ESTIMATE     counted/metered quantity x AWS Pricing API list rate
  PENDING_ATTRIBUTE      the Attribute figure is not available for that day; the cell shows
                         the list-rate estimate of the same quantity until it is
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from . import (SOURCE_APP_METERED, SOURCE_ATTRIBUTE_APPORTIONED, SOURCE_LIST_RATE_ESTIMATE,
               SOURCE_PENDING_ATTRIBUTE)
from .apportionment import apportion, even_split
from .rates import (BEDROCK_RATE_SOURCE, EVENTBRIDGE_AWS_SERVICE_EVENT_USD, EVENTBRIDGE_RATE_SOURCE, RateCard,
                    bedrock_cost, fargate_task_rate_per_second, ingest_cost, lambda_invocation_cost,
                    storage_cost_per_day)

# Task sizes verified in infra/lda_infra/main_stack.py and in the ECS task records:
# embed 16384 CPU / 32768 MiB / 21 GiB ephemeral, ARM64. Coordinator is 1024 CPU / 2048 MiB in
# attrb_sandbox (bumped from 512/1024 for the Attribute sensor; ATTRB_SENSOR_PROFILES), ARM64.
TASK_SIZES = {
    "lda-embed": {"vcpu": 16.0, "memory_gib": 32.0, "ephemeral_gib": 21.0},
    "lda-coordinator": {"vcpu": 1.0, "memory_gib": 2.0, "ephemeral_gib": 20.0},
}
PIPELINE_LOG_GROUPS = ("/lda/agent/coordinator", "/lda/ecs/coordinator", "/lda/ecs/embed", "/lda/lambda/knn",
                       "/lda/lambda/ticket")
OTEL_LOG_GROUPS = ("/lda/agent/coordinator",)  # OTLP logs endpoint, billed as DataProcessingOtel-Bytes
DRIVER_LOG_GROUPS = ("/aws/lambda/lda-batch-step", "/aws/lambda/lda-batch-watchdog", "/aws/lambda/lda-batch-deadman",
                     "/aws/vendedlogs/states/lda-batch-pass")
BANK_CLIPS = 5  # bank manifest clips loaded by lda-knn on a cold start (_load_bank)
SQS_HEARTBEAT_S = 300

SERVICE_COLUMNS = [
    # column key, label
    ("bedrock_opus", "Bedrock Opus 5.5"),
    ("fargate_embed", "Fargate embed"),
    ("fargate_coordinator", "Fargate coordinator"),
    ("vpc_public_ipv4", "VPC public IPv4"),
    ("lambda_knn_ticket", "Lambda knn+ticket"),
    ("s3_requests", "S3 requests"),
    ("s3_storage_day", "S3 storage /day"),
    ("sqs", "SQS"),
    ("eventbridge", "EventBridge"),
    ("dynamodb", "DynamoDB"),
    ("cloudwatch_logs", "CloudWatch Logs"),
    ("xray_spans", "X-Ray spans"),
    ("step_functions", "Step Functions"),
    ("scheduler", "Scheduler"),
    ("lambda_batch_driver", "Lambda batch driver"),
]
TAG_ABBR = {SOURCE_ATTRIBUTE_APPORTIONED: "AA", SOURCE_APP_METERED: "AM", SOURCE_LIST_RATE_ESTIMATE: "LR",
            SOURCE_PENDING_ATTRIBUTE: "PA", "N/A": "na"}


def _load(raw: Path, name: str, default=None):
    p = raw / name
    if not p.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"{p} missing; run `cli.py collect` first")
    return json.loads(p.read_text())


def _dt(s: str | None) -> datetime | None:
    if not s or s == "None":
        return None
    return datetime.fromisoformat(s.replace(" ", "T")).astimezone(timezone.utc)


def run_class(run_id: str) -> str:
    m = re.match(r"run-\d{8}T\d{6}-(healthy|anomaly|stress)_", run_id)
    return m.group(1) if m else "unknown"


def run_day(run_id: str) -> str:
    m = re.match(r"run-(\d{4})(\d{2})(\d{2})T", run_id)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def run_start(run_id: str) -> datetime:
    m = re.match(r"run-(\d{8}T\d{6})", run_id)
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def _day_value(series: list[dict], day: str, stat: str = "Sum") -> float | None:
    for p in series:
        if p["Timestamp"][:10] == day:
            return float(p[stat])
    return None


# -- Fargate task time per service per UTC day ---------------------------------------------------

def _clip_to_day(start: datetime, stop: datetime, day: str) -> float:
    d0 = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    d1 = d0 + timedelta(days=1)
    return max(0.0, (min(stop, d1) - max(start, d0)).total_seconds())


def fargate_task_time(batch: dict, metrics: dict, days: list[str]) -> dict:
    """Billed task-seconds per (service, day). Exact where the batch driver recorded ECS task
    pull-start/stop times; elsewhere LiveTaskCount minutes plus a start/stop offset calibrated on
    the exact windows (LiveTaskCount misses the image-pull and stopping phases, which bill)."""
    exact = []  # (service, start, stop, label)
    for key, tasks in batch.items():
        if not key.endswith("ecs_tasks.json"):
            continue
        for t in tasks:
            start = _dt(t.get("pull_started_at")) or _dt(t.get("started_at")) or _dt(t.get("created_at"))
            stop = _dt(t.get("stopped_at"))
            if start and stop:
                exact.append((t["service"], start, stop, key.split("/")[-2]))
    result: dict = {"by_service_day": {}, "windows": [], "calibration": {}}
    for svc, pts in metrics["ecs_live_task_count_1m"].items():
        on = [(_dt(p["Timestamp"]), p["Maximum"]) for p in pts if p["Maximum"] > 0]
        windows = []
        for ts, v in on:
            if windows and (ts - windows[-1]["last"]).total_seconds() <= 60:
                windows[-1]["last"] = ts
                windows[-1]["live_minutes"] += v
            else:
                windows.append({"first": ts, "last": ts, "live_minutes": v})
        offsets = []
        for w in windows:
            w_start, w_stop = w["first"], w["last"] + timedelta(minutes=1)
            match = [e for e in exact if e[0] == svc and e[1] < w_stop + timedelta(minutes=3)
                     and e[2] > w_start - timedelta(minutes=3)]
            w["service"] = svc
            if match:
                w["method"] = "exact ECS task record (" + ",".join(sorted({m[3] for m in match})) + ")"
                w["intervals"] = [(m[1], m[2]) for m in match]
                w["billed_s"] = sum((m[2] - m[1]).total_seconds() for m in match)
                offsets.append(w["billed_s"] - w["live_minutes"] * 60)
            else:
                w["method"] = None
            windows_append = w
            result["windows"].append(windows_append)
        mean_off = statistics.mean(offsets) if offsets else 0.0
        result["calibration"][svc] = {"windows_with_exact": len(offsets), "mean_offset_s": mean_off,
                                      "offsets_s": offsets}
        for w in windows:
            if w["method"] is None:
                w["billed_s"] = w["live_minutes"] * 60 + mean_off
                w["method"] = f"LiveTaskCount minutes + calibrated offset {mean_off:.0f}s (estimate)"
                w["intervals"] = None
        for day in days:
            tot = 0.0
            methods = set()
            for w in windows:
                if w["intervals"]:
                    s = sum(_clip_to_day(a, b, day) for a, b in w["intervals"])
                    if s:
                        tot += s
                        methods.add("exact")
                elif w["first"].date().isoformat() == day:
                    if w["last"].date() != w["first"].date():
                        raise ValueError(f"estimated window crosses midnight: {w}")
                    tot += w["billed_s"]
                    methods.add("estimate")
            result["by_service_day"][(svc, day)] = {"billed_s": tot, "methods": sorted(methods)}
    return result


# -- Attribute (if reachable) -----------------------------------------------------------------

def attribute_day_status(attr: dict, day: str, today: date) -> dict:
    """Returns {'fargate': {svc: usd}|None, 'bedrock': usd|None, 'settled': bool, 'reason': str}."""
    settled = date.fromisoformat(day) <= today - timedelta(days=2)
    if attr.get("status") != "OK":
        return {"fargate": None, "bedrock": None, "settled": settled,
                "reason": f"Attribute REST API {attr.get('status', 'not queried')}"}
    d = (attr.get("days") or {}).get(day) or {}
    wl = d.get("workloads") or {}
    ai = d.get("aicost") or {}
    farg = None
    if wl.get("status") == "OK":
        farg = defaultdict(float)
        for r in wl["account_rows"]:
            name = r.get("workload_name") or r.get("workload") or r.get("name")
            if name in TASK_SIZES and r.get("resource_type") == "EC2-ECS":
                farg[name] += float(r.get("amortized_cost") or 0)
        farg = dict(farg) or None
    bed = None
    if ai.get("status") == "OK":
        rows = [r for r in ai["account_rows"] if r.get("resource_type") == "AmazonBedrock"]
        bed = sum(float(r.get("amortized_cost") or 0) for r in rows) if rows else None
    reason = "settled" if settled else "inside the 2-day settlement window (partial day)"
    if farg is None:
        reason = f"workloads/daily {wl.get('status', 'missing')}; " + reason
    return {"fargate": farg, "bedrock": bed, "settled": settled, "reason": reason}


# -- main join ----------------------------------------------------------------------------------

def build_report(out_dir: Path, today: date | None = None) -> list[Path]:
    raw = out_dir / "raw"
    runs = _load(raw, "runs.json")
    rates = RateCard(_load(raw, "pricing.json"))
    pricing = _load(raw, "pricing.json")
    s3 = _load(raw, "s3.json")
    batch = _load(raw, "batch.json")
    metrics = _load(raw, "metrics.json")
    lcfg = _load(raw, "lambda_config.json")
    lrep = _load(raw, "lambda_reports.json")
    xray = _load(raw, "xray.json")["traces"]
    sfn = _load(raw, "sfn.json")["executions"]
    attr = _load(raw, "attribute.json", {"status": "not queried"})
    ce = _load(raw, "cost_explorer.json", {})
    meta = _load(raw, "collect_meta.json")
    today = today or datetime.fromisoformat(meta["collected_at"]).date()
    days = sorted({run_day(r["run_id"]) for r in runs})
    all_days = meta["days"]

    reports_by_rid = {}
    for group, v in lrep.items():
        for rep in v.get("reports", []):
            rep = dict(rep, group=group)
            reports_by_rid[rep["request_id"]] = rep

    # Pass membership from the batch driver manifests
    run_pass: dict[str, str] = {}
    pass_manifest: dict[str, dict] = {}
    for key, m in batch.items():
        if key.endswith("manifest.json") and m.get("runs"):
            pid = f"{m['batch_id']}/{m['pass_id']}"
            pass_manifest[pid] = m
            for v in m["runs"].values():
                run_pass[v["run_id"]] = pid
    pass_runs = defaultdict(list)
    for rid, pid in run_pass.items():
        pass_runs[pid].append(rid)
    exec_by_pass = {f"{e['batch_id']}/{e['pass_id']}": e for e in sfn if e.get("batch_id")}

    ftime = fargate_task_time(batch, metrics, all_days)
    rows: dict[str, dict] = {}
    notes: dict[str, list[str]] = defaultdict(list)
    q: dict[str, dict] = {}  # per-run quantities

    # ---- per-run quantities ----
    for r in runs:
        rid = r["run_id"]
        t = r["timings"]
        u = (r["reasoner"] or {}).get("usage") or {}
        x = xray.get(rid, {})
        alert = bool((r["detector"] or {}).get("alert"))
        diffs = int((r["detector"] or {}).get("evidence_diffs") or 0)
        objs = s3["runs"].get(rid, [])
        knn_calls = [k for k, v in (x.get("lambda_calls") or {}).items() if v["function"] == "lda-knn"]
        tic_calls = [k for k, v in (x.get("lambda_calls") or {}).items() if v["function"] == "lda-ticket"]
        knn_rep = [reports_by_rid[k] for k in knn_calls if k in reports_by_rid]
        tic_rep = [reports_by_rid[k] for k in tic_calls if k in reports_by_rid]
        knn_cold = any(rep.get("init_ms") for rep in knn_rep)
        predicted_put = 1 + 2 + (1 if alert else 0) + 2 + 2 + ((1 + diffs) if alert else 0)
        get_est = 1 + 3 + (1 if alert else 0) + (3 if alert else 0) + ((2 + BANK_CLIPS) if knn_cold else 0)
        xray_get = (x.get("s3_ops") or {}).get("GetObject")
        q[rid] = {
            "class": run_class(rid), "day": run_day(rid), "pass": run_pass.get(rid, "(not batch-driven)"),
            "decision": r.get("decision"), "alert": alert,
            "input_tokens": u.get("input_tokens", 0) or 0, "output_tokens": u.get("output_tokens", 0) or 0,
            "thinking_tokens": (u.get("output_tokens_details") or {}).get("thinking_tokens"),
            "embed_busy_s": (t.get("download", 0) + t.get("decode", 0) + sum(t.get("encode_windows", []))
                             + t.get("put", 0)),
            "coordinator_busy_s": t.get("total", 0.0),
            "knn_billed_ms": sum(rep["billed_ms"] for rep in knn_rep), "knn_cold": knn_cold,
            "knn_init_ms": sum(rep.get("init_ms") or 0 for rep in knn_rep),
            "ticket_billed_ms": sum(rep["billed_ms"] for rep in tic_rep),
            "knn_calls": len(knn_calls), "ticket_calls": len(tic_calls),
            "knn_unmatched": len(knn_calls) - len(knn_rep), "ticket_unmatched": len(tic_calls) - len(tic_rep),
            "s3_put_exact": len(objs), "s3_put_predicted": predicted_put, "s3_get_est": get_est,
            "s3_get_xray_coord_embed": xray_get, "s3_get_code_coord_embed": 1 + (3 if alert else 0),
            "s3_bytes": sum(o["size"] for o in objs),
            "sqs_requests": 3 + int(t.get("total", 0) // SQS_HEARTBEAT_S),
            "ddb_wru": 2 if r.get("ticket_id") else 0,
            "span_bytes": x.get("segment_document_bytes", 0), "span_count": x.get("span_count", 0),
        }
        if predicted_put != len(objs):
            notes[rid].append(f"S3 objects listed {len(objs)} != code-path PUT prediction {predicted_put}")
        if xray_get is not None and xray_get != q[rid]["s3_get_code_coord_embed"]:
            notes[rid].append(f"X-Ray coordinator+embed GetObject {xray_get} != code path "
                              f"{q[rid]['s3_get_code_coord_embed']}")
        for name, n in (("knn", q[rid]["knn_unmatched"]), ("ticket", q[rid]["ticket_unmatched"])):
            if n:
                notes[rid].append(f"{n} {name} invocation(s) with no Lambda REPORT line; priced at $0 here")
        if not x:
            notes[rid].append("no X-Ray trace found; Lambda and span lines are $0 here (unmeasured)")

    # ---- Fargate daily cost and apportionment ----
    fargate_rate_s = {svc: fargate_task_rate_per_second(**sz, rates=rates) for svc, sz in TASK_SIZES.items()}
    attr_status = {d: attribute_day_status(attr, d, today) for d in all_days}
    fargate_day: dict = {}
    ipv4_day: dict = {}
    for svc in TASK_SIZES:
        for d in all_days:
            ts = ftime["by_service_day"].get((svc, d), {"billed_s": 0.0, "methods": []})
            list_cost = ts["billed_s"] * fargate_rate_s[svc]
            a = attr_status[d]
            use_attr = bool(a["fargate"] and svc in a["fargate"] and a["settled"])
            fargate_day[(svc, d)] = {
                "billed_s": ts["billed_s"], "methods": ts["methods"], "list_cost": list_cost,
                "attribute_cost": (a["fargate"] or {}).get(svc) if a["fargate"] else None,
                "cost": a["fargate"][svc] if use_attr else list_cost,
                "source": SOURCE_ATTRIBUTE_APPORTIONED if use_attr else SOURCE_PENDING_ATTRIBUTE,
                "reason": a["reason"]}
            ipv4_day[(svc, d)] = ts["billed_s"] / 3600 * rates["public_ipv4_hour"]
    app = {}
    for svc, key in (("lda-embed", "embed_busy_s"), ("lda-coordinator", "coordinator_busy_s")):
        for d in all_days:
            busy = {rid: v[key] for rid, v in q.items() if v["day"] == d}
            fd = fargate_day[(svc, d)]
            app[(svc, d)] = apportion(svc, fd["cost"], fd["billed_s"], busy)
            app[("ipv4/" + svc, d)] = apportion("ipv4/" + svc, ipv4_day[(svc, d)], fd["billed_s"], busy)

    # ---- per-pass shared driver overhead ----
    sfn_rate = rates["sfn_state_transition"]
    driver = defaultdict(lambda: {"step_functions": 0.0, "scheduler": 0.0, "lambda_batch_driver": 0.0})
    driver_lambda_used = set()
    for pid, rids in pass_runs.items():
        e = exec_by_pass.get(pid)
        if not e:
            continue
        sf_cost = e["state_transitions"] * sfn_rate
        start, stop = _dt(e["start"]), _dt(e["stop"]) or _dt(e["start"])
        lam = 0.0
        for group, fn in (("/aws/lambda/lda-batch-step", "lda-batch-step"),
                          ("/aws/lambda/lda-batch-watchdog", "lda-batch-watchdog")):
            cfg = lcfg[fn]
            for rep in lrep.get(group, {}).get("reports", []):
                ts = _dt(rep["ts"])
                if start - timedelta(seconds=5) <= ts <= stop + timedelta(seconds=5):
                    lam += lambda_invocation_cost(rep["billed_ms"], cfg["memory_mb"], cfg["arch"], rates)
                    driver_lambda_used.add(rep["request_id"])
        for rid, v in even_split(sf_cost, rids).items():
            driver[rid]["step_functions"] += v
        for rid, v in even_split(rates["scheduler_invocation"], rids).items():  # one at() fire per pass
            driver[rid]["scheduler"] += v
        for rid, v in even_split(lam, rids).items():
            driver[rid]["lambda_batch_driver"] += v

    # ---- day-level log/span ingestion ----
    def logs_day(group, d):
        return _day_value(metrics["daily"].get(f"logs/{group}/IncomingBytes", []), d) or 0.0

    runs_per_day = defaultdict(list)
    for rid, v in q.items():
        runs_per_day[v["day"]].append(rid)
    logs_cost_day = {}
    for d in all_days:
        c = 0.0
        for g in PIPELINE_LOG_GROUPS:
            c += ingest_cost(logs_day(g, d), rates["logs_otel_ingest_gb" if g in OTEL_LOG_GROUPS else "logs_ingest_gb"])
        logs_cost_day[d] = c

    # ---- assemble rows ----
    for rid, v in q.items():
        d = v["day"]
        row = {"run_id": rid, "class": v["class"], "day": d, "pass": v["pass"], "decision": v["decision"]}
        cells = {}
        cells["bedrock_opus"] = (bedrock_cost(BEDROCK_MODEL_ID_FOR(v), v["input_tokens"], v["output_tokens"])
                                 if v["input_tokens"] or v["output_tokens"] else 0.0, SOURCE_APP_METERED)
        for svc, col in (("lda-embed", "fargate_embed"), ("lda-coordinator", "fargate_coordinator")):
            a = app[(svc, d)]
            fd = fargate_day[(svc, d)]
            cells[col] = (a.absorbed_by_run.get(rid, 0.0), fd["source"])
            row[f"{col}_marginal_usd"] = a.marginal_by_run.get(rid, 0.0)
        ip = sum(app[("ipv4/" + s, d)].absorbed_by_run.get(rid, 0.0) for s in TASK_SIZES)
        row["vpc_public_ipv4_marginal_usd"] = sum(app[("ipv4/" + s, d)].marginal_by_run.get(rid, 0.0)
                                                  for s in TASK_SIZES)
        cells["vpc_public_ipv4"] = (ip, SOURCE_LIST_RATE_ESTIMATE)
        knn_cfg, tic_cfg = lcfg["lda-knn"], lcfg["lda-ticket"]
        lam = 0.0
        if v["knn_billed_ms"]:
            lam += lambda_invocation_cost(v["knn_billed_ms"], knn_cfg["memory_mb"], knn_cfg["arch"], rates,
                                          ephemeral_mb=knn_cfg["ephemeral_mb"], requests=v["knn_calls"])
        if v["ticket_billed_ms"]:
            lam += lambda_invocation_cost(v["ticket_billed_ms"], tic_cfg["memory_mb"], tic_cfg["arch"], rates,
                                          ephemeral_mb=tic_cfg["ephemeral_mb"], requests=v["ticket_calls"])
        cells["lambda_knn_ticket"] = (lam, SOURCE_LIST_RATE_ESTIMATE)
        cells["s3_requests"] = (v["s3_put_exact"] * rates["s3_tier1_request"] + v["s3_get_est"] * rates["s3_tier2_request"],
                                SOURCE_LIST_RATE_ESTIMATE)
        cells["s3_storage_day"] = (storage_cost_per_day(v["s3_bytes"], rates["s3_standard_gb_month"]),
                                   SOURCE_LIST_RATE_ESTIMATE)
        cells["sqs"] = (v["sqs_requests"] * rates["sqs_standard_request"], SOURCE_LIST_RATE_ESTIMATE)
        cells["eventbridge"] = (EVENTBRIDGE_AWS_SERVICE_EVENT_USD, SOURCE_LIST_RATE_ESTIMATE)
        cells["dynamodb"] = (v["ddb_wru"] * rates["ddb_write_request_unit"], SOURCE_LIST_RATE_ESTIMATE)
        cells["cloudwatch_logs"] = (logs_cost_day[d] / len(runs_per_day[d]), SOURCE_LIST_RATE_ESTIMATE)
        cells["xray_spans"] = (ingest_cost(v["span_bytes"], rates["logs_otel_ingest_gb"]), SOURCE_LIST_RATE_ESTIMATE)
        dv = driver.get(rid)
        for col in ("step_functions", "scheduler", "lambda_batch_driver"):
            cells[col] = (dv[col], SOURCE_LIST_RATE_ESTIMATE) if dv else (0.0, "N/A")
        for col, (usd, src) in cells.items():
            row[f"{col}_usd"] = usd
            row[f"{col}_source"] = src
        row["total_absorbed_usd"] = sum(usd for usd, _ in cells.values())
        row["total_marginal_usd"] = (row["total_absorbed_usd"]
                                     - cells["fargate_embed"][0] - cells["fargate_coordinator"][0] - ip
                                     + row["fargate_embed_marginal_usd"] + row["fargate_coordinator_marginal_usd"]
                                     + row["vpc_public_ipv4_marginal_usd"])
        row["notes"] = "; ".join(notes[rid])
        row["_q"] = v
        rows[rid] = row

    # ---- unallocated / idle lines and day totals ----
    unalloc = []
    for d in all_days:
        for svc in TASK_SIZES:
            a = app[(svc, d)]
            fd = fargate_day[(svc, d)]
            if fd["billed_s"] or a.daily_cost_usd:
                unalloc.append({"day": d, "line": f"Fargate {svc} idle (task time not covered by any run)",
                                "usd": a.idle_residual_usd, "source": fd["source"],
                                "detail": f"billed {fd['billed_s']:.0f}s ({'/'.join(fd['methods']) or '-'}), busy "
                                          f"{a.busy_seconds_total:.0f}s, utilisation {a.utilisation:.1%}; absorbed "
                                          f"into runs in the fully-absorbed view"})
            ai = app[("ipv4/" + svc, d)]
            if ai.daily_cost_usd:
                unalloc.append({"day": d, "line": f"Public IPv4 on {svc} tasks, idle share", "usd": ai.idle_residual_usd,
                                "source": SOURCE_LIST_RATE_ESTIMATE, "detail": "absorbed in the fully-absorbed view"})
        # Lambda time not covered by run traces (bank builds, retries), from REPORT lines
        for group, fn in (("/lda/lambda/knn", "lda-knn"), ("/lda/lambda/ticket", "lda-ticket")):
            cfg = lcfg[fn]
            used = {k for r in runs if run_day(r["run_id"]) == d
                    for k, c in (xray.get(r["run_id"], {}).get("lambda_calls") or {}).items() if c["function"] == fn}
            rest = [rep for rep in lrep.get(group, {}).get("reports", []) if rep["ts"][:10] == d
                    and rep["request_id"] not in used]
            if rest:
                c = sum(lambda_invocation_cost(rep["billed_ms"], cfg["memory_mb"], cfg["arch"], rates,
                                               ephemeral_mb=cfg["ephemeral_mb"]) for rep in rest)
                unalloc.append({"day": d, "line": f"Lambda {fn} invocations not linked to a run trace", "usd": c,
                                "source": SOURCE_LIST_RATE_ESTIMATE,
                                "detail": f"{len(rest)} REPORT lines, {sum(r['billed_ms'] for r in rest)} billed ms "
                                          "(bank build / pre-trace smoke)"})
        # Reaper: no REPORT lines in its log group; use CloudWatch Duration/Invocations (Duration excludes init)
        inv = _day_value(metrics["daily"].get("lambda/lda-reaper/Invocations", []), d)
        dur = _day_value(metrics["daily"].get("lambda/lda-reaper/Duration", []), d)
        if inv:
            cfg = lcfg["lda-reaper"]
            c = lambda_invocation_cost(dur or 0, cfg["memory_mb"], cfg["arch"], rates, requests=int(inv))
            unalloc.append({"day": d, "line": "Lambda lda-reaper (EventBridge rate(15 min), runs all day)", "usd": c,
                            "source": SOURCE_LIST_RATE_ESTIMATE,
                            "detail": f"{inv:.0f} invocations, {dur or 0:.0f} ms (CloudWatch Duration, excl. init)"})
            unalloc.append({"day": d, "line": "EventBridge rule lda-idle-reaper", "usd": 0.0,
                            "source": SOURCE_LIST_RATE_ESTIMATE, "detail": "scheduled rules on the default bus: $0"})
        # Batch driver Lambdas outside any pass execution (watchdog ticks, dead-man, dry runs)
        c, n = 0.0, 0
        for group, fn in (("/aws/lambda/lda-batch-step", "lda-batch-step"),
                          ("/aws/lambda/lda-batch-watchdog", "lda-batch-watchdog"),
                          ("/aws/lambda/lda-batch-deadman", "lda-batch-deadman")):
            cfg = lcfg[fn]
            for rep in lrep.get(group, {}).get("reports", []):
                if rep["ts"][:10] == d and rep["request_id"] not in driver_lambda_used:
                    c += lambda_invocation_cost(rep["billed_ms"], cfg["memory_mb"], cfg["arch"], rates)
                    n += 1
        if n:
            unalloc.append({"day": d, "line": "Lambda batch driver outside pass executions (watchdog ticks, dead-man, tests)",
                            "usd": c, "source": SOURCE_LIST_RATE_ESTIMATE, "detail": f"{n} REPORT lines"})
        # Step Functions executions with no runs (kill-switch tests)
        sf = sum(e["state_transitions"] for e in sfn if (e["start"] or "")[:10] == d
                 and f"{e.get('batch_id')}/{e.get('pass_id')}" not in pass_runs)
        if sf:
            unalloc.append({"day": d, "line": "Step Functions executions with no runs (kill-switch tests)",
                            "usd": sf * sfn_rate, "source": SOURCE_LIST_RATE_ESTIMATE, "detail": f"{sf} transitions"})
        # Scheduler invocations not attributable to a pass fire
        sched = sum((_day_value(metrics["daily"].get(f"scheduler/{g}/InvocationAttemptCount", []), d) or 0)
                    for g in ("lda-batch", "lda-batch-deadman", "lda-batch-test"))
        fires = sum(1 for pid in pass_runs if (exec_by_pass.get(pid) or {}).get("start", "")[:10] == d)
        if sched:
            unalloc.append({"day": d, "line": "EventBridge Scheduler invocations other than pass fires", "usd":
                            max(0.0, sched - fires) * rates["scheduler_invocation"], "source": SOURCE_LIST_RATE_ESTIMATE,
                            "detail": f"{sched:.0f} attempts, {fires} pass fires attributed to runs; first 14M/month "
                                      "are $0 at list"})
        # SQS idle polling
        empty = _day_value(metrics["daily"].get("sqs/lda-door-events/NumberOfEmptyReceives", []), d)
        if empty:
            unalloc.append({"day": d, "line": "SQS empty long-poll receives (coordinator idle)", "usd": empty *
                            rates["sqs_standard_request"], "source": SOURCE_LIST_RATE_ESTIMATE,
                            "detail": f"{empty:.0f} NumberOfEmptyReceives"})
        # Driver logs and span residual
        dl = sum(logs_day(g, d) for g in DRIVER_LOG_GROUPS)
        if dl:
            unalloc.append({"day": d, "line": "CloudWatch Logs ingestion, batch driver log groups", "usd":
                            ingest_cost(dl, rates["logs_ingest_gb"]), "source": SOURCE_LIST_RATE_ESTIMATE,
                            "detail": f"{dl:.0f} bytes"})
        spans_in = logs_day("aws/spans", d)
        spans_runs = sum(q[rid]["span_bytes"] for rid in runs_per_day.get(d, []))
        if spans_in:
            unalloc.append({"day": d, "line": "X-Ray spans ingested (aws/spans) not covered by run traces",
                            "usd": ingest_cost(max(0.0, spans_in - spans_runs), rates["logs_otel_ingest_gb"]),
                            "source": SOURCE_LIST_RATE_ESTIMATE,
                            "detail": f"aws/spans IncomingBytes {spans_in:.0f} vs run trace documents {spans_runs:.0f} "
                                      "(account-wide log group; trace document size is a proxy for span bytes)"})
    # S3 storage that belongs to no run (models, bank, dataset, batch, build)
    other_bytes = sum(v["bytes"] for k, v in s3["bucket_totals_by_prefix"].items() if k not in ("runs", "incoming"))
    run_bytes_all = sum(v["s3_bytes"] for v in q.values())
    listed_run_bytes = sum(v["bytes"] for k, v in s3["bucket_totals_by_prefix"].items() if k in ("runs", "incoming"))
    s3_fixed = {"line": "S3 storage per day, objects not under any of these 55 runs", "usd":
                storage_cost_per_day(other_bytes + max(0, listed_run_bytes - run_bytes_all), rates["s3_standard_gb_month"]),
                "source": SOURCE_LIST_RATE_ESTIMATE,
                "detail": "; ".join(f"{k}/ {v['bytes'] / 1e9:.2f} GB" for k, v in sorted(s3["bucket_totals_by_prefix"].items()))
                + f"; runs+incoming not in this set {(listed_run_bytes - run_bytes_all) / 1e9:.2f} GB"}

    paths = _write_outputs(out_dir, rows, unalloc, s3_fixed, fargate_day, ftime, attr_status, attr, ce, rates, pricing,
                           q, days, all_days, today, metrics, lrep, sfn, fargate_rate_s)
    return paths


def BEDROCK_MODEL_ID_FOR(v: dict) -> str:  # noqa: N802 - single model in lda; kept explicit for the raise path
    from .rates import BEDROCK_MODEL_ID

    return BEDROCK_MODEL_ID


# -- output writers -----------------------------------------------------------------------------

def _fmt(x: float) -> str:
    if x == 0:
        return "0"
    if abs(x) >= 0.01:
        return f"{x:.4f}"
    return f"{x:.2e}"


def _stats(vals: list[float]) -> tuple[float, float, float]:
    return (statistics.median(vals), min(vals), max(vals)) if vals else (0.0, 0.0, 0.0)


def _ce_rows(ce: dict, name: str):
    for day in (ce.get("results") or {}).get(name, []):
        for g in day.get("Groups", []):
            yield day["TimePeriod"]["Start"], g["Keys"], float(g["Metrics"]["UnblendedCost"]["Amount"]), \
                float(g["Metrics"]["UsageQuantity"]["Amount"]), day.get("Estimated")


def _write_outputs(out_dir, rows, unalloc, s3_fixed, fargate_day, ftime, attr_status, attr, ce, rates, pricing, q,
                   days, all_days, today, metrics, lrep, sfn, fargate_rate_s) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    order = sorted(rows)
    paths = []

    # CSV
    csv_path = out_dir / "per_invocation.csv"
    cols = ["run_id", "class", "day", "pass", "decision"]
    for key, _ in SERVICE_COLUMNS:
        cols += [f"{key}_usd", f"{key}_source"]
    cols += ["fargate_embed_marginal_usd", "fargate_coordinator_marginal_usd", "vpc_public_ipv4_marginal_usd",
             "total_absorbed_usd", "total_marginal_usd", "input_tokens", "output_tokens", "embed_busy_s",
             "coordinator_busy_s", "knn_billed_ms", "knn_cold_start", "ticket_billed_ms", "s3_put_exact",
             "s3_get_est", "s3_bytes", "sqs_requests", "ddb_wru", "span_bytes", "notes"]
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for rid in order:
            r, v = rows[rid], rows[rid]["_q"]
            extra = {"input_tokens": v["input_tokens"], "output_tokens": v["output_tokens"],
                     "embed_busy_s": round(v["embed_busy_s"], 3), "coordinator_busy_s": round(v["coordinator_busy_s"], 3),
                     "knn_billed_ms": v["knn_billed_ms"], "knn_cold_start": v["knn_cold"],
                     "ticket_billed_ms": v["ticket_billed_ms"], "s3_put_exact": v["s3_put_exact"],
                     "s3_get_est": v["s3_get_est"], "s3_bytes": v["s3_bytes"], "sqs_requests": v["sqs_requests"],
                     "ddb_wru": v["ddb_wru"], "span_bytes": v["span_bytes"]}
            w.writerow([(f"{r[c]:.9f}" if isinstance(r.get(c), float) else r.get(c, extra.get(c))) for c in cols])
    paths.append(csv_path)

    # Per-invocation markdown
    md = out_dir / "per_invocation.md"
    L = ["# lda per-invocation cost, fully loaded", "",
         f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from `out/raw/` (collected "
         f"{_load(out_dir / 'raw', 'collect_meta.json')['collected_at'][:19]}Z). Account <ACCOUNT_ID>, us-east-1. "
         f"{len(rows)} runs. USD.", "",
         "Source tags: **AM** APP_METERED (lda token counts x app rate), **LR** LIST_RATE_ESTIMATE (counted quantity "
         "x AWS Pricing API list rate), **PA** PENDING_ATTRIBUTE (Attribute figure not available; the cell holds the "
         "list-rate estimate of the same task time, apportioned by busy-seconds), **AA** ATTRIBUTE_APPORTIONED "
         "(Attribute daily workload cost apportioned by lda busy-seconds), **na** not batch-driven.", "",
         "Fargate cells are the fully-absorbed view: the day's task cost is shared over that day's runs in proportion "
         "to busy-seconds, so idle task time is included. `marginal` = busy-seconds x per-second task rate only. The "
         "gap between the two is the idle line in the table further down. None of these figures is an amount "
         "Attribute attributed to a run; Attribute has no per-run join.", "",
         "Lines under US$0.0001 are kept to show the mechanism, not because they cost anything.", ""]
    hdr = ["run", "class", "pass"] + [lab for _, lab in SERVICE_COLUMNS] + ["total (absorbed)", "total (marginal)"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    for rid in order:
        r = rows[rid]
        cells = [rid.replace("run-", ""), r["class"], r["pass"].split("/")[-1]]
        for key, _ in SERVICE_COLUMNS:
            cells.append(f"{_fmt(r[f'{key}_usd'])} {TAG_ABBR[r[f'{key}_source']]}")
        cells += [f"**{r['total_absorbed_usd']:.4f}**", f"{r['total_marginal_usd']:.4f}"]
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "## Idle and unallocated lines (not in any run's marginal cost)", "",
          "| day | line | USD | source | detail |", "|---|---|---|---|---|"]
    for u in unalloc:
        L.append(f"| {u['day']} | {u['line']} | {_fmt(u['usd'])} | {u['source']} | {u['detail']} |")
    L.append(f"| every day | {s3_fixed['line']} | {_fmt(s3_fixed['usd'])} | {s3_fixed['source']} | {s3_fixed['detail']} |")
    L += ["", "## Daily totals", "",
          "| day | runs | Σ runs, fully absorbed | of which Bedrock | of which Fargate (embed+coord) | Σ runs, marginal | "
          "idle + unallocated lines | day total (Σ marginal + all idle/unallocated lines) |", "|---|---|---|---|---|---|---|---|"]
    for d in all_days:
        rr = [rows[k] for k in order if rows[k]["day"] == d]
        if not rr and not any(u["day"] == d for u in unalloc):
            continue
        absorbed = sum(r["total_absorbed_usd"] for r in rr)
        marg = sum(r["total_marginal_usd"] for r in rr)
        bed = sum(r["bedrock_opus_usd"] for r in rr)
        farg = sum(r["fargate_embed_usd"] + r["fargate_coordinator_usd"] for r in rr)
        un = sum(u["usd"] for u in unalloc if u["day"] == d)
        L.append(f"| {d} | {len(rr)} | {absorbed:.4f} | {bed:.4f} | {farg:.4f} | {marg:.4f} | {un:.4f} | "
                 f"{marg + un + s3_fixed['usd']:.4f} |")
    L += ["", "Day total includes the S3 storage line for objects outside these runs. It excludes costs that are not "
          "per-invocation and that Cost Explorer shows on the lda tag: ECR image storage (~US$0.003/day) and CodeBuild "
          "image builds (US$0.48 on 7 Nov). Not priced at all: the Route 53 private hosted zone that the Cloud Map "
          "namespace `lda.local` creates (monthly fee; CE's Route 53 line is untagged so it cannot be pinned to lda), "
          "Cloud Map DNS lookups, cross-AZ transfer (CE `DataTransfer-Regional-Bytes` US$0.0008 on 8 Nov) and "
          "Lambda/ECR image pulls.", "",
          "## Per-run notes", ""]
    for rid in order:
        if rows[rid]["notes"]:
            L.append(f"- `{rid}`: {rows[rid]['notes']}")
    md.write_text("\n".join(L) + "\n")
    paths.append(md)

    paths.append(_write_reconciliation(out_dir, rows, order, fargate_day, ftime, attr_status, attr, ce, rates, q,
                                       all_days, metrics, lrep, sfn, fargate_rate_s, unalloc))
    paths.append(_write_rates(out_dir, rates, pricing, fargate_rate_s))
    paths.append(_write_summary(out_dir, rows, order, attr_status, all_days, ce))
    return paths


def _write_reconciliation(out_dir, rows, order, fargate_day, ftime, attr_status, attr, ce, rates, q, all_days,
                          metrics, lrep, sfn, fargate_rate_s, unalloc) -> Path:
    p = out_dir / "reconciliation.md"
    L = ["# Reconciliation", "",
         f"Attribute REST API status at collection: **{attr.get('status', 'not queried')}**. "
         + ("The token in SSM `/luminara/attribute/api_token` (issued "
            f"{attr.get('token_iat', '?')[:10]}, 90-day expiry) returns 401 on `/api/v1/health`, so no Attribute "
            "figure is available for any day. Every Attribute comparison below is therefore **UNAVAILABLE**, not "
            "zero. A new REST token (dashboard Setup > API Tokens) written to that SSM parameter, then "
            "`cli.py all --cost-explorer`, fills these in." if str(attr.get("status", "")).startswith("UNAUTH") else ""),
         "",
         "Cost Explorer (`ce:GetCostAndUsage`, tag `Project=lift-door-agentic` plus the untagged Marketplace Bedrock "
         "line) is shown as a second, AWS-native billing cross-check. It is NOT one of the four cell sources. All "
         "Cost Explorer days below are flagged `Estimated` by AWS and still filling in.", ""]
    # Bedrock
    L += ["## 1. Bedrock tokens: Σ app-metered vs billing", "",
          "| UTC day | runs calling Opus | input tok | output tok | Σ app-metered USD | Attribute aicost Bedrock | "
          "delta vs Attribute | Cost Explorer 'Claude Opus 5.5 (Amazon Bedrock Edition)' (account-wide) | delta vs CE |",
          "|---|---|---|---|---|---|---|---|---|"]
    ce_bed = defaultdict(float)
    for d, keys, amt, _, _ in _ce_rows(ce, "by_service_and_project_tag"):
        if "Opus" in keys[0] or "Bedrock" in keys[0]:
            ce_bed[d] += amt
    for d in all_days:
        rr = [rows[k] for k in order if rows[k]["day"] == d]
        calls = [r for r in rr if r["_q"]["input_tokens"]]
        s = sum(r["bedrock_opus_usd"] for r in rr)
        a = attr_status[d]["bedrock"]
        cev = ce_bed.get(d)
        a_txt = "UNAVAILABLE" if a is None else f"{a:.5f}"
        da_txt = f"{s - a:+.5f} ({(s - a) / a:+.1%})" if a else "n/a"
        ce_txt = "not queried" if not ce else (f"{cev:.5f}" if cev is not None else "no line")
        dce_txt = f"{s - cev:+.5f} ({(s - cev) / cev:+.1%})" if cev else "n/a"
        L.append(f"| {d} | {len(calls)} | {sum(r['_q']['input_tokens'] for r in rr)} | "
                 f"{sum(r['_q']['output_tokens'] for r in rr)} | {s:.5f} | {a_txt} | {da_txt} | {ce_txt} | {dce_txt} |")
    # Cumulative by pass for days where CE is short: shows where CE's partial day appears to stop.
    for d in all_days:
        rr = sorted((rows[k] for k in order if rows[k]["day"] == d), key=lambda r: r["run_id"])
        if not ce or not ce_bed.get(d) or len({r["pass"] for r in rr}) < 3:
            continue
        L += ["", f"Cumulative Σ app-metered Opus on {d} by pass, against CE {ce_bed[d]:.5f} (hypothesis check: where "
              "does the partial CE day stop?):", "", "| through | Σ app-metered | minus CE |", "|---|---|---|"]
        cum, seen = 0.0, []
        for r in rr:
            if r["pass"] not in seen:
                if seen:
                    L.append(f"| {seen[-1]} | {cum:.5f} | {cum - ce_bed[d]:+.5f} |")
                seen.append(r["pass"])
            cum += r["bedrock_opus_usd"]
        L.append(f"| {seen[-1]} | {cum:.5f} | {cum - ce_bed[d]:+.5f} |")
    L += ["", "Notes: the Cost Explorer Bedrock line is account-wide and untagged (Marketplace product), so any other "
          "Opus 5.5 caller in <ACCOUNT_ID> on that day lands in it too, and any lda Opus call made outside these 55 "
          "result.json records (for example `scripts/bedrock_preflight.py`) is in CE but not in Σ app-metered. "
          "The 8 Nov CE figure is partial. The app rate is a frozen constant (see rates_used.md); a persistent "
          "same-sign delta on a settled day would mean the constant is wrong.", ""]
    # Fargate
    L += ["## 2. Fargate: Σ per-run shares vs workload day cost", "",
          "| UTC day | workload | billed task time | method | list-rate day cost | Σ per-run fully-absorbed | Σ per-run "
          "marginal | idle | Attribute EC2-ECS day cost | delta (list vs Attribute) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for d in all_days:
        for svc, col in (("lda-embed", "fargate_embed"), ("lda-coordinator", "fargate_coordinator")):
            fd = fargate_day[(svc, d)]
            if not fd["billed_s"]:
                continue
            rr = [rows[k] for k in order if rows[k]["day"] == d]
            ab = sum(r[f"{col}_usd"] for r in rr)
            mg = sum(r[f"{col}_marginal_usd"] for r in rr)
            at = fd["attribute_cost"]
            at_txt = "UNAVAILABLE" if at is None else f"{at:.5f}"
            delta_txt = "n/a" if at is None else f"{fd['list_cost'] - at:+.5f}"
            L.append(f"| {d} | {svc} | {fd['billed_s']:.0f} s | {'/'.join(fd['methods'])} | {fd['list_cost']:.5f} | "
                     f"{ab:.5f} | {mg:.5f} | {fd['cost'] - mg:.5f} | {at_txt} | {delta_txt} |")
    L += ["", "Σ fully-absorbed equals the day cost by construction (runs carry the idle). Task-time windows:", ""]
    for w in ftime["windows"]:
        L.append(f"- {w['service']} {w['first'].isoformat()[:16]}Z..{(w['last'] + timedelta(minutes=1)).isoformat()[11:16]}Z: "
                 f"{w['billed_s']:.0f} s billed, {w['method']}")
    L.append("")
    for svc, c in ftime["calibration"].items():
        L.append(f"- calibration {svc}: LiveTaskCount misses on average {c['mean_offset_s']:.0f} s per window "
                 f"(image pull + stopping) across {c['windows_with_exact']} windows with exact records "
                 f"({', '.join(f'{o:.0f}' for o in c['offsets_s'])} s)")
    # CE quantity cross-check
    if ce:
        L += ["", "## 3. Cost Explorer quantities on the lda tag vs this report's own counts", "",
              "| UTC day | usage type | CE quantity | CE USD (after free tier) | this report |", "|---|---|---|---|---|"]
        ours = {}
        for d in all_days:
            vcpu_h = sum(fargate_day[(s, d)]["billed_s"] / 3600 * TASK_SIZES[s]["vcpu"] for s in TASK_SIZES)
            gb_h = sum(fargate_day[(s, d)]["billed_s"] / 3600 * TASK_SIZES[s]["memory_gib"] for s in TASK_SIZES)
            ipv4_h = sum(fargate_day[(s, d)]["billed_s"] / 3600 for s in TASK_SIZES)
            rr = [rows[k]["_q"] for k in order if rows[k]["day"] == d]
            st = sum(e["state_transitions"] for e in sfn if (e["start"] or "")[:10] == d)
            ours[d] = {"USE1-Fargate-ARM-vCPU-Hours:perCPU": f"{vcpu_h:.4f} vCPU-h (task records)",
                       "USE1-Fargate-ARM-GB-Hours": f"{gb_h:.4f} GB-h",
                       "USE1-PublicIPv4:InUseAddress": f"{ipv4_h:.4f} h",
                       "Requests-Tier1": f"{sum(v['s3_put_exact'] for v in rr)} pipeline PUT/COPY (runs/+incoming/ "
                                         "listing; excludes batch/ driver writes and bank/model uploads)",
                       "Requests-Tier2": f"{sum(v['s3_get_est'] for v in rr)} pipeline GET (code path; excludes "
                                         "driver HEAD polling, checkpoint download, operator reads)",
                       "USE1-StateTransition": f"{st} (GetExecutionHistory *StateEntered)",
                       "WriteRequestUnits": f"{sum(v['ddb_wru'] for v in rr)} (2 per ticket: table + GSI)"}
        for d, keys, amt, qty, est in _ce_rows(ce, "lda_tag_by_usage_type"):
            if d in ours and (amt > 0 or qty > 0):
                L.append(f"| {d} | {keys[0]} | {qty:.4f} | {amt:.6f} | {ours[d].get(keys[0], '')} |")
        L += ["", "Cost Explorer tag-filtered ECS/Fargate cost per day:"]
        for d, keys, amt, qty, est in _ce_rows(ce, "by_service_and_project_tag"):
            if "lift-door" in keys[1] and amt > 0:
                L.append(f"- {d} {keys[0]}: US${amt:.5f} (Estimated={est})")
    # Telemetry self-consistency
    qq = [rows[k]["_q"] for k in order]
    put_ok = sum(1 for v in qq if v["s3_put_exact"] == v["s3_put_predicted"])
    xr = [v for v in qq if v["s3_get_xray_coord_embed"] is not None]
    get_ok = sum(1 for v in xr if v["s3_get_xray_coord_embed"] == v["s3_get_code_coord_embed"])
    knn_ok = sum(1 for v in qq if v["knn_calls"] and not v["knn_unmatched"])
    cold = sum(1 for v in qq if v["knn_cold"])
    L += ["", "## 4. Telemetry self-checks (do the counted quantities agree with each other?)", "",
          f"- S3 PUT: listed objects under `runs/<id>/` + `incoming/<id>/` equal the code-path PUT prediction "
          f"(coordinator claim/result[/reasoner meta], embed frames/tokens, knn anomaly map/detector[/map+diffs], "
          f"incoming clip) for **{put_ok}/{len(qq)}** runs.",
          f"- S3 GET: X-Ray `GetObject` spans from coordinator+embed equal the code path (1 clip GET, +3 on alert) for "
          f"**{get_ok}/{len(xr)}** traced runs. lda-knn's S3 calls are not instrumented in X-Ray, so its 3-4 GETs per "
          f"run (+7 bank GETs on a cold start, {cold} cold starts among run invocations) remain code-path estimates. "
          "The coordinator's `claim.json` PUT happens before its root span opens, which is why X-Ray shows one PUT "
          "fewer than the listing.",
          f"- Lambda: lda-knn invocations on run traces matched to a REPORT line by request id: **{knn_ok}/"
          f"{sum(1 for v in qq if v['knn_calls'])}**. Billed ms includes cold-start init (image functions bill init).",
          f"- X-Ray: traces retrieved for {len(xr)}/{len(qq)} runs."]
    L += ["", "## 5. Lambda billed time vs CloudWatch", "",
          "| UTC day | function | REPORT lines | Σ billed ms (incl. init) | CloudWatch Invocations | CloudWatch Duration ms "
          "(excl. init) | Σ billed ms on run traces |", "|---|---|---|---|---|---|---|"]
    for d in all_days:
        for group, fn in (("/lda/lambda/knn", "lda-knn"), ("/lda/lambda/ticket", "lda-ticket")):
            reps = [r for r in lrep.get(group, {}).get("reports", []) if r["ts"][:10] == d]
            inv = _day_value(metrics["daily"].get(f"lambda/{fn}/Invocations", []), d)
            dur = _day_value(metrics["daily"].get(f"lambda/{fn}/Duration", []), d)
            on_runs = sum(rows[k]["_q"]["knn_billed_ms" if fn == "lda-knn" else "ticket_billed_ms"] for k in order
                          if rows[k]["day"] == d)
            if reps or inv:
                L.append(f"| {d} | {fn} | {len(reps)} | {sum(r['billed_ms'] for r in reps)} | "
                         f"{'-' if inv is None else f'{inv:.0f}'} | {'-' if dur is None else f'{dur:.0f}'} | {on_runs} |")
    p.write_text("\n".join(L) + "\n")
    return p


def _write_rates(out_dir, rates, pricing, fargate_rate_s) -> Path:
    p = out_dir / "rates_used.md"
    L = ["# Rates used", "", f"AWS list rates: {pricing['source']}, fetched {pricing['fetched_at'][:19]}Z. "
         "Free tiers are not applied (Cost Explorer shows Lambda, SQS, Step Functions and Scheduler billing $0.00 "
         "on 7-8 Nov because of them).", "",
         "| key | USD | unit | usage type | publication date | description |", "|---|---|---|---|---|---|"]
    for r in rates.all():
        L.append(f"| {r.key} | {r.usd:.10g} | {r.unit} | {r.service} {r.usagetype} | {r.publication_date[:10]} | "
                 f"{r.description} |")
    L += ["", f"Bedrock: {BEDROCK_RATE_SOURCE}.", f"EventBridge: {EVENTBRIDGE_RATE_SOURCE}.",
          "X-Ray: spans go to the `aws/spans` log group (`GetTraceSegmentDestination` = CloudWatchLogs, Transaction "
          "Search). Priced at the CloudWatch OTel ingest rate on the trace's segment-document bytes (a proxy for "
          "ingested span bytes). Span indexing is assumed to be the free default 1%.", "",
          "Derived Fargate per-second rates (ARM64, Linux):"]
    for svc, r in fargate_rate_s.items():
        L.append(f"- {svc} {TASK_SIZES[svc]}: US${r:.8f}/s = US${r * 3600:.5f}/task-hour")
    p.write_text("\n".join(L) + "\n")
    return p


def _ce_bedrock_sentence(rr, ce) -> str:
    if not ce:
        return ""
    ce_bed = defaultdict(float)
    for d, keys, amt, _, _ in _ce_rows(ce, "by_service_and_project_tag"):
        if "Opus" in keys[0] or "Bedrock" in keys[0]:
            ce_bed[d] += amt
    parts, prefix_notes = [], []
    for d in sorted(ce_bed):
        day_rows = sorted((r for r in rr if r["day"] == d), key=lambda r: r["run_id"])
        s = sum(r["bedrock_opus_usd"] for r in day_rows)
        if not (ce_bed[d] and s):
            continue
        parts.append(f"{d} app-metered US${s:.5f} vs Cost Explorer US${ce_bed[d]:.5f} ({(s - ce_bed[d]) / ce_bed[d]:+.1%})")
        if s - ce_bed[d] > 0.001:  # CE short: does a pass prefix match it?
            cum, last = 0.0, None
            for i, r in enumerate(day_rows):
                cum += r["bedrock_opus_usd"]
                nxt = day_rows[i + 1]["pass"] if i + 1 < len(day_rows) else None
                if nxt != r["pass"] and abs(cum - ce_bed[d]) < 0.0001:
                    last = r["pass"].split("/")[-1]
            if last:
                prefix_notes.append(f"The {d} CE figure equals the app-metered sum through {last} to within "
                                    "US$0.0001, which fits CE not yet including the later passes")
    if not parts:
        return ""
    return (" As a stand-in for Attribute's aicost check, Cost Explorer's account-level Opus 5.5 line (Estimated, "
            "still filling in) gives: " + "; ".join(parts) + "."
            + (" " + "; ".join(prefix_notes) + "." if prefix_notes else "") + " See reconciliation.md.")


def _write_summary(out_dir, rows, order, attr_status, all_days, ce) -> Path:
    p = out_dir / "SUMMARY.md"
    rr = [rows[k] for k in order]
    by = {}
    for key, lab in SERVICE_COLUMNS:
        by[lab] = _stats([r[f"{key}_usd"] for r in rr])
    tot = _stats([r["total_absorbed_usd"] for r in rr])
    marg = _stats([r["total_marginal_usd"] for r in rr])
    cls = {}
    for c in ("healthy", "anomaly", "stress"):
        vals = [r for r in rr if r["class"] == c]
        cls[c] = (len(vals), _stats([r["total_absorbed_usd"] for r in vals]), _stats([r["total_marginal_usd"] for r in vals]),
                  _stats([r["bedrock_opus_usd"] for r in vals]))
    pending = sorted({d for r in rr for d in [r["day"]] if r["fargate_embed_source"] == SOURCE_PENDING_ATTRIBUTE})
    svc_txt = "; ".join(f"{lab} {_fmt(m)} ({_fmt(lo)}-{_fmt(hi)})" for lab, (m, lo, hi) in by.items())
    cls_txt = "; ".join(f"{c} (n={n}) {a[0]:.4f} ({a[1]:.4f}-{a[2]:.4f}) absorbed, {mg[0]:.4f} marginal, of which "
                        f"Opus {b[0]:.4f}" for c, (n, a, mg, b) in cls.items())
    reason = attr_status[pending[0]]["reason"] if pending else ""
    text = (
        f"Across {len(rr)} lda invocations (4 smoke runs on 7 Nov; 1 smoke and 2 end-to-end test runs on 8 Nov; the "
        f"8 Nov batch of 12 clips x 4 passes), the fully loaded cost per invocation has a median of US${tot[0]:.4f} "
        f"(range {tot[1]:.4f}-{tot[2]:.4f}) "
        f"when each day's Fargate task time, idle included, is absorbed by that day's runs in proportion to busy-seconds, "
        f"and US${marg[0]:.4f} ({marg[1]:.4f}-{marg[2]:.4f}) marginal, counting busy-seconds only. "
        f"Median (min-max) by service in USD: {svc_txt}. By class: {cls_txt}. "
        f"Opus is only called when the V-JEPA kNN detector alerts, which accounts for most of the spread between "
        f"classes (the healthy maximum is healthy_ball_reopen, which alerts and is then cleared by Opus as expected "
        f"behaviour; the 7 Nov runs also absorb a bank-build task window). Fargate embed (16 vCPU / 32 GiB) is the largest non-token line and sits mostly idle between "
        f"passes (see the idle lines). Every request-priced service (S3, SQS, Lambda, DynamoDB, Step Functions, "
        f"Scheduler, EventBridge, CloudWatch Logs, X-Ray) is below a tenth of a cent per run: mechanism, not money. "
        f"Sources: Bedrock is APP_METERED (token counts x the app's frozen rate). Fargate cells are PENDING_ATTRIBUTE "
        f"for {', '.join(pending) or 'no day'}: {reason}. Until a fresh token lands they hold the list-rate cost of "
        f"the recorded ECS task time, apportioned the way the Attribute figure will be. Everything else is "
        f"LIST_RATE_ESTIMATE. None of these numbers is an amount Attribute attributed to a run."
        + _ce_bedrock_sentence(rr, ce))
    p.write_text("# Summary\n\n" + text + "\n")
    return p
