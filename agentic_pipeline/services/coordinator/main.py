from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.parse
from datetime import datetime, timezone

import numpy as np
import requests
from opentelemetry import baggage, context
from opentelemetry.context import Context
from opentelemetry.trace import SpanKind, Status, StatusCode

from lda_core.design import design_sha256
from lda_core.detector import unit
from lda_core.evidence import build_evidence
from lda_core.policy import decide
from lda_core.reasoner import MODEL, PRICE_IN, PRICE_OUT, build_request, cost_usd, fingerprints, validate
from lda_core.storage import from_env
from lda_core.telemetry import extract_context, flush, init, inject_traceparent, set_genai_attrs, trace_id_hex

TRACER = init("lift-door-coordinator")
logger = logging.getLogger(__name__)


def _json_bytes(value) -> bytes:
    return json.dumps(value, separators=(",", ":"), default=str).encode()


def _parse_s3_record(record: dict) -> tuple[str, str]:
    bucket = record["s3"]["bucket"]["name"]
    key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
    parts = key.split("/")
    if len(parts) < 3 or parts[0] != "incoming" or parts[-1] != "clip.mp4":
        raise ValueError(f"unsupported incoming object key: {key}")
    return parts[1], key


class VisibilityHeartbeat:
    def __init__(self, sqs_client, queue_url: str, receipt_handle: str):
        self.client = sqs_client
        self.queue_url = queue_url
        self.receipt_handle = receipt_handle
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.wait(300):
            self.client.change_message_visibility(
                QueueUrl=self.queue_url, ReceiptHandle=self.receipt_handle, VisibilityTimeout=1800,
            )

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2)


def _tool_span(name: str):
    span = TRACER.start_as_current_span(f"execute_tool {name}", kind=SpanKind.CLIENT)
    return span


def _invoke_knn(payload: dict) -> dict:
    url = os.environ.get("LDA_KNN_URL")
    if url:
        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
        return response.json()
    import boto3
    response = boto3.client("lambda").invoke(
        FunctionName=os.environ["LDA_KNN_FUNCTION"],
        InvocationType="RequestResponse",
        Payload=json.dumps(payload).encode(),
    )
    value = json.loads(response["Payload"].read())
    if response.get("FunctionError"):
        raise RuntimeError(value.get("errorMessage", str(value)))
    if "body" in value and isinstance(value["body"], str):
        value = json.loads(value["body"])
    return value


def _invoke_ticket(payload: dict) -> dict:
    url = os.environ.get("LDA_TICKET_URL")
    if url:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    import boto3
    response = boto3.client("lambda").invoke(
        FunctionName=os.environ["LDA_TICKET_FUNCTION"],
        InvocationType="RequestResponse",
        Payload=json.dumps(payload).encode(),
    )
    value = json.loads(response["Payload"].read())
    if response.get("FunctionError"):
        raise RuntimeError(value.get("errorMessage", str(value)))
    if "body" in value and isinstance(value["body"], str):
        value = json.loads(value["body"])
    return value


def _prepare_reasoner(storage, run_id: str, detector: dict):
    if not detector.get("window_s"):
        raise ValueError("alerting detector response is missing window_s")
    frames = storage.get_npy(f"runs/{run_id}/frames.npy").astype(np.uint8)
    reference_key = detector.get("reference_frames_key")
    if not reference_key:
        raise ValueError("detector response is missing reference_frames_key")
    reference = storage.get_npy(reference_key).astype(np.uint8)
    excess = storage.get_npy(f"runs/{run_id}/anomaly_map.npy").astype(np.float32)
    evidence = build_evidence(frames, reference, excess, detector["peak_time_s"])
    payload, image_count, _ = build_request(frames, reference, evidence, detector["window_s"])
    text_blocks = [item["text"] for item in payload["messages"][0]["content"] if item["type"] == "text"]
    image_labels = []
    for index, item in enumerate(payload["messages"][0]["content"]):
        if item["type"] == "image":
            prior = payload["messages"][0]["content"][index - 1]
            image_labels.append(prior.get("text", "image"))
    metadata = {
        "text_blocks": text_blocks,
        "image_labels": image_labels,
        "image_count": image_count,
        "fingerprints": fingerprints(),
    }
    return payload, metadata


def _call_reasoner(payload: dict) -> tuple[dict | None, dict]:
    if os.environ.get("LDA_REASONER_MODE", "").lower() == "dry-run":
        if os.environ.get("AWS_EXECUTION_ENV"):
            raise RuntimeError("LDA_REASONER_MODE=dry-run is forbidden in AWS")
        return None, {"error": "DRY_RUN"}
    from anthropic import AnthropicBedrockMantle
    region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    client = AnthropicBedrockMantle(aws_region=region, max_retries=0)
    start = time.perf_counter()
    raw = client.messages.create(**payload).model_dump(mode="json")
    text = "".join(item["text"] for item in raw["content"] if item["type"] == "text")
    output = json.loads(text)
    validate(output)
    usage = raw["usage"]
    return output, {
        "usage": usage, "cost_usd": round(cost_usd(usage), 4),
        "latency_s": time.perf_counter() - start,
    }


def _process_record(
    record: dict,
    *,
    storage=None,
    embed_call=None,
    detect_call=None,
    reasoner_call=None,
    ticket_call=None,
    ack=None,
    heartbeat=None,
) -> dict:
    storage = storage or from_env()
    run_id, clip_key = _parse_s3_record(record)
    claim_key = f"runs/{run_id}/claim.json"
    claim = _json_bytes({"run_id": run_id, "claimed_at": datetime.now(timezone.utc).isoformat()})
    if not storage.put_bytes_if_absent(claim_key, claim, "application/json"):
        print(json.dumps({"duplicate_delivery": run_id}, separators=(",", ":")), flush=True)
        if ack:
            ack()
        return {"duplicate_delivery": run_id}

    if heartbeat:
        heartbeat.start()
    baggage_ctx = baggage.set_baggage("lift.run_id", run_id, context=Context())
    baggage_ctx = baggage.set_baggage("session.id", run_id, context=baggage_ctx)
    baggage_token = context.attach(baggage_ctx)
    started = time.perf_counter()
    result = {"run_id": run_id, "clip_key": clip_key}
    reasoner_result = {
        "output": None, "usage": None, "cost_usd_est": 0.0, "latency_s": None,
        "image_count": 0, "fingerprints": fingerprints(), "error": None,
    }
    try:
        with TRACER.start_as_current_span(
            "invoke_agent lift-door-coordinator",
            context=baggage_ctx,
            kind=SpanKind.SERVER,
        ) as root:
            trace_id = trace_id_hex(root)
            set_genai_attrs(
                root,
                **{
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.agent.name": "lift-door-coordinator",
                    "openinference.span.kind": "AGENT",
                    "lift.run_id": run_id,
                    "lift.clip_key": clip_key,
                },
            )
            try:
                with _tool_span("embed") as span:
                    set_genai_attrs(
                        span,
                        **{
                            "gen_ai.operation.name": "execute_tool",
                            "gen_ai.tool.name": "embed",
                            "openinference.span.kind": "TOOL",
                        },
                    )
                    if embed_call:
                        embed_result = embed_call(run_id, clip_key, inject_traceparent())
                    else:
                        response = requests.post(
                            os.environ.get("LDA_EMBED_URL", "http://embed.lda.local:8080") + "/embed",
                            json={"run_id": run_id, "clip_key": clip_key},
                            headers=inject_traceparent(),
                            timeout=900,
                        )
                        response.raise_for_status()
                        embed_result = response.json()
                with _tool_span("detect") as span:
                    set_genai_attrs(
                        span,
                        **{
                            "gen_ai.operation.name": "execute_tool",
                            "gen_ai.tool.name": "detect",
                            "openinference.span.kind": "TOOL",
                        },
                    )
                    detect_event = {
                        "mode": "detect", "run_id": run_id, "clip_key": clip_key,
                        "tokens_key": embed_result["tokens_key"], "frames_key": embed_result["frames_key"],
                        "mp4_sha256": embed_result["mp4_sha256"], "_otel": inject_traceparent(),
                    }
                    detector = detect_call(detect_event) if detect_call else _invoke_knn(detect_event)
                    set_genai_attrs(
                        span,
                        **{
                            "lift.alert": detector["alert"],
                            "lift.max_score_over_tau": detector["max_score_over_tau"],
                            "lift.window_s": detector.get("window_s") or [],
                        },
                    )

                result["detector"] = detector
                decision_result = None
                if detector["alert"]:
                    with TRACER.start_as_current_span("chat anthropic.claude-opus-5-5") as llm_span:
                        payload, request_meta = _prepare_reasoner(storage, run_id, detector)
                        reasoner_result["image_count"] = request_meta["image_count"]
                        reasoner_result["fingerprints"] = request_meta["fingerprints"]
                        storage.put_bytes(
                            f"runs/{run_id}/reasoner_request_meta.json",
                            _json_bytes(request_meta), "application/json",
                        )
                        set_genai_attrs(
                            llm_span,
                            **{
                                "gen_ai.operation.name": "chat",
                                "gen_ai.provider.name": "aws.bedrock",
                                "gen_ai.request.model": MODEL,
                                "openinference.span.kind": "LLM",
                            },
                        )
                        call = reasoner_call or _call_reasoner
                        r_start = time.perf_counter()
                        output, detail = call(payload)
                        reasoner_result.update(detail)
                        reasoner_result["latency_s"] = reasoner_result.get("latency_s") or (
                            time.perf_counter() - r_start
                        )
                        reasoner_result["output"] = output
                        reasoner_result["error"] = detail.get("error")
                        if detail.get("usage"):
                            set_genai_attrs(
                                llm_span,
                                **{
                                    "gen_ai.usage.input_tokens": detail["usage"].get("input_tokens"),
                                    "gen_ai.usage.output_tokens": detail["usage"].get("output_tokens"),
                                },
                            )
                        if output:
                            set_genai_attrs(
                                llm_span, **{"lift.classification": output.get("classification")}
                            )
                    decision_result = decide(detector, output, reasoner_result["error"])
                else:
                    decision_result = decide(detector, None)

                result["decision"] = decision_result
                ticket_id = None
                if decision_result["decision"] == "TICKET":
                    ticket_event = {
                        "run_id": run_id, "clip_key": clip_key,
                        "classification": decision_result["effective_class"],
                        "window_s": detector.get("window_s"),
                        "recommended_action": (reasoner_result.get("output") or {}).get(
                            "recommended_action", "HUMAN_REVIEW"
                        ),
                        "trace_id": trace_id,
                    }
                    with _tool_span("create_ticket") as span:
                        set_genai_attrs(
                            span,
                            **{
                                "gen_ai.operation.name": "execute_tool",
                                "gen_ai.tool.name": "create_ticket",
                                "openinference.span.kind": "TOOL",
                            },
                        )
                        ticket_event["_otel"] = inject_traceparent()
                        ticket_result = ticket_call(ticket_event) if ticket_call else _invoke_ticket(ticket_event)
                        ticket_id = ticket_result["ticket_id"]
                        set_genai_attrs(span, **{"lift.ticket_id": ticket_id})
                result.update({
                    "trace_id": trace_id,
                    "reasoner": reasoner_result,
                    "ticket_id": ticket_id,
                    "timings": {"total": time.perf_counter() - started, **embed_result.get("seconds", {})},
                    "design_sha256": design_sha256(),
                    "bank_prefix": detector.get("bank_prefix"),
                })
                set_genai_attrs(root, **{"lift.decision": decision_result["decision"]})
                storage.put_bytes(
                    f"runs/{run_id}/result.json", _json_bytes(result), "application/json",
                )
                if ack:
                    ack()
                return result
            except Exception as exc:
                logger.exception("coordinator message processing failed for run_id=%s", run_id)
                root.record_exception(exc)
                root.set_status(Status(StatusCode.ERROR, str(exc)[:4096]))
                result.update({
                    "trace_id": trace_id, "error": f"{type(exc).__name__}: {exc}",
                    "reasoner": reasoner_result, "timings": {"total": time.perf_counter() - started},
                    "design_sha256": design_sha256(),
                })
                storage.put_bytes(f"runs/{run_id}/result.json", _json_bytes(result), "application/json")
                raise
    finally:
        context.detach(baggage_token)
        if heartbeat:
            heartbeat.stop()
        flush()


def _decode_delivery(message: dict) -> dict:
    if "Body" in message:
        return json.loads(message["Body"])
    return message


def parse_event_records(event: dict) -> list[dict]:
    """Normalize the original S3 notification and the STEP_04 EventBridge form."""
    if isinstance(event.get("Records"), list):
        return event["Records"]
    if event.get("detail-type") == "Object Created" and isinstance(event.get("detail"), dict):
        detail = event["detail"]
        bucket = (detail.get("bucket") or {}).get("name")
        key = (detail.get("object") or {}).get("key")
        if bucket and key:
            return [{"s3": {"bucket": {"name": bucket}, "object": {"key": key}}}]
    raise ValueError("expected S3 ObjectCreated Records or EventBridge Object Created event")


def _process_delivery(message: dict, ack=None, storage=None, heartbeat=None):
    event = _decode_delivery(message)
    records = parse_event_records(event)
    outcomes = []
    for record in records:
        outcomes.append(_process_record(
            record, storage=storage, ack=ack, heartbeat=heartbeat,
        ))
    return outcomes


def _file_queue_run_id(path, message) -> str:
    try:
        records = parse_event_records(_decode_delivery(message))
        return _parse_s3_record(records[0])[0]
    except Exception:
        return path.stem


def _process_file_queue_item(path, store, attempts: dict[str, int]) -> None:
    message = {}
    try:
        message = json.loads(path.read_text())
        _process_delivery(
            message,
            ack=lambda p=path: p.unlink(missing_ok=True),
            storage=store,
        )
        attempts.pop(path.name, None)
    except Exception as exc:
        attempt = attempts.get(path.name, 0) + 1
        attempts[path.name] = attempt
        print(json.dumps({
            "message_file": path.name,
            "attempt": attempt,
            "error": f"{type(exc).__name__}: {exc}",
        }), flush=True)
        if attempt < 2:
            time.sleep(2)
            return

        run_id = _file_queue_run_id(path, message)
        deadletter = {
            "run_id": run_id,
            "error": f"{type(exc).__name__}: {exc}",
            "attempts": attempt,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        store.put_bytes(f"deadletter/{run_id}.json", _json_bytes(deadletter), "application/json")
        dead_dir = path.parent / "dead"
        dead_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
        path.replace(dead_dir / path.name)
        attempts.pop(path.name, None)


def _run_file_queue(queue_dir: str):
    from pathlib import Path

    queue = Path(queue_dir)
    store = from_env()
    attempts = {}
    while True:
        pending = sorted(queue.glob("*.json")) if queue.exists() else []
        for path in pending:
            _process_file_queue_item(path, store, attempts)
        time.sleep(1)


def main():
    queue = os.environ.get("LDA_QUEUE", "")
    if queue.startswith("file:"):
        _run_file_queue(queue[len("file:"):])
        return
    queue_url = os.environ["LDA_QUEUE_URL"]
    import boto3
    sqs = boto3.client("sqs")
    while True:
        response = sqs.receive_message(
            QueueUrl=queue_url, MaxNumberOfMessages=1, WaitTimeSeconds=20, VisibilityTimeout=1800,
        )
        for message in response.get("Messages", []):
            heartbeat = VisibilityHeartbeat(sqs, queue_url, message["ReceiptHandle"])
            ack = lambda m=message: sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=m["ReceiptHandle"])
            _process_delivery(message, ack=ack, heartbeat=heartbeat)


if __name__ == "__main__":
    main()
