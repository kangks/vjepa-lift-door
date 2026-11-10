from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from decimal import Decimal

from opentelemetry import baggage, context
from opentelemetry.trace import Status, StatusCode

from lda_core.storage import FileStorage
from lda_core.telemetry import flush, init_lambda, set_genai_attrs, start_server_span

TRACER = None


def _window_decimal(window):
    if window is None:
        return None
    return [Decimal(str(value)) for value in window]


def _file_put(path: str, item: dict) -> bool:
    storage = FileStorage(path)
    key = f"{item['ticket_id']}.json"
    created = storage.put_bytes_if_absent(key, json.dumps(item, default=str).encode(), "application/json")
    return created


def _put_dynamodb(table_name: str, item: dict) -> bool:
    import boto3
    from botocore.exceptions import ClientError

    table = boto3.resource("dynamodb").Table(table_name)
    try:
        table.put_item(Item=item, ConditionExpression="attribute_not_exists(ticket_id)")
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            return False
        raise


def handler(event, context_obj=None):
    global TRACER
    try:
        TRACER = TRACER or init_lambda("lda-ticket")
        run_id = str(event["run_id"])
        baggage_ctx = baggage.set_baggage("lift.run_id", run_id)
        baggage_token = context.attach(baggage_ctx)
        try:
            with start_server_span(TRACER, "create_ticket", event.get("_otel")) as span:
                ticket_id = "MOCK-" + hashlib.sha256(run_id.encode()).hexdigest()[:16]
                set_genai_attrs(
                    span,
                    **{
                        "lift.run_id": run_id, "lift.ticket_id": ticket_id, "lift.mock": True,
                    },
                )
                item = {
                    "ticket_id": ticket_id,
                    "run_id": run_id,
                    "clip_key": event["clip_key"],
                    "classification": event["classification"],
                    "window_s": _window_decimal(event.get("window_s")),
                    "recommended_action": event["recommended_action"],
                    "trace_id": event["trace_id"],
                    "mock": True,
                    "system": "MOCK-CMMS",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                file_mode = os.environ.get("LDA_TICKET_STORE", "")
                if file_mode.startswith("file:"):
                    created = _file_put(file_mode[len("file:"):], item)
                else:
                    created = _put_dynamodb(os.environ["LDA_TICKET_TABLE"], item)
                if not created:
                    if file_mode.startswith("file:"):
                        existing = json.loads(
                            FileStorage(file_mode[len("file:"):]).get_bytes(f"{ticket_id}.json")
                        )
                        ticket_id = existing["ticket_id"]
                    else:
                        import boto3
                        existing = boto3.resource("dynamodb").Table(os.environ["LDA_TICKET_TABLE"]).get_item(
                            Key={"ticket_id": ticket_id}, ConsistentRead=True,
                        )["Item"]
                        ticket_id = existing["ticket_id"]
                    set_genai_attrs(span, **{"lift.ticket_duplicate": True})
                    return {"ticket_id": ticket_id, "duplicate": True}
                return {"ticket_id": ticket_id, "duplicate": False}
        except Exception as exc:
            span = __import__("opentelemetry.trace", fromlist=["get_current_span"]).get_current_span()
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)[:4096]))
            raise
        finally:
            context.detach(baggage_token)
    finally:
        flush()
