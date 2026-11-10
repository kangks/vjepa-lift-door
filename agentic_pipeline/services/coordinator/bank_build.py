from __future__ import annotations

import argparse
import json
import os
import time

import requests
from opentelemetry.trace import SpanKind

from lda_core.design import BANK
from lda_core.telemetry import init, inject_traceparent, set_genai_attrs, trace_id_hex

TRACER = init("lda-coordinator")


def _call_knn(payload):
    url = os.environ.get("LDA_KNN_URL")
    if url:
        response = requests.post(url, json=payload, headers=inject_traceparent(), timeout=900)
        response.raise_for_status()
        return response.json()
    import boto3
    result = boto3.client("lambda").invoke(
        FunctionName=os.environ["LDA_KNN_FUNCTION"],
        InvocationType="RequestResponse",
        Payload=json.dumps({**payload, "_otel": inject_traceparent()}).encode(),
    )
    value = json.loads(result["Payload"].read())
    if result.get("FunctionError"):
        raise RuntimeError(value.get("errorMessage", str(value)))
    if "body" in value and isinstance(value["body"], str):
        value = json.loads(value["body"])
    return value


def build_bank(bank_build_id: str, *, embed_call=None, knn_call=None) -> dict:
    clips = {}
    embed_url = os.environ.get("LDA_EMBED_URL", "http://embed.lda.local:8080").rstrip("/")
    for name in BANK:
        run_id = f"bank-{bank_build_id}-{name}"
        clip_key = f"dataset/lift_dataset/healthy/{name}/{name}.mp4"
        if embed_call:
            result = embed_call(run_id, clip_key, inject_traceparent())
        else:
            response = requests.post(
                f"{embed_url}/embed", json={"run_id": run_id, "clip_key": clip_key},
                headers=inject_traceparent(), timeout=900,
            )
            response.raise_for_status()
            result = response.json()
        clips[name] = {
            "tokens_key": result["tokens_key"],
            "frames_key": result["frames_key"],
            "mp4_sha256": result["mp4_sha256"],
        }
    payload = {
        "mode": "build_bank",
        "bank_build_id": bank_build_id,
        "clips": clips,
        "ckpt_sha256": os.environ.get("LDA_CKPT_SHA256"),
        "embed_image": os.environ.get("LDA_EMBED_IMAGE"),
    }
    if knn_call:
        result = knn_call(payload)
    else:
        result = _call_knn(payload)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-build-id", required=True)
    args = parser.parse_args(argv)
    start = time.perf_counter()
    with TRACER.start_as_current_span("bank_build", kind=SpanKind.INTERNAL) as span:
        set_genai_attrs(span, **{"lift.bank_build_id": args.bank_build_id})
        result = build_bank(args.bank_build_id)
        set_genai_attrs(
            span,
            **{
                "lift.bank_prefix": result.get("bank_prefix"),
                "lift.design_sha256": result.get("design_sha256"),
                "lift.duration_s": time.perf_counter() - start,
            },
        )
        result["trace_id"] = trace_id_hex(span)
    print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
