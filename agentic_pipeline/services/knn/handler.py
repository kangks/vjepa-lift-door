from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
from opentelemetry import baggage, context
from opentelemetry.trace import Status, StatusCode

from lda_core.design import BANK, design_sha256
from lda_core.detector import build_envelope, detect, unit
from lda_core.evidence import build_evidence
from lda_core.storage import from_env
from lda_core.telemetry import flush, init_lambda, set_genai_attrs, start_server_span

_BANK_CACHE = {}
TRACER = None


def _json_bytes(value) -> bytes:
    return json.dumps(value, separators=(",", ":"), default=str).encode()


def _load_bank(storage, bank_prefix: str):
    cache_key = (os.environ.get("LDA_STORAGE", ""), bank_prefix)
    if cache_key in _BANK_CACHE:
        return _BANK_CACHE[cache_key]
    manifest = json.loads(storage.get_bytes(f"{bank_prefix}/manifest.json"))
    bank_tokens = {
        name: unit(storage.get_npy(f"{bank_prefix}/tokens/{name}.npy").astype(np.float32))
        for name in manifest["clips"]
    }
    envelope = storage.get_npy(f"{bank_prefix}/envelope.npy").astype(np.float32)
    ref_frames_key = f"{bank_prefix}/ref_frames/healthy_clean.npy"
    value = (manifest, bank_tokens, envelope, ref_frames_key)
    _BANK_CACHE[cache_key] = value
    return value


def _resolve_bank_prefix(storage, event) -> str:
    if event.get("bank_prefix"):
        return event["bank_prefix"].rstrip("/")
    current = json.loads(storage.get_bytes("bank/CURRENT.json"))
    return current["bank_prefix"].rstrip("/")


def _without_excess(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "excess"}


def _build_bank(event, storage):
    clips = event["clips"]
    bank_tokens = {}
    manifest_clips = {}
    for name in BANK:
        row = clips[name]
        bank_tokens[name] = unit(storage.get_npy(row["tokens_key"]).astype(np.float32))
        frames = storage.get_npy(row["frames_key"])
        manifest_clips[name] = {
            "mp4_sha256": row["mp4_sha256"], "frames_sha16": hashlib.sha256(frames.tobytes()).hexdigest()[:16],
        }
    envelope = build_envelope(bank_tokens)
    prefix = f"bank/v2/{event['bank_build_id']}"
    for name in BANK:
        storage.put_npy(f"{prefix}/tokens/{name}.npy", bank_tokens[name].astype(np.float16))
    storage.put_npy(f"{prefix}/envelope.npy", envelope.astype(np.float32))
    ref = storage.get_npy(clips["healthy_clean"]["frames_key"]).astype(np.uint8)
    storage.put_npy(f"{prefix}/ref_frames/healthy_clean.npy", ref)
    manifest = {
        "design_sha256": design_sha256(),
        "ckpt_sha256": event.get("ckpt_sha256") or os.environ.get("LDA_CKPT_SHA256"),
        "clips": manifest_clips,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embed_image": event.get("embed_image"),
    }
    storage.put_bytes(f"{prefix}/manifest.json", _json_bytes(manifest), "application/json")
    storage.put_bytes("bank/CURRENT.json", _json_bytes({"bank_prefix": prefix}), "application/json")
    return {"bank_prefix": prefix, **manifest}


def _detect(event, storage):
    with TRACER.start_as_current_span("knn.load_bank"):
        bank_prefix = _resolve_bank_prefix(storage, event)
        manifest, bank_tokens, envelope, ref_frames_key = _load_bank(storage, bank_prefix)
    with TRACER.start_as_current_span("knn.score"):
        tokens = unit(storage.get_npy(event["tokens_key"]).astype(np.float32))
        frames = storage.get_npy(event["frames_key"]).astype(np.uint8)
        excluded = [
            name for name, clip in manifest["clips"].items()
            if clip.get("mp4_sha256") == event.get("mp4_sha256")
        ]
        result = detect(tokens, bank_tokens, envelope, exclude=excluded)
        base = f"runs/{event['run_id']}"
        anomaly_key = f"{base}/anomaly_map.npy"
        storage.put_npy(anomaly_key, result["excess"].astype(np.float32))
    response = _without_excess(result)
    response.update({
        "run_id": event["run_id"], "bank_prefix": bank_prefix,
        "anomaly_map_key": anomaly_key, "reference_frames_key": ref_frames_key,
        "evidence_keys": {},
    })
    if result["alert"]:
        with TRACER.start_as_current_span("knn.render_evidence"):
            ref_frames = storage.get_npy(ref_frames_key).astype(np.uint8)
            evidence = build_evidence(frames, ref_frames, result["excess"], result["peak_time_s"])
            map_key = f"{base}/evidence/map.png"
            storage.put_bytes(map_key, evidence["map_png"], "image/png")
            keys = {"map": map_key, "diffs": []}
            for k, png in evidence["diffs"]:
                key = f"{base}/evidence/diff_t{k / 4:.2f}.png"
                storage.put_bytes(key, png, "image/png")
                keys["diffs"].append(key)
            response["evidence_keys"] = keys
    storage.put_bytes(f"{base}/detector.json", _json_bytes(response), "application/json")
    return response


def handler(event, context_obj=None):
    global TRACER
    try:
        storage = from_env()
        TRACER = TRACER or init_lambda("lda-knn")
        run_id = event.get("run_id") or event.get("bank_build_id", "bank-build")
        baggage_ctx = baggage.set_baggage("lift.run_id", str(run_id))
        baggage_token = context.attach(baggage_ctx)
        try:
            with start_server_span(
                TRACER,
                "build_bank" if event.get("mode") == "build_bank" else "detect",
                event.get("_otel"),
            ) as root:
                set_genai_attrs(root, **{"lift.run_id": run_id})
                if event.get("mode") == "build_bank":
                    return _build_bank(event, storage)
                if event.get("mode", "detect") != "detect":
                    raise ValueError(f"unsupported knn mode: {event['mode']}")
                response = _detect(event, storage)
                set_genai_attrs(
                    root,
                    **{
                        "lift.run_id": run_id,
                        "lift.alert": response["alert"],
                        "lift.max_score_over_tau": response["max_score_over_tau"],
                        "lift.window_s": response["window_s"] or [],
                        "lift.bank_mode": response["bank_mode"],
                    },
                )
                return response
        except Exception as exc:
            span = __import__("opentelemetry.trace", fromlist=["get_current_span"]).get_current_span()
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)[:4096]))
            raise
        finally:
            context.detach(baggage_token)
    finally:
        flush()
