from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from lda_core.encoder import build_encoder, dense_tokens
from lda_core.frames import decode_mp4, frames_sha16
from lda_core.storage import from_env
from lda_core.telemetry import flush, init, set_genai_attrs, start_server_span
from opentelemetry import baggage, context
from opentelemetry.trace import Status, StatusCode

SERVICE = "lda-embed"
logger = logging.getLogger(__name__)
TRACER = init(SERVICE)
STORAGE = from_env()
LOCK = asyncio.Lock()
ENCODER = None
CKPT_SHA256 = None
TORCH_THREADS = 1
CPU_COUNT = os.cpu_count()
app = FastAPI(title="Lift-door V-JEPA embed service")


class EmbedRequest(BaseModel):
    run_id: str
    clip_key: str


def _checkpoint() -> tuple[Path, str]:
    local_path = os.environ.get("LDA_CKPT_PATH")
    if not local_path:
        raise RuntimeError("LDA_CKPT_PATH is required")
    path = Path(local_path)
    if not path.is_absolute():
        raise ValueError("LDA_CKPT_PATH must be an absolute path")
    s3_key = os.environ.get("LDA_CKPT_S3_KEY")
    if not path.is_file():
        if not s3_key:
            raise FileNotFoundError(
                f"checkpoint file not found at {path}; LDA_CKPT_S3_KEY is required to download it"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(STORAGE.get_bytes(s3_key))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = os.environ.get("LDA_CKPT_SHA256")
    if not expected:
        manifest = json.loads(STORAGE.get_bytes("models/vjepa21/manifest.json"))
        expected = manifest["sha256"]
    if digest != expected:
        raise ValueError(f"checkpoint sha256 mismatch: expected {expected}, got {digest}")
    return path, digest


@app.on_event("startup")
def startup() -> None:
    global ENCODER, CKPT_SHA256, TORCH_THREADS
    try:
        requested = os.environ.get("LDA_TORCH_THREADS")
        TORCH_THREADS = max(1, int(requested)) if requested else max(1, CPU_COUNT or 1)
        torch.set_num_threads(TORCH_THREADS)
        checkpoint, CKPT_SHA256 = _checkpoint()
        ENCODER = build_encoder("vitl", checkpoint, img_size=384, num_frames=16).to("cpu")
    except Exception as exc:
        ENCODER = None
        raise RuntimeError(f"V-JEPA embed service startup failed: {exc}") from exc


@app.get("/health")
def health():
    if ENCODER is None:
        raise HTTPException(status_code=503, detail={
            "model_loaded": False, "torch_threads": TORCH_THREADS, "cpu_count": CPU_COUNT,
        })
    return {
        "model_loaded": True, "ckpt_sha256": CKPT_SHA256,
        "torch_threads": TORCH_THREADS, "cpu_count": CPU_COUNT,
    }


@app.post("/embed")
async def embed(payload: EmbedRequest, request: Request):
    if ENCODER is None:
        raise HTTPException(status_code=503, detail="encoder not loaded")
    async with LOCK:
        baggage_ctx = baggage.set_baggage("lift.run_id", payload.run_id)
        token = context.attach(baggage_ctx)
        try:
            with start_server_span(TRACER, "embed", dict(request.headers)) as server_span:
                server_span.set_attribute("lift.run_id", payload.run_id[:4096])
                try:
                    start = time.perf_counter()
                    clip_bytes = await asyncio.to_thread(STORAGE.get_bytes, payload.clip_key)
                    download_s = time.perf_counter() - start
                    mp4_sha256 = hashlib.sha256(clip_bytes).hexdigest()
                    with tempfile.NamedTemporaryFile(suffix=".mp4") as fp:
                        fp.write(clip_bytes)
                        fp.flush()
                        decode_start = time.perf_counter()
                        with TRACER.start_as_current_span("embed.decode_clip") as span:
                            set_genai_attrs(span, **{"lift.run_id": payload.run_id})
                            frames = await asyncio.to_thread(decode_mp4, fp.name)
                        decode_s = time.perf_counter() - decode_start

                    tokens = []
                    encode_windows = []
                    for index in range(4):
                        window = frames[index * 16:(index + 1) * 16]
                        encode_start = time.perf_counter()
                        with TRACER.start_as_current_span("embed.vjepa_encode_window") as span:
                            set_genai_attrs(
                                span, **{"lift.run_id": payload.run_id, "lift.window_index": index}
                            )
                            values = await asyncio.to_thread(dense_tokens, ENCODER, window, "cpu")
                        encode_windows.append(time.perf_counter() - encode_start)
                        tokens.append(values)

                    put_start = time.perf_counter()
                    frames_key = f"runs/{payload.run_id}/frames.npy"
                    tokens_key = f"runs/{payload.run_id}/tokens.npy"
                    await asyncio.to_thread(STORAGE.put_npy, frames_key, frames.astype(np.uint8))
                    await asyncio.to_thread(
                        STORAGE.put_npy, tokens_key, np.concatenate(tokens).astype(np.float16)
                    )
                    put_s = time.perf_counter() - put_start
                    return {
                        "run_id": payload.run_id, "frames_key": frames_key, "tokens_key": tokens_key,
                        "mp4_sha256": mp4_sha256, "frames_sha16": frames_sha16(frames),
                        "seconds": {
                            "download": download_s, "decode": decode_s,
                            "encode_windows": encode_windows, "put": put_s,
                        },
                    }
                except Exception as exc:
                    logger.exception("embed request failed")
                    server_span.record_exception(exc)
                    server_span.set_status(Status(StatusCode.ERROR, str(exc)[:4096]))
                    return JSONResponse(
                        status_code=500,
                        content={"error": type(exc).__name__, "message": str(exc)[:500]},
                    )
        finally:
            context.detach(token)
            flush()
