from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

from lda_core.design import BANK


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _post_json(url: str, payload: dict, timeout: int = 900) -> dict:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    value = response.json()
    if isinstance(value, dict) and "body" in value and isinstance(value["body"], str):
        value = json.loads(value["body"])
    if not isinstance(value, dict):
        raise ValueError(f"service returned a non-object JSON response from {url}")
    return value


def _wait_for_embed_ready(
    embed_url: str,
    expected_ckpt_sha: str,
    *,
    ready_timeout: float = 300,
    poll_interval: float = 5,
) -> None:
    if not expected_ckpt_sha:
        raise ValueError("embed readiness requires --ckpt-sha or LDA_CKPT_SHA256")
    health_url = embed_url.rstrip("/") + "/health"
    deadline = time.monotonic() + ready_timeout
    last_status = "no health response"
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"embed readiness timed out after {ready_timeout:g}s at {health_url}: {last_status}"
            )
        try:
            response = requests.get(health_url, timeout=min(5, remaining))
            if response.status_code == 200:
                try:
                    health = response.json()
                except ValueError:
                    health = {}
                actual_sha = health.get("ckpt_sha256")
                if actual_sha == expected_ckpt_sha:
                    return
                last_status = (
                    f"HTTP 200 with ckpt_sha256={actual_sha!r}; "
                    f"expected {expected_ckpt_sha}"
                )
            else:
                last_status = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_status = f"{type(exc).__name__}: {exc}"

        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(poll_interval, remaining))


def _check_knn_tcp(knn_url: str, *, timeout: float = 5) -> None:
    parsed = urlparse(knn_url)
    if not parsed.hostname:
        raise ValueError(f"invalid knn URL: {knn_url}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return
    except OSError as exc:
        raise ConnectionError(
            f"knn TCP check failed for {parsed.hostname}:{port}: {exc}"
        ) from exc


def build_bank(
    store_dir: Path,
    *,
    embed_url: str = "http://embed:8080",
    knn_url: str = "http://knn:8080/2015-03-31/functions/function/invocations",
    build_id: str | None = None,
    ckpt_sha: str | None = None,
    ready_timeout: float = 300,
    poll_interval: float = 5,
) -> dict:
    store_dir = store_dir.expanduser().resolve()
    build_id = build_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(2)
    )
    manifest_path = store_dir / "models" / "vjepa21" / "manifest.json"
    model_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    ckpt_sha = ckpt_sha or os.environ.get("LDA_CKPT_SHA256") or model_manifest.get("sha256")
    _wait_for_embed_ready(
        embed_url,
        ckpt_sha or "",
        ready_timeout=ready_timeout,
        poll_interval=poll_interval,
    )
    _check_knn_tcp(knn_url)
    clips = {}
    for name in BANK:
        matches = list((store_dir / "dataset" / "lift_dataset").glob(f"*/{name}/{name}.mp4"))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one seeded clip for {name}, found {len(matches)}")
        video = matches[0]
        key = video.relative_to(store_dir).as_posix()
        embedded = _post_json(
            embed_url.rstrip("/") + "/embed",
            {"run_id": f"bank-{build_id}-{name}", "clip_key": key},
        )
        clips[name] = {
            "frames_key": embedded["frames_key"],
            "tokens_key": embedded["tokens_key"],
            "mp4_sha256": embedded.get("mp4_sha256") or _sha256(video),
        }

    payload = {
        "mode": "build_bank",
        "bank_build_id": build_id,
        "clips": clips,
        "ckpt_sha256": ckpt_sha,
    }
    result = _post_json(knn_url, payload)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the five-clip healthy bank via local services.")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--embed-url", default="http://embed:8080")
    parser.add_argument(
        "--knn-url",
        default="http://knn:8080/2015-03-31/functions/function/invocations",
    )
    parser.add_argument("--ckpt-sha", default=None)
    parser.add_argument("--ready-timeout", type=float, default=300)
    parser.add_argument("--build-id")
    args = parser.parse_args(argv)
    build_bank(
        args.store,
        embed_url=args.embed_url,
        knn_url=args.knn_url,
        build_id=args.build_id,
        ckpt_sha=args.ckpt_sha,
        ready_timeout=args.ready_timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
