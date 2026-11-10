"""Read-only DoiT Attribute REST client for the lda cost joiner.

Adapted (copied, not imported) from
<LOCAL_PATH>/Documents/workplace/attributes/app/cost_joiner/attribute_client.py
and probe.py. Kept from the original, because each one is a real failure from
that project:

- 404 on submit is OVERLOADED (job incomplete OR no data): raise
  IndeterminateJobError, never return an empty list.
- A FAILED/ERROR job raises JobFailedError. The original once returned rows=[]
  here and priced every Fargate workload at $0.00 without an error.
- Rows live under "results" (checked first); the page token is "nextToken".
- 429 gets exponential backoff with jitter; one job in flight at a time.

Changed for lda:
- The token is read from SSM SecureString /luminara/attribute/api_token
  (account <ACCOUNT_ID>) inside this process only. It is never printed, logged,
  or written anywhere. `token_from_ssm` returns it to the caller's variable.
- The raw org-wide response cache lives OUTSIDE the git worktree (default
  /tmp/lda-attribute-cache). Attribute endpoints are org-scoped, so raw rows
  include other accounts' spend; only account-filtered rows reach out/.
- allow_partial is the caller's explicit choice and is stamped into the cache
  envelope. 8 Nov and 9 Nov are both inside the settlement window today.
- The disk cache is keyed by the fetch timestamp's hour as well, so a partial
  day is re-fetched rather than served stale forever.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

API_BASE = "https://api.app.attrb.io"
SSM_TOKEN_PARAMETER = "/luminara/attribute/api_token"
DEFAULT_CACHE_DIR = Path(os.environ.get("LDA_ATTRIBUTE_CACHE_DIR", "/tmp/lda-attribute-cache"))


class IndeterminateJobError(RuntimeError):
    """404 on submit or a poll timeout: not zero, not data. Unknown."""


class JobFailedError(RuntimeError):
    """Terminal FAILED/ERROR job: a server-side failure, not an empty day."""


@dataclass
class JobResult:
    endpoint: str
    target_date: str
    job_id: str
    reason: str
    rows: list[dict]
    allow_partial: bool
    from_cache: bool
    fetched_at: str


def token_from_ssm(profile: str = "attrb_sandbox", region: str = "us-east-1") -> str:
    """GetParameter WithDecryption. Returns the token to the caller only."""
    import boto3

    ssm = boto3.Session(profile_name=profile, region_name=region).client("ssm")
    value = ssm.get_parameter(Name=SSM_TOKEN_PARAMETER, WithDecryption=True)["Parameter"]["Value"]
    if not value:
        raise RuntimeError(f"SSM parameter {SSM_TOKEN_PARAMETER} is empty")
    return value


class AttributeApiClient:
    def __init__(self, token: str, *, cache_dir: Path | None = None, max_retries: int = 6,
                 base_backoff_s: float = 1.0) -> None:
        if not token:
            raise ValueError("AttributeApiClient requires the REST API JWT")
        self._token = token
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.max_retries = max_retries
        self.base_backoff_s = base_backoff_s

    def __repr__(self) -> str:  # never leak the token through a repr
        return f"AttributeApiClient(cache_dir={self.cache_dir!s})"

    def _get(self, path: str) -> tuple[int, dict | str]:
        req = urllib.request.Request(
            f"{API_BASE}{path}", method="GET",
            headers={"Accept": "application/json", "Authorization": f"Bearer {self._token}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body, status = resp.read().decode("utf-8"), resp.status
        except urllib.error.HTTPError as exc:
            body, status = exc.read().decode("utf-8"), exc.code
        except urllib.error.URLError as exc:
            return -1, str(exc.reason)
        try:
            return status, json.loads(body)
        except json.JSONDecodeError:
            return status, body

    def _get_with_backoff(self, path: str) -> tuple[int, dict | str]:
        attempt = 0
        while True:
            status, body = self._get(path)
            if status != 429:
                return status, body
            attempt += 1
            if attempt > self.max_retries:
                return status, body
            time.sleep(self.base_backoff_s * (2 ** (attempt - 1)) + random.uniform(0, 1.0))

    def _cache_path(self, endpoint: str, target_date: str, stamp: str) -> Path:
        return self.cache_dir / f"{endpoint}_daily_{target_date}_{stamp}.json"

    def get_daily(self, endpoint: str, target_date: str, *, allow_partial: bool,
                  max_results: int = 1000, poll_timeout_s: int = 180,
                  poll_interval_s: float = 3.0, use_cache: bool = True) -> JobResult:
        now = datetime.now(timezone.utc)
        stamp = now.strftime("%Y%m%dT%H")
        cache_path = self._cache_path(endpoint, target_date, stamp)
        if use_cache and cache_path.exists():
            cached = json.loads(cache_path.read_text())
            return JobResult(endpoint, target_date, cached["job_id"], cached["reason"], cached["rows"],
                             cached["allow_partial"], True, cached["fetched_at"])

        status, body = self._get_with_backoff(f"/api/v1/{endpoint}/daily/{target_date}")
        if status == 404:
            raise IndeterminateJobError(
                f"404 on submit for {endpoint}/daily/{target_date} (overloaded: incomplete OR no data)")
        if status != 200 or not isinstance(body, dict):
            raise RuntimeError(f"unexpected submit response for {endpoint}/daily/{target_date}: "
                               f"status={status} body={str(body)[:300]!r}")
        job_id = body.get("id") or body.get("jobId") or body.get("job_id")
        if not job_id:
            raise RuntimeError(f"no job id in JobDescription for {endpoint}/daily/{target_date}")
        reason = self._poll(job_id, timeout_s=poll_timeout_s, interval_s=poll_interval_s)
        if reason.upper() in {"FAILED", "ERROR"}:
            raise JobFailedError(f"job {job_id} for {endpoint}/daily/{target_date} terminated {reason!r}")
        if reason == "TIMEOUT_INDETERMINATE":
            raise IndeterminateJobError(f"job {job_id} for {endpoint}/daily/{target_date} poll timeout")
        rows = self._fetch_all_pages(job_id, max_results=max_results)
        fetched_at = now.isoformat()
        cache_path.write_text(json.dumps({"job_id": job_id, "reason": reason, "rows": rows,
                                          "allow_partial": allow_partial, "fetched_at": fetched_at}))
        os.chmod(cache_path, 0o600)
        return JobResult(endpoint, target_date, job_id, reason, rows, allow_partial, False, fetched_at)

    def _poll(self, job_id: str, *, timeout_s: int, interval_s: float) -> str:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status, body = self._get_with_backoff(f"/api/v1/job/{job_id}/status")
            if status != 404 and isinstance(body, dict):
                reason = body.get("status") or body.get("reason")
                if reason and str(reason).lower() not in {"pending", "running", "in_progress"}:
                    return str(reason)
            time.sleep(interval_s)
        return "TIMEOUT_INDETERMINATE"

    def _fetch_all_pages(self, job_id: str, *, max_results: int) -> list[dict]:
        rows: list[dict] = []
        token: str | None = None
        while True:
            suffix = f"?maxResults={max_results}" + (f"&token={token}" if token else "")
            status, body = self._get_with_backoff(f"/api/v1/job/{job_id}/fetch{suffix}")
            if status != 200 or not isinstance(body, dict):
                raise RuntimeError(f"unexpected fetch response for job {job_id}: status={status}")
            rows.extend(body.get("results") or body.get("rows") or body.get("data") or body.get("items") or [])
            token = body.get("nextToken") or body.get("token")
            if not token:
                return rows
