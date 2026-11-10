from __future__ import annotations

import json
import os
from datetime import datetime, timezone


def handler(event, context=None):
    import boto3

    parameter = boto3.client("ssm").get_parameter(Name=os.environ.get("LDA_LEASE_PARAMETER", "/lda/lease_until"))
    lease_until = datetime.fromisoformat(parameter["Parameter"]["Value"].replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    if now <= lease_until:
        return {"action": "keep_running", "lease_until": lease_until.isoformat()}
    ecs = boto3.client("ecs")
    updated = []
    for item in json.loads(os.environ.get("LDA_SERVICES", "[]")):
        ecs.update_service(cluster=item["cluster"], service=item["service"], desiredCount=0)
        updated.append(item["service"])
    result = {"action": "scale_to_zero", "services": updated, "lease_until": lease_until.isoformat()}
    print(json.dumps(result, separators=(",", ":")))
    return result
