#!/usr/bin/env bash
set -euo pipefail
[ "$#" -ge 1 ] && [ "$#" -le 2 ] || { echo "usage: $0 <profile> [hours]" >&2; exit 2; }
PROFILE=$1
HOURS=${2:-2}
A=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=$(cd "$A/../../.." && pwd)
REGION=us-east-1
OUT="$A/outputs/${PROFILE}.json"
aws_args=(--profile "$PROFILE" --region "$REGION")
IFS=$'\t' read -r CLUSTER EMBED COORD < <(PYTHONPATH="$A" python3 - "$OUT" <<'PY'
import sys
from lda_core.output_utils import read_stack_outputs
v=read_stack_outputs(sys.argv[1], "LiftDoorAgentic")
print("\t".join(v[key] for key in ("ClusterName","EmbedServiceName","CoordinatorServiceName")))
PY
)
for SERVICE in "$EMBED" "$COORD"; do
  aws ecs update-service "${aws_args[@]}" --cluster "$CLUSTER" --service "$SERVICE" --desired-count 1 >/dev/null
done
python3 - "$HOURS" <<'PY' > /tmp/lda_lease_until.txt
from datetime import datetime, timedelta, timezone
import sys
print((datetime.now(timezone.utc)+timedelta(hours=float(sys.argv[1]))).isoformat().replace("+00:00","Z"))
PY
LEASE=$(cat /tmp/lda_lease_until.txt)
aws ssm put-parameter "${aws_args[@]}" --name /lda/lease_until --type String --overwrite --value "$LEASE" >/dev/null
echo "lease_until=$LEASE"
