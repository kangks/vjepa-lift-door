#!/usr/bin/env bash
set -euo pipefail
[ "$#" = 1 ] || { echo "usage: $0 <profile>" >&2; exit 2; }
PROFILE=$1
A=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT="$A/outputs/${PROFILE}.json"
IFS=$'\t' read -r CLUSTER EMBED COORD < <(PYTHONPATH="$A" python3 - "$OUT" <<'PY'
import sys
from lda_core.output_utils import read_stack_outputs
v=read_stack_outputs(sys.argv[1], "LiftDoorAgentic")
print("\t".join(v[key] for key in ("ClusterName","EmbedServiceName","CoordinatorServiceName")))
PY
)
for SERVICE in "$EMBED" "$COORD"; do
  aws ecs update-service --profile "$PROFILE" --region us-east-1 \
    --cluster "$CLUSTER" --service "$SERVICE" --desired-count 0 >/dev/null
done
echo "desiredCount=0 for $EMBED and $COORD"
