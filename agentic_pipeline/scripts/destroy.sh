#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: $0 <profile> [--purge-bucket]" >&2
  exit 2
fi
PROFILE=$1
PURGE=${2:-}
A=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=$(cd "$A/../../.." && pwd)
ACCOUNT=$(python3 - "$A/infra/accounts.json" "$PROFILE" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))[sys.argv[2]])
PY
)
cd "$A/infra"
CONTEXT=(-c "profile=$PROFILE" -c "region=us-east-1")
cdk destroy LiftDoorAgentic --profile "$PROFILE" "${CONTEXT[@]}" --force
cdk destroy LiftDoorAgenticBuild --profile "$PROFILE" "${CONTEXT[@]}" --force
if [ "$PURGE" = "--purge-bucket" ]; then
  BUCKET=$(PYTHONPATH="$A" python3 - "$A/outputs/${PROFILE}-build.json" <<'PY'
import sys
from lda_core.output_utils import read_stack_outputs
value=read_stack_outputs(sys.argv[1], "LiftDoorAgenticBuild")
print(value.get("BucketName") or value.get("bucket") or "")
PY
)
  [ -n "$BUCKET" ] || { echo "bucket name not found in build outputs" >&2; exit 1; }
  [ "$(aws sts get-caller-identity --profile "$PROFILE" --query Account --output text)" = "$ACCOUNT" ]
  aws s3 rm "s3://$BUCKET" --recursive --profile "$PROFILE" --region us-east-1
  aws s3api delete-bucket --bucket "$BUCKET" --profile "$PROFILE" --region us-east-1
fi
