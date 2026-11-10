#!/usr/bin/env bash
set -euo pipefail
[ "$#" = 2 ] || { echo "usage: $0 <profile> <bank_build_id>" >&2; exit 2; }
PROFILE=$1
BANK_BUILD_ID=$2
A=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MAIN_OUT="$A/outputs/${PROFILE}.json"
BUILD_OUT="$A/outputs/${PROFILE}-build.json"
REGION=us-east-1
IFS=$'\t' read -r CLUSTER SERVICE TASK SUBNETS SG < <(PYTHONPATH="$A" python3 - "$MAIN_OUT" <<'PY'
import sys
from lda_core.output_utils import read_stack_outputs
o=read_stack_outputs(sys.argv[1], "LiftDoorAgentic")
print("\t".join(str(o[key]) for key in (
    "ClusterName", "CoordinatorServiceName", "CoordinatorTaskDefinitionArn",
    "PublicSubnetIds", "CoordinatorSecurityGroupId",
)))
PY
)
BUCKET=$(PYTHONPATH="$A" python3 - "$BUILD_OUT" <<'PY'
import sys
from lda_core.output_utils import read_stack_outputs
o=read_stack_outputs(sys.argv[1], "LiftDoorAgenticBuild")
print(o.get("BucketName") or o.get("bucket") or "")
PY
)
[ -n "$BUCKET" ] || { echo "build output lacks BucketName" >&2; exit 1; }
NETWORK="awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SG],assignPublicIp=ENABLED}"
OVERRIDES=$(python3 - "$BANK_BUILD_ID" <<'PY'
import json,sys
print(json.dumps({"containerOverrides":[{
    "name":"CoordinatorContainer",
    "command":["python","-m","services.coordinator.bank_build","--bank-build-id",sys.argv[1]],
}]}))
PY
)
TASK_ARN=$(aws ecs run-task --profile "$PROFILE" --region "$REGION" \
  --cluster "$CLUSTER" --task-definition "$TASK" --launch-type FARGATE \
  --network-configuration "$NETWORK" --overrides "$OVERRIDES" \
  --propagate-tags TASK_DEFINITION --enable-ecs-managed-tags \
  --query 'tasks[0].taskArn' --output text)
echo "task=$TASK_ARN"
aws ecs wait tasks-stopped --profile "$PROFILE" --region "$REGION" --cluster "$CLUSTER" --tasks "$TASK_ARN"
EXIT_CODE=$(aws ecs describe-tasks --profile "$PROFILE" --region "$REGION" --cluster "$CLUSTER" \
  --tasks "$TASK_ARN" --query "tasks[0].containers[?name=='CoordinatorContainer'].exitCode | [0]" --output text)
echo "exit_code=$EXIT_CODE"
[ "$EXIT_CODE" = "0" ] || exit 1
aws s3 cp "s3://$BUCKET/bank/CURRENT.json" -
