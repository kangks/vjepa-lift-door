#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <profile> --diff-only|--deploy --stacks build|main|batch --diff-out FILE|--reviewed-diff FILE [-c key=value ...]" >&2
  exit 2
}

[ "$#" -ge 4 ] || usage
PROFILE=$1
ACTION=$2
shift 2
STACKS=""
DIFF_OUT=""
REVIEWED_DIFF=""
IMAGE_TAG=""
IMAGE_DIGESTS=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --stacks)
      [ "$#" -ge 2 ] || usage
      STACKS=$2
      shift 2
      ;;
    --diff-out)
      [ "$#" -ge 2 ] || usage
      DIFF_OUT=$2
      shift 2
      ;;
    --reviewed-diff)
      [ "$#" -ge 2 ] || usage
      REVIEWED_DIFF=$2
      shift 2
      ;;
    --image-tag)
      [ "$#" -ge 2 ] || usage
      IMAGE_TAG=$2
      shift 2
      ;;
    --image-digests)
      [ "$#" -ge 2 ] || usage
      IMAGE_DIGESTS=$2
      shift 2
      ;;
    -c)
      [ "$#" -ge 2 ] || usage
      assignment=$2
      key=${assignment%%=*}
      value=${assignment#*=}
      [ "$key" != "$assignment" ] || usage
      case "$key" in
        imageTag) IMAGE_TAG=$value ;;
        imageDigests) IMAGE_DIGESTS=$value ;;
        *) echo "unsupported CDK context key: $key" >&2; exit 2 ;;
      esac
      shift 2
      ;;
    *) usage ;;
  esac
done

[ "$ACTION" = "--diff-only" ] || [ "$ACTION" = "--deploy" ] || usage
[ "$STACKS" = "build" ] || [ "$STACKS" = "main" ] || [ "$STACKS" = "batch" ] || usage
if [ "$ACTION" = "--diff-only" ]; then
  [ -n "$DIFF_OUT" ] || { echo "--diff-out is required for --diff-only" >&2; exit 2; }
  [ -z "$REVIEWED_DIFF" ] || { echo "--reviewed-diff is only used for --deploy" >&2; exit 2; }
else
  [ -n "$REVIEWED_DIFF" ] || { echo "--reviewed-diff is required for --deploy" >&2; exit 2; }
  [ -z "$DIFF_OUT" ] || { echo "--diff-out is only used for --diff-only" >&2; exit 2; }
fi

A=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=$(cd "$A/../../.." && pwd)
ACCOUNTS="$A/infra/accounts.json"
ACCOUNT=$(python3 - "$ACCOUNTS" "$PROFILE" <<'PY'
import json
import sys
with open(sys.argv[1]) as source:
    print(json.load(source)[sys.argv[2]])
PY
)

CONTEXT=(-c "profile=$PROFILE" -c "region=us-east-1")
if [ "$STACKS" = "build" ]; then
  CONTEXT+=(-c "stacks=build")
  TARGETS=(LiftDoorAgenticBudget LiftDoorAgenticBuild)
  OUTPUT="$A/outputs/${PROFILE}-build.json"
elif [ "$STACKS" = "batch" ]; then
  CONTEXT+=(-c "stacks=batch")
  TARGETS=(LiftDoorAgenticBatch)
  OUTPUT="$A/outputs/${PROFILE}-batch.json"
else
  [ -n "$IMAGE_TAG" ] || { echo "main stack requires -c imageTag=<tag>" >&2; exit 2; }
  [ -n "$IMAGE_DIGESTS" ] || { echo "main stack requires -c imageDigests=<path>" >&2; exit 2; }
  CONTEXT+=(-c "imageTag=$IMAGE_TAG" -c "imageDigests=$IMAGE_DIGESTS")
  TARGETS=(LiftDoorAgentic)
  OUTPUT="$A/outputs/${PROFILE}.json"
fi

cd "$A/infra"

run_template_diff() {
  local output_path=$1
  cdk diff --method=template --profile "$PROFILE" "${CONTEXT[@]}" "${TARGETS[@]}" \
    > "$output_path" 2>&1
}

normalize_diff() {
  local input_path=$1
  local output_path=$2
  python3 - "$input_path" "$output_path" <<'PY'
from pathlib import Path
import re
import sys

source, target = map(Path, sys.argv[1:])
kept = []
for raw in source.read_text(errors="replace").splitlines():
    line = raw.rstrip()
    # Preserve all CloudFormation resource/change lines even if other text
    # around them resembles progress output.
    if line.startswith(("[+]", "[-]", "[~]")) or "│" in line:
        kept.append(line)
        continue
    if not line.strip():
        continue
    if line.startswith(("✨", "Stack ", "Hold on", "Number of stacks", "Total time")):
        continue
    if re.match(r"^(?:Synthesis|Deployment) time:", line):
        continue
    if re.match(r"^\s*\d+(?:\.\d+)?(?:ms|s)\s*$", line):
        continue
    kept.append(line)
target.write_text("\n".join(kept) + ("\n" if kept else ""))
PY
}

sha256() {
  shasum -a 256 "$1" | awk '{print $1}'
}

if [ "$ACTION" = "--diff-only" ]; then
  mkdir -p "$(dirname "$DIFF_OUT")"
  cdk synth --no-lookups "${CONTEXT[@]}" -o "cdk.out.${PROFILE}.${STACKS}"
  run_template_diff "$DIFF_OUT"
  echo "DIFF_FILE=$DIFF_OUT"
  echo "DIFF_SHA256=$(sha256 "$DIFF_OUT")"
  exit 0
fi

[ -f "$REVIEWED_DIFF" ] || { echo "reviewed diff does not exist: $REVIEWED_DIFF" >&2; exit 1; }
ACTUAL=$(mktemp "$A/.deploy_diff.XXXXXX")
NORMAL_REVIEWED=$(mktemp "$A/.deploy_reviewed_normalized.XXXXXX")
NORMAL_ACTUAL=$(mktemp "$A/.deploy_actual_normalized.XXXXXX")
cleanup() {
  python3 - "$A" "$ACTUAL" "$NORMAL_REVIEWED" "$NORMAL_ACTUAL" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
for value in sys.argv[2:]:
    path = Path(value).resolve()
    if path.parent == root and path.name.startswith((
        ".deploy_diff.", ".deploy_reviewed_normalized.", ".deploy_actual_normalized.",
    )):
        path.unlink(missing_ok=True)
PY
}
trap cleanup EXIT

run_template_diff "$ACTUAL"
echo "REVIEWED_DIFF_SHA256=$(sha256 "$REVIEWED_DIFF")"
echo "ACTUAL_DIFF_SHA256=$(sha256 "$ACTUAL")"
normalize_diff "$REVIEWED_DIFF" "$NORMAL_REVIEWED"
normalize_diff "$ACTUAL" "$NORMAL_ACTUAL"
REVIEWED_LINES=$(wc -l < "$NORMAL_REVIEWED" | tr -d ' ')
ACTUAL_LINES=$(wc -l < "$NORMAL_ACTUAL" | tr -d ' ')
if [ "$REVIEWED_LINES" -lt 3 ] || [ "$ACTUAL_LINES" -lt 3 ]; then
  echo "refusing to deploy: normalized reviewed/actual diff has fewer than 3 lines ($REVIEWED_LINES/$ACTUAL_LINES)" >&2
  exit 1
fi
if ! diff -u "$NORMAL_REVIEWED" "$NORMAL_ACTUAL"; then
  echo "refusing to deploy: reviewed diff changed" >&2
  exit 1
fi

IDENTITY=$(aws sts get-caller-identity --profile "$PROFILE" --query Account --output text)
[ "$IDENTITY" = "$ACCOUNT" ] || { echo "profile account mismatch" >&2; exit 1; }
mkdir -p "$A/outputs"
cdk deploy --require-approval never --profile "$PROFILE" "${CONTEXT[@]}" \
  "${TARGETS[@]}" --outputs-file "$OUTPUT"
echo "OUTPUT=$OUTPUT"
