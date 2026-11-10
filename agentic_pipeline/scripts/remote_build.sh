#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <profile> <images|e2e|chain> [--tag-suffix rN] [--dry-run]" >&2
  exit 2
}

[ "$#" -ge 2 ] || usage
PROFILE=$1
KIND=$2
shift 2
SUFFIX=""
DRY_RUN=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag-suffix)
      [ "$#" -ge 2 ] || usage
      SUFFIX=$2
      [[ "$SUFFIX" =~ ^r[0-9]+$ ]] || {
        echo "tag suffix must look like rN" >&2
        exit 2
      }
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *) usage ;;
  esac
done

case "$KIND" in
  chain)
    image_args=("$PROFILE" images)
    e2e_args=("$PROFILE" e2e)
    if [ -n "$SUFFIX" ]; then
      image_args+=(--tag-suffix "$SUFFIX")
      e2e_args+=(--tag-suffix "$SUFFIX")
    fi
    if [ "$DRY_RUN" = 1 ]; then
      image_args+=(--dry-run)
      e2e_args+=(--dry-run)
    fi
    bash "$0" "${image_args[@]}"
    bash "$0" "${e2e_args[@]}"
    exit 0
    ;;
  images) PROJECT_KEY=ImageBuildProject ;;
  e2e) PROJECT_KEY=E2eTestProject ;;
  *) echo "kind must be images, e2e or chain" >&2; exit 2 ;;
esac

A=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=$(cd "$A/../../.." && pwd)
PL="$ROOT/.claude/worktrees/vjepa-talk-draft/agentic_plan"
CONFIG="$A/outputs/${PROFILE}-build.json"
ZIP="$A/dist/src.zip"
CHECKSUM="$A/dist/src.zip.sha256"
test -f "$CONFIG" || { echo "missing build profile: $CONFIG" >&2; exit 1; }
test -f "$ZIP" && test -f "$CHECKSUM" || {
  echo "package source first with scripts/package_source.sh" >&2
  exit 1
}

SHA=$(shasum -a 256 "$ZIP" | awk '{print $1}')
RECORDED_SHA=$(awk '{print $1}' "$CHECKSUM")
test "$SHA" = "$RECORDED_SHA" || {
  echo "dist/src.zip does not match dist/src.zip.sha256" >&2
  exit 1
}
TAG="src-${SHA:0:12}"
[ -z "$SUFFIX" ] || TAG="${TAG}-${SUFFIX}"

BUCKET=$(jq -r '.LiftDoorAgenticBuild.BucketName // empty' "$CONFIG")
PROJECT=$(jq -r ".LiftDoorAgenticBuild.${PROJECT_KEY} // empty" "$CONFIG")
REGION=$(aws configure get region --profile "$PROFILE" || true)
REGION=${REGION:-us-east-1}
test -n "$BUCKET" || { echo "build profile has no LiftDoorAgenticBuild.BucketName: $CONFIG" >&2; exit 1; }
test -n "$PROJECT" || { echo "build profile has no LiftDoorAgenticBuild.${PROJECT_KEY}: $CONFIG" >&2; exit 1; }
test -n "$REGION" || { echo "could not resolve region for profile $PROFILE" >&2; exit 1; }

KEY="build/src/${TAG}.zip"
DEST="s3://${BUCKET}/${KEY}"
S3_ENDPOINT="https://s3.${REGION}.amazonaws.com"
START_COMMAND="aws codebuild start-build --profile ${PROFILE} --region ${REGION} --project-name ${PROJECT} --source-location-override ${BUCKET}/${KEY} --environment-variables-override name=IMAGE_TAG,value=${TAG},type=PLAINTEXT name=SRC_SHA256,value=${SHA},type=PLAINTEXT"

if [ "$DRY_RUN" = 1 ]; then
  echo "DRY_RUN=true"
  echo "BUCKET=$BUCKET"
  echo "REGION=$REGION"
  echo "S3_ENDPOINT=$S3_ENDPOINT"
  echo "TAG=$TAG"
  echo "SOURCE=$DEST"
  echo "PROJECT=$PROJECT"
  echo "START_BUILD_COMMAND=$START_COMMAND"
  exit 0
fi

EXISTING_SHA=$(aws s3api head-object --profile "$PROFILE" --region "$REGION" \
  --bucket "$BUCKET" --key "$KEY" --query 'Metadata.sha256' --output text 2>/dev/null || true)
if [ "$EXISTING_SHA" != "$SHA" ]; then
  aws s3 cp "$ZIP" "$DEST" --profile "$PROFILE" --region "$REGION" \
    --metadata "sha256=$SHA"
else
  echo "source zip already present with matching sha256: $DEST"
fi

BUILD_ID=$(aws codebuild start-build --profile "$PROFILE" --region "$REGION" \
  --project-name "$PROJECT" \
  --source-location-override "${BUCKET}/${KEY}" \
  --environment-variables-override \
    "name=IMAGE_TAG,value=${TAG},type=PLAINTEXT" \
    "name=SRC_SHA256,value=${SHA},type=PLAINTEXT" \
  --query 'build.id' --output text)
echo "TAG=$TAG"
echo "BUILD_ID=$BUILD_ID"

LAST_PHASE=""
while true; do
  RESPONSE=$(aws codebuild batch-get-builds --profile "$PROFILE" --region "$REGION" \
    --ids "$BUILD_ID" --output json)
  STATUS=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["builds"][0]["buildStatus"])' <<<"$RESPONSE")
  PHASE=$(python3 -c '
import json,sys
build=json.load(sys.stdin)["builds"][0]
active=[p["phaseType"] for p in build.get("phases",[]) if p.get("phaseStatus") == "IN_PROGRESS"]
print(active[-1] if active else "")
' <<<"$RESPONSE")
  if [ "$PHASE" != "$LAST_PHASE" ] && [ -n "$PHASE" ]; then
    echo "PHASE=$PHASE"
    LAST_PHASE=$PHASE
  fi
  case "$STATUS" in
    SUCCEEDED|FAILED|FAULT|TIMED_OUT|STOPPED) break ;;
  esac
  sleep 30
done

OUT="$PL/evidence/${PROFILE}/build_${TAG}"
mkdir -p "$OUT"
aws s3 cp "s3://${BUCKET}/build/out/${TAG}/" "$OUT/" \
  --recursive --profile "$PROFILE" --region "$REGION"
echo "OUTPUT_DIR=$OUT"
echo "BUILD_STATUS=$STATUS"
if [ "$STATUS" != "SUCCEEDED" ]; then
  exit 1
fi
