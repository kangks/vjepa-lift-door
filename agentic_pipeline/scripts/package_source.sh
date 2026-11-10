#!/usr/bin/env bash
set -euo pipefail

A=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ALLOW_LIST=(
  services/
  lda_core/
  vendor/
  scripts/
  tests/
  local/
  fixtures/
  buildspec-images.yml
  buildspec-e2e.yml
  'requirements*.txt'
  pyproject.toml
  README.md
)
printf 'ALLOW_LIST:\n'
printf '  %s\n' "${ALLOW_LIST[@]}"

DIST="$A/dist"
mkdir -p "$DIST"
WORK=$(mktemp -d "$A/.package_tmp.XXXXXX")

cleanup() {
  python3 - "$A" "$WORK" <<'PY'
from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
if target.parent != root or not target.name.startswith(".package_tmp."):
    raise SystemExit("refusing to remove a path outside the package temp directory")
shutil.rmtree(target, ignore_errors=True)
PY
}
trap cleanup EXIT

copy_tree() {
  local destination=$1
  mkdir -p "$destination"
  (
    cd "$A"
    {
      for entry in "${ALLOW_LIST[@]}"; do
        if [[ "$entry" == */ ]]; then
          directory="./${entry%/}"
          find "$directory" \
            \( -name '.DS_Store' -o -name '__pycache__' -o -name '.pytest_cache' \
               -o -name '.venv' -o -name 'dist' -o -name 'outputs' \
               -o -name 'local_runs' \) -prune \
            -o -type f ! -name '*.npy' ! -name '*.pt' -print
        elif [[ "$entry" == "requirements"*.txt ]]; then
          find . -maxdepth 1 -type f -name "$entry" -print
        elif [[ -f "$entry" ]]; then
          printf './%s\n' "$entry"
        fi
      done
    } | LC_ALL=C sort |
      while IFS= read -r relative; do
        mkdir -p "$destination/$(dirname "$relative")"
        cp "$A/${relative#./}" "$destination/${relative#./}"
      done
  )
  find "$destination" -type f -exec env TZ=UTC touch -t 202611060000 {} +
  chmod -R u=rwX,go=rX "$destination"
}

zip_tree() {
  local source=$1
  local output=$2
  (
    cd "$source"
    find . -type f -print | LC_ALL=C sort | zip -X -D -q "$output" -@
  )
}

FIRST="$WORK/first"
SECOND="$WORK/second"
FIRST_ZIP="$WORK/src-first.zip"
copy_tree "$FIRST"
zip_tree "$FIRST" "$FIRST_ZIP"
FIRST_SHA=$(shasum -a 256 "$FIRST_ZIP" | awk '{print $1}')

copy_tree "$SECOND"
zip_tree "$SECOND" "$WORK/src-second.zip"
SECOND_SHA=$(shasum -a 256 "$WORK/src-second.zip" | awk '{print $1}')
test "$FIRST_SHA" = "$SECOND_SHA"

cp "$FIRST_ZIP" "$DIST/src.zip"
printf '%s  src.zip\n' "$FIRST_SHA" > "$DIST/src.zip.sha256"
printf 'sha256=%s\nTAG=src-%s\n' "$FIRST_SHA" "${FIRST_SHA:0:12}"
