# Lift-door agentic core

This is the standalone, local-first core for the lift-door pipeline. It ports the
frozen detector, evidence renderer, reasoner request, and decision policy into
`lda_core`; no AWS resources are created by this project scaffold.

Run the local checks with:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-core.txt -r requirements-encoder.txt -r requirements-test.txt
.venv/bin/python -m pytest -q -m "not slow"
.venv/bin/python -m pytest -q -m slow -s
```

The slow encoder check uses the local V-JEPA checkpoint. The dataset and
offline results are read-only test inputs.

## Third-party binaries

The embed image installs Debian Bookworm's `ffmpeg`, built with
`--enable-gpl` and `libx264` (GPL-3.0-or-later). The images are private and are
not distributed; they are pushed only to the project's private ECR repositories.
`aws-opentelemetry-distro` is licensed under Apache-2.0.

## CDK deployment

Use the profile name to select the account. Review each generated CDK diff in
the planner's approved deployment step before running `--deploy`:

```bash
bash scripts/deploy.sh ml-sandbox --diff-only --stacks build --diff-out /path/to/reviewed-build-diff.txt
bash scripts/deploy.sh ml-sandbox --deploy --stacks build --reviewed-diff /path/to/reviewed-build-diff.txt
bash scripts/deploy.sh ml-sandbox --diff-only --stacks main --diff-out /path/to/reviewed-main-diff.txt -c imageTag=TAG -c imageDigests=PATH
bash scripts/deploy.sh ml-sandbox --deploy --stacks main --reviewed-diff /path/to/reviewed-main-diff.txt -c imageTag=TAG -c imageDigests=PATH
bash scripts/destroy.sh ml-sandbox
```

The same commands accept `attrb_sandbox`. Destroy runs the main stack before the
build stack. The budget stack remains. The project data bucket and all eight
CloudWatch log groups are retained; ECR repositories and the mock ticket table
are deleted with their stack.

## Remote image source bundle

`scripts/package_source.sh` includes only `lda_core/`, `vendor/`, `services/`,
`local/`, `fixtures/`, `tests/`, top-level `requirements*.txt`, `pyproject.toml`,
both CodeBuild buildspecs, and the scripts they invoke
(`scripts/local_*.py`, `scripts/check_trace.py`, and
`scripts/parity_report.py`). Deployment and operations tooling stays outside
the image source bundle, so changing those tools does not change the image tag.
