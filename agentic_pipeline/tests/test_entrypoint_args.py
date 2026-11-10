from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_entrypoint_args_exec_overrides(tmp_path):
    stub = tmp_path / "capture-command"
    stub.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$(basename "$0")" "$@" > "$ENTRYPOINT_CAPTURE"\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    instrument = tmp_path / "opentelemetry-instrument"
    instrument.write_text(stub.read_text(encoding="utf-8"), encoding="utf-8")
    instrument.chmod(0o755)

    args = ["capture-command", "alpha", "beta"]
    for relative in (
        "services/coordinator/entrypoint.sh",
        "services/embed/entrypoint.sh",
    ):
        entrypoint = ROOT / relative
        source = entrypoint.read_text()
        assert 'if [ "$#" -gt 0 ]; then' in source
        assert 'exec "$@"' in source
        assert 'exec opentelemetry-instrument "$@"' in source

        for mode in ("file", "aws"):
            capture = tmp_path / f"{entrypoint.parent.name}-{mode}.txt"
            environment = os.environ.copy()
            environment["LDA_OTEL_MODE"] = mode
            environment["ENTRYPOINT_CAPTURE"] = str(capture)
            environment["PATH"] = f"{tmp_path}{os.pathsep}{environment.get('PATH', '')}"
            subprocess.run(
                ["/bin/sh", str(entrypoint), *args],
                check=True,
                env=environment,
                capture_output=True,
                text=True,
            )
            expected = [*args] if mode == "file" else ["opentelemetry-instrument", *args]
            assert capture.read_text().splitlines() == expected
