from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def test_static_config_consistency_gate_passes() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/check_config_consistency.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "| Check | Files | Result |" in output
    assert "| checkpoint path values |" in output
    assert "FAIL" not in output
    if (root / "infra").is_dir():
        assert "| CDK embed checkpoint environment | infra/lda_infra/main_stack.py | PASS:" in output
    else:
        assert "SKIPPED: no infra/ in bundle; enforced on the working tree by gate G2" in output


def test_consistency_bundle_mode_without_infra(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    ignored = shutil.ignore_patterns("__pycache__", ".pytest_cache")
    for directory in ("services", "lda_core", "vendor", "scripts", "tests", "local", "fixtures"):
        shutil.copytree(root / directory, tmp_path / directory, ignore=ignored)
    for pattern in ("buildspec-*.yml", "requirements*.txt"):
        for source in root.glob(pattern):
            shutil.copy2(source, tmp_path / source.name)
    for filename in ("pyproject.toml", "README.md"):
        source = root / filename
        if source.is_file():
            shutil.copy2(source, tmp_path / filename)

    result = subprocess.run(
        [sys.executable, str(tmp_path / "scripts/check_config_consistency.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Traceback" not in output
    assert "FAIL" not in output
    trace_endpoint_row = next(
        line for line in output.splitlines()
        if "| CDK ADOT trace endpoint uses stack region |" in line
    )
    assert "SKIPPED" in trace_endpoint_row
