from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_shallow_conftest_collects_with_and_without_repo_root_override(
    tmp_path: Path,
) -> None:
    source_conftest = Path(__file__).with_name("conftest.py")
    shallow_tests = tmp_path / "src" / "tests"
    shallow_tests.mkdir(parents=True)
    shutil.copy2(source_conftest, shallow_tests / "conftest.py")

    project_root = Path(__file__).resolve().parents[1]
    pythonpath = os.pathsep.join(
        filter(None, (str(project_root), os.environ.get("PYTHONPATH", "")))
    )

    for repo_root in (None, tmp_path / "repo-root-override"):
        env = os.environ.copy()
        env["PYTHONPATH"] = pythonpath
        env.pop("LDA_REPO_ROOT", None)
        if repo_root is not None:
            repo_root.mkdir()
            env["LDA_REPO_ROOT"] = str(repo_root)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                str(shallow_tests),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        output = result.stdout + result.stderr
        assert result.returncode in (0, 5), output
        assert "IndexError" not in output, output
