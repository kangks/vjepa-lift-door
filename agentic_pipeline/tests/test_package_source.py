from __future__ import annotations

import hashlib
import re
import subprocess
import zipfile
from pathlib import Path

import pytest


ALLOW_DIRS = {
    "services", "lda_core", "vendor", "scripts", "tests", "local", "fixtures",
}
ROOT_FILES = {
    "pyproject.toml", "buildspec-images.yml", "buildspec-e2e.yml", "README.md",
}
ROOT_REQUIREMENTS = re.compile(r"requirements[^/]*\.txt$")
EXCLUDED_PARTS = {
    ".DS_Store", "__pycache__", ".pytest_cache", ".venv", "dist",
    "outputs", "local_runs",
}


def _in_image_bundle(name: str) -> bool:
    parts = Path(name).parts
    if not parts:
        return False
    if parts[0] in ALLOW_DIRS:
        return (
            not any(part in EXCLUDED_PARTS for part in parts)
            and not name.endswith((".npy", ".pt", ".pyc"))
        )
    if len(parts) == 1 and (name in ROOT_FILES or ROOT_REQUIREMENTS.fullmatch(name)):
        return True
    return False


def test_source_zip_is_deterministic_and_matches_directory_allow_list():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/package_source.sh"
    archive = root / "dist/src.zip"
    if not script.is_file() or not archive.is_file():
        pytest.skip("package tooling and dist/src.zip are excluded from the remote image bundle")

    manifest = re.search(r"^ALLOW_LIST=\((.*?)^\)", script.read_text(), flags=re.MULTILINE | re.DOTALL)
    assert manifest, "package_source.sh must define its allow-list array"
    manifest_entries = {
        line.strip().strip("'\"")
        for line in manifest.group(1).splitlines()
        if line.strip()
    }
    assert manifest_entries == {
        *(f"{directory}/" for directory in ALLOW_DIRS),
        *ROOT_FILES,
        "requirements*.txt",
    }

    runs = []
    for _ in range(2):
        result = subprocess.run(
            ["bash", str(script)], cwd=root, check=True, capture_output=True, text=True,
        )
        match = re.search(r"^sha256=([0-9a-f]{64})$", result.stdout, flags=re.MULTILINE)
        tag = re.search(r"^TAG=(src-[0-9a-f]{12})$", result.stdout, flags=re.MULTILINE)
        assert match and tag
        runs.append((match.group(1), tag.group(1)))
    assert runs[0] == runs[1]

    actual_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert actual_sha == runs[0][0]
    names = {
        name[2:] if name.startswith("./") else name
        for name in zipfile.ZipFile(archive).namelist()
    }
    assert names
    assert all(_in_image_bundle(name) for name in names), sorted(
        name for name in names if not _in_image_bundle(name)
    )
    present_directories = {Path(name).parts[0] for name in names if "/" in name}
    assert ALLOW_DIRS <= present_directories
    assert "requirements-core.txt" in names
    assert "buildspec-images.yml" in names
    assert "buildspec-e2e.yml" in names
    assert "README.md" in names
    assert "scripts/check_config_consistency.py" in names
    assert "scripts/package_source.sh" in names
    assert "scripts/remote_build.sh" in names
    assert "scripts/deploy.sh" in names
    assert "scripts/destroy.sh" in names
    assert "scripts/upload_data.py" in names
    assert "infra" not in {Path(name).parts[0] for name in names}
    assert "outputs" not in {Path(name).parts[0] for name in names}
    assert "dist" not in {Path(name).parts[0] for name in names}
    assert ".venv" not in {Path(name).parts[0] for name in names}
    assert all(not name.endswith((".npy", ".pt", ".pyc")) for name in names)
