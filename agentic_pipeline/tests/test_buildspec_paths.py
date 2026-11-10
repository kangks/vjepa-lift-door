from __future__ import annotations

import re
import subprocess
import zipfile
from pathlib import Path

import pytest
import yaml


REFERENCE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"((?:scripts|local|services|fixtures)/[A-Za-z0-9_./-]+|requirements[A-Za-z0-9_.-]*\.txt)"
)
TEST_SCRIPT_REFERENCE = re.compile(r"scripts/[A-Za-z0-9_./-]+\.(?:py|sh)")


def _buildspec_paths(root: Path) -> list[str]:
    found = set()
    for name in ("buildspec-images.yml", "buildspec-e2e.yml"):
        document = yaml.safe_load((root / name).read_text())
        assert document["version"] == 0.2
        found.update(REFERENCE.findall((root / name).read_text()))
    return sorted(found)


def test_buildspec_referenced_paths_exist_on_disk_and_in_source_zip():
    root = Path(__file__).resolve().parents[1]
    paths = _buildspec_paths(root)
    package_script = root / "scripts" / "package_source.sh"
    if not package_script.is_file():
        pytest.skip("scripts/package_source.sh is excluded from this test layout")
    subprocess.run(
        ["bash", str(package_script)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    zip_path = root / "dist" / "src.zip"
    if not zip_path.is_file():
        pytest.fail("package_source.sh did not create dist/src.zip")
    with zipfile.ZipFile(zip_path) as archive:
        archived = {
            name[2:] if name.startswith("./") else name
            for name in archive.namelist()
        }

    missing_disk = [path for path in paths if not (root / path).is_file()]
    missing_zip = [path for path in paths if path not in archived]
    test_script_references = sorted({
        reference
        for test_file in (root / "tests").glob("*.py")
        for reference in TEST_SCRIPT_REFERENCE.findall(test_file.read_text())
    })
    missing_test_scripts = [
        path for path in test_script_references if path not in archived
    ]
    print("Buildspec source references:")
    for path in paths:
        print(
            f"  {path}: "
            f"{'present' if (root / path).is_file() else 'MISSING'} on disk, "
            f"{'present' if path in archived else 'MISSING'} in dist/src.zip"
        )
    assert not missing_disk, f"buildspec paths missing on disk: {missing_disk}"
    assert not missing_zip, f"buildspec paths missing from dist/src.zip: {missing_zip}"
    assert not missing_test_scripts, (
        f"test-referenced scripts missing from dist/src.zip: {missing_test_scripts}"
    )
