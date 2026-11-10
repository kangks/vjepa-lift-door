from __future__ import annotations

from pathlib import Path


def test_runtime_and_tests_do_not_use_home_checkpoint_defaults() -> None:
    root = Path(__file__).resolve().parents[1]
    scanned = [
        path
        for folder in ("services", "lda_core", "tests")
        for path in (root / folder).rglob("*.py")
    ]
    forbidden = (
        "Path" + ".home(",
        "expand" + "user(",
        "~/" + ".cache/",
    )
    matches = [
        f"{path.relative_to(root)} contains {token}"
        for path in scanned
        for token in forbidden
        if token in path.read_text()
    ]
    assert not matches, "\n".join(matches)
