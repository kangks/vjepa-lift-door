from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

_parents = Path(__file__).resolve().parents
ROOT = (
    Path(os.environ["LDA_REPO_ROOT"])
    if os.environ.get("LDA_REPO_ROOT")
    else (_parents[4] if len(_parents) > 4 else None)
)
DEFAULT_EVAL = (
    ROOT / ".claude/worktrees/vjepa-talk-draft/workshop/lift-door/eval_suite"
    if ROOT is not None
    else None
)
DEFAULT_DATA = (
    ROOT / "workshop/lift-door/lift_scene/lift_dataset"
    if ROOT is not None
    else None
)


@pytest.fixture(scope="session")
def eval_dir() -> Path:
    configured_path = os.environ.get("LDA_EVAL_DIR")
    path = Path(configured_path) if configured_path else DEFAULT_EVAL
    if path is None:
        pytest.skip("offline evaluation inputs unavailable: repository root is unknown")
    if not path.is_dir():
        pytest.skip(f"offline evaluation inputs missing: {path}")
    return path


@pytest.fixture(scope="session")
def data_dir() -> Path:
    configured_path = os.environ.get("LDA_DATA_DIR")
    path = Path(configured_path) if configured_path else DEFAULT_DATA
    if path is None:
        pytest.skip("lift-door dataset unavailable: repository root is unknown")
    if not path.is_dir():
        pytest.skip(f"lift-door dataset missing: {path}")
    return path


@pytest.fixture(scope="session")
def cases(data_dir: Path) -> dict[str, Path]:
    found = {p.parent.name: p.parent for p in sorted(data_dir.glob("*/*/ground_truth.json"))}
    if not found:
        pytest.skip(f"no ground_truth.json fixtures found under {data_dir}")
    return found


@pytest.fixture(scope="session")
def summary(eval_dir: Path) -> dict:
    return json.loads((eval_dir / "results_v2/summary.json").read_text())


@pytest.fixture(scope="session")
def reasoner_rows(eval_dir: Path) -> dict:
    path = eval_dir / "results_v2/reasoner_v2_calls.jsonl"
    return {row["scenario"]: row for row in (json.loads(line) for line in path.read_text().splitlines())}
