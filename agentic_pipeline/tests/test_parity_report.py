from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "fixtures/results_v2_summary.json"
SCENARIOS = {
    "C1": ("anomaly_door_desync", "run-20261106T090723-anomaly_door_desync-ea4d", "meta"),
    "C2": ("anomaly_interior_light_flicker", "run-20261106T090840-anomaly_interior_light_flicker-2d64", "run_id_suffix"),
    "C3": ("healthy_ball_reopen", "run-20261106T090956-healthy_ball_reopen-dff0", "legacy"),
    "C4": ("healthy_clean", "run-20261106T091113-healthy_clean-e270", "meta"),
}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _ground_truth(offline: dict | None) -> dict:
    events = [
        {
            "type": event["type"],
            "start_sec": event["start"],
            "end_sec_exclusive": event["end"],
        }
        for event in (offline.get("events", []) if offline else [])
    ]
    frames = [{"anomaly": False, "observable": True} for _ in range(64)]
    if offline:
        faults = [
            event for event in events
            if event["type"] in {"door_desync", "interior_light_flicker"}
        ]
        false_indices = []
        for value in offline.get("alert_times_s", []):
            index = int(round(float(value) * 4))
            time_s = index / 4.0
            near_fault = any(
                event["start_sec"] - 0.5 <= time_s < event["end_sec_exclusive"] + 0.5
                for event in faults
            )
            if 0 <= index < len(frames) and not near_fault:
                false_indices.append(index)
        target = int(offline.get("false_alert_samples", len(false_indices)))
        for index in false_indices[:max(0, len(false_indices) - target)]:
            frames[index]["observable"] = False
    return {"fps": 4, "frames": frames, "event_intervals": events}


def _fake_store(root: Path) -> None:
    summary = json.loads(SUMMARY.read_text())["detectors"]["D_patchKNN"]["envelope"]
    for scenario, run_id, source in SCENARIOS.values():
        dataset_dir = root / "dataset/lift_dataset/healthy" / scenario
        dataset_dir.mkdir(parents=True, exist_ok=True)
        run_dir = root / "runs" / run_id
        offline = summary.get(scenario)
        _write_json(dataset_dir / "ground_truth.json", _ground_truth(offline))
        _write_json(run_dir / "result.json", {"run_id": run_id, "clip_key": f"incoming/{run_id}/clip.mp4"})
        _write_json(run_dir / "detector.json", {
            "run_id": run_id,
            "alert": bool(offline and offline.get("alert_times_s")),
            "alert_times_s": offline.get("alert_times_s", []) if offline else [],
            "window_s": [3.0, 7.0] if scenario in {"anomaly_door_desync", "healthy_ball_reopen"}
            else [11.0, 15.0] if scenario == "anomaly_interior_light_flicker"
            else None,
            "peak_time_s": offline.get("peak_time_s", 0.0) if offline else 0.0,
            "peak_block_xy": offline.get("peak_block_xy") if offline else None,
            "max_score_over_tau": offline.get("max_score_over_tau", 0.0) if offline else 0.0,
            "false_alert_samples": offline.get("false_alert_samples", 0) if offline else 0,
            "bank_mode": "loo" if scenario == "healthy_clean" else "inclusive",
            "excluded_bank_clips": ["healthy_clean"] if scenario == "healthy_clean" else [],
        })
        if source == "meta":
            _write_json(root / "meta" / f"{run_id}.json", {"scenario": scenario})
        elif source == "legacy":
            _write_json(run_dir / "scenario.json", {"scenario": scenario})

    # Bank build output must be ignored by the run report.
    _write_json(root / "runs/bank-fixture/detector.json", {"bank_mode": "not-a-run"})
    _write_json(root / "runs/bank-fixture/result.json", {"run_id": "bank-fixture"})


def test_parity_report_c1_to_c4_table_and_scenario_lookup(tmp_path: Path) -> None:
    store = tmp_path / "store"
    _fake_store(store)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/parity_report.py"),
            "--store",
            str(store),
            "--summary",
            str(SUMMARY),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output

    lines = result.stdout.splitlines()
    header_line = next(line for line in lines if line.startswith("| run_id |"))
    headers = [cell.strip() for cell in header_line.strip("|").split("|")]
    for column in ("scenario", "scenario_source", "t0", "alert", "max_score_over_tau", "bank_mode"):
        assert column in headers, header_line
    indexes = {name: headers.index(name) for name in headers}
    for label, (scenario, _run_id, source) in SCENARIOS.items():
        row_line = next(line for line in lines if f"| {scenario} |" in line)
        cells = [cell.strip() for cell in row_line.strip("|").split("|")]
        assert cells[indexes["scenario_source"]] == source, (label, row_line)
        for column in ("t0", "alert", "max_score_over_tau", "bank_mode"):
            assert cells[indexes[column]], (label, column, row_line)
    assert "bank-fixture" not in result.stdout
    assert "no local detector results" not in result.stdout


def test_parity_report_uses_legacy_scenario_metadata(tmp_path: Path) -> None:
    store = tmp_path / "store"
    run_id = "run-20261106T092000-anomaly_desync_tree_shadow-1234"
    scenario = "anomaly_desync_tree_shadow"
    dataset_dir = store / "dataset/lift_dataset/anomaly" / scenario
    dataset_dir.mkdir(parents=True)
    offline = json.loads(SUMMARY.read_text())["detectors"]["D_patchKNN"]["envelope"][scenario]
    _write_json(dataset_dir / "ground_truth.json", _ground_truth(offline))
    _write_json(store / "runs" / run_id / "scenario.json", {"scenario": scenario})
    _write_json(store / "runs" / run_id / "result.json", {"run_id": run_id})
    _write_json(store / "runs" / run_id / "detector.json", {
        "alert": True, "alert_times_s": [3.0], "window_s": [3.0, 7.0],
        "peak_time_s": 5.0, "peak_block_xy": [1, 2], "max_score_over_tau": 1.0,
        "bank_mode": "inclusive",
    })

    from scripts.parity_report import build_report

    report = build_report(store, SUMMARY)
    row = next(line for line in report.splitlines() if f"| {scenario} |" in line)
    assert "| legacy |" in row


def test_parity_report_masks_unobservable_false_alert_samples(tmp_path: Path) -> None:
    store = tmp_path / "store"
    scenario = "stress_balloon_full_occlusion"
    run_id = "run-20261106T095501-stress_balloon_full_occlusion-e988"
    summary = json.loads(SUMMARY.read_text())["detectors"]["D_patchKNN"]["envelope"][scenario]
    dataset_dir = store / "dataset/lift_dataset/healthy" / scenario
    frames = [{"anomaly": False, "observable": True} for _ in range(64)]
    for index in range(23, 29):
        frames[index]["observable"] = False
    _write_json(dataset_dir / "ground_truth.json", {
        "fps": 4, "frames": frames, "event_intervals": [],
    })
    _write_json(store / "meta" / f"{run_id}.json", {"scenario": scenario})
    _write_json(store / "runs" / run_id / "result.json", {"run_id": run_id})
    _write_json(store / "runs" / run_id / "detector.json", {
        "alert": True,
        "alert_times_s": summary["alert_times_s"],
        "window_s": [5.0, 9.0],
        "peak_time_s": summary["peak_time_s"],
        "peak_block_xy": summary["peak_block_xy"],
        "max_score_over_tau": summary["max_score_over_tau"],
        "bank_mode": "full",
    })

    from scripts.parity_report import build_report

    report = build_report(store, SUMMARY)
    row = next(line for line in report.splitlines() if f"| {scenario} |" in line)
    assert f"| {summary['false_alert_samples']}/{summary['false_alert_samples']} |" in row
    assert f"| {len(summary['alert_times_s'])}/{len(summary['alert_times_s'])} |" in row


def test_parity_report_fails_clearly_when_scenario_is_unresolvable(tmp_path: Path) -> None:
    store = tmp_path / "store"
    run_id = "run-20261106T093000-unknown_clip-ffff"
    _write_json(store / "runs" / run_id / "result.json", {"run_id": run_id})
    _write_json(store / "runs" / run_id / "detector.json", {"alert": False})

    from scripts.parity_report import build_report

    try:
        build_report(store, SUMMARY)
    except ValueError as exc:
        assert f"scenario metadata missing for run {run_id}" in str(exc)
    else:
        raise AssertionError("unresolvable scenario was not rejected")
