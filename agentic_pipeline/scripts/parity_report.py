from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lda_core.detector import evaluate, sample_labels


EXPECTED_SUMMARY_SHA256 = "bb876097593b912d5d6b8ad21ed54c28580e54a78a5d05a0d00105a41c75f716"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scenario_for(store: Path, run_dir: Path) -> tuple[str, str]:
    run_id = run_dir.name
    info_path = store / "meta" / f"{run_id}.json"
    if info_path.exists():
        return json.loads(info_path.read_text())["scenario"], "meta"

    legacy_path = run_dir / "scenario.json"
    if legacy_path.exists():
        return json.loads(legacy_path.read_text())["scenario"], "legacy"

    datasets = {path.name for path in store.glob("dataset/lift_dataset/*/*")}
    match = re.fullmatch(r"run-[^-]+-(?P<scenario>.+)-(?P<suffix>[0-9a-fA-F]{4})", run_id)
    if match:
        scenario = match.group("scenario")
        if scenario in datasets:
            return scenario, "run_id_suffix"
    raise ValueError(
        f"scenario metadata missing for run {run_id}; checked meta/{run_id}.json, "
        "legacy scenario.json, and the run-id suffix against dataset names"
    )


def _t0(peak_time: float) -> float:
    return max(0.0, min(float(peak_time) - 2.0, 12.0))


def _event_hits(times: list[float], events: list[dict]) -> str:
    output = []
    for event in events:
        start = float(event.get("start", event.get("start_sec", 0))) - 0.5
        end = float(event.get("end", event.get("end_sec_exclusive", 0))) + 0.5
        hit = any(start <= float(value) < end for value in times)
        output.append(f"{event.get('type', 'event')}={'HIT' if hit else 'MISS'}")
    return ", ".join(output) or "none"


def _labels_for_scenario(store: Path, scenario: str) -> dict:
    matches = sorted(store.glob(f"dataset/lift_dataset/*/{scenario}/ground_truth.json"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one ground_truth.json for scenario {scenario}, found {len(matches)}"
        )
    return sample_labels(json.loads(matches[0].read_text()))


def _false_alert_counts(times: list[float], labels: dict) -> tuple[int, int]:
    alert_samples = np.zeros(64, dtype=bool)
    for value in times:
        index = int(round(float(value) * 4.0))
        if 0 <= index < len(alert_samples):
            alert_samples[index] = True

    masked = evaluate(alert_samples, labels)["false_alert_samples"]
    raw_labels = dict(labels)
    raw_labels["observable"] = np.ones_like(labels["observable"], dtype=bool)
    raw = evaluate(alert_samples, raw_labels)["false_alert_samples"]
    return masked, raw


def build_report(store: Path, summary_path: Path) -> str:
    store = store.expanduser().resolve()
    summary_path = summary_path.expanduser().resolve()
    actual_hash = sha256(summary_path)
    if actual_hash != EXPECTED_SUMMARY_SHA256:
        raise ValueError(
            f"summary fixture sha256 mismatch: expected {EXPECTED_SUMMARY_SHA256}, got {actual_hash}"
        )
    summary = json.loads(summary_path.read_text())
    offline = summary["detectors"]["D_patchKNN"]["envelope"]
    rows = []
    for detector_path in sorted(store.glob("runs/run-*/detector.json")):
        run_dir = detector_path.parent
        result_path = run_dir / "result.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        scenario, scenario_source = _scenario_for(store, run_dir)
        expected = offline.get(scenario)
        actual = json.loads(detector_path.read_text())
        labels = _labels_for_scenario(store, scenario)
        events = expected.get("events", []) if expected else []
        offline_times = expected.get("alert_times_s", []) if expected else []
        actual_times = actual.get("alert_times_s", [])
        offline_fa_masked, offline_fa_raw = _false_alert_counts(offline_times, labels)
        actual_fa_masked, actual_fa_raw = _false_alert_counts(actual_times, labels)
        expected_score = (
            float(expected["max_score_over_tau"])
            if expected and "max_score_over_tau" in expected else None
        )
        actual_score = float(actual.get("max_score_over_tau", 0))
        score_delta = (
            100.0 * (actual_score - expected_score) / expected_score
            if expected_score else None
        )
        expected_peak = float(expected.get("peak_time_s", 0)) if expected else None
        actual_peak = float(actual.get("peak_time_s", 0))
        expected_block = expected.get("peak_block_xy") if expected else None
        actual_block = actual.get("peak_block_xy")
        if (
            expected
            and "false_alert_samples" in expected
            and offline_fa_masked != int(expected["false_alert_samples"])
        ):
            raise ValueError(
                f"offline observable-mask FA mismatch for {scenario}: "
                f"summary={expected['false_alert_samples']} ground_truth={offline_fa_masked}"
            )
        actual_window = actual.get("window_s")
        actual_t0 = float(actual_window[0]) if actual_window else None
        expected_t0 = _t0(expected_peak) if offline_times and expected_peak is not None else None
        rows.append({
            "run_id": result.get("run_id", run_dir.name),
            "scenario": scenario,
            "scenario_source": scenario_source,
            "alert": (
                f"{bool(offline_times)}/{bool(actual.get('alert'))}"
                if expected else f"n/a/{bool(actual.get('alert'))}"
            ),
            "t0": (
                f"{expected_t0:.2f}/{actual_t0:.2f}" if expected_t0 is not None and actual_t0 is not None
                else f"{expected_t0 if expected_t0 is not None else 'n/a'}/"
                f"{actual_t0 if actual_t0 is not None else 'n/a'}"
            ),
            "peak_offline_runtime_s": (
                f"{expected_peak:.2f}/{actual_peak:.2f}"
                if expected_peak is not None else f"n/a/{actual_peak:.2f}"
            ),
            "block_offline_runtime_xy": f"{expected_block}/{actual_block}",
            "max_score_over_tau": (
                f"{expected_score:.4f}/{actual_score:.4f}"
                if expected_score is not None else f"n/a/{actual_score:.4f}"
            ),
            "score_delta_pct": f"{score_delta:+.2f}%" if score_delta is not None else "n/a",
            "events_offline": _event_hits(offline_times, events) if expected else "n/a",
            "events_runtime": _event_hits(actual_times, events),
            "fa_masked": (
                f"{offline_fa_masked}/{actual_fa_masked}"
                if expected else f"n/a/{actual_fa_masked}"
            ),
            "fa_raw": f"{offline_fa_raw}/{actual_fa_raw}" if expected else f"n/a/{actual_fa_raw}",
            "bank_mode": actual.get("bank_mode", "n/a"),
            "excluded_bank_clips": ",".join(actual.get("excluded_bank_clips", [])) or "none",
        })
    headers = [
        "run_id", "scenario", "scenario_source", "alert", "t0",
        "peak_offline_runtime_s", "block_offline_runtime_xy",
        "max_score_over_tau", "score_delta_pct",
        "events_offline", "events_runtime", "fa_masked", "fa_raw",
        "bank_mode", "excluded_bank_clips",
    ]
    lines = [
        f"# Offline parity report (summary sha256 `{actual_hash}`)",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(row[header]) for header in headers) + " |" for row in rows)
    if not rows:
        lines.append("| no local detector results | " + " | ".join("n/a" for _ in headers[1:]) + " |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare local detector output to the frozen v2 summary.")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    print(build_report(args.store, args.summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
