from __future__ import annotations

import json
import importlib
import sys
from pathlib import Path

import numpy as np

from lda_core.design import BANK
from lda_core.detector import build_envelope, cycles, detect, evaluate, persist, sample_labels, unit
from lda_core.frames import frames_sha16, load_png_frames


def _load_tokens(name: str, frames: np.ndarray, cache_dir: Path) -> np.ndarray:
    token_path = cache_dir / f"{name}_{frames_sha16(frames)}.npy"
    assert token_path.is_file(), f"missing token cache: {token_path}"
    return unit(np.load(token_path).astype(np.float32))


def _load_original_modules(eval_dir):
    sys.path.insert(0, str(eval_dir))
    original_v1 = importlib.import_module("lift_vjepa_suite")
    original_v2 = importlib.import_module("lift_vjepa_suite_v2")
    return original_v1, original_v2


def test_port_matches_original_code(eval_dir, cases, reasoner_rows):
    cache_dir = eval_dir / "results_v1/token_cache"
    original_v1, original_v2 = _load_original_modules(eval_dir)
    frames = {name: load_png_frames(case) for name, case in cases.items()}
    labels = {
        name: sample_labels(json.loads((case / "ground_truth.json").read_text()))
        for name, case in cases.items()
    }
    tokens = {name: _load_tokens(name, frames[name], cache_dir) for name in cases}
    bank_tokens = {name: tokens[name] for name in BANK}
    original_tokens = {
        name: original_v1.unit(np.load(cache_dir / f"{name}_{frames_sha16(frames[name])}.npy").astype(np.float32))
        for name in cases
    }
    original_bank = {name: original_tokens[name] for name in BANK}
    original_loo = []
    for bank_name in BANK:
        for cycle_idx in range(2):
            others = [
                original_v2.cycles(original_tokens[other])[other_cycle]
                for other in BANK if other != bank_name
                for other_cycle in range(2)
            ]
            original_loo.append(
                original_v1.map_for(
                    "D_patchKNN", original_v2.cycles(original_tokens[bank_name])[cycle_idx], others
                )
            )
    original_envelope = np.max(original_loo, 0) + 1e-4
    envelope = build_envelope(bank_tokens)
    assert np.array_equal(envelope, original_envelope)

    rows = []
    for name in cases:
        if name in BANK:
            continue
        all_bank = [
            original_v2.cycles(original_tokens[bank_name])[cycle_idx]
            for bank_name in BANK for cycle_idx in range(2)
        ]
        original_maps = [
            original_v1.map_for("D_patchKNN", cycle, all_bank)
            for cycle in original_v2.cycles(original_tokens[name])
        ]
        original_map = np.concatenate(original_maps, 0)
        original_tau = np.concatenate([original_envelope, original_envelope], 0)
        tok_alert = original_v2.persist((original_map > original_tau).reshape(32, -1).any(1))
        original_alert_samples = np.repeat(tok_alert, 2)
        original_result = original_v1.evaluate(original_alert_samples, labels[name])
        ti, y, x = np.unravel_index(int(np.argmax(original_map - original_tau)), original_map.shape)
        original_result.update({
            "peak_time_s": float(ti * 0.5),
            "peak_block_xy": [int(x), int(y)],
            "max_score_over_tau": float((original_map / original_tau).max()),
        })
        port_result = detect(tokens[name], bank_tokens, envelope)
        port_alert_samples = np.isin(np.arange(64) / 4.0, port_result["alert_times_s"])
        port_evaluate = evaluate(port_alert_samples, labels[name])
        assert port_result["alert_times_s"] == original_result["alert_times_s"], name
        assert port_result["peak_time_s"] == original_result["peak_time_s"], name
        assert port_result["peak_block_xy"] == original_result["peak_block_xy"], name
        assert abs(port_result["max_score_over_tau"] - original_result["max_score_over_tau"]) <= 1e-6, name
        assert port_evaluate["events"] == original_result["events"], name
        assert port_evaluate["false_alert_samples"] == original_result["false_alert_samples"], name
        max_excess_delta = float(np.abs(port_result["excess"] - (original_map - original_tau)).max())
        excess_equal = bool(np.array_equal(port_result["excess"], original_map - original_tau))
        assert max_excess_delta <= 1e-6, (name, max_excess_delta)
        rows.append((
            name, port_result["alert"], port_result["peak_time_s"], port_result["peak_block_xy"],
            port_result["max_score_over_tau"], max_excess_delta, excess_equal,
        ))
    print("PORT VS ORIGINAL CODE (clip, alert, peak_s, peak_xy, max_score, max_abs_excess_delta, excess_array_equal)")
    for row in rows:
        print(row)


def test_cached_tokens_reproduce_frozen_v2_summary(eval_dir, cases, summary, reasoner_rows):
    cache_dir = eval_dir / "results_v1/token_cache"
    frames = {name: load_png_frames(case) for name, case in cases.items()}
    labels = {
        name: sample_labels(json.loads((case / "ground_truth.json").read_text()))
        for name, case in cases.items()
    }
    tokens = {name: _load_tokens(name, frames[name], cache_dir) for name in cases}
    bank_tokens = {name: tokens[name] for name in BANK}
    envelope = build_envelope(bank_tokens)
    expected = summary["detectors"]["D_patchKNN"]["envelope"]
    parity_rows = []
    for name in cases:
        if name in BANK:
            continue
        result = detect(tokens[name], bank_tokens, envelope)
        scored = evaluate(np.isin(np.arange(64) / 4.0, result["alert_times_s"]), labels[name])
        reference = expected[name]
        assert result["alert"] == bool(reference["alert_times_s"])
        assert result["alert_times_s"] == reference["alert_times_s"], name
        assert result["peak_time_s"] == reference["peak_time_s"], name
        assert result["peak_block_xy"] == reference["peak_block_xy"], name
        abs_delta = abs(result["max_score_over_tau"] - reference["max_score_over_tau"])
        rel_delta = abs_delta / max(abs(reference["max_score_over_tau"]), 1e-12)
        assert rel_delta <= 1e-4, (name, abs_delta, rel_delta)
        assert scored["events"] == reference["events"], name
        assert scored["false_alert_samples"] == reference["false_alert_samples"], name
        if name in reasoner_rows:
            assert result["window_s"] == reasoner_rows[name]["window_s"], name
        parity_rows.append(
            (name, result["alert"], result["alert_times_s"], result["peak_time_s"],
             result["peak_block_xy"], result["max_score_over_tau"],
             abs_delta, rel_delta)
        )
    print("PORT VS summary.json (clip, alert, alert_times_s, peak_time_s, peak_xy, max_score, abs_delta, rel_delta)")
    for row in parity_rows:
        print(row)


def test_acceptance_decode_png_frame_differences(eval_dir, cases):
    """Non-gating R4 diagnostic is included so it is available with the test evidence."""
    from lda_core.frames import decode_mp4

    names = (
        "anomaly_door_desync",
        "anomaly_interior_light_flicker",
        "healthy_ball_reopen",
        "healthy_clean",
    )
    rows = []
    for name in names:
        decoded = decode_mp4(cases[name] / f"{name}.mp4")
        png = load_png_frames(cases[name])
        delta = np.abs(decoded.astype(np.int16) - png.astype(np.int16))
        rows.append((name, float(delta.mean()), int(delta.max()), frames_sha16(decoded)))
    print("MP4 vs PNG pixel diff (clip, mean_abs, max_abs, mp4_frames_sha16)")
    for row in rows:
        print(row)
    # healthy_clean has two offline caches. Report any exact mp4-frame hash match.
    clean_hash = dict((name, sha) for name, _, _, sha in rows)["healthy_clean"]
    candidates = sorted((eval_dir / "results_v1/token_cache").glob("healthy_clean_*.npy"))
    matched = [path.name for path in candidates if path.stem.endswith(clean_hash)]
    print(f"healthy_clean mp4 cache match: {matched if matched else 'none of the two caches'}")
