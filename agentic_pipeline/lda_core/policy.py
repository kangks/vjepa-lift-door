from __future__ import annotations

from .reasoner import validate

STRONG_SIGNAL = 2.0


def decide(detector: dict, reasoner_output: dict | None, reasoner_error: str | None = None) -> dict:
    if not detector.get("alert", False):
        return {
            "decision": "NO_ALERT_LOG_ONLY", "rule": "no_detector_alert",
            "guard_passed": None, "effective_class": None,
        }

    classification = "UNCERTAIN"
    output = reasoner_output or {}
    if reasoner_error is None and reasoner_output is not None:
        try:
            validate(reasoner_output)
            classification = reasoner_output["classification"]
        except Exception as exc:
            reasoner_error = f"{type(exc).__name__}: {exc}"

    if classification == "SUSPECTED_MALFUNCTION":
        return {
            "decision": "TICKET", "rule": "suspected_malfunction",
            "guard_passed": None, "effective_class": classification,
        }

    guard_passed = None
    if classification == "EXPECTED_BEHAVIOR":
        evidence_ids = output.get("evidence_frame_ids", [])
        guard_passed = bool(
            output.get("obstruction_visible") is True
            and any(isinstance(item, str) and item.startswith("CUR_") for item in evidence_ids)
            and bool(output.get("observed_events"))
        )
        if guard_passed:
            return {
                "decision": "LOG_ONLY", "rule": "expected_behavior_visible_benign_cause",
                "guard_passed": True, "effective_class": "EXPECTED_BEHAVIOR",
            }
        classification = "UNCERTAIN"

    if classification == "UNCERTAIN":
        score = float(detector.get("max_score_over_tau", 0.0))
        if score >= STRONG_SIGNAL:
            return {
                "decision": "TICKET", "rule": "uncertain_strong_detector_signal",
                "guard_passed": guard_passed, "effective_class": "UNCERTAIN",
            }
        return {
            "decision": "HUMAN_REVIEW", "rule": "uncertain_weak_detector_signal",
            "guard_passed": guard_passed, "effective_class": "UNCERTAIN",
        }

    return {
        "decision": "HUMAN_REVIEW", "rule": "unhandled_reasoner_classification",
        "guard_passed": guard_passed, "effective_class": classification,
    }
