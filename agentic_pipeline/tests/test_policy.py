import json

import pytest

from lda_core.policy import STRONG_SIGNAL, decide


def _reasoner(classification, *, obstruction=False, evidence=None, events=None):
    return {
        "classification": classification,
        "observed_events": [] if events is None else events,
        "obstruction_visible": obstruction,
        "door_behaviour": "visible behavior",
        "cabin_light_behaviour": "steady",
        "evidence_frame_ids": [] if evidence is None else evidence,
        "rationale": "visible evidence",
        "recommended_action": "NO_ACTION_LOG_ONLY" if classification == "EXPECTED_BEHAVIOR" else "HUMAN_REVIEW",
    }


@pytest.mark.parametrize(
    ("detector", "output", "error", "expected", "guard"),
    [
        ({"alert": False, "max_score_over_tau": 0.0}, None, None, "NO_ALERT_LOG_ONLY", None),
        ({"alert": True, "max_score_over_tau": 20.0}, _reasoner("SUSPECTED_MALFUNCTION"), None, "TICKET", None),
        ({"alert": True, "max_score_over_tau": 3.0},
         _reasoner("EXPECTED_BEHAVIOR", obstruction=True, evidence=["CUR_06"], events=["ball enters path"]),
         None, "LOG_ONLY", True),
        ({"alert": True, "max_score_over_tau": 20.0},
         _reasoner("EXPECTED_BEHAVIOR", obstruction=False, evidence=["CUR_06"], events=["something"]),
         None, "TICKET", False),
        ({"alert": True, "max_score_over_tau": 3.0},
         _reasoner("EXPECTED_BEHAVIOR", obstruction=True, evidence=["REF_00", "DIFF_00"], events=["benign"]),
         None, "TICKET", False),
        ({"alert": True, "max_score_over_tau": 2.0}, _reasoner("UNCERTAIN"), None, "TICKET", None),
        ({"alert": True, "max_score_over_tau": 1.99}, _reasoner("UNCERTAIN"), None, "HUMAN_REVIEW", None),
        ({"alert": True, "max_score_over_tau": 16.0}, None, "timeout", "TICKET", None),
        ({"alert": True, "max_score_over_tau": 16.0}, {"classification": "not-in-schema"}, None, "TICKET", None),
    ],
)
def test_policy_table(detector, output, error, expected, guard):
    result = decide(detector, output, error)
    assert result["decision"] == expected
    assert result["guard_passed"] is guard


def test_policy_real_offline_reasoner_outputs(eval_dir):
    rows = [json.loads(line) for line in (eval_dir / "results_v2/reasoner_v2_calls.jsonl").read_text().splitlines()]
    expected = {
        "anomaly_door_desync": "TICKET",
        "anomaly_interior_light_flicker": "TICKET",
        "healthy_ball_reopen": "LOG_ONLY",
    }
    for row in rows:
        result = decide(
            {"alert": True, "max_score_over_tau": 20.0},
            row["output"],
        )
        assert result["decision"] == expected[row["scenario"]]


def test_strong_signal_boundary_is_frozen():
    assert STRONG_SIGNAL == 2.0
