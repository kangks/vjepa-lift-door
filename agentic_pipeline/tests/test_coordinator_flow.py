from __future__ import annotations

import json
from contextlib import contextmanager

import numpy as np

from lda_core.storage import FileStorage
from services.coordinator import main as coordinator


class _FakeSpan:
    def set_attribute(self, key, value):
        return None

    def record_exception(self, exc):
        return None

    def set_status(self, status):
        return None

    def get_span_context(self):
        class _Ctx:
            is_valid = False
        return _Ctx()


class _FakeTracer:
    @contextmanager
    def start_as_current_span(self, *args, **kwargs):
        yield _FakeSpan()


def _record(run_id):
    return {
        "s3": {
            "bucket": {"name": "test-bucket"},
            "object": {"key": f"incoming/{run_id}/clip.mp4"},
        }
    }


def _output(classification, obstruction=False, evidence_ids=None, events=None):
    return {
        "classification": classification,
        "observed_events": events or [],
        "obstruction_visible": obstruction,
        "door_behaviour": "observed",
        "cabin_light_behaviour": "steady",
        "evidence_frame_ids": evidence_ids or [],
        "rationale": "visible evidence",
        "recommended_action": "NO_ACTION_LOG_ONLY" if classification == "EXPECTED_BEHAVIOR"
        else "INSPECT_DOOR_MECHANISM",
    }


def test_four_coordinator_flows_are_blind_and_gate_calls(tmp_path, monkeypatch):
    coordinator.TRACER = _FakeTracer()
    frames = np.zeros((64, 384, 384, 3), dtype=np.uint8)
    tokens = np.zeros((32, 24, 24, 1024), dtype=np.float32)
    reference_key = "bank/v2/test/ref_frames/healthy_clean.npy"

    scenarios = [
        ("fault_desync", True, _output("SUSPECTED_MALFUNCTION", evidence_ids=["CUR_01"]),
         None, "TICKET"),
        ("ball_reopen", True, _output("EXPECTED_BEHAVIOR", True, ["CUR_06"], ["ball in door path"]),
         None, "LOG_ONLY"),
        ("no_alert", False, None, None, "NO_ALERT_LOG_ONLY"),
        ("reasoner_error", True, None, "synthetic timeout", "TICKET"),
    ]
    for scenario, alert, output, error, expected_decision in scenarios:
        run_id = f"run-test-{scenario}-0000"
        clip_key = f"incoming/{run_id}/clip.mp4"
        storage = FileStorage(tmp_path / scenario)
        storage.put_npy(reference_key, frames)
        calls = []

        def embed_call(received_run_id, received_clip_key, headers):
            assert received_run_id == run_id
            assert received_clip_key == clip_key
            calls.append("embed")
            storage.put_npy(f"runs/{run_id}/frames.npy", frames)
            storage.put_npy(f"runs/{run_id}/tokens.npy", tokens.astype(np.float16))
            return {
                "run_id": run_id, "frames_key": f"runs/{run_id}/frames.npy",
                "tokens_key": f"runs/{run_id}/tokens.npy", "mp4_sha256": "a" * 64,
                "seconds": {"download": 0.1, "decode": 0.2, "encode_windows": [0.1] * 4, "put": 0.1},
            }

        def detect_call(event):
            calls.append("detect")
            storage.put_npy(f"runs/{run_id}/anomaly_map.npy", np.zeros((32, 23, 23), dtype=np.float32))
            return {
                "alert": alert, "alert_times_s": [5.0] if alert else [],
                "peak_time_s": 5.0, "peak_block_xy": [1, 1], "max_score_over_tau": 3.0 if alert else 0.0,
                "window_s": [3.0, 7.0] if alert else None, "bank_mode": "full",
                "bank_prefix": "bank/v2/test", "reference_frames_key": reference_key,
            }

        def reasoner_call(payload):
            calls.append("reasoner")
            texts = [item["text"] for item in payload["messages"][0]["content"] if item["type"] == "text"]
            assert all(run_id not in text for text in texts)
            assert all(clip_key not in text for text in texts)
            assert all(scenario not in text for text in texts)
            if error:
                return None, {"error": error}
            return output, {"usage": None, "cost_usd": 0.0, "latency_s": 0.01}

        def ticket_call(event):
            calls.append("ticket")
            return {"ticket_id": "MOCK-" + scenario, "duplicate": False}

        result = coordinator._process_record(
            _record(run_id), storage=storage, embed_call=embed_call, detect_call=detect_call,
            reasoner_call=reasoner_call, ticket_call=ticket_call,
        )
        assert result["decision"]["decision"] == expected_decision
        assert storage.exists(f"runs/{run_id}/result.json")
        if not alert:
            assert "reasoner" not in calls and "ticket" not in calls
        elif expected_decision == "LOG_ONLY":
            assert calls == ["embed", "detect", "reasoner"]
        else:
            assert calls == ["embed", "detect", "reasoner", "ticket"]
        record = json.loads(storage.get_bytes(f"runs/{run_id}/result.json"))
        assert record["clip_key"] == clip_key
