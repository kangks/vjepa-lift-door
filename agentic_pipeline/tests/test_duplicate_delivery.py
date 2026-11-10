from contextlib import contextmanager

import numpy as np

from lda_core.storage import FileStorage
from services.coordinator import main as coordinator


class _CountingTracer:
    def __init__(self):
        self.starts = 0

    @contextmanager
    def start_as_current_span(self, *args, **kwargs):
        self.starts += 1
        class _Span:
            def set_attribute(self, *a): pass
            def record_exception(self, *a): pass
            def set_status(self, *a): pass
            def get_span_context(self):
                class _Ctx:
                    is_valid = False
                return _Ctx()
        yield _Span()


def test_duplicate_delivery_claim_skips_work_and_spans(tmp_path):
    coordinator.TRACER = _CountingTracer()
    storage = FileStorage(tmp_path)
    run_id = "run-test-duplicate-0000"
    record = {
        "s3": {"bucket": {"name": "test"}, "object": {"key": f"incoming/{run_id}/clip.mp4"}}
    }
    calls = []
    acked = []

    def embed(*args):
        calls.append("embed")
        return {"frames_key": "f.npy", "tokens_key": "t.npy", "mp4_sha256": "a" * 64, "seconds": {}}

    def detect(event):
        calls.append("detect")
        return {
            "alert": False, "alert_times_s": [], "peak_time_s": 0.0, "peak_block_xy": [0, 0],
            "max_score_over_tau": 0.0, "window_s": None, "bank_mode": "full",
        }

    first = coordinator._process_record(
        record, storage=storage, embed_call=embed, detect_call=detect, ack=lambda: acked.append("first"),
    )
    spans_after_first = coordinator.TRACER.starts
    second = coordinator._process_record(
        record, storage=storage, embed_call=embed, detect_call=detect, ack=lambda: acked.append("second"),
    )
    assert first["decision"]["decision"] == "NO_ALERT_LOG_ONLY"
    assert second == {"duplicate_delivery": run_id}
    assert calls == ["embed", "detect"]
    assert coordinator.TRACER.starts == spans_after_first
    assert acked == ["first", "second"]
