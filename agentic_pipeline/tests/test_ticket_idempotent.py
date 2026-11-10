import json

from services.ticket.handler import handler


def test_ticket_put_is_idempotent_per_run_id(tmp_path, monkeypatch):
    ticket_dir = tmp_path / "tickets"
    monkeypatch.setenv("LDA_TICKET_STORE", f"file:{ticket_dir}")
    monkeypatch.setenv("LDA_OTEL_MODE", "file")
    monkeypatch.setenv("LDA_OTEL_FILE_DIR", str(tmp_path / "spans"))
    event = {
        "run_id": "run-test-idempotent-0000", "clip_key": "incoming/run-test-idempotent-0000/clip.mp4",
        "classification": "SUSPECTED_MALFUNCTION", "window_s": [3.0, 7.0],
        "recommended_action": "INSPECT_DOOR_MECHANISM", "trace_id": "b" * 32,
    }
    first = handler(event)
    second = handler(event)
    assert first["ticket_id"] == second["ticket_id"]
    assert first["duplicate"] is False
    assert second["duplicate"] is True
    items = list(ticket_dir.glob("*.json"))
    assert len(items) == 1
    saved = json.loads(items[0].read_text())
    assert saved["run_id"] == event["run_id"]
