import json

from services.ticket.handler import handler


def test_ticket_file_store_contains_mock_acceptance_fields(tmp_path, monkeypatch):
    ticket_dir = tmp_path / "tickets"
    monkeypatch.setenv("LDA_TICKET_STORE", f"file:{ticket_dir}")
    monkeypatch.setenv("LDA_OTEL_MODE", "file")
    monkeypatch.setenv("LDA_OTEL_FILE_DIR", str(tmp_path / "spans"))
    event = {
        "run_id": "run-test-ticket-0000", "clip_key": "incoming/run-test-ticket-0000/clip.mp4",
        "classification": "SUSPECTED_MALFUNCTION", "window_s": [3.0, 7.0],
        "recommended_action": "INSPECT_DOOR_MECHANISM", "trace_id": "a" * 32,
    }
    result = handler(event)
    assert result["duplicate"] is False
    item = json.loads((ticket_dir / f"{result['ticket_id']}.json").read_text())
    assert item["run_id"] == event["run_id"]
    assert item["clip_key"] == event["clip_key"]
    assert item["classification"] == event["classification"]
    assert item["window_s"] == ["3.0", "7.0"]
    assert item["recommended_action"] == event["recommended_action"]
    assert item["trace_id"] == event["trace_id"]
    assert item["mock"] is True
    assert item["system"] == "MOCK-CMMS"
