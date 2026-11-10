from services.coordinator.main import parse_event_records, _parse_s3_record


def test_eventbridge_object_created_and_s3_record_formats_parse_to_same_run():
    run_id = "run-20261106T000000-anomaly_door_desync-a1b2"
    key = f"incoming/{run_id}/clip.mp4"
    s3_event = {
        "Records": [
            {"s3": {"bucket": {"name": "lda-bucket"}, "object": {"key": key}}}
        ]
    }
    eventbridge = {
        "source": "aws.s3",
        "detail-type": "Object Created",
        "detail": {"bucket": {"name": "lda-bucket"}, "object": {"key": key}},
    }
    original = parse_event_records(s3_event)
    converted = parse_event_records(eventbridge)
    assert _parse_s3_record(original[0]) == _parse_s3_record(converted[0])
    assert _parse_s3_record(converted[0]) == (run_id, key)
