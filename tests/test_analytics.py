import json

import pytest

from traffic_analysis.analytics import (
    classify_density,
    load_events,
    summarize_events,
)


@pytest.mark.parametrize(
    ("vehicle_count", "expected"),
    [
        (0, "LOW"),
        (5, "LOW"),
        (6, "MEDIUM"),
        (8, "MEDIUM"),
        (9, "HIGH"),
        (20, "HIGH"),
    ],
)
def test_classify_density(vehicle_count, expected):
    assert classify_density(vehicle_count) == expected


def test_negative_density_count_is_rejected():
    with pytest.raises(ValueError):
        classify_density(-1)


def test_summarize_events():
    events = [
        {"event_type": "vehicle_crossing", "vehicle_class": "Car",
         "session_id": "s1"},
        {"event_type": "vehicle_crossing", "vehicle_class": "Bike",
         "session_id": "s1"},
        {"event_type": "vehicle_crossing", "vehicle_class": "Car",
         "session_id": "s1"},
        {"event_type": "wrong_way", "vehicle_class": "Bike",
         "session_id": "s1"},
    ]

    summary = summarize_events(events)

    assert summary["total_crossings"] == 3
    assert summary["vehicle_counts"] == {
        "Car": 2, "Bike": 1, "Bus": 0, "Truck": 0
    }
    assert summary["wrong_way_events"] == 1
    assert summary["session_ids"] == ["s1"]


def test_load_events_rejects_duplicate_ids(tmp_path):
    event = {
        "event_id": "duplicate",
        "event_type": "vehicle_crossing",
        "timestamp": "2026-10-09T00:00:00Z",
        "session_id": "s1",
        "track_id": 1,
        "vehicle_class": "Car",
        "direction": "UP",
        "counting_line": "main_road_line",
    }
    path = tmp_path / "events.json"
    path.write_text(json.dumps([event, event]), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate event_id"):
        load_events(path)


def valid_event():
    return {
        "event_id": "event-1",
        "event_type": "vehicle_crossing",
        "timestamp": "2026-10-09T05:09:49.478Z",
        "session_id": "session-1",
        "track_id": 1,
        "vehicle_class": "Car",
        "direction": "UNKNOWN",
        "counting_line": "main_road_line",
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("direction", "SIDEWAYS", "Invalid direction"),
        ("timestamp", "not-a-date", "Invalid timestamp"),
        ("track_id", -1, "Invalid track_id"),
        ("track_id", True, "Invalid track_id"),
        ("session_id", "", "Invalid session_id"),
    ],
)
def test_load_events_rejects_invalid_fields(
    tmp_path, field, value, message
):
    event = valid_event()
    event[field] = value
    path = tmp_path / "events.json"
    path.write_text(json.dumps([event]), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_events(path)


def test_load_events_accepts_unknown_direction(tmp_path):
    event = valid_event()
    path = tmp_path / "events.json"
    path.write_text(json.dumps([event]), encoding="utf-8")

    assert load_events(path)[0]["direction"] == "UNKNOWN"


def test_summarize_events_counts_helmet_violations():
    events = [
        {
            "event_type": "vehicle_crossing",
            "vehicle_class": "Bike",
            "session_id": "s1",
        },
        {
            "event_type": "helmet_violation",
            "vehicle_class": "Bike",
            "session_id": "s1",
        },
        {
            "event_type": "helmet_violation",
            "vehicle_class": "Bike",
            "session_id": "s1",
        },
    ]

    summary = summarize_events(events)

    assert summary["total_crossings"] == 1
    assert summary["wrong_way_events"] == 0
    assert summary["helmet_violations"] == 2
