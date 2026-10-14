import pytest

from traffic_analysis.events import TrafficEvent


def make_helmet_event(**overrides):
    data = {
        "event_type": "helmet_violation",
        "session_id": "session-1",
        "track_id": 10,
        "vehicle_class": "Bike",
        "direction": "UP",
    }
    data.update(overrides)
    return TrafficEvent.create(**data)


def test_create_helmet_violation_event():
    event = make_helmet_event()

    assert event.event_type == "helmet_violation"
    assert event.vehicle_class == "Bike"
    assert event.counting_line is None
    assert event.event_id
    assert event.timestamp.endswith("Z")


def test_helmet_violation_serializes_correctly():
    event = make_helmet_event()
    data = event.to_dict()

    assert data["schema_version"] == "1.0"
    assert data["event_type"] == "helmet_violation"
    assert data["track_id"] == 10
    assert data["counting_line"] is None
    assert data["session_id"] == "session-1"


def test_helmet_violation_rejects_counting_line():
    with pytest.raises(ValueError, match="counting_line=None"):
        make_helmet_event(counting_line="main_road_line")


def test_helmet_violation_rejects_invalid_vehicle_class():
    with pytest.raises(ValueError, match="Invalid vehicle_class"):
        make_helmet_event(vehicle_class="Person")


def test_helmet_violation_rejects_invalid_direction():
    with pytest.raises(ValueError, match="Invalid direction"):
        make_helmet_event(direction="SIDEWAYS")


def test_serialized_event_validates_against_contract():
    from app.event_validation import validate_event

    crossing_event = TrafficEvent.create(
        event_type="vehicle_crossing",
        session_id="session-1",
        track_id=5,
        vehicle_class="Car",
        direction="UP",
        counting_line="main_line",
    )
    assert validate_event(crossing_event.to_dict()) is True

    helmet_event = make_helmet_event()
    assert validate_event(helmet_event.to_dict()) is True
