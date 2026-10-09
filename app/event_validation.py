import re
from datetime import datetime
from typing import Any


EVENT_TYPES = {"vehicle_crossing", "wrong_way"}
VEHICLE_CLASSES = {"CAR", "BIKE", "BUS", "TRUCK"}
DIRECTIONS = {"UP", "DOWN", "LEFT", "RIGHT", "UNKNOWN"}

TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"
)

REQUIRED_FIELDS = {
    "schema_version",
    "event_id",
    "session_id",
    "event_type",
    "timestamp",
    "track_id",
    "vehicle_class",
    "direction",
    "counting_line",
}


def validate_event(event: dict[str, Any]) -> bool:
    """Validate an event against Traffic Analysis System Event Contract v1.0."""
    if not isinstance(event, dict):
        raise ValueError("Event must be an object")

    if set(event) != REQUIRED_FIELDS:
        raise ValueError("Missing or unexpected fields")

    if event["schema_version"] != "1.0":
        raise ValueError("Unsupported schema version")

    for field in ("event_id", "session_id"):
        value = event[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")

    if event["event_type"] not in EVENT_TYPES:
        raise ValueError("Invalid event_type")

    if type(event["track_id"]) is not int or event["track_id"] < 0:
        raise ValueError("track_id must be a non-negative integer")

    if event["vehicle_class"] not in VEHICLE_CLASSES:
        raise ValueError("Invalid vehicle_class")

    if event["direction"] not in DIRECTIONS:
        raise ValueError("Invalid direction")

    timestamp = event["timestamp"]
    if not isinstance(timestamp, str) or not TIMESTAMP_PATTERN.fullmatch(timestamp):
        raise ValueError("Timestamp must be UTC ISO 8601 with milliseconds and Z")

    try:
        datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise ValueError("Invalid timestamp date or time") from exc

    line = event["counting_line"]

    if event["event_type"] == "vehicle_crossing":
        if not isinstance(line, str) or not line.strip():
            raise ValueError("Crossing events require a counting_line")
    elif line is not None:
        raise ValueError("Wrong-way events require counting_line=null")

    return True
