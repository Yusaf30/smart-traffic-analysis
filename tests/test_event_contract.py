import json
import re
import unittest
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "contracts" / "event_schema.json"
SAMPLES_PATH = ROOT / "samples" / "sample_events.json"

EVENT_TYPES = {"vehicle_crossing", "wrong_way"}
VEHICLE_CLASSES = {"CAR", "BIKE", "BUS", "TRUCK"}
DIRECTIONS = {"UP", "DOWN", "LEFT", "RIGHT", "UNKNOWN"}
TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"
)


def validate_event(event):
    required = {
        "schema_version", "event_id", "session_id", "event_type",
        "timestamp", "track_id", "vehicle_class", "direction",
        "counting_line"
    }

    if not isinstance(event, dict):
        raise ValueError("Event must be an object")

    if set(event) != required:
        raise ValueError("Missing or unexpected fields")

    if event["schema_version"] != "1.0":
        raise ValueError("Unsupported schema version")

    for field in ("event_id", "session_id"):
        if not isinstance(event[field], str) or not event[field].strip():
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
        raise ValueError("Timestamp must use UTC ISO 8601 with milliseconds and Z")

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


class EventProcessor:
    """Minimal in-memory example of session-scoped event deduplication."""

    def __init__(self):
        self.processed_keys = set()

    def process(self, event):
        validate_event(event)
        key = (event["session_id"], event["event_id"])

        if key in self.processed_keys:
            return False

        self.processed_keys.add(key)
        return True


class EventContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with SCHEMA_PATH.open(encoding="utf-8-sig") as file:
            cls.schema = json.load(file)
        with SAMPLES_PATH.open(encoding="utf-8-sig") as file:
            cls.samples = json.load(file)

    def test_schema_and_samples_are_valid_json(self):
        self.assertEqual(self.schema["properties"]["schema_version"]["const"], "1.0")
        self.assertEqual(len(self.samples), 2)

    def test_sample_events_are_valid(self):
        for event in self.samples:
            self.assertTrue(validate_event(event))

    def test_missing_field_is_rejected(self):
        event = self.samples[0].copy()
        del event["track_id"]
        with self.assertRaises(ValueError):
            validate_event(event)

    def test_invalid_event_type_is_rejected(self):
        event = self.samples[0].copy()
        event["event_type"] = "unknown_event"
        with self.assertRaises(ValueError):
            validate_event(event)

    def test_invalid_vehicle_class_is_rejected(self):
        event = self.samples[0].copy()
        event["vehicle_class"] = "VAN"
        with self.assertRaises(ValueError):
            validate_event(event)

    def test_invalid_direction_is_rejected(self):
        event = self.samples[0].copy()
        event["direction"] = "NORTH"
        with self.assertRaises(ValueError):
            validate_event(event)

    def test_invalid_timestamp_is_rejected(self):
        event = self.samples[0].copy()
        event["timestamp"] = "2026-10-09T04:35:12+05:30"
        with self.assertRaises(ValueError):
            validate_event(event)

    def test_invalid_calendar_date_is_rejected(self):
        event = self.samples[0].copy()
        event["timestamp"] = "2026-02-30T04:35:12.250Z"
        with self.assertRaises(ValueError):
            validate_event(event)

    def test_negative_track_id_is_rejected(self):
        event = self.samples[0].copy()
        event["track_id"] = -1
        with self.assertRaises(ValueError):
            validate_event(event)

    def test_boolean_track_id_is_rejected(self):
        event = self.samples[0].copy()
        event["track_id"] = True
        with self.assertRaises(ValueError):
            validate_event(event)

    def test_crossing_requires_counting_line(self):
        event = self.samples[0].copy()
        event["counting_line"] = None
        with self.assertRaises(ValueError):
            validate_event(event)

    def test_wrong_way_requires_null_counting_line(self):
        event = self.samples[1].copy()
        event["counting_line"] = "main_line"
        with self.assertRaises(ValueError):
            validate_event(event)

    def test_duplicate_event_is_ignored(self):
        processor = EventProcessor()
        event = self.samples[0]
        self.assertTrue(processor.process(event))
        self.assertFalse(processor.process(event))

    def test_same_event_id_in_different_sessions_is_allowed(self):
        processor = EventProcessor()
        event = self.samples[0].copy()
        self.assertTrue(processor.process(event))

        second_session = event.copy()
        second_session["session_id"] = "session-20261009-002"
        self.assertTrue(processor.process(second_session))

    def test_distinct_crossings_for_same_track_are_allowed(self):
        processor = EventProcessor()
        event = self.samples[0].copy()
        self.assertTrue(processor.process(event))

        recrossing = event.copy()
        recrossing["event_id"] = "evt-000003"
        recrossing["direction"] = "DOWN"
        self.assertTrue(processor.process(recrossing))

    def test_wrong_way_episode_can_emit_a_new_event_id(self):
        processor = EventProcessor()
        event = self.samples[1].copy()
        self.assertTrue(processor.process(event))

        new_episode = event.copy()
        new_episode["event_id"] = "evt-000004"
        new_episode["timestamp"] = "2026-10-09T04:40:01.100Z"
        self.assertTrue(processor.process(new_episode))


if __name__ == "__main__":
    unittest.main(verbosity=2)
