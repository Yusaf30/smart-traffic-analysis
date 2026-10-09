import json
import unittest
from pathlib import Path

from app.event_validation import validate_event
from app.analytics.counter import VehicleCounter


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "contracts" / "event_schema.json"
SAMPLES_PATH = ROOT / "samples" / "sample_events.json"


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
        counter = VehicleCounter()
        event = self.samples[0]
        self.assertTrue(counter.process_event(event))
        self.assertFalse(counter.process_event(event))
        self.assertEqual(counter.get_stats()["total"], 1)

    def test_same_event_id_in_different_sessions_is_allowed(self):
        counter = VehicleCounter()
        event = self.samples[0].copy()
        self.assertTrue(counter.process_event(event))

        second_session = event.copy()
        second_session["session_id"] = "session-20261009-002"
        self.assertTrue(counter.process_event(second_session))
        self.assertEqual(counter.get_stats()["total"], 2)

    def test_distinct_crossings_for_same_track_are_allowed(self):
        counter = VehicleCounter()
        event = self.samples[0].copy()
        self.assertTrue(counter.process_event(event))

        recrossing = event.copy()
        recrossing["event_id"] = "evt-000003"
        recrossing["direction"] = "DOWN"
        self.assertTrue(counter.process_event(recrossing))
        self.assertEqual(counter.get_stats()["total"], 2)

    def test_wrong_way_event_does_not_increment_crossings(self):
        counter = VehicleCounter()
        self.assertFalse(counter.process_event(self.samples[1]))
        self.assertEqual(counter.get_stats()["total"], 0)

    def test_counts_by_class_and_direction(self):
        counter = VehicleCounter()
        counter.process_event(self.samples[0])

        stats = counter.get_stats()
        self.assertEqual(stats["by_class"]["CAR"], 1)
        self.assertEqual(stats["by_direction"]["UP"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
