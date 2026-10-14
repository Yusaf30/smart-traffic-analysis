import json
import tempfile
import unittest
from pathlib import Path

from app.storage.event_logger import EventLogger


def make_event(event_id="event-001", session_id="session-001"):
    return {
        "schema_version": "1.0",
        "event_id": event_id,
        "session_id": session_id,
        "event_type": "vehicle_crossing",
        "timestamp": "2026-10-09T04:35:12.250Z",
        "track_id": 7,
        "vehicle_class": "Car",
        "direction": "DOWN",
        "counting_line": "line-1",
    }


class TestEventLogger(unittest.TestCase):
    def test_logs_event_as_json_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            logger = EventLogger(path)

            self.assertTrue(logger.log_event(make_event()))

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0]), make_event())

    def test_duplicate_event_is_not_logged_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            logger = EventLogger(path)
            event = make_event()

            self.assertTrue(logger.log_event(event))
            self.assertFalse(logger.log_event(event))
            self.assertEqual(len(path.read_text().splitlines()), 1)

    def test_existing_events_are_loaded_for_deduplication(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text(json.dumps(make_event()) + "\n", encoding="utf-8")

            logger = EventLogger(path)

            self.assertFalse(logger.log_event(make_event()))
            self.assertEqual(len(path.read_text().splitlines()), 1)

    def test_same_event_id_in_different_sessions_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            logger = EventLogger(path)

            self.assertTrue(logger.log_event(make_event(session_id="session-001")))
            self.assertTrue(logger.log_event(make_event(session_id="session-002")))
            self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_invalid_event_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            logger = EventLogger(path)
            event = make_event()
            event["vehicle_class"] = "PLANE"

            with self.assertRaises(ValueError):
                logger.log_event(event)

            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
