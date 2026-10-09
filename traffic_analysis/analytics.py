"""Analytics helpers for traffic events."""

import json
from collections import Counter
from pathlib import Path

VEHICLE_CLASSES = ("Car", "Bike", "Bus", "Truck")
EVENT_TYPES = ("vehicle_crossing", "wrong_way")


def classify_density(vehicle_count: int) -> str:
    """Classify vehicles currently visible in a frame."""
    if vehicle_count < 0:
        raise ValueError("vehicle_count must be non-negative")
    if vehicle_count < 6:
        return "LOW"
    if vehicle_count < 9:
        return "MEDIUM"
    return "HIGH"


def load_events(path: str | Path = "events.json") -> list[dict]:
    """Load events from a JSON file and check the basic data structure."""
    with open(path, encoding="utf-8") as event_file:
        events = json.load(event_file)

    if not isinstance(events, list):
        raise ValueError("Events JSON must contain a list")

    required_fields = {
        "event_id",
        "event_type",
        "timestamp",
        "session_id",
        "track_id",
        "vehicle_class",
        "direction",
        "counting_line",
    }
    seen_ids = set()

    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"Event at index {index} must be an object")

        missing = required_fields - event.keys()
        if missing:
            raise ValueError(
                f"Event at index {index} is missing: {sorted(missing)}"
            )

        if event["event_type"] not in EVENT_TYPES:
            raise ValueError(
                f"Invalid event_type at index {index}: "
                f"{event['event_type']}"
            )

        if event["vehicle_class"] not in VEHICLE_CLASSES:
            raise ValueError(
                f"Invalid vehicle_class at index {index}: "
                f"{event['vehicle_class']}"
            )

        event_id = event["event_id"]
        if not isinstance(event_id, str) or not event_id:
            raise ValueError(f"Invalid event_id at index {index}")
        if event_id in seen_ids:
            raise ValueError(f"Duplicate event_id: {event_id}")
        seen_ids.add(event_id)

        if event["event_type"] == "vehicle_crossing":
            if not event["counting_line"]:
                raise ValueError(
                    f"Crossing event at index {index} needs counting_line"
                )
        elif event["counting_line"] is not None:
            raise ValueError(
                f"Wrong-way event at index {index} must have "
                "counting_line=None"
            )

    return events


def summarize_events(events: list[dict]) -> dict:
    """Summarize crossing counts and recorded wrong-way events."""
    crossings = [
        event for event in events
        if event["event_type"] == "vehicle_crossing"
    ]
    wrong_way_events = [
        event for event in events
        if event["event_type"] == "wrong_way"
    ]

    class_counts = Counter(event["vehicle_class"] for event in crossings)

    return {
        "total_crossings": len(crossings),
        "vehicle_counts": {
            vehicle_class: class_counts.get(vehicle_class, 0)
            for vehicle_class in VEHICLE_CLASSES
        },
        "wrong_way_events": len(wrong_way_events),
        "session_ids": sorted({
            event["session_id"] for event in events
        }),
    }


def main() -> None:
    events = load_events()
    summary = summarize_events(events)

    print("TRAFFIC ANALYTICS")
    print("=" * 30)
    print(f"Total crossings: {summary['total_crossings']}")
    for vehicle_class, count in summary["vehicle_counts"].items():
        print(f"{vehicle_class}: {count}")
    print(f"Wrong-way events: {summary['wrong_way_events']}")
    print(f"Sessions: {len(summary['session_ids'])}")


if __name__ == "__main__":
    main()
