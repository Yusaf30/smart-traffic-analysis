from collections import Counter
from typing import Any

from app.event_validation import validate_event


class VehicleCounter:
    """Aggregate vehicle_crossing events without counting duplicates."""

    def __init__(self) -> None:
        self._processed_events: set[tuple[str, str]] = set()
        self._total = 0
        self._by_class: Counter[str] = Counter()
        self._by_direction: Counter[str] = Counter()

    def process_event(self, event: dict[str, Any]) -> bool:
        """Process one event. Return True if a crossing was counted."""
        validate_event(event)

        if event["event_type"] != "vehicle_crossing":
            return False

        key = (event["session_id"], event["event_id"])
        if key in self._processed_events:
            return False

        self._processed_events.add(key)
        self._total += 1
        self._by_class[event["vehicle_class"]] += 1
        self._by_direction[event["direction"]] += 1
        return True

    def get_stats(self) -> dict[str, Any]:
        """Return a snapshot of the aggregated counts."""
        return {
            "total": self._total,
            "by_class": dict(self._by_class),
            "by_direction": dict(self._by_direction),
        }


