from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4


EventType = Literal["vehicle_crossing", "wrong_way"]
VehicleClass = Literal["Car", "Bike", "Bus", "Truck"]
Direction = Literal["UP", "DOWN", "LEFT", "RIGHT", "UNKNOWN"]

VALID_EVENT_TYPES = {"vehicle_crossing", "wrong_way"}
VALID_VEHICLE_CLASSES = {"Car", "Bike", "Bus", "Truck"}
VALID_DIRECTIONS = {"UP", "DOWN", "LEFT", "RIGHT", "UNKNOWN"}


@dataclass(frozen=True)
class TrafficEvent:
    event_type: EventType
    timestamp: str
    session_id: str
    track_id: int
    vehicle_class: VehicleClass
    direction: Direction
    counting_line: str | None = None
    event_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string")

        if not isinstance(self.event_type, str) or self.event_type not in VALID_EVENT_TYPES:
            raise ValueError("Invalid event_type")

        if not isinstance(self.vehicle_class, str) or self.vehicle_class not in VALID_VEHICLE_CLASSES:
            raise ValueError("Invalid vehicle_class")

        if not isinstance(self.direction, str) or self.direction not in VALID_DIRECTIONS:
            raise ValueError("Invalid direction")

        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int) or self.track_id < 0:
            raise ValueError("track_id must be a non-negative integer")

        if not isinstance(self.timestamp, str):
            raise ValueError("timestamp must be a string")

        try:
            datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp must be a valid ISO 8601 timestamp") from exc

        if self.event_type == "vehicle_crossing":
            if not isinstance(self.counting_line, str) or not self.counting_line.strip():
                raise ValueError("vehicle_crossing requires counting_line")
        elif self.counting_line is not None:
            raise ValueError("wrong_way must have counting_line=None")

        if not self.event_id:
            object.__setattr__(
                self, "event_id", f"{self.session_id}-{uuid4().hex}"
            )
        elif not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("event_id must be a non-empty string")

    @classmethod
    def create(
        cls,
        *,
        event_type: EventType,
        session_id: str,
        track_id: int,
        vehicle_class: VehicleClass,
        direction: Direction,
        counting_line: str | None = None,
    ) -> "TrafficEvent":
        timestamp = (
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        return cls(
            event_type=event_type,
            timestamp=timestamp,
            session_id=session_id,
            track_id=track_id,
            vehicle_class=vehicle_class,
            direction=direction,
            counting_line=counting_line,
        )

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "track_id": self.track_id,
            "vehicle_class": self.vehicle_class,
            "direction": self.direction,
            "counting_line": self.counting_line,
        }
