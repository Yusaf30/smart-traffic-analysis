
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4


EventType = Literal["vehicle_crossing", "wrong_way"]
VehicleClass = Literal["Car", "Bike", "Bus", "Truck"]
Direction = Literal["UP", "DOWN", "LEFT", "RIGHT", "UNKNOWN"]


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
        if not self.event_id:
            object.__setattr__(
                self, "event_id", f"{self.session_id}-{uuid4().hex}"
            )

        if self.track_id < 0:
            raise ValueError("track_id must be non-negative")

        if self.event_type == "vehicle_crossing" and not self.counting_line:
            raise ValueError("vehicle_crossing requires counting_line")

        if self.event_type == "wrong_way" and self.counting_line is not None:
            raise ValueError("wrong_way must have counting_line=None")

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