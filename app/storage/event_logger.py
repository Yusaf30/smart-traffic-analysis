import json
from pathlib import Path
from typing import Any

from app.event_validation import validate_event


class EventLogger:
    """Append validated events to a JSON Lines file."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self._seen_events: set[tuple[str, str]] = set()

        if self.file_path.exists():
            with self.file_path.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    validate_event(event)
                    self._seen_events.add(
                        (event["session_id"], event["event_id"])
                    )

    def log_event(self, event: dict[str, Any]) -> bool:
        """Append a new event; return False if it is a duplicate."""
        validate_event(event)

        key = (event["session_id"], event["event_id"])
        if key in self._seen_events:
            return False

        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(event, separators=(",", ":")) + "\n")

        self._seen_events.add(key)
        return True
