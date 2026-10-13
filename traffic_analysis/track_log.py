"""CSV logging for per-frame vehicle tracker observations."""

import csv
from operator import index
from pathlib import Path
from typing import Sequence


class TrackLogger:
    """Write one row per tracked detection to a new CSV file."""

    COLUMNS = ("frame", "track_id", "class_id", "conf", "x1", "y1", "x2", "y2")

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = self.path.open("x", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file, lineterminator="\n")
        self._closed = False
        try:
            self._writer.writerow(self.COLUMNS)
            self._file.flush()
        except BaseException:
            self.close()
            raise

    def write(
        self,
        frame: int,
        track_id: int,
        class_id: int,
        conf: float,
        bbox: Sequence[float],
    ) -> None:
        """Write a frame observation; bbox is (x1, y1, x2, y2)."""
        if self._closed:
            raise ValueError("Cannot write to a closed TrackLogger")

        coordinates = tuple(bbox)
        if len(coordinates) != 4:
            raise ValueError("bbox must contain exactly four coordinates")

        self._writer.writerow((
            index(frame),
            index(track_id),
            index(class_id),
            float(conf),
            *(float(coordinate) for coordinate in coordinates),
        ))

    def close(self) -> None:
        """Flush and close the CSV file. Safe to call more than once."""
        if not self._closed:
            try:
                self._file.close()
            finally:
                self._closed = True