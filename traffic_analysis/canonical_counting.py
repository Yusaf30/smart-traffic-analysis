"""Line-crossing counter keyed by canonical identity (offline second pass).

Raw tracker IDs stay in the returned events; only the state keys
(last position, counted set) use the canonical ID.
"""
from __future__ import annotations

import csv
from pathlib import Path

from traffic_analysis.crossing_validation import (
    CrossingObservation,
    validate_crossing,
)


class CanonicalMapError(ValueError):
    """The identity map is missing, malformed, or inconsistent."""


def load_canonical_map(path, require_nonempty=True):
    """Load identity_map.csv -> {raw_id: canonical_id}. Strict: raises on any problem.

    Accepted rows:
      * source == "approved_mapping": any track_id -> canonical_id
      * source == "identity": only when track_id == canonical_id
    Anything else (unknown source, conflicting rows, chains) raises. A map with no
    data rows raises unless require_nonempty=False.
    """
    p = Path(path)
    if not p.is_file():
        raise CanonicalMapError(f"identity map not found: {p}")
    mapping = {}
    with p.open(newline="") as f:
        reader = csv.DictReader(f)
        if not {"track_id", "canonical_id", "source"} <= set(reader.fieldnames or []):
            raise CanonicalMapError("identity map needs columns track_id,canonical_id,source")
        for n, row in enumerate(reader, start=2):
            try:
                tid, cid = int(row["track_id"]), int(row["canonical_id"])
            except (TypeError, ValueError):
                raise CanonicalMapError(f"line {n}: non-integer id") from None
            src = row["source"]
            if src == "identity":
                if tid != cid:
                    raise CanonicalMapError(
                        f"line {n}: identity row must have track_id == canonical_id ({tid} != {cid})")
            elif src != "approved_mapping":
                raise CanonicalMapError(f"line {n}: unknown source {src!r}")
            if mapping.get(tid, cid) != cid:
                raise CanonicalMapError(f"line {n}: conflicting mapping for id {tid}")
            mapping[tid] = cid
    if require_nonempty and not mapping:
        raise CanonicalMapError("identity map has no data rows")
    for cid in set(mapping.values()):
        if mapping.get(cid, cid) != cid:
            raise CanonicalMapError(f"canonical id {cid} is itself remapped (chain)")
    return mapping


class LineCounter:
    """Mirrors the live crossing rule in Terr_v2.py; state is per canonical ID."""

    def __init__(self, line_y, class_names, canonical=None, axis="y"):
        self.line_y = line_y
        self.axis = axis
        self.class_names = class_names
        self.canonical = dict(canonical or {})
        self.counts = {n: 0 for n in dict.fromkeys(class_names.values())}
        self.counted = set()
        self.conflicts = []  # same canonical id, different raw ids, same frame
        self.unconfirmed_crossings = []
        self._pos, self._frame, self._raw = {}, {}, {}
        self._box = {}

    def update(self, frame, raw_id, cls, cx, cy, box=None):
        """Feed one observation; return an event dict if it triggers a count."""
        key = self.canonical.get(raw_id, raw_id)
        if self._frame.get(key) == frame and self._raw[key] != raw_id:
            self.conflicts.append((frame, key, self._raw[key], raw_id))
            return None
        prev = self._pos.get(key)
        previous = None if prev is None else CrossingObservation(
            self._frame[key], prev, self._box[key]
        )
        current_box = None if box is None else tuple(float(v) for v in box)
        current = CrossingObservation(frame, (cx, cy), current_box)
        validation = (
            None if previous is None
            else validate_crossing(previous, current, self.line_y, axis=self.axis)
        )
        self._pos[key], self._frame[key], self._raw[key] = (cx, cy), frame, raw_id
        self._box[key] = current_box
        if previous is None or key in self.counted or not validation.crossed:
            return None
        if not validation.confirmed:
            self.unconfirmed_crossings.append({
                "frame": int(frame),
                "raw_id": int(raw_id),
                "canonical_id": int(key),
                "reason": validation.reason,
            })
            return None
        self.counted.add(key)
        name = self.class_names.get(cls, "Vehicle")
        self.counts[name] = self.counts.get(name, 0) + 1
        return {"frame": frame, "raw_id": int(raw_id), "canonical_id": int(key),
                "vehicle_class": name}