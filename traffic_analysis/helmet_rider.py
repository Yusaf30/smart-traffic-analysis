"""Conservative rider association and helmet-violation confirmation.

Pure Python (no cv2 / ultralytics) so it can be unit-tested without models.
Boxes are (x1, y1, x2, y2) in pixels, y grows downward.

Policy: a person is a "rider" only if several independent geometric cues agree
and exactly one motorcycle is clearly the best match. Everything else is
rejected with an explicit reason; rejected people are never violators.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Hashable, Sequence

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class RiderConfig:
    # Rider-association gates (calibrate on more footage before trusting).
    min_foot_ratio: float = 0.40      # person bottom depth inside bike box: 0=top edge, 1=bottom edge
    min_height_ratio: float = 0.75    # person height / bike height
    max_center_offset: float = 0.60   # |person cx - bike cx| / bike width
    min_score: float = 0.55
    min_margin: float = 0.12          # best bike must beat runner-up by this much
    max_riders_per_bike: int = 3
    # Head region, as fractions of the person box.
    head_height_frac: float = 0.35
    head_width_trim: float = 0.10
    head_top_pad: float = 0.05
    # Head classification.
    no_helmet_conf: float = 0.45
    helmet_conf: float = 0.30
    decision_margin: float = 0.10
    # Multi-frame confirmation.
    window: int = 6
    min_no_helmet_obs: int = 3
    min_no_helmet_ratio: float = 0.70


@dataclass(frozen=True)
class Association:
    person_idx: int
    bike_idx: int | None
    accepted: bool
    score: float
    reason: str          # "ok" or a rejection reason
    foot_ratio: float
    height_ratio: float
    center_offset: float


def box_w(b: Box) -> float:
    return max(0.0, b[2] - b[0])


def box_h(b: Box) -> float:
    return max(0.0, b[3] - b[1])


def box_center(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


def pair_features(person: Box, bike: Box) -> tuple[float, float, float]:
    """(foot_ratio, height_ratio, center_offset) for a person/bike pair."""
    bw, bh = max(1.0, box_w(bike)), max(1.0, box_h(bike))
    foot_ratio = (person[3] - bike[1]) / bh
    height_ratio = box_h(person) / bh
    center_offset = abs(box_center(person)[0] - box_center(bike)[0]) / bw
    return foot_ratio, height_ratio, center_offset


def score_pair(person: Box, bike: Box, cfg: RiderConfig) -> tuple[float, str, tuple[float, float, float]]:
    """Return (score, reason, features). reason == "ok" if all hard gates pass."""
    feats = pair_features(person, bike)
    foot, hratio, offset = feats
    if offset > cfg.max_center_offset:
        return 0.0, "person_not_over_bike", feats
    if foot < cfg.min_foot_ratio:
        return 0.0, "person_ends_above_vehicle", feats
    if hratio < cfg.min_height_ratio:
        return 0.0, "person_too_short_for_rider", feats
    score = (
        0.4 * _clip01(foot / 0.8)
        + 0.3 * _clip01(hratio / 1.0)
        + 0.3 * (1.0 - _clip01(offset / cfg.max_center_offset))
    )
    if score < cfg.min_score:
        return score, "low_score", feats
    return score, "ok", feats


def associate(persons: Sequence[Box], bikes: Sequence[Box], cfg: RiderConfig = RiderConfig()) -> list[Association]:
    """Associate each person with at most one motorcycle, or reject with a reason."""
    out: list[Association] = []
    for pi, person in enumerate(persons):
        if not bikes:
            out.append(Association(pi, None, False, 0.0, "no_bike_detected", 0.0, 0.0, 0.0))
            continue
        scored = [(bi, *score_pair(person, bike, cfg)) for bi, bike in enumerate(bikes)]
        passing = sorted(
            [s for s in scored if s[2] == "ok"], key=lambda s: (-s[1], s[0])
        )
        if not passing:
            # Report the reason from the geometrically nearest bike (smallest center offset).
            bi, sc, reason, feats = min(scored, key=lambda s: s[3][2])
            out.append(Association(pi, None, False, sc, reason, *feats))
            continue
        bi, sc, _, feats = passing[0]
        if len(passing) > 1 and sc - passing[1][1] < cfg.min_margin:
            out.append(Association(pi, None, False, sc, "ambiguous_multiple_bikes", *feats))
            continue
        out.append(Association(pi, bi, True, sc, "ok", *feats))

    # Capacity: keep the best-scoring riders per bike, reject the rest.
    by_bike: dict[int, list[Association]] = defaultdict(list)
    for a in out:
        if a.accepted and a.bike_idx is not None:
            by_bike[a.bike_idx].append(a)
    for bike_idx, group in by_bike.items():
        group.sort(key=lambda a: (-a.score, a.person_idx))
        for extra in group[cfg.max_riders_per_bike:]:
            out[extra.person_idx] = Association(
                extra.person_idx, None, False, extra.score, "over_capacity",
                extra.foot_ratio, extra.height_ratio, extra.center_offset,
            )
    return out


def head_region(person: Box, cfg: RiderConfig = RiderConfig()) -> Box:
    """Top part of a person box, trimmed horizontally."""
    w, h = box_w(person), box_h(person)
    return (
        person[0] + cfg.head_width_trim * w,
        person[1] - cfg.head_top_pad * h,
        person[2] - cfg.head_width_trim * w,
        person[1] + cfg.head_height_frac * h,
    )


def classify_head(
    dets: Sequence[tuple[str, float, Box]],
    head: Box,
    cfg: RiderConfig = RiderConfig(),
) -> tuple[str, float, float]:
    """dets: (class_name, conf, box) from the helmet model, in frame coordinates.

    Returns (label, no_helmet_conf, helmet_conf); label is "no_helmet",
    "helmet" or "unknown". Only detections whose center is in the (slightly
    enlarged) head region are considered.
    """
    pad_x, pad_y = 0.2 * box_w(head), 0.2 * box_h(head)
    region = (head[0] - pad_x, head[1] - pad_y, head[2] + pad_x, head[3] + pad_y)
    no_c = yes_c = 0.0
    for name, conf, box in dets:
        cx, cy = box_center(box)
        if not (region[0] <= cx <= region[2] and region[1] <= cy <= region[3]):
            continue
        lname = name.lower()
        if lname == "without helmet":
            no_c = max(no_c, conf)
        elif lname == "with helmet":
            yes_c = max(yes_c, conf)
    if no_c >= cfg.no_helmet_conf and no_c - yes_c >= cfg.decision_margin:
        return "no_helmet", no_c, yes_c
    if yes_c >= cfg.helmet_conf and yes_c - no_c >= cfg.decision_margin:
        return "helmet", no_c, yes_c
    return "unknown", no_c, yes_c


class ViolationConfirmer:
    """Confirms at most one violation per key (e.g. a motorcycle track id).

    Feed it only observations from accepted rider associations. "unknown"
    observations are ignored; "helmet" observations dilute the ratio.
    """

    def __init__(self, cfg: RiderConfig = RiderConfig()):
        self.cfg = cfg
        self._hist: dict[Hashable, deque[str]] = defaultdict(lambda: deque(maxlen=cfg.window))
        self.reported: set[Hashable] = set()

    def update(self, key: Hashable, label: str) -> bool:
        """Return True exactly once per key, when the violation is confirmed."""
        if key in self.reported or label not in {"no_helmet", "helmet"}:
            return False
        hist = self._hist[key]
        hist.append(label)
        n_no = sum(1 for x in hist if x == "no_helmet")
        if n_no >= self.cfg.min_no_helmet_obs and n_no / len(hist) >= self.cfg.min_no_helmet_ratio:
            self.reported.add(key)
            return True
        return False


def iou(a: Box, b: Box) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = box_w(a) * box_h(a) + box_w(b) * box_h(b) - inter
    return inter / union if union > 0 else 0.0


class RiderTracker:
    """Gives accepted riders a stable key across sampled frames (greedy IoU)
    and decides what, if anything, may be drawn for each rider.

    Display policy: only "helmet" (majority of recent decisive observations)
    or "no_helmet" (only after the violation is confirmed). Unknown / unproven
    riders are never drawn.
    """

    def __init__(self, max_age: int = 6, iou_thresh: float = 0.3, vote_window: int = 5):
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.vote_window = vote_window
        self._next = 1
        self._riders: dict[int, dict] = {}

    def match(self, boxes: Sequence[Box], frame_id: int) -> list[int]:
        for key in [k for k, r in self._riders.items() if frame_id - r["last"] > 5 * self.max_age]:
            del self._riders[key]
        pairs = sorted(
            (
                (iou(b, r["box"]), bi, key)
                for bi, b in enumerate(boxes)
                for key, r in self._riders.items()
                if frame_id - r["last"] <= self.max_age
            ),
            reverse=True,
        )
        keys: list[int | None] = [None] * len(boxes)
        used: set[int] = set()
        for score, bi, key in pairs:
            if score < self.iou_thresh:
                break
            if keys[bi] is None and key not in used:
                keys[bi] = key
                used.add(key)
        out: list[int] = []
        for k in keys:
            if k is None:
                k = self._next
                self._next += 1
            out.append(k)
        return out

    def record(self, key: int, bike_id: int, person_box: Box, head_box: Box, label: str, frame_id: int) -> None:
        r = self._riders.get(key)
        if r is None:
            r = {"labels": deque(maxlen=self.vote_window), "confirmed": False}
            self._riders[key] = r
        r.update(box=person_box, head=head_box, bike_id=bike_id, last=frame_id)
        if label in {"helmet", "no_helmet"}:
            r["labels"].append(label)

    def mark_confirmed(self, key: int) -> None:
        if key in self._riders:
            self._riders[key]["confirmed"] = True

    def visible(self, frame_id: int) -> list[tuple[int, Box, str]]:
        """(rider_key, head_box, "helmet" | "no_helmet") for riders that may be drawn."""
        out = []
        for key, r in self._riders.items():
            if frame_id - r["last"] > self.max_age:
                continue
            if r["confirmed"]:
                out.append((key, r["head"], "no_helmet"))
                continue
            n_yes = sum(1 for x in r["labels"] if x == "helmet")
            n_no = sum(1 for x in r["labels"] if x == "no_helmet")
            if n_yes > n_no:
                out.append((key, r["head"], "helmet"))
        return out
