"""Runs rider-based helmet checks inside the main video loop.

Detects persons with its own YOLO instance (the vehicle tracker is untouched),
associates them with the tracker's Bike boxes, classifies each accepted
rider's head, confirms violations over several frames, and draws ONLY riders
that are confirmed with helmet / confirmed without helmet.
"""
from __future__ import annotations

import cv2

from traffic_analysis.helmet_rider import (
    RiderConfig,
    RiderTracker,
    ViolationConfirmer,
    associate,
    classify_head,
    head_region,
)

FONT = cv2.FONT_HERSHEY_SIMPLEX
DRAW = {"helmet": ((0, 200, 0), "HELMET"), "no_helmet": ((0, 0, 255), "NO HELMET")}


def head_crop_dets(helmet_model, frame, head, imgsz=320, conf=0.10, pad=0.5, min_side=64):
    """Run the helmet model on a padded head crop; return (name, conf, box) in frame coords."""
    h, w = frame.shape[:2]
    hw, hh = head[2] - head[0], head[3] - head[1]
    cx, cy = (head[0] + head[2]) / 2, (head[1] + head[3]) / 2
    half_w = max(min_side / 2, hw * (0.5 + pad))
    half_h = max(min_side / 2, hh * (0.5 + pad))
    x1, y1 = max(0, int(cx - half_w)), max(0, int(cy - half_h))
    x2, y2 = min(w, int(cx + half_w)), min(h, int(cy + half_h))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return []
    res = helmet_model.predict(frame[y1:y2, x1:x2], conf=conf, imgsz=imgsz, verbose=False)[0]
    dets = []
    for b in res.boxes:
        bx = b.xyxy[0].tolist()
        dets.append((
            helmet_model.names[int(b.cls.item())],
            float(b.conf.item()),
            (bx[0] + x1, bx[1] + y1, bx[2] + x1, bx[3] + y1),
        ))
    return dets


class HelmetRiderMonitor:
    def __init__(self, helmet_model, person_model, cfg: RiderConfig | None = None,
                 every: int = 3, person_conf: float = 0.40):
        self.cfg = cfg or RiderConfig()
        self.helmet_model = helmet_model
        self.person_model = person_model
        self.every = every
        self.person_conf = person_conf
        self.tracker = RiderTracker(max_age=2 * every)
        self.confirmer = ViolationConfirmer(self.cfg)
        self.reported_bikes: set[int] = set()

    def update(self, frame, bike_tracks, frame_id: int) -> list[int]:
        """bike_tracks: [(track_id, (x1, y1, x2, y2))]. Returns bike track ids
        whose helmet violation was confirmed on this call (at most once per bike)."""
        if frame_id % self.every != 0 or not bike_tracks:
            return []
        res = self.person_model.predict(frame, conf=self.person_conf, classes=[0], verbose=False)[0]
        persons = [tuple(map(float, b.xyxy[0].tolist())) for b in res.boxes]
        bike_ids = [t[0] for t in bike_tracks]
        bike_boxes = [t[1] for t in bike_tracks]

        accepted = [a for a in associate(persons, bike_boxes, self.cfg) if a.accepted]
        keys = self.tracker.match([persons[a.person_idx] for a in accepted], frame_id)

        confirmed_now: list[int] = []
        for a, key in zip(accepted, keys):
            person = persons[a.person_idx]
            head = head_region(person, self.cfg)
            dets = head_crop_dets(self.helmet_model, frame, head)
            label, _, _ = classify_head(dets, head, self.cfg)
            bike_id = bike_ids[a.bike_idx]
            self.tracker.record(key, bike_id, person, head, label, frame_id)
            if self.confirmer.update(key, label):
                self.tracker.mark_confirmed(key)
                if bike_id not in self.reported_bikes:
                    self.reported_bikes.add(bike_id)
                    confirmed_now.append(bike_id)
        return confirmed_now

    def draw(self, annotated, frame_id: int) -> None:
        for _, head, state in self.tracker.visible(frame_id):
            color, text = DRAW[state]
            x1, y1, x2, y2 = (int(v) for v in head)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (tw, th), base = cv2.getTextSize(text, FONT, 0.6, 2)
            ty = max(0, y1 - th - base - 6)
            cv2.rectangle(annotated, (x1, ty), (x1 + tw + 10, ty + th + base + 6), color, -1)
            cv2.putText(annotated, text, (x1 + 5, ty + th + 2), FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
