#!/usr/bin/env python3
"""Run the rider-association + head-crop helmet pipeline on chosen frames.

Does not touch Terr_v2.py. Writes one readable image per frame (photo on top,
text legend underneath), plus a CSV, under diag_helmet/rider_pipeline/.

Usage (from project root):
  python3 tools/helmet_rider_pipeline_diag.py --frames 189,237
  python3 tools/helmet_rider_pipeline_diag.py --every 25
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traffic_analysis.helmet_rider import (  # noqa: E402
    RiderConfig, associate, classify_head, head_region,
)

VIDEO = "input.mp4"
OUT = Path("diag_helmet/rider_pipeline")
DET_CONF = 0.40          # person / motorcycle detector confidence
HEAD_CONF = 0.10         # helmet model confidence on head crops (decision is made in classify_head)
HEAD_IMGSZ = 320
CROP_PAD = 0.5           # extra context around the head region, fraction of its size
MIN_CROP = 64            # minimum crop side in pixels

FONT = cv2.FONT_HERSHEY_SIMPLEX
STATE_COLOR = {          # BGR
    "no_helmet": (0, 0, 255),
    "helmet": (0, 200, 0),
    "unknown": (0, 215, 255),
    "rejected": (170, 170, 170),
}
BIKE_COLOR = (255, 150, 50)

cfg = RiderConfig()


def clamp_box(b, w, h):
    return (max(0, int(b[0])), max(0, int(b[1])), min(w, int(b[2])), min(h, int(b[3])))


def head_crop_dets(helmet_model, frame, head):
    h, w = frame.shape[:2]
    hw, hh = head[2] - head[0], head[3] - head[1]
    cx, cy = (head[0] + head[2]) / 2, (head[1] + head[3]) / 2
    half_w = max(MIN_CROP / 2, hw * (0.5 + CROP_PAD))
    half_h = max(MIN_CROP / 2, hh * (0.5 + CROP_PAD))
    x1, y1, x2, y2 = clamp_box((cx - half_w, cy - half_h, cx + half_w, cy + half_h), w, h)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return []
    res = helmet_model.predict(frame[y1:y2, x1:x2], conf=HEAD_CONF, imgsz=HEAD_IMGSZ, verbose=False)[0]
    dets = []
    for b in res.boxes:
        bx = b.xyxy[0].tolist()
        dets.append((
            helmet_model.names[int(b.cls.item())],
            float(b.conf.item()),
            (bx[0] + x1, bx[1] + y1, bx[2] + x1, bx[3] + y1),
        ))
    return dets


def _tag(img, text, x, y, bg, placed, scale=0.9, thick=2):
    """Filled label with dark text; slides down if it would cover another tag."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    w, h = tw + 12, th + base + 8
    x = max(0, min(img.shape[1] - w, int(x)))
    y = max(0, int(y))
    while any(not (x + w < px or px + pw < x or y + h < py or py + ph < y) for px, py, pw, ph in placed):
        y += h + 2
        if y + h > img.shape[0]:
            break
    placed.append((x, y, w, h))
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    cv2.putText(img, text, (x + 6, y + h - base - 4), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)


def render_overlay(frame, frame_id, persons, bikes, assocs, heads):
    """Photo with boxes + short ID tags, and a black legend panel underneath.

    heads: {person_idx: (head_box, label, no_conf, yes_conf)} for accepted riders.
    """
    vis = frame.copy()
    placed = []
    for bk in bikes:
        cv2.rectangle(vis, (int(bk[0]), int(bk[1])), (int(bk[2]), int(bk[3])), BIKE_COLOR, 2)
    for a in assocs:
        p = persons[a.person_idx]
        state = heads[a.person_idx][1] if a.accepted else "rejected"
        thick = 4 if a.accepted else 2
        cv2.rectangle(vis, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), STATE_COLOR[state], thick)
        if a.accepted:
            hb = heads[a.person_idx][0]
            cv2.rectangle(vis, (int(hb[0]), int(hb[1])), (int(hb[2]), int(hb[3])), (255, 255, 0), 2)
    # Tags last so no box line is drawn over them.
    for bi, bk in enumerate(bikes):
        _tag(vis, f"B{bi}", bk[0] + 2, bk[3] - 34, BIKE_COLOR, placed)
    for a in assocs:
        state = heads[a.person_idx][1] if a.accepted else "rejected"
        p = persons[a.person_idx]
        _tag(vis, f"P{a.person_idx}", p[0], p[1] + 4, STATE_COLOR[state], placed)

    # Legend panel (large text, one row per person).
    row_h = 36
    n_acc = sum(a.accepted for a in assocs)
    panel = np.zeros((70 + row_h * max(1, len(assocs)), vis.shape[1], 3), np.uint8)
    head_txt = f"FRAME {frame_id}   persons {len(persons)}   bikes {len(bikes)}   riders accepted {n_acc}"
    cv2.putText(panel, head_txt, (12, 40), FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    for i, a in enumerate(assocs):
        y = 70 + row_h * i + 24
        if a.accepted:
            _, label, no_c, yes_c = heads[a.person_idx]
            txt = (f"P{a.person_idx} -> B{a.bike_idx}   {label.upper()}   "
                   f"no {no_c:.2f}  yes {yes_c:.2f}   score {a.score:.2f}")
            color = STATE_COLOR[label]
        else:
            txt = (f"P{a.person_idx}   REJECTED: {a.reason}   "
                   f"(depth {a.foot_ratio:.2f}, height {a.height_ratio:.2f})")
            color = STATE_COLOR["rejected"]
        cv2.putText(panel, txt, (12, y), FONT, 0.8, color, 2, cv2.LINE_AA)
    return np.vstack([vis, panel])


def main():
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="189,237", help="comma-separated 1-based frame numbers")
    ap.add_argument("--every", type=int, default=0, help="also sample every N frames")
    args = ap.parse_args()

    targets = {int(x) for x in args.frames.split(",") if x.strip()}
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {VIDEO}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.every > 0:
        targets |= set(range(args.every, total + 1, args.every))

    OUT.mkdir(parents=True, exist_ok=True)
    det_model = YOLO("yolov8n.pt")
    helmet_model = YOLO("models/candidates/helmet_iam_tsr.pt")

    rows = []
    frame_id = 0
    last = max(targets)
    while frame_id < last:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        if frame_id not in targets:
            continue
        frame = cv2.resize(frame, (1280, 720))
        res = det_model.predict(frame, conf=DET_CONF, classes=[0, 3], verbose=False)[0]
        persons, bikes = [], []
        for b in res.boxes:
            box = tuple(map(float, b.xyxy[0].tolist()))
            (persons if int(b.cls.item()) == 0 else bikes).append(box)

        assocs = associate(persons, bikes, cfg)
        heads = {}
        for a in assocs:
            if a.accepted:
                head = head_region(persons[a.person_idx], cfg)
                dets = head_crop_dets(helmet_model, frame, head)
                label, no_c, yes_c = classify_head(dets, head, cfg)
                heads[a.person_idx] = (head, label, no_c, yes_c)
            p = persons[a.person_idx]
            h = heads.get(a.person_idx)
            rows.append([
                frame_id, a.person_idx, [round(v, 1) for v in p], a.bike_idx,
                a.accepted, round(a.score, 3), a.reason,
                round(a.foot_ratio, 3), round(a.height_ratio, 3), round(a.center_offset, 3),
                h[1] if h else "-", round(h[2], 3) if h else 0.0, round(h[3], 3) if h else 0.0,
            ])

        img = render_overlay(frame, frame_id, persons, bikes, assocs, heads)
        cv2.imwrite(str(OUT / f"frame_{frame_id:04d}_rider.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"frame {frame_id}: persons={len(persons)} bikes={len(bikes)} "
              f"accepted={sum(a.accepted for a in assocs)}")

    cap.release()
    with (OUT / "rider_pipeline.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "person_idx", "person_box", "bike_idx", "accepted", "score", "reason",
                    "foot_ratio", "height_ratio", "center_offset", "head_label", "no_conf", "yes_conf"])
        w.writerows(rows)
    print(f"Saved: {OUT}/frame_XXXX_rider.jpg and {OUT}/rider_pipeline.csv")


if __name__ == "__main__":
    main()
