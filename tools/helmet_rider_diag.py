import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO = "input.mp4"
OUT = Path("diag_helmet/rider_association.csv")
TARGET_FRAMES = {189, 237}
VEHICLE_CONF = 0.40
PERSON_CONF = 0.40
HELMET_CONF = 0.05
MIN_PERSON_BIKE_OVERLAP = 0.10


def person_overlap(person, bike):
    px1, py1, px2, py2 = person
    bx1, by1, bx2, by2 = bike
    iw = max(0, min(px2, bx2) - max(px1, bx1))
    ih = max(0, min(py2, by2) - max(py1, by1))
    area = max(1, (px2 - px1) * (py2 - py1))
    return iw * ih / area


def center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


vehicle_model = YOLO("yolov8n.pt")
helmet_model = YOLO("models/candidates/helmet_iam_tsr.pt")

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {VIDEO}")

OUT.parent.mkdir(parents=True, exist_ok=True)

try:
    with OUT.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame", "helmet_cls", "helmet_conf", "helmet_cx", "helmet_cy",
            "person_box", "bike_box", "overlap", "association"
        ])

        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1

            if frame_id not in TARGET_FRAMES:
                continue

            frame = cv2.resize(frame, (1280, 720))

            bike_result = vehicle_model.predict(
                frame, conf=VEHICLE_CONF, classes=[3], verbose=False
            )[0]
            person_result = vehicle_model.predict(
                frame, conf=PERSON_CONF, classes=[0], verbose=False
            )[0]
            helmet_result = helmet_model.predict(
                frame, conf=HELMET_CONF, imgsz=960, verbose=False
            )[0]

            bikes = [
                tuple(map(float, b.xyxy[0].tolist()))
                for b in bike_result.boxes
            ]
            people = [
                tuple(map(float, b.xyxy[0].tolist()))
                for b in person_result.boxes
            ]

            print(
                f"Frame {frame_id}: motorcycles={len(bikes)}, "
                f"people={len(people)}, helmet_detections={len(helmet_result.boxes)}"
            )

            for hb in helmet_result.boxes:
                hbox = tuple(map(float, hb.xyxy[0].tolist()))
                hx, hy = center(hbox)
                hcls = helmet_model.names[int(hb.cls.item())]
                hconf = float(hb.conf.item())

                candidate_people = [
                    p for p in people
                    if p[0] <= hx <= p[2] and p[1] <= hy <= p[3]
                ]

                if not candidate_people:
                    writer.writerow([
                        frame_id, hcls, round(hconf, 3), round(hx, 1),
                        round(hy, 1), "", "", "", "NO_PERSON_MATCH"
                    ])
                    continue

                for person in candidate_people:
                    matches = [
                        (bike, person_overlap(person, bike))
                        for bike in bikes
                    ]
                    matches = [
                        (bike, overlap) for bike, overlap in matches
                        if overlap >= MIN_PERSON_BIKE_OVERLAP
                    ]

                    if len(matches) == 1:
                        bike, overlap = matches[0]
                        status = "SINGLE_CANDIDATE"
                    elif len(matches) > 1:
                        bike, overlap = None, None
                        status = "AMBIGUOUS_MULTIPLE_BIKES"
                    else:
                        bike, overlap = None, None
                        status = "NO_BIKE_MATCH"

                    writer.writerow([
                        frame_id, hcls, round(hconf, 3), round(hx, 1),
                        round(hy, 1), person, bike,
                        round(overlap, 3) if overlap is not None else "",
                        status
                    ])
finally:
    cap.release()

print(f"Saved: {OUT}")
