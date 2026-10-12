import cv2
import json
import numpy as np
import time
from collections import defaultdict, deque
from ultralytics import YOLO
from uuid import uuid4
from traffic_analysis.events import TrafficEvent
from traffic_analysis.helmet_monitor import HelmetRiderMonitor

# =========================
# CONFIG
# =========================
VIDEO_PATH = "input.mp4"

CONF_THRESHOLD = 0.4
VEHICLE_CLASSES = [2, 3, 5, 7]
CLASS_NAMES = {2: "Car", 3: "Bike", 5: "Bus", 7: "Truck"}

EXPECTED_DIRECTION = "UP"     # change to "DOWN" if normal traffic shows WRONG WAY
STOP_LINE_Y = 450
COUNT_LINE_Y = 330            # vehicles are counted when they cross this line

DENSITY_LOW = 6               # fewer than this many vehicles in frame = LOW
DENSITY_HIGH = 9             # this many or more = HIGH

RED_TIME = 10
GREEN_TIME = 10

LOG_EVERY = 1                 # print a live report every N frames

# =========================
# HOMOGRAPHY (CALIBRATE!)
# =========================
src_pts = np.float32([
    [200, 500],
    [1000, 500],
    [300, 800],
    [900, 800]
])

dst_pts = np.float32([
    [0, 0],
    [12, 0],
    [0, 40],
    [12, 40]
])

H, _ = cv2.findHomography(src_pts, dst_pts)

# =========================
# FUNCTIONS
# =========================
def to_world(pt):
    p = np.array([pt[0], pt[1], 1]).reshape(3, 1)
    w = H @ p
    w /= w[2]
    return w[0][0], w[1][0]


def real_speed(prev, curr, fps):
    if prev is None:
        return 0
    p1 = to_world(prev)
    p2 = to_world(curr)
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    return dist * fps * 3.6


def get_direction(hist):
    if len(hist) < 15:
        return "NONE"
    dy = hist[-1][1] - hist[0][1]
    if abs(dy) < 40:
        return "NONE"
    return "DOWN" if dy > 0 else "UP"


def wrong_way(direction):
    return (
        direction in {"UP", "DOWN", "LEFT", "RIGHT"}
        and direction != EXPECTED_DIRECTION
    )


def traffic_density(n):
    if n < DENSITY_LOW:
        return "LOW"
    if n < DENSITY_HIGH:
        return "MEDIUM"
    return "HIGH"


def traffic_light_state(start):
    t = int(time.time() - start)
    cycle = t % (RED_TIME + GREEN_TIME)
    return "RED" if cycle < RED_TIME else "GREEN"


def red_light_violation(curr, prev, light):
    if light != "RED" or prev is None:
        return False
    return prev[1] < STOP_LINE_Y and curr[1] >= STOP_LINE_Y


# =========================
# INIT
# =========================
model = YOLO("yolov8n.pt")
helmet_model = YOLO("models/candidates/helmet_iam_tsr.pt")
person_model = YOLO("yolov8n.pt")   # separate instance; vehicle tracker stays untouched
helmet_monitor = HelmetRiderMonitor(helmet_model, person_model, every=3)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25

track_history = defaultdict(lambda: deque(maxlen=30))
counted_ids = set()
recent_counts = deque(maxlen=30)
wrong_ids = set()
counts = {"Car": 0, "Bike": 0, "Bus": 0, "Truck": 0}
wrong_streak = defaultdict(int)
start_time = time.time()
frame_id = 0
session_id = uuid4().hex
events = []
wrong_event_ids = set()
helmet_violation_ids = set()

log_file = open("frame_log.csv", "w")
log_file.write("frame,in_frame,total,car,bike,bus,truck,density,wrong_total,wrong_now,fps\n")

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        break

    frame = cv2.resize(frame, (1280, 720))
    frame_id += 1
    annotated = frame.copy()

    # Traffic light simulation
    light = traffic_light_state(start_time)
    light_color = (0, 0, 255) if light == "RED" else (0, 255, 0)
    cv2.circle(annotated, (50, 50), 20, light_color, -1)
    cv2.putText(annotated, light, (80, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, light_color, 2)

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONF_THRESHOLD,
        classes=VEHICLE_CLASSES,
        verbose=False
    )

    n_in_frame = 0
    wrong_now = []
    bike_tracks = []

    if results[0].boxes is not None and results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)
        n_in_frame = len(ids)

        for box, tid, cls in zip(boxes, ids, clss):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            prev = track_history[tid][-1] if track_history[tid] else None
            track_history[tid].append((cx, cy))

            speed = real_speed(prev, (cx, cy), fps)

            direction = get_direction(track_history[tid])
            if direction == "NONE":
                direction = "UNKNOWN"
            is_wrong = wrong_way(direction)

            is_red = red_light_violation((cx, cy), prev, light)

            name = CLASS_NAMES.get(cls, "Vehicle")

            if name == "Bike":
                bike_tracks.append((int(tid), (x1, y1, x2, y2)))

            # Count each tracked vehicle once and emit a crossing event.
            if prev is not None and tid not in counted_ids:
                crossed_line = (
                    (prev[1] < COUNT_LINE_Y <= cy)
                    or (prev[1] >= COUNT_LINE_Y > cy)
                )

                if crossed_line:
                    counted_ids.add(tid)
                    counts[name] += 1

                    crossing_event = TrafficEvent.create(
                        event_type="vehicle_crossing",
                        session_id=session_id,
                        track_id=int(tid),
                        vehicle_class=name,
                        direction=direction,
                        counting_line="main_road_line",
                    )
                    events.append(crossing_event.to_dict())

            if is_wrong:
                wrong_streak[tid] += 1
            else:
                wrong_streak[tid] = 0
            is_wrong = wrong_streak[tid] >= 20   # must stay wrong for 20 frames

            if is_wrong:
                wrong_ids.add(int(tid))
                wrong_now.append(int(tid))

                if tid not in wrong_event_ids:
                    wrong_event = TrafficEvent.create(
                        event_type="wrong_way",
                        session_id=session_id,
                        track_id=int(tid),
                        vehicle_class=name,
                        direction=direction,
                        counting_line=None,
                    )
                    events.append(wrong_event.to_dict())
                    wrong_event_ids.add(tid)
                
            color = (0, 255, 0)
            if is_wrong:
                color = (255, 0, 255)
            if is_red:
                color = (0, 255, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{name} ID:{tid} {int(speed)}km/h {direction}"
            if is_wrong:
                label += " WRONG WAY"
            if tid in helmet_violation_ids:
                label += " NO HELMET"
            if is_red:
                label += " RED VIOLATION"

            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Rider-based helmet check (person + bike association; runs every 3rd frame).
    for bike_id in helmet_monitor.update(frame, bike_tracks, frame_id):
        if bike_id in helmet_violation_ids:
            continue
        bike_dir = get_direction(track_history[bike_id])
        if bike_dir == "NONE":
            bike_dir = "UNKNOWN"
        helmet_event = TrafficEvent.create(
            event_type="helmet_violation",
            session_id=session_id,
            track_id=int(bike_id),
            vehicle_class="Bike",
            direction=bike_dir,
        )
        events.append(helmet_event.to_dict())
        helmet_violation_ids.add(int(bike_id))
        print(f"HELMET VIOLATION | bike track_id={bike_id}")
    helmet_monitor.draw(annotated, frame_id)

    # Stop line (red) and count line (blue)
    cv2.line(annotated, (0, STOP_LINE_Y),
             (annotated.shape[1], STOP_LINE_Y), (0, 0, 255), 2)
    cv2.line(annotated, (0, COUNT_LINE_Y),
             (annotated.shape[1], COUNT_LINE_Y), (255, 100, 0), 2)

    # Stats panel
    total = sum(counts.values())
    recent_counts.append(n_in_frame)
    density = traffic_density(sum(recent_counts) / len(recent_counts))
    panel = [f"Total: {total}",
             f"Cars: {counts['Car']}  Bikes: {counts['Bike']}",
             f"Buses: {counts['Bus']}  Trucks: {counts['Truck']}",
             f"Density: {density}",
             f"Wrong-way: {len(wrong_ids)}"]
    for i, line in enumerate(panel):
        cv2.putText(annotated, line, (20, 140 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Save stats for the dashboard
    if frame_id % 10 == 0:
        with open("stats.json", "w") as f:
            json.dump({"total": total, **counts, "density": density,
                       "wrong_way": len(wrong_ids)}, f)

    # FPS
    fps_live = frame_id / (time.time() - start_time)
    cv2.putText(annotated, f"FPS:{fps_live:.2f}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Live report (terminal + CSV)
    if frame_id % LOG_EVERY == 0:
        print(f"F{frame_id:04d} | in_frame={n_in_frame:2d} | total={total:3d} | "
              f"car={counts['Car']} bike={counts['Bike']} bus={counts['Bus']} "
              f"truck={counts['Truck']} | density={density:6s} | "
              f"wrong_total={len(wrong_ids)} now={wrong_now} | fps={fps_live:.1f}")
    log_file.write(f"{frame_id},{n_in_frame},{total},{counts['Car']},{counts['Bike']},"
                   f"{counts['Bus']},{counts['Truck']},{density},{len(wrong_ids)},"
                   f"{len(wrong_now)},{fps_live:.1f}\n")

    cv2.imshow("Traffic System", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
log_file.close()

print("=" * 60)
print("FINAL COUNTS:", counts, "TOTAL:", sum(counts.values()))
print("WRONG-WAY IDs:", sorted(wrong_ids))
print("TRACKED IDs:", len(track_history))
print("Frames processed:", frame_id)
print("=" * 60)
print("Processing Complete ✅")

with open("events.json", "w", encoding="utf-8") as event_file:
    json.dump(events, event_file, indent=2)

print(f"Events saved: {len(events)}")
print("Event output: events.json")