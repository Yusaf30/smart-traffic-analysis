import csv
import json
import os
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from ultralytics import YOLO
from traffic_analysis.events import TrafficEvent
from traffic_analysis.analytics import (
    count_untracked_observations,
    summarize_metrics,
)
from traffic_analysis.helmet_monitor import HelmetRiderMonitor
from traffic_analysis.reid import CanonicalCounter
from traffic_analysis.run_history import register_completed_run

# =========================
# CONFIG
# =========================
VIDEO_PATH = os.environ.get("VIDEO_PATH", "india_trafic.mp4")
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
HEADLESS = os.environ.get("HEADLESS", "0") == "1"
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "0"))

CONF_THRESHOLD = 0.4
VEHICLE_CLASSES = [2, 3, 5, 7]
CLASS_NAMES = {2: "Car", 3: "Bike", 5: "Bus", 7: "Truck"}

# ReID experimental defaults (not calibrated thresholds)
REID_ENABLED = True
REID_MAX_GAP_SECONDS = 1.0
REID_MAX_SPEED_PX_PER_SEC = 200.0
REID_HIST_THRESHOLD = 0.70

EXPECTED_DIRECTION = "UP"     # change to "DOWN" if normal traffic shows WRONG WAY
STOP_LINE_Y = 450
COUNT_LINE_Y = int(os.environ.get("COUNT_LINE_Y", "500"))

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
    return direction != "NONE" and direction != EXPECTED_DIRECTION


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
person_model = YOLO("yolov8n.pt")
helmet_monitor = HelmetRiderMonitor(helmet_model, person_model, every=3)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Could not open video: {VIDEO_PATH}")

run_started_at = datetime.now(UTC)
run_id = f"{run_started_at.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid4().hex[:8]}"
run_output_dir = RUNS_DIR / run_id
run_output_dir.mkdir(parents=True, exist_ok=False)
EVENTS_PATH = run_output_dir / "events.json"
STATS_PATH = run_output_dir / "stats.json"
FRAME_LOG_PATH = run_output_dir / "frame_log.csv"
reid_log_name = Path(
    os.environ.get("REID_FRAME_LOG_PATH", "reid_frame_log.csv")
).name or "reid_frame_log.csv"
REID_FRAME_LOG_PATH = run_output_dir / reid_log_name

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25

track_history = defaultdict(lambda: deque(maxlen=30))
counted_ids = set()
recent_counts = deque(maxlen=30)
wrong_ids = set()
counts = {"Car": 0, "Bike": 0, "Bus": 0, "Truck": 0}
wrong_streak = defaultdict(int)

unique_counter = CanonicalCounter(
    enabled=REID_ENABLED,
    max_gap_seconds=REID_MAX_GAP_SECONDS,
    max_speed_px_per_sec=REID_MAX_SPEED_PX_PER_SEC,
    hist_threshold=REID_HIST_THRESHOLD,
    fps=fps,
) if REID_ENABLED else None
unique_vehicle_count = None if not REID_ENABLED else 0
total_frame_detections = 0
untracked_detection_observations = 0
start_time = time.time()
frame_id = 0

session_id = uuid4().hex
events = []
wrong_event_ids = set()
helmet_violation_ids = set()


def current_metric_summary():
    canonical_vehicles = (
        unique_counter.get_canonical_vehicles(CLASS_NAMES)
        if unique_counter is not None
        else []
    )
    metrics = summarize_metrics(
        unique_vehicles=canonical_vehicles,
        crossing_events=events,
        untracked_detections=untracked_detection_observations,
    )
    if unique_counter is None:
        metrics["unique_vehicle_total"] = None
        metrics["unique_vehicles_by_class"] = None
    return metrics


log_file = open(FRAME_LOG_PATH, "w", encoding="utf-8")
log_file.write("frame,in_frame,total,car,bike,bus,truck,density,wrong_total,wrong_now,fps\n")
reid_log_file = open(REID_FRAME_LOG_PATH, "w", encoding="utf-8", newline="")
reid_log_writer = csv.writer(reid_log_file)
reid_log_writer.writerow([
    "frame", "raw_track_id", "canonical_id", "class_id", "class_name",
    "x1", "y1", "x2", "y2", "centroid_x", "centroid_y",
    "matched_existing", "match_similarity", "decision_reason",
    "crossing_detected",
])

# =========================
# MAIN LOOP
# =========================
run_completed_successfully = False
try:
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
        in_frame = 0
        wrong_now = []
        bike_tracks = []

        boxes = None
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            in_frame = len(boxes)   # ALL valid vehicle detections, including no-ID

        total_frame_detections += in_frame

        if boxes is not None and results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            n_in_frame = len(ids)
            reid_observations = [
                (
                    int(tid),
                    int(cls),
                    tuple(map(int, box)),
                    (
                        (int(box[0]) + int(box[2])) // 2,
                        (int(box[1]) + int(box[3])) // 2,
                    ),
                )
                for box, tid, cls in zip(boxes, ids, clss)
            ]
            canonical_ids = {}
            reid_decisions = {}
            if unique_counter is not None:
                canonical_ids = unique_counter.register_frame(
                    reid_observations, frame_id, frame
                )
                reid_decisions = {
                    decision.track_id: decision
                    for decision in unique_counter.get_frame_diagnostics()
                }
                unique_vehicle_count = unique_counter.get_unique_count()

            for box, tid, cls in zip(boxes, ids, clss):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                canonical_id = canonical_ids.get(int(tid))
                crossing_detected = False

                prev = track_history[tid][-1] if track_history[tid] else None
                track_history[tid].append((cx, cy))
    
                speed = real_speed(prev, (cx, cy), fps)
    
                direction = get_direction(track_history[tid])
                is_wrong = wrong_way(direction)
    
                is_red = red_light_violation((cx, cy), prev, light)
    
                name = CLASS_NAMES.get(cls, "Vehicle")
                if name == "Bike":
                    bike_tracks.append((int(tid), (x1, y1, x2, y2)))
    
                # Count each raw tracker ID once when it crosses the line.
                if prev is not None and tid not in counted_ids:
                    crossed_down = prev[1] < COUNT_LINE_Y <= cy
                    crossed_up = prev[1] >= COUNT_LINE_Y > cy

                    if crossed_down or crossed_up:
                        crossing_detected = True
                        counted_ids.add(tid)
                        counts[name] += 1

                        # Determine direction from the actual line crossing.
                        event_direction = "DOWN" if crossed_down else "UP"

                        crossing_event = TrafficEvent.create(
                            event_type="vehicle_crossing",
                            session_id=session_id,
                            track_id=int(tid),
                            vehicle_class=name,
                            direction=event_direction,
                            counting_line="main_line",
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
                        event_direction = direction if direction != "NONE" else "UNKNOWN"
                        wrong_event = TrafficEvent.create(
                            event_type="wrong_way",
                            session_id=session_id,
                            track_id=int(tid),
                            vehicle_class=name,
                            direction=event_direction,
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
    
                identity_label = f"ID:{tid}"
                if canonical_id is not None:
                    identity_label += f" C:{canonical_id}"
                label = f"{name} {identity_label} {int(speed)}km/h {direction}"
                if is_wrong:
                    label += " WRONG WAY"
                if tid in helmet_violation_ids:
                    label += " NO HELMET"
                if is_red:
                    label += " RED VIOLATION"
    
                cv2.putText(annotated, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if unique_counter is not None:
                    decision = reid_decisions[int(tid)]
                    reid_log_writer.writerow([
                        frame_id,
                        int(tid),
                        canonical_id,
                        int(cls),
                        name,
                        x1,
                        y1,
                        x2,
                        y2,
                        cx,
                        cy,
                        decision.matched_existing,
                        (
                            ""
                            if decision.similarity is None
                            else f"{decision.similarity:.6f}"
                        ),
                        decision.reason,
                        crossing_detected,
                    ])

        untracked_detection_observations += count_untracked_observations(
            in_frame, n_in_frame
        )

        # Rider-based helmet check (person + bike association; runs every 3rd frame)
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
                counting_line=None,
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
            with open(STATS_PATH, "w", encoding="utf-8") as f:
                json.dump({"total": total, "total_frame_detections": total_frame_detections,
                           "raw_tracker_ids_observed": len(track_history),
                           "unique_vehicle_count": unique_vehicle_count, **counts,
                           **current_metric_summary(),
                           "density": density,
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
        log_file.write(f"{frame_id},{in_frame},{total},{counts['Car']},{counts['Bike']},"
                       f"{counts['Bus']},{counts['Truck']},{density},{len(wrong_ids)},"
                       f"{len(wrong_now)},{fps_live:.1f}\n")
    
        if not HEADLESS:
            cv2.imshow("Traffic System", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        if MAX_FRAMES > 0 and frame_id >= MAX_FRAMES:
            print(f"Reached MAX_FRAMES ({MAX_FRAMES}). Stopping.")
            break

    run_completed_successfully = True
    
finally:
    # =========================
    # CLEANUP
    # =========================
    cap.release()
    cv2.destroyAllWindows()
    if not log_file.closed:
        log_file.close()
    if not reid_log_file.closed:
        reid_log_file.close()
    
    total = sum(counts.values())
    recent_density = traffic_density(sum(recent_counts) / len(recent_counts)) if recent_counts else 'LOW'
    with open(STATS_PATH, 'w', encoding='utf-8') as f:
        json.dump({'total': total, 'total_frame_detections': total_frame_detections,
                   'raw_tracker_ids_observed': len(track_history),
                   'unique_vehicle_count': unique_vehicle_count, **counts,
                   **current_metric_summary(),
                   'density': recent_density,
                   'wrong_way': len(wrong_ids)}, f, indent=2)
    
    with open(EVENTS_PATH, 'w', encoding='utf-8') as event_file:
        json.dump(events, event_file, indent=2)

    if run_completed_successfully:
        completed_at = datetime.now(UTC).isoformat().replace(
            "+00:00", "Z"
        )
        register_completed_run(
            OUTPUTS_DIR,
            run_id=run_id,
            run_directory=run_output_dir,
            completed_at=completed_at,
            source_video=VIDEO_PATH,
        )
    
    print('=' * 60)
    print('FINAL COUNTS:', counts, 'TOTAL:', sum(counts.values()))
    print('WRONG-WAY IDs:', sorted(wrong_ids))
    print('HELMET VIOLATION IDs:', sorted(helmet_violation_ids))
    print('TRACKED IDs:', len(track_history))
    print('Frames processed:', frame_id)
    print(f'Events saved: {len(events)}')
    print(f'Event output: {EVENTS_PATH}')
    print(f'Stats output: {STATS_PATH}')
    print(f'Frame log output: {FRAME_LOG_PATH}')
    print(f'ReID frame log output: {REID_FRAME_LOG_PATH}')
    print('=' * 60)
    if run_completed_successfully:
        print('Processing Complete')
    else:
        print('Processing did not complete; run was not added to history.')
