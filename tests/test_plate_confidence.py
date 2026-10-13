
from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO = "input.mp4"
MODEL_PATH = "models/candidates/plate_koushim.pt"
OUTPUT = Path("runs/plate_confidence")
SECONDS = [5, 7, 10, 12, 14]
THRESHOLDS = [0.10, 0.25, 0.40]

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total <= 0:
        cap.release()
        raise RuntimeError("Invalid video metadata")

    try:
        for sec in SECONDS:
            if sec * fps >= total:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
            ok, frame = cap.read()
            if not ok:
                continue

            for conf in THRESHOLDS:
                result = model.predict(
                    frame, imgsz=960, conf=conf, verbose=False
                )[0]

                print(f"{sec}s | conf={conf:.2f} | detections={len(result.boxes)}")

                # Draw boxes without large text labels.
                annotated = frame.copy()
                for box in result.boxes:
                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].cpu().tolist()
                    )
                    score = float(box.conf[0])
                    cv2.rectangle(
                        annotated, (x1, y1), (x2, y2), (0, 255, 0), 2
                    )
                    cv2.putText(
                        annotated, f"{score:.2f}",
                        (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1
                    )

                cv2.imwrite(
                    str(OUTPUT / f"sec{sec:02d}_conf{conf:.2f}.jpg"),
                    annotated,
                )
    finally:
        cap.release()

if __name__ == "__main__":
    main()
