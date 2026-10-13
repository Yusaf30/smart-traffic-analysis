
from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO = "input.mp4"
MODEL_PATH = "models/candidates/plate_morsetech_yolo11n.pt"
OUTPUT_DIR = Path("runs/plate_yolo11")
SECONDS = [0, 2, 4, 5, 7, 9, 10, 12, 14]
CONFIDENCE = 0.25
IMAGE_SIZE = 960


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError("Invalid video metadata")

    try:
        for sec in SECONDS:
            frame_index = int(sec * fps)
            if frame_index >= total_frames:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                print(f"{sec}s: unable to read frame")
                continue

            result = model.predict(
                frame,
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                verbose=False,
            )[0]

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
                    annotated,
                    f"{score:.2f}",
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            output = OUTPUT_DIR / f"second_{sec:02d}.jpg"
            if not cv2.imwrite(str(output), annotated):
                print(f"Failed to save {output}")
                continue

            print(
                f"{sec:02d}s | detections={len(result.boxes)} "
                f"| saved {output}"
            )
    finally:
        cap.release()


if __name__ == "__main__":
    main()
