
from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO_PATH = "input.mp4"
MODEL_PATH = "models/candidates/plate_koushim.pt"
OUTPUT_DIR = Path("runs/plate_diagnostics")

SECONDS = [2, 5, 7, 10, 12, 14]
THRESHOLDS = [0.05, 0.10, 0.25]
IMAGE_SIZES = [640, 960]

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total <= 0:
        cap.release()
        raise RuntimeError("Invalid video metadata")

    model = YOLO(MODEL_PATH)

    try:
        for sec in SECONDS:
            if sec * fps >= total:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
            ok, frame = cap.read()
            if not ok:
                continue

            for imgsz in IMAGE_SIZES:
                for conf in THRESHOLDS:
                    result = model.predict(
                        frame,
                        imgsz=imgsz,
                        conf=conf,
                        verbose=False,
                    )[0]

                    count = len(result.boxes)
                    print(
                        f"{sec:02d}s | imgsz={imgsz} | "
                        f"conf={conf:.2f} | detections={count}"
                    )

                    if count:
                        annotated = result.plot()
                        name = f"sec{sec:02d}_size{imgsz}_conf{conf:.2f}.jpg"
                        cv2.imwrite(str(OUTPUT_DIR / name), annotated)
    finally:
        cap.release()

if __name__ == "__main__":
    main()
