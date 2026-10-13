
from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO_PATH = "input.mp4"
OUTPUT_DIR = Path("runs/candidate_models")
SAMPLE_SECONDS = list(range(0, 10))
IMGSZ = 640

MODELS = {
    "helmet": (
        "models/candidates/helmet_iam_tsr.pt",
        0.25,
    ),
    "plate": (
        "models/candidates/plate_koushim.pt",
        0.25,
    ),
}


def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError("Invalid video FPS or frame count")

    duration = total_frames / fps
    sample_seconds = [s for s in SAMPLE_SECONDS if s < duration]

    models = {
        name: (YOLO(path), threshold)
        for name, (path, threshold) in MODELS.items()
    }

    print(f"Video duration: {duration:.1f}s")
    print(f"Sampling seconds: {sample_seconds}")

    try:
        for second in sample_seconds:
            frame_index = min(int(second * fps), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()

            if not ok:
                print(f"Could not read frame at {second}s")
                continue

            for name, (model, threshold) in models.items():
                result = model.predict(
                    frame,
                    conf=threshold,
                    imgsz=IMGSZ,
                    verbose=False,
                )[0]

                annotated = result.plot()
                output_path = output_dir / f"{name}_{second:02d}.jpg"

                if not cv2.imwrite(str(output_path), annotated):
                    print(f"Failed to save {output_path}")
                    continue

                counts = {}
                for cls_id in result.boxes.cls.cpu().tolist():
                    label = model.names[int(cls_id)]
                    counts[label] = counts.get(label, 0) + 1

                print(
                    f"{second:02d}s | {name:6s} | "
                    f"{counts} | saved {output_path}"
                )
    finally:
        cap.release()


if __name__ == "__main__":
    main()
