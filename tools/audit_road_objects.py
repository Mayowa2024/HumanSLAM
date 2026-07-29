import argparse
import json
from collections import Counter
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-dir", default="image_0")
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    images = sorted((args.dataset / args.image_dir).glob("*"))
    images = images[::max(1, args.stride)]
    model = YOLO(args.model)
    counts = Counter()
    frames = Counter()
    for index, image_path in enumerate(images, start=1):
        seen = set()
        for result in model(str(image_path), conf=args.confidence, verbose=False):
            if result.boxes is None:
                continue
            for class_tensor in result.boxes.cls:
                class_name = model.names[int(class_tensor)]
                counts[class_name] += 1
                seen.add(class_name)
        frames.update(seen)
        if index == 1 or index % 100 == 0:
            print(f"Object audit progress: {index}/{len(images)}")

    report = {
        "sampled_frames": len(images),
        "detection_counts": dict(counts.most_common()),
        "frames_containing_class": dict(frames.most_common()),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
