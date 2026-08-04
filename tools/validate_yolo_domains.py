#!/usr/bin/env python3
"""Run a frozen YOLO segmentation checkpoint on named external domains."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


DEFAULT_DATASETS = {
    "MAPILLARY_VISTAS_VALIDATION": Path(
        "/home/teleopbike/Documents/Mayowa/mapillary vistas dataset/validation/images"
    ),
    "KITTI_ODOMETRY_00_NORMAL_LEFT": Path(
        "/home/teleopbike/Documents/slam_experiments/datasets/KITTI/"
        "dataset/sequences/00/image_0"
    ),
    "KITTI_ODOMETRY_06_NORMAL_LEFT": Path(
        "/home/teleopbike/Documents/slam_experiments/datasets/KITTI/"
        "dataset/sequences/06/image_0"
    ),
    "KITTI_ODOMETRY_06_DARK_LOOPREGION_LEFT": Path(
        "/home/teleopbike/Documents/slam_experiments/datasets/KITTI_variants/"
        "dataset/sequences/06_dark_loopregion/image_0"
    ),
    "CARLA_05_WEATHER_CYCLE_LEFT": Path(
        "/home/teleopbike/Documents/Mayowa/05_weather_cycle/image_0"
    ),
    "CARLA_05_WEATHER_CYCLE_DYNAMIC_LEFT": Path(
        "/home/teleopbike/Documents/Mayowa/05_weather_cycle_dynamic/image_0"
    ),
    "CARLA_EPISODE_0001_RGB": Path(
        "/home/teleopbike/Documents/Mayowa/episode_0001/images"
    ),
    "OXFORD_RADAR_ROBOTCAR_2019_01_10_PARTIAL_CENTRE_RGB": Path(
        "/home/teleopbike/Downloads/"
        "2019-01-10-14-36-48-radar-oxford-10k-partial_centre_RGB/"
        "2019-01-10-14-36-48-radar-oxford-10k-partial_centre_RGB"
    ),
}


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=Path,
        default=root
        / "runs/mapillary_humanslam/yolo26n_seg_full_seed42_b8/weights/best.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "validation_results/yolo26n_seg_mapillary_domain_test",
    )
    parser.add_argument("--samples-per-dataset", type=int, default=30)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def image_files(directory):
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def evenly_spaced(paths, count):
    if count <= 0 or len(paths) <= count:
        return paths
    if count == 1:
        return [paths[len(paths) // 2]]
    indices = [round(index * (len(paths) - 1) / (count - 1)) for index in range(count)]
    return [paths[index] for index in indices]


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    weights = args.weights.resolve()
    output_root = args.output.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    if args.samples_per_dataset < 1:
        raise ValueError("samples-per-dataset must be at least 1")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but PyTorch cannot access a CUDA device")

    output_root.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights), task="segment")
    manifest_rows = []
    summaries = {}

    for dataset_name, source_directory in DEFAULT_DATASETS.items():
        source_directory = source_directory.resolve()
        if not source_directory.is_dir():
            summaries[dataset_name] = {
                "source_image_directory": str(source_directory),
                "status": "missing",
            }
            continue

        available = image_files(source_directory)
        selected = evenly_spaced(available, args.samples_per_dataset)
        dataset_output = output_root / dataset_name
        dataset_output.mkdir(parents=True, exist_ok=True)
        class_counts = Counter()
        confidences = []

        results = model.predict(
            source=[str(path) for path in selected],
            conf=args.confidence,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
            stream=True,
        )
        for sample_index, (source_path, result) in enumerate(zip(selected, results)):
            output_path = dataset_output / f"{sample_index:03d}_{source_path.stem}_prediction.jpg"
            plotted = result.plot()
            if not cv2.imwrite(str(output_path), plotted):
                raise RuntimeError(f"Failed to write prediction overlay: {output_path}")

            detections = []
            if result.boxes is not None:
                class_ids = result.boxes.cls.detach().cpu().tolist()
                scores = result.boxes.conf.detach().cpu().tolist()
                for class_id, score in zip(class_ids, scores):
                    class_name = result.names[int(class_id)]
                    class_counts[class_name] += 1
                    confidences.append(float(score))
                    detections.append(f"{class_name}:{float(score):.4f}")

            manifest_rows.append(
                {
                    "dataset_name": dataset_name,
                    "source_image_directory": str(source_directory),
                    "source_image": str(source_path),
                    "sample_index": sample_index,
                    "output_overlay": str(output_path),
                    "detection_count": len(detections),
                    "detections": ";".join(detections),
                }
            )

        summaries[dataset_name] = {
            "source_image_directory": str(source_directory),
            "status": "completed",
            "available_images": len(available),
            "sampled_images": len(selected),
            "images_with_detections": sum(
                row["detection_count"] > 0
                for row in manifest_rows
                if row["dataset_name"] == dataset_name
            ),
            "total_detections": sum(class_counts.values()),
            "mean_detection_confidence": (
                sum(confidences) / len(confidences) if confidences else None
            ),
            "detections_by_class": dict(sorted(class_counts.items())),
            "overlay_directory": str(dataset_output),
        }

    manifest_path = output_root / "prediction_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "dataset_name",
            "source_image_directory",
            "source_image",
            "sample_index",
            "output_overlay",
            "detection_count",
            "detections",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(weights),
        "checkpoint_sha256": checkpoint_sha256(weights),
        "confidence_threshold": args.confidence,
        "image_size": args.imgsz,
        "device": args.device,
        "sampling": "deterministic evenly spaced across lexically sorted filenames",
        "datasets": summaries,
        "prediction_manifest": str(manifest_path),
        "interpretation_limit": (
            "External datasets have no converted YOLO ground-truth labels in this run; "
            "counts and overlays support domain-transfer inspection, not accuracy claims."
        ),
    }
    report_path = output_root / "validation_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
