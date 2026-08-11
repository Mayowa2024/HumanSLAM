#!/usr/bin/env python3
"""Controlled perceptual-variation benchmark for TensorRT VPR descriptors.

Each selected image is the ground-truth match for its own transformed query;
all other selected images act as hard/ordinary negatives.  This screening test
does not run ORB-SLAM3 and therefore isolates descriptor robustness and cost.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_SRC = PACKAGE_DIR.parent
if str(WORKSPACE_SRC) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_SRC))

from slam.global_place_descriptor import (  # noqa: E402
    DescriptorSpec,
    TensorRTGlobalDescriptor,
)


def natural_key(path):
    digits = "".join(value for value in path.stem if value.isdigit())
    return (int(digits), path.name) if digits else (10**18, path.name)


def transform(image, condition):
    if condition == "original":
        return image.copy()
    if condition.startswith("dark_"):
        percent = int(condition.rsplit("_", 1)[1])
        return np.clip(image.astype(np.float32) * (1.0 - percent / 100.0),
                       0, 255).astype(np.uint8)
    if condition.startswith("blur_"):
        length = int(condition.rsplit("_", 1)[1])
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0 / length
        return cv2.filter2D(image, -1, kernel)
    if condition.startswith("combo_"):
        _, blur, dark = condition.split("_")
        blurred = transform(image, f"blur_{blur}")
        return transform(blurred, f"dark_{dark}")
    if condition == "low_contrast":
        return cv2.convertScaleAbs(image, alpha=0.35, beta=75)
    if condition == "fog":
        haze = np.full_like(image, 210)
        return cv2.addWeighted(image, 0.45, haze, 0.55, 0)
    if condition == "occlusion_25":
        result = image.copy()
        height, width = result.shape[:2]
        result[height // 4:3 * height // 4, 3 * width // 8:5 * width // 8] = 0
        return result
    raise ValueError(f"Unknown condition: {condition}")


def timed_description(runtime, image, warmups=0):
    for _ in range(warmups):
        runtime.describe(image)
    started = time.perf_counter()
    descriptor = runtime.describe(image)
    return descriptor, (time.perf_counter() - started) * 1000.0


def percentile(values, value):
    return float(np.mean(np.asarray(values) <= value)) if values else 0.0


def run_model(spec, images, conditions, warmups):
    runtime = TensorRTGlobalDescriptor(spec)
    cold_started = time.perf_counter()
    gallery_first = runtime.describe(images[0][1])
    cold_ms = (time.perf_counter() - cold_started) * 1000.0
    for _ in range(warmups):
        runtime.describe(images[0][1])

    gallery = [gallery_first]
    gallery_ms = [cold_ms]
    for _, image in images[1:]:
        descriptor, latency = timed_description(runtime, image)
        gallery.append(descriptor)
        gallery_ms.append(latency)
    gallery = np.stack(gallery)

    rows = []
    for condition in conditions:
        for index, (path, image) in enumerate(images):
            query = transform(image, condition)
            descriptor, latency = timed_description(runtime, query)
            scores = gallery @ descriptor
            order = np.argsort(-scores)
            rank = int(np.flatnonzero(order == index)[0]) + 1
            negative_scores = np.delete(scores, index)
            hardest_negative = float(np.max(negative_scores))
            positive = float(scores[index])
            rows.append({
                "model": spec.name,
                "condition": condition,
                "image": str(path),
                "positive_similarity": positive,
                "hardest_negative_similarity": hardest_negative,
                "separation_margin": positive - hardest_negative,
                "positive_rank": rank,
                "recall_at_1": int(rank == 1),
                "recall_at_5": int(rank <= 5),
                "inference_ms": latency,
                "descriptor_dimension": int(descriptor.size),
            })
    return rows, cold_ms, gallery_ms


def summarise(rows, cold_by_model):
    result = []
    keys = sorted({(row["model"], row["condition"]) for row in rows})
    for model, condition in keys:
        subset = [row for row in rows
                  if row["model"] == model and row["condition"] == condition]
        latency = np.asarray([row["inference_ms"] for row in subset])
        result.append({
            "model": model,
            "condition": condition,
            "queries": len(subset),
            "recall_at_1": float(np.mean([row["recall_at_1"] for row in subset])),
            "recall_at_5": float(np.mean([row["recall_at_5"] for row in subset])),
            "mean_positive_similarity": float(np.mean([
                row["positive_similarity"] for row in subset
            ])),
            "mean_hardest_negative_similarity": float(np.mean([
                row["hardest_negative_similarity"] for row in subset
            ])),
            "mean_separation_margin": float(np.mean([
                row["separation_margin"] for row in subset
            ])),
            "warm_mean_ms": float(np.mean(latency)),
            "warm_p95_ms": float(np.percentile(latency, 95)),
            "cold_first_ms": cold_by_model[model],
            "descriptor_dimension": subset[0]["descriptor_dimension"],
        })
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--engine-spec", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--conditions", nargs="+", default=[
        "original", "dark_50", "dark_80", "blur_15", "blur_35",
        "combo_15_50", "combo_35_80", "low_contrast", "fog", "occlusion_25",
    ])
    args = parser.parse_args()

    paths = sorted(
        [path for path in args.images.iterdir()
         if path.suffix.lower() in {".png", ".jpg", ".jpeg"}],
        key=natural_key,
    )[::max(1, args.stride)][:args.count]
    if len(paths) < 6:
        raise RuntimeError("Select at least six images so negatives are meaningful")
    images = [(path, cv2.imread(str(path), cv2.IMREAD_COLOR)) for path in paths]
    unreadable = [str(path) for path, image in images if image is None]
    if unreadable:
        raise RuntimeError(f"Unreadable images: {unreadable}")

    args.output.mkdir(parents=True, exist_ok=True)
    rows, cold = [], {}
    for spec_path in args.engine_spec:
        spec = DescriptorSpec.from_json(spec_path)
        print(f"Benchmarking {spec.name}: {spec.engine_path}")
        model_rows, cold_ms, _ = run_model(
            spec, images, args.conditions, args.warmups
        )
        rows.extend(model_rows)
        cold[spec.name] = cold_ms
    summary = summarise(rows, cold)
    write_csv(args.output / "pair_results.csv", rows)
    write_csv(args.output / "condition_summary.csv", summary)
    metadata = {
        "image_directory": str(args.images.resolve()),
        "selected_images": [str(path.resolve()) for path in paths],
        "conditions": args.conditions,
        "engine_specs": [str(path.resolve()) for path in args.engine_spec],
        "selection_rule": "highest mean separation/recall subject to warm latency",
    }
    (args.output / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
