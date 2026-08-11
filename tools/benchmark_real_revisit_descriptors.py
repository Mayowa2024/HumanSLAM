#!/usr/bin/env python3
"""Benchmark TensorRT descriptors on real KITTI revisits and hard negatives."""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/humanslam_matplotlib")

import cv2
import matplotlib.pyplot as plt
import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR.parent))

from slam.global_place_descriptor import DescriptorSpec, TensorRTGlobalDescriptor


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
        _, blur, darkness = condition.split("_")
        return transform(transform(image, f"blur_{blur}"), f"dark_{darkness}")
    raise ValueError(condition)


DISPLAY_NAMES = {
    "places365_resnet50_embedding": "Places365",
    "eigenplaces_r18_512": "EigenPlaces",
    "mixvpr_r50_512": "MixVPR",
    "salad": "SALAD",
}
COLORS = {
    "places365_resnet50_embedding": "#7f7f7f",
    "eigenplaces_r18_512": "#1f77b4",
    "mixvpr_r50_512": "#2ca02c",
    "salad": "#d62728",
}


def load_poses(path):
    values = np.asarray([
        [float(value) for value in line.split()]
        for line in path.read_text().splitlines() if line.strip()
    ], dtype=np.float64).reshape(-1, 3, 4)
    return values


def yaw_degrees(rotation):
    return float(np.degrees(np.arctan2(rotation[0, 2], rotation[2, 2])))


def angle_difference(first, second):
    return abs((first - second + 180.0) % 360.0 - 180.0)


def build_triplets(candidate_csv, poses, images, count, positive_radius,
                   negative_radius, min_separation):
    rows_by_query = defaultdict(list)
    with candidate_csv.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            rows_by_query[int(row["query_frame_id"])].append(row)
    positions = poses[:, :, 3]
    possible = []
    for query, rows in rows_by_query.items():
        eligible = np.arange(0, max(0, query - min_separation + 1))
        if eligible.size == 0:
            continue
        distances = np.linalg.norm(positions[eligible] - positions[query], axis=1)
        positive_indices = eligible[distances <= positive_radius]
        if positive_indices.size == 0:
            continue
        positive_distances = np.linalg.norm(
            positions[positive_indices] - positions[query], axis=1
        )
        positive = int(positive_indices[np.argmin(positive_distances)])
        negatives = []
        for row in rows:
            candidate = int(row["candidate_source_frame_id"])
            distance = float(np.linalg.norm(positions[candidate] - positions[query]))
            if distance >= negative_radius:
                negatives.append((float(row["unified_score"]), candidate, distance))
        if not negatives:
            continue
        old_score, negative, negative_distance = max(negatives)
        paths = [images / f"{frame:06d}.png"
                 for frame in (query, positive, negative)]
        if not all(path.is_file() for path in paths):
            continue
        query_yaw = yaw_degrees(poses[query, :, :3])
        positive_yaw = yaw_degrees(poses[positive, :, :3])
        possible.append({
            "query_frame": query,
            "positive_frame": positive,
            "negative_frame": negative,
            "positive_distance_m": float(np.linalg.norm(
                positions[positive] - positions[query]
            )),
            "negative_distance_m": negative_distance,
            "viewpoint_difference_deg": angle_difference(query_yaw, positive_yaw),
            "old_humanslam_negative_score": old_score,
            "query_path": str(paths[0]),
            "positive_path": str(paths[1]),
            "negative_path": str(paths[2]),
        })
    possible.sort(key=lambda row: row["old_humanslam_negative_score"], reverse=True)
    return possible[:count]


def timed(runtime, image):
    started = time.perf_counter()
    descriptor = runtime.describe(image)
    return descriptor, (time.perf_counter() - started) * 1000.0


def run_model(spec, triplets, conditions, warmups):
    runtime = TensorRTGlobalDescriptor(spec)
    warm_image = cv2.imread(triplets[0]["query_path"])
    for _ in range(warmups):
        runtime.describe(warm_image)
    rows = []
    for triplet in triplets:
        positive_image = cv2.imread(triplet["positive_path"])
        negative_image = cv2.imread(triplet["negative_path"])
        positive_descriptor, positive_ms = timed(runtime, positive_image)
        negative_descriptor, negative_ms = timed(runtime, negative_image)
        query_image = cv2.imread(triplet["query_path"])
        for condition in conditions:
            query_descriptor, query_ms = timed(
                runtime, transform(query_image, condition)
            )
            positive_similarity = float(query_descriptor @ positive_descriptor)
            negative_similarity = float(query_descriptor @ negative_descriptor)
            rows.append({
                "model": spec.name,
                "condition": condition,
                **{key: triplet[key] for key in (
                    "query_frame", "positive_frame", "negative_frame",
                    "positive_distance_m", "negative_distance_m",
                    "viewpoint_difference_deg", "old_humanslam_negative_score",
                )},
                "positive_similarity": positive_similarity,
                "hard_negative_similarity": negative_similarity,
                "separation_margin": positive_similarity - negative_similarity,
                "positive_ranked_first": int(positive_similarity > negative_similarity),
                "query_inference_ms": query_ms,
                "reference_inference_ms": (positive_ms + negative_ms) / 2.0,
                "descriptor_dimension": int(query_descriptor.size),
            })
    return rows


def summarise(rows):
    summaries = []
    keys = sorted({(row["model"], row["condition"]) for row in rows})
    for model, condition in keys:
        subset = [row for row in rows
                  if row["model"] == model and row["condition"] == condition]
        summaries.append({
            "model": model,
            "condition": condition,
            "triplets": len(subset),
            "pairwise_accuracy": float(np.mean([
                row["positive_ranked_first"] for row in subset
            ])),
            "mean_positive_similarity": float(np.mean([
                row["positive_similarity"] for row in subset
            ])),
            "mean_hard_negative_similarity": float(np.mean([
                row["hard_negative_similarity"] for row in subset
            ])),
            "mean_separation_margin": float(np.mean([
                row["separation_margin"] for row in subset
            ])),
            "warm_mean_ms": float(np.mean([
                row["query_inference_ms"] for row in subset
            ])),
            "warm_p95_ms": float(np.percentile([
                row["query_inference_ms"] for row in subset
            ], 95)),
            "descriptor_dimension": subset[0]["descriptor_dimension"],
        })
    return summaries


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_triplets(triplets, output, limit=8):
    selected = triplets[:limit]
    cell_width, image_height, label_height = 480, 270, 64
    canvas = np.full(
        (48 + len(selected) * (image_height + label_height), cell_width * 3, 3),
        245, dtype=np.uint8,
    )
    cv2.putText(canvas, "KITTI 06 real revisit triplets", (16, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    for row_index, row in enumerate(selected):
        y = 48 + row_index * (image_height + label_height)
        entries = [
            ("QUERY", row["query_frame"], row["query_path"]),
            ("POSITIVE", row["positive_frame"], row["positive_path"]),
            ("HARD NEGATIVE", row["negative_frame"], row["negative_path"]),
        ]
        for column, (label, frame, path) in enumerate(entries):
            image = cv2.imread(path)
            image = cv2.resize(image, (cell_width, image_height))
            x = column * cell_width
            canvas[y:y + image_height, x:x + cell_width] = image
            cv2.putText(canvas, f"{label} frame {frame}",
                        (x + 8, y + image_height + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1,
                        cv2.LINE_AA)
        detail = (
            f"positive={row['positive_distance_m']:.2f} m | "
            f"negative={row['negative_distance_m']:.1f} m | "
            f"old negative score={row['old_humanslam_negative_score']:.3f}"
        )
        cv2.putText(canvas, detail, (8, y + image_height + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (55, 55, 55), 1,
                    cv2.LINE_AA)
    cv2.imwrite(str(output), canvas)


def plot(summaries, output):
    models = list(DISPLAY_NAMES)
    models = [model for model in models
              if any(row["model"] == model for row in summaries)]
    conditions = ["original", "dark_50", "dark_80", "blur_15", "blur_35",
                  "combo_15_50", "combo_35_80"]
    labels = ["Original", "Dark 50%", "Dark 80%", "Blur 15", "Blur 35",
              "Blur 15 + dark 50", "Blur 35 + dark 80"]
    lookup = {(row["model"], row["condition"]): row for row in summaries}
    x_values = np.arange(len(conditions))
    width = 0.8 / len(models)
    for metric, ylabel, name, limits in (
        ("pairwise_accuracy", "Positive ranked above hard negative ↑",
         "real_revisit_pairwise_accuracy", (0, 1.08)),
        ("mean_separation_margin", "Positive − hard-negative similarity ↑",
         "real_revisit_separation", None),
    ):
        fig, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
        for index, model in enumerate(models):
            axis.bar(
                x_values + (index - (len(models) - 1) / 2) * width,
                [lookup[(model, condition)][metric] for condition in conditions],
                width, label=DISPLAY_NAMES[model], color=COLORS[model],
            )
        axis.set_xticks(x_values, labels, rotation=15, ha="right")
        axis.set_ylabel(ylabel)
        if limits:
            axis.set_ylim(*limits)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, ncols=4)
        fig.savefig(output / f"{name}.png", dpi=220)
        fig.savefig(output / f"{name}.pdf")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--poses", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--engine-spec", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--positive-radius", type=float, default=5.0)
    parser.add_argument("--negative-radius", type=float, default=100.0)
    parser.add_argument("--min-separation", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--conditions", nargs="+", default=[
        "original", "dark_50", "dark_80", "blur_15", "blur_35",
        "combo_15_50", "combo_35_80",
    ])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    triplets = build_triplets(
        args.candidates, load_poses(args.poses), args.images, args.count,
        args.positive_radius, args.negative_radius, args.min_separation,
    )
    if len(triplets) < 10:
        raise RuntimeError(f"Only {len(triplets)} valid real-revisit triplets")
    write_csv(args.output / "triplet_manifest.csv", triplets)
    render_triplets(triplets, args.output / "triplet_visual_review.png")
    rows = []
    for spec_path in args.engine_spec:
        spec = DescriptorSpec.from_json(spec_path)
        print(f"Benchmarking {spec.name}")
        rows.extend(run_model(spec, triplets, args.conditions, args.warmups))
    summaries = summarise(rows)
    write_csv(args.output / "pair_results.csv", rows)
    write_csv(args.output / "condition_summary.csv", summaries)
    plot(summaries, args.output)
    (args.output / "metadata.json").write_text(json.dumps({
        "experiment": "real revisit versus pose-verified hard negative",
        "images": str(args.images.resolve()),
        "poses": str(args.poses.resolve()),
        "candidate_source": str(args.candidates.resolve()),
        "positive_radius_m": args.positive_radius,
        "negative_radius_m": args.negative_radius,
        "min_frame_separation": args.min_separation,
        "conditions": args.conditions,
        "triplets": len(triplets),
    }, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
