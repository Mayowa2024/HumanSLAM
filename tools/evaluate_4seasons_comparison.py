#!/usr/bin/env python3
"""Evaluate paired 4Seasons ORB-SLAM3 baseline and HumanSLAM runs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    return parser.parse_args()


def camera_times(path):
    return np.array([
        float(line.split()[1]) for line in path.read_text().splitlines() if line.strip()
    ])


def reference_positions(path, target_times):
    data = np.loadtxt(path)
    times, xyz = data[:, 0], data[:, 1:4]
    return np.column_stack([
        np.interp(target_times, times, xyz[:, axis]) for axis in range(3)
    ])


def estimated_positions(path):
    matrices = np.loadtxt(path).reshape(-1, 3, 4)
    return matrices[:, :, 3]


def tracked_frame_ids(path):
    ids = []
    latencies = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("event") != "track":
                continue
            ids.append(int(row["frame_id"]))
            value = row.get("track_stereo_ms", "")
            if value:
                latencies.append(float(value))
    return np.asarray(ids, dtype=int), np.asarray(latencies)


def rigid_align(source, target):
    source_mean, target_mean = source.mean(0), target.mean(0)
    covariance = (source - source_mean).T @ (target - target_mean)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_mean - rotation @ source_mean
    return (rotation @ source.T).T + translation


def load_run(directory, times, reference_file):
    estimate = estimated_positions(directory / "trajectory_kitti.txt")
    frame_ids, latency = tracked_frame_ids(directory / "orb_events_latency.csv")
    # ORB's KITTI export omits initialization frames. Its final poses align
    # with the final successful TrackStereo calls in chronological order.
    frame_ids = frame_ids[-len(estimate):]
    valid = (frame_ids >= 0) & (frame_ids < len(times))
    frame_ids, estimate = frame_ids[valid], estimate[valid]
    target_times = times[frame_ids]
    reference = reference_positions(reference_file, target_times)
    aligned = rigid_align(estimate, reference)
    errors = np.linalg.norm(aligned - reference, axis=1)
    rpe = np.linalg.norm(
        np.diff(aligned, axis=0) - np.diff(reference, axis=0), axis=1
    )
    return {
        "frame_ids": frame_ids, "estimate": aligned, "reference": reference,
        "errors": errors, "rpe": rpe, "latency": latency,
    }


def metrics(data):
    error, rpe, latency = data["errors"], data["rpe"], data["latency"]
    return {
        "matched_poses": int(len(error)),
        "ape_rmse_m": float(np.sqrt(np.mean(error ** 2))),
        "ape_mean_m": float(np.mean(error)),
        "ape_median_m": float(np.median(error)),
        "ape_max_m": float(np.max(error)),
        "rpe_translation_rmse_m": float(np.sqrt(np.mean(rpe ** 2))),
        "tracking_latency_mean_ms": float(np.mean(latency)),
        "tracking_latency_p95_ms": float(np.percentile(latency, 95)),
    }


def main():
    args = arguments()
    plots = args.results / "plots"
    metrics_dir = args.results / "metrics"
    plots.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    times = camera_times(args.sequence / "times.txt")
    reference_file = args.sequence / "result.txt"
    runs = {
        "ORB-SLAM3": load_run(args.results / "baseline", times, reference_file),
        "ORB-SLAM3 + HumanSLAM": load_run(
            args.results / "humanslam", times, reference_file
        ),
    }
    summary = {name: metrics(data) for name, data in runs.items()}
    (metrics_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    reference = next(iter(runs.values()))["reference"]
    axes[0].plot(reference[:, 0], reference[:, 1], "k", lw=1.5, label="Reference")
    colours = ("tab:blue", "tab:orange")
    for (name, data), colour in zip(runs.items(), colours):
        axes[0].plot(data["estimate"][:, 0], data["estimate"][:, 1], colour, lw=1, label=name)
        axes[1].plot(data["frame_ids"], data["errors"], colour, lw=.9, label=name)
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Aligned trajectory")
    axes[0].axis("equal")
    axes[1].set(xlabel="Dataset frame", ylabel="translation error (m)", title="APE versus frame")
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(plots / "trajectory_and_ape.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    labels = list(runs)
    means = [summary[name]["tracking_latency_mean_ms"] for name in labels]
    p95 = [summary[name]["tracking_latency_p95_ms"] for name in labels]
    x = np.arange(len(labels))
    axis.bar(x - .18, means, .36, label="Mean")
    axis.bar(x + .18, p95, .36, label="95th percentile")
    axis.set_xticks(x, labels)
    axis.set(ylabel="TrackStereo latency (ms)", title="Tracking latency")
    axis.grid(axis="y", alpha=.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots / "tracking_latency.png", dpi=180)
    plt.close(fig)
    print(args.results)


if __name__ == "__main__":
    main()
