#!/usr/bin/env python3
"""Measure local-feature survivability for known-correct KITTI revisit pairs."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark_global_descriptors import transform


DEFAULT_PAIRS = ((858, 28), (888, 60), (950, 122), (955, 127),
                 (1026, 210), (1058, 242), (1078, 267), (1084, 267))
CONDITIONS = ("original", "combo_15_50", "combo_25_65", "combo_35_80", "fog")


def coverage(keypoints, matches, mask, shape, grid=4):
    occupied = set()
    height, width = shape[:2]
    for match, keep in zip(matches, mask):
        if not keep:
            continue
        x, y = keypoints[match.queryIdx].pt
        occupied.add((min(grid - 1, int(x * grid / width)),
                      min(grid - 1, int(y * grid / height))))
    return len(occupied) / float(grid * grid)


def analyse(orb, candidate, query, ratio, ransac_threshold, minimum_inliers):
    kp_q, des_q = orb.detectAndCompute(query, None)
    kp_c, des_c = orb.detectAndCompute(candidate, None)
    result = {
        "query_keypoints": len(kp_q), "candidate_keypoints": len(kp_c),
        "knn_pairs": 0, "ratio_matches": 0, "ransac_inliers": 0,
        "inlier_ratio": 0.0, "query_spatial_coverage": 0.0,
        "proxy_pass": False, "failure_stage": "descriptor_unavailable",
    }
    if des_q is None or des_c is None:
        return result, kp_q, kp_c, [], []
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_q, des_c, k=2)
    result["knn_pairs"] = len(knn)
    good = [first for pair in knn if len(pair) == 2
            for first, second in [pair] if first.distance < ratio * second.distance]
    result["ratio_matches"] = len(good)
    result["failure_stage"] = "insufficient_ratio_matches"
    if len(good) < 8:
        return result, kp_q, kp_c, good, [0] * len(good)
    points_q = np.float32([kp_q[m.queryIdx].pt for m in good])
    points_c = np.float32([kp_c[m.trainIdx].pt for m in good])
    _, mask = cv2.findFundamentalMat(
        points_q, points_c, cv2.FM_RANSAC, ransac_threshold, 0.99
    )
    if mask is None:
        return result, kp_q, kp_c, good, [0] * len(good)
    inlier_mask = mask.ravel().astype(bool).tolist()
    inliers = int(sum(inlier_mask))
    result["ransac_inliers"] = inliers
    result["inlier_ratio"] = inliers / len(good)
    result["query_spatial_coverage"] = coverage(
        kp_q, good, inlier_mask, query.shape
    )
    result["proxy_pass"] = inliers >= minimum_inliers
    result["failure_stage"] = (
        "proxy_pass" if result["proxy_pass"] else "insufficient_ransac_inliers"
    )
    return result, kp_q, kp_c, good, inlier_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--poses", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ratio", type=float, default=0.80)
    parser.add_argument("--ransac-threshold", type=float, default=1.5)
    parser.add_argument("--minimum-inliers", type=int, default=30)
    parser.add_argument("--features", type=int, default=2000)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    visual_dir = args.output / "match_visualisations"
    visual_dir.mkdir(exist_ok=True)
    image_paths = sorted(p for p in args.images.iterdir()
                         if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    poses = np.loadtxt(args.poses).reshape(-1, 3, 4)
    translations = poses[:, :3, 3]
    orb = cv2.ORB_create(nfeatures=args.features, fastThreshold=20)
    rows = []
    for query_id, candidate_id in DEFAULT_PAIRS:
        clean_query = cv2.imread(str(image_paths[query_id]), cv2.IMREAD_GRAYSCALE)
        candidate = cv2.imread(str(image_paths[candidate_id]), cv2.IMREAD_GRAYSCALE)
        for condition in args.conditions:
            query = transform(clean_query, condition)
            metrics, kp_q, kp_c, matches, mask = analyse(
                orb, candidate, query, args.ratio,
                args.ransac_threshold, args.minimum_inliers,
            )
            row = {
                "query_frame": query_id, "candidate_frame": candidate_id,
                "ground_truth_distance_m": float(np.linalg.norm(
                    translations[query_id] - translations[candidate_id]
                )),
                "condition": condition, **metrics,
            }
            rows.append(row)
            inlier_matches = [m for m, keep in zip(matches, mask) if keep][:100]
            visual = cv2.drawMatches(
                query, kp_q, candidate, kp_c, inlier_matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            cv2.putText(
                visual,
                f"q{query_id} -> c{candidate_id} | {condition} | "
                f"kp={len(kp_q)} ratio={len(matches)} inliers={metrics['ransac_inliers']}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(visual_dir / f"q{query_id}_c{candidate_id}_{condition}.png"), visual)

    with (args.output / "pair_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for condition in args.conditions:
        subset = [row for row in rows if row["condition"] == condition]
        summary.append({
            "condition": condition, "pairs": len(subset),
            "pass_rate": float(np.mean([row["proxy_pass"] for row in subset])),
            "mean_query_keypoints": float(np.mean([row["query_keypoints"] for row in subset])),
            "mean_ratio_matches": float(np.mean([row["ratio_matches"] for row in subset])),
            "mean_ransac_inliers": float(np.mean([row["ransac_inliers"] for row in subset])),
            "median_ransac_inliers": float(np.median([row["ransac_inliers"] for row in subset])),
            "mean_spatial_coverage": float(np.mean([row["query_spatial_coverage"] for row in subset])),
        })
    with (args.output / "condition_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    (args.output / "metadata.json").write_text(json.dumps({
        "images": str(args.images.resolve()), "poses": str(args.poses.resolve()),
        "pairs": DEFAULT_PAIRS, "conditions": args.conditions,
        "ratio": args.ratio, "ransac_threshold": args.ransac_threshold,
        "minimum_inliers": args.minimum_inliers, "features": args.features,
        "scope": "OpenCV ORB/F-matrix proxy; not ORB-SLAM3 Sim3 verification",
    }, indent=2) + "\n")

    labels = [item["condition"].replace("combo_", "blur/dark ") for item in summary]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, [item["mean_ransac_inliers"] for item in summary], color="#4472c4")
    ax.axhline(args.minimum_inliers, color="#c00000", linestyle="--", label=f"{args.minimum_inliers}-inlier proxy gate")
    ax.set(ylabel="Mean F-matrix RANSAC inliers", title="Known-correct revisit geometry survivability")
    ax.tick_params(axis="x", rotation=18); ax.grid(axis="y", alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(args.output / "geometry_survivability.png", dpi=180); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
