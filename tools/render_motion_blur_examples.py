#!/usr/bin/env python3
"""Export labelled horizontal motion-blur examples without rerunning inference."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from run_motion_blur_revisit_microtests import apply_blur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kernels", nargs="+", type=int,
                        default=[0, 5, 9, 15, 25, 35])
    args = parser.parse_args()
    images = sorted(path for path in args.images.iterdir()
                    if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    args.output.mkdir(parents=True, exist_ok=True)
    with args.pairs.open(newline="") as stream:
        pairs = list(csv.DictReader(stream))
    for row in pairs:
        query_id = int(row["query_frame"])
        candidate_id = int(row["candidate_frame"])
        query = cv2.imread(str(images[query_id]))
        candidate = cv2.imread(str(images[candidate_id]))
        pair_dir = args.output / f"q{query_id:06d}_c{candidate_id:06d}"
        pair_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(pair_dir / "candidate_daylight.png"), candidate)
        panels = []
        for kernel in args.kernels:
            result = query if kernel == 0 else apply_blur(query, kernel, 0.0)
            cv2.imwrite(str(pair_dir / f"query_blur_{kernel:02d}px.png"), result)
            panel = cv2.resize(result, (480, 146), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((184, 480, 3), dtype=np.uint8)
            canvas[:146] = panel
            label = "Original later query" if kernel == 0 else f"{kernel}-pixel horizontal motion blur"
            cv2.putText(canvas, label, (10, 172), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(canvas)
        sheet = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:6])))
        cv2.imwrite(str(pair_dir / "all_blur_levels.png"), sheet)
    print(f"Rendered {len(pairs)} comparison sheets to {args.output}")


if __name__ == "__main__":
    main()
