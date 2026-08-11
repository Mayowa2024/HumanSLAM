#!/usr/bin/env python3
"""Create a KITTI-like stereo scenario using benchmark appearance transforms."""

import argparse
import os
import shutil
from pathlib import Path

import cv2

from benchmark_global_descriptors import transform


def link_or_copy(source: Path, destination: Path):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--condition", required=True,
        choices=("combo_15_50", "combo_15_55", "combo_35_80", "fog", "low_contrast", "occlusion_25"),
    )
    parser.add_argument("--start-frame", type=int, default=828)
    parser.add_argument("--end-frame", type=int, default=-1)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    counts = []
    for directory in ("image_0", "image_1"):
        source_dir = args.source / directory
        output_dir = args.output / directory
        output_dir.mkdir()
        images = sorted(path for path in source_dir.iterdir()
                        if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
        end = len(images) - 1 if args.end_frame < 0 else min(
            args.end_frame, len(images) - 1
        )
        for frame_id, source in enumerate(images):
            destination = output_dir / source.name
            if args.start_frame <= frame_id <= end:
                image = cv2.imread(str(source), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Cannot read {source}")
                if not cv2.imwrite(str(destination), transform(image, args.condition)):
                    raise RuntimeError(f"Cannot write {destination}")
            else:
                link_or_copy(source, destination)
        counts.append(len(images))
    if counts[0] != counts[1]:
        raise SystemExit(f"Stereo counts differ: {counts}")
    shutil.copy2(args.source / "times.txt", args.output / "times.txt")
    (args.output / "SCENARIO.txt").write_text(
        f"source={args.source.resolve()}\ncondition={args.condition}\n"
        f"start_frame={args.start_frame}\nend_frame={args.end_frame}\n"
        "transform_source=tools/benchmark_global_descriptors.py\n",
        encoding="utf-8",
    )
    print(f"Created {args.condition}: {counts[0]} stereo pairs at {args.output}")


if __name__ == "__main__":
    main()
