#!/usr/bin/env python3
"""Create a KITTI-like stereo sequence with blur/exposure perturbations."""

import argparse
import os
import shutil
from pathlib import Path

import cv2

from run_motion_blur_revisit_microtests import apply_blur, apply_brightness


def link_or_copy(source: Path, destination: Path):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=828)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--kernel", type=int, default=25)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--brightness-retained", type=float, default=1.0)
    args = parser.parse_args()
    if not 0 < args.brightness_retained <= 1:
        raise SystemExit("--brightness-retained must be in (0, 1]")
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
        end = len(images) - 1 if args.end_frame < 0 else min(args.end_frame, len(images) - 1)
        for frame_id, source in enumerate(images):
            destination = output_dir / source.name
            if args.start_frame <= frame_id <= end:
                image = cv2.imread(str(source), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Cannot read {source}")
                perturbed = apply_blur(image, args.kernel, args.angle)
                perturbed = apply_brightness(perturbed, args.brightness_retained)
                if not cv2.imwrite(str(destination), perturbed):
                    raise RuntimeError(f"Cannot write {destination}")
            else:
                link_or_copy(source, destination)
        counts.append(len(images))
    if counts[0] != counts[1]:
        raise SystemExit(f"Stereo counts differ: {counts}")
    shutil.copy2(args.source / "times.txt", args.output / "times.txt")
    (args.output / "SCENARIO.txt").write_text(
        f"source={args.source.resolve()}\nstart_frame={args.start_frame}\n"
        f"end_frame={args.end_frame}\nkernel_px={args.kernel}\n"
        f"angle_deg={args.angle}\n"
        f"brightness_retained={args.brightness_retained}\n"
        f"darkening_percent={(1.0 - args.brightness_retained) * 100.0}\n"
    )
    print(f"Created {counts[0]} stereo pairs at {args.output}")


if __name__ == "__main__":
    main()
