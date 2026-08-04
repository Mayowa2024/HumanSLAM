#!/usr/bin/env python3
"""Create a KITTI-like sequence with a forced loss before a dark revisit."""

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Copy a stereo KITTI-like sequence, black out a short interval to "
            "force tracking loss, and darken the subsequent loop revisit."
        )
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dropout-start", type=int, default=800)
    parser.add_argument("--dropout-end", type=int, default=814)
    parser.add_argument("--night-start", type=int, default=815)
    parser.add_argument("--night-end", type=int, default=900)
    parser.add_argument("--brightness", type=float, default=-60.0)
    parser.add_argument("--contrast", type=float, default=0.75)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument(
        "--revisit-source-start",
        type=int,
        default=None,
        help=(
            "Replay this earlier mapped frame at night-start, advancing one "
            "source frame per output frame. Omit to perturb the original revisit."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate(args):
    if args.dropout_start < 0:
        raise ValueError("dropout-start must be non-negative")
    if args.dropout_end < args.dropout_start:
        raise ValueError("dropout-end must be >= dropout-start")
    if args.night_start <= args.dropout_end:
        raise ValueError("night-start must be after dropout-end")
    if args.night_end < args.night_start:
        raise ValueError("night-end must be >= night-start")
    if args.gamma <= 0:
        raise ValueError("gamma must be positive")
    if args.revisit_source_start is not None and args.revisit_source_start < 0:
        raise ValueError("revisit-source-start must be non-negative")


def image_paths(directory):
    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in allowed)


def night_transform(image, brightness, contrast, gamma):
    adjusted = image.astype(np.float32) * contrast + brightness
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    lut = np.array(
        [np.clip((index / 255.0) ** gamma * 255.0, 0, 255) for index in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(adjusted, lut)


def link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build(args):
    validate(args)
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        shutil.rmtree(output)

    left = image_paths(source / "image_0")
    right = image_paths(source / "image_1")
    times = (source / "times.txt").read_text(encoding="utf-8").splitlines()
    if not left or len(left) != len(right) or len(left) != len(times):
        raise ValueError(
            f"Invalid stereo sequence counts: left={len(left)}, "
            f"right={len(right)}, times={len(times)}"
        )
    if args.night_end >= len(left):
        raise ValueError(f"night-end {args.night_end} exceeds last frame {len(left)-1}")

    for directory in ("image_0", "image_1"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    modified = 0
    for stereo_dir, paths in (("image_0", left), ("image_1", right)):
        for index, source_image in enumerate(paths):
            destination = output / stereo_dir / source_image.name
            if args.dropout_start <= index <= args.dropout_end:
                image = cv2.imread(str(source_image), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise RuntimeError(f"Could not read {source_image}")
                if not cv2.imwrite(str(destination), np.zeros_like(image)):
                    raise RuntimeError(f"Could not write {destination}")
                modified += 1
            elif args.night_start <= index <= args.night_end:
                replay_index = (
                    args.revisit_source_start + index - args.night_start
                    if args.revisit_source_start is not None
                    else index
                )
                if replay_index >= len(paths):
                    raise ValueError(
                        f"Revisit source frame {replay_index} exceeds sequence"
                    )
                image = cv2.imread(
                    str(paths[replay_index]), cv2.IMREAD_UNCHANGED
                )
                if image is None:
                    raise RuntimeError(f"Could not read {source_image}")
                image = night_transform(
                    image, args.brightness, args.contrast, args.gamma
                )
                if not cv2.imwrite(str(destination), image):
                    raise RuntimeError(f"Could not write {destination}")
                modified += 1
            else:
                link_or_copy(source_image, destination)

    shutil.copy2(source / "times.txt", output / "times.txt")
    manifest = {
        "purpose": "forced tracking loss followed by appearance-changed revisit",
        "source": str(source),
        "frame_count": len(left),
        "dropout": [args.dropout_start, args.dropout_end],
        "night_revisit": [args.night_start, args.night_end],
        "night_parameters": {
            "brightness": args.brightness,
            "contrast": args.contrast,
            "gamma": args.gamma,
        },
        "revisit_source_start": args.revisit_source_start,
        "modified_stereo_images": modified,
    }
    (output / "scenario.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    build(parse_args())
