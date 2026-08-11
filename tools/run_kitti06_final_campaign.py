#!/usr/bin/env python3
"""Queue the final KITTI 06 perturbation batches after the clean batch."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


CONDITIONS = (
    ("K06-1_blur15", 15, 1.00),
    ("K06-2_dark50", 1, 0.50),
    ("K06-3_blur15_dark50", 15, 0.50),
    ("K06-4_blur15_dark80", 15, 0.20),
    ("K06-5_blur35_dark80", 35, 0.20),
    ("K06-6_blur35_dark90", 35, 0.10),
)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--scenario-root", default=Path("test_scenarios/final_kitti06"), type=Path)
    parser.add_argument("--clean-status", required=True, type=Path)
    parser.add_argument("--runs", default=10, type=int)
    parser.add_argument("--start-frame", default=828, type=int)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    campaign_status = args.output_root / "campaign_status.json"
    state: dict[str, object] = {
        "state": "waiting_for_clean",
        "created": datetime.now().astimezone().isoformat(),
        "completed_conditions": [],
        "failed_conditions": [],
    }
    write_json(campaign_status, state)

    while True:
        try:
            clean = json.loads(args.clean_status.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            clean = {}
        if clean.get("state") in {"complete", "complete_with_failures"}:
            if clean.get("state") != "complete":
                raise SystemExit("Clean batch completed with failures; refusing automatic continuation")
            break
        time.sleep(30)

    maker = Path(__file__).with_name("make_motion_blur_scenario.py")
    runner = Path(__file__).with_name("run_repeated_offline_benchmark.py")
    state["state"] = "running_perturbations"
    write_json(campaign_status, state)

    for condition, kernel, brightness in CONDITIONS:
        scenario = args.scenario_root / condition
        marker = scenario / "SCENARIO.txt"
        if not marker.is_file():
            command = [
                sys.executable, str(maker), "--source", str(args.source),
                "--output", str(scenario), "--start-frame", str(args.start_frame),
                "--kernel", str(kernel), "--brightness-retained", str(brightness),
            ]
            subprocess.run(command, check=True)

        left_count = len(list((scenario / "image_0").glob("*.png")))
        right_count = len(list((scenario / "image_1").glob("*.png")))
        time_count = len((scenario / "times.txt").read_text(encoding="utf-8").splitlines())
        if not left_count or len({left_count, right_count, time_count}) != 1:
            raise RuntimeError(
                f"Invalid scenario {condition}: left={left_count}, right={right_count}, times={time_count}"
            )

        output = args.output_root / condition / "formal_10x"
        result = subprocess.run([
            sys.executable, str(runner), "--dataset", str(scenario),
            "--settings", str(args.settings), "--ground-truth", str(args.ground_truth),
            "--output", str(output), "--runs", str(args.runs),
            "--perturb-start", str(args.start_frame),
        ], check=False)
        key = "completed_conditions" if result.returncode == 0 else "failed_conditions"
        state[key].append({"condition": condition, "return_code": result.returncode})
        write_json(campaign_status, state)
        if result.returncode != 0:
            break

    state["state"] = "complete" if not state["failed_conditions"] else "stopped_on_failure"
    state["finished"] = datetime.now().astimezone().isoformat()
    write_json(campaign_status, state)


if __name__ == "__main__":
    main()
