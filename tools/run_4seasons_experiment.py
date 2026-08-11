#!/usr/bin/env python3
"""Run reproducible 4Seasons baseline/HumanSLAM trials with saved outputs."""

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ROS_PACKAGE = PROJECT.parent
DEFAULT_DATA_ROOT = Path(
    "/home/teleopbike/Documents/slam_experiments/datasets/4seasons"
)
DEFAULT_RESULTS_ROOT = PROJECT / "test_results"
DEFAULT_VOCABULARY = Path("/home/teleopbike/ORB_SLAM3/Vocabulary/ORBvoc.txt")
DEFAULT_SETTINGS = ROS_PACKAGE / "config/4seasons_stereo_inertial.yaml"
DEFAULT_HUMAN_CONFIG = ROS_PACKAGE / "config/human_slam_params.yaml"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence", choices=("neighborhood_1_train", "neighborhood_3_train"),
        default="neighborhood_1_train",
    )
    parser.add_argument("--mode", choices=("baseline", "humanslam", "both"), default="both")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--playback-rate", type=float, default=1.0)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--human-config", type=Path, default=DEFAULT_HUMAN_CONFIG)
    return parser.parse_args()


def sequence_directory(root, name):
    recording = {
        "neighborhood_1_train": "recording_2020-03-26_13-32-55",
        "neighborhood_3_train": "recording_2020-10-07_14-53-52",
    }[name]
    return root / name / recording


def ros_parameter(name, value):
    if isinstance(value, bool):
        value = "true" if value else "false"
    return ["-p", f"{name}:={value}"]


def start_process(command, log_path, environment, cwd=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("w", encoding="utf-8", buffering=1)
    process = subprocess.Popen(
        command, stdout=stream, stderr=subprocess.STDOUT, env=environment,
        cwd=cwd, start_new_session=True,
    )
    process._humanslam_log_stream = stream
    return process


def wait_for_log(process, path, marker, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Process exited before '{marker}'; inspect {path}"
            )
        if path.exists() and marker in path.read_text(
                encoding="utf-8", errors="replace"):
            return
        time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for '{marker}' in {path}")


def stop_process(process, timeout=20):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)


def close_log(process):
    stream = getattr(process, "_humanslam_log_stream", None)
    if stream:
        stream.close()


def run_trial(args, sequence, root, assisted):
    label = "humanslam" if assisted else "baseline"
    output = root / label
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    source_parent = str(PROJECT.parent)
    environment["PYTHONPATH"] = source_parent + os.pathsep + environment.get("PYTHONPATH", "")

    common_ros = ["--ros-args"]
    orb_command = ["ros2", "run", "orbslam3_zed_stereo", "zed_stereo_node"] + common_ros
    for key, value in {
        "voc_file": args.vocabulary,
        "settings_file": args.settings,
        "left_topic": "/dataset/left/image_raw",
        "right_topic": "/dataset/right/image_raw",
        "imu_topic": "/dataset/imu",
        "use_imu": True,
        "semantic_assistance_enabled": assisted,
        "trajectory_output": output / "trajectory_kitti.txt",
        "latency_output": output / "orb_events_latency.csv",
        "tracked_features_output": output / "orb_tracked_features.csv",
    }.items():
        orb_command += ros_parameter(key, value)

    human_command = None
    if assisted:
        human_command = [
            sys.executable, "-m", "slam.human_slam_node", "--ros-args",
            "--params-file", str(args.human_config),
        ]
        for key, value in {
            "latency_output": output / "human_latency.csv",
            "candidate_output": output / "human_candidates.csv",
            "debug_output_dir": output / "semantic_frames",
            "source_image_dir": sequence / "undistorted_images/cam0",
        }.items():
            human_command += ros_parameter(key, value)

    player_command = [
        sys.executable, "-m", "slam.kitti_dataset_player", "--ros-args",
    ]
    for key, value in {
        "dataset_path": sequence,
        "left_dir": "undistorted_images/cam0",
        "right_dir": "undistorted_images/cam1",
        "times_file": "times.txt",
        "imu_file": "imu.txt",
        "imu_topic": "/dataset/imu",
        "left_topic": "/dataset/left/image_raw",
        "right_topic": "/dataset/right/image_raw",
        "playback_rate": args.playback_rate,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "startup_delay": 2.0,
        "completion_delay": 5.0,
        "stereo_publish_gap": 0.01,
        "results_dir": output,
    }.items():
        player_command += ros_parameter(key, value)

    processes = []
    try:
        orb = start_process(orb_command, output / "orbslam3.log", environment)
        processes.append(orb)
        human = None
        if human_command:
            human = start_process(human_command, output / "humanslam.log", environment, PROJECT.parent)
            processes.append(human)
        wait_for_log(
            orb, output / "orbslam3.log", "ORB-SLAM3 active", timeout=120
        )
        if human is not None:
            wait_for_log(
                human, output / "humanslam.log", "HumanSLAM active", timeout=180
            )
        player = start_process(player_command, output / "player.log", environment, PROJECT.parent)
        processes.append(player)
        return_code = player.wait()
        if return_code:
            raise RuntimeError(f"Dataset player failed ({return_code}); inspect {output / 'player.log'}")
        time.sleep(2.0)
    finally:
        for process in reversed(processes):
            stop_process(process)
        for process in processes:
            close_log(process)

    video_command = [
        sys.executable, str(PROJECT / "tools/render_slam_annotated_video.py"),
        "--images", str(sequence / "undistorted_images/cam0"),
        "--orb-events", str(output / "orb_events_latency.csv"),
        "--orb-features", str(output / "orb_tracked_features.csv"),
        "--output", str(output / f"{label}_annotated.mp4"),
        "--label", "ORB-SLAM3 + HumanSLAM" if assisted else "ORB-SLAM3 baseline",
        "--start-frame", str(args.start_frame),
        "--end-frame", str(args.end_frame),
    ]
    if assisted:
        video_command += ["--semantic-frames", str(output / "semantic_frames")]
    subprocess.run(video_command, check=True, env=environment)
    if assisted:
        subprocess.run([
            sys.executable, str(PROJECT / "tools/render_candidate_matches.py"),
            "--candidates", str(output / "human_candidates.csv"),
            "--output", str(output / "humanslam_candidate_matches.mp4"),
            "--images", str(sequence / "undistorted_images/cam0"),
            "--semantic-frames", str(output / "semantic_frames"),
        ], check=True, env=environment)
    print(f"Completed {label}: {output}")


def main():
    args = arguments()
    sequence = sequence_directory(args.dataset_root.expanduser(), args.sequence)
    required = [
        sequence / "times.txt", sequence / "imu.txt",
        sequence / "undistorted_images/cam0", sequence / "undistorted_images/cam1",
        args.vocabulary, args.settings, args.human_config,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required resources:\n" + "\n".join(map(str, missing)))
    run_name = args.run_name or (
        datetime.now().strftime("%Y-%m-%d_%H%M%S") + f"_4seasons_{args.sequence}"
    )
    root = args.results_root.expanduser() / run_name
    modes = (False, True) if args.mode == "both" else (args.mode == "humanslam",)
    for assisted in modes:
        run_trial(args, sequence, root, assisted)
    if args.mode == "both":
        subprocess.run([
            sys.executable,
            str(PROJECT / "tools/evaluate_4seasons_comparison.py"),
            "--sequence", str(sequence),
            "--results", str(root),
        ], check=True)
    print(f"Experiment complete: {root}")


if __name__ == "__main__":
    main()
