#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/teleopbike/orbslam3_ros2_ws/install/setup.bash
source /home/teleopbike/Documents/Mayowa/ros2_ws/install/setup.bash
set -u
export PYTHONPATH="/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam:${PYTHONPATH:-}"

exec python3 tools/run_separated_duplicate_microtests.py \
  --images /home/teleopbike/Documents/slam_experiments/datasets/4seasons/neighborhood_3_train/recording_2020-10-07_14-53-52/undistorted_images/cam0 \
  --params /home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/config/human_slam_params.yaml \
  --output test_results/microtest/2026-08-08_duplicate_revisit_after_10_consecutive_frames_query_denominator_corrected \
  --base-frame-ids 500 1400 2300 3200 4100 5000 5900 6800 7700 8600 \
  --gap 10
