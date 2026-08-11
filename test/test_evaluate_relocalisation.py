import csv
from pathlib import Path

import numpy as np
import pytest

from tools.evaluate_relocalisation import evaluate


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("source", ["native_bow", "humanslam"])
def test_recovery_episode_and_source_are_measured_for_both_methods(tmp_path, source):
    events = tmp_path / "orb_events.csv"
    keyframes = tmp_path / "orb_keyframes.csv"
    index = tmp_path / "trajectory_frame_ids.csv"
    ground_truth = tmp_path / "poses.txt"
    fields = [
        "frame_id", "dataset_time", "module", "event", "state",
        "state_name", "current_kf", "matched_kf", "matches_inliers", "details",
    ]
    write_csv(events, fields, [
        {"frame_id": 8, "dataset_time": 0.8, "module": "Tracking",
         "event": "TRACKING_RECENTLY_LOST", "state": 3,
         "state_name": "RECENTLY_LOST", "current_kf": 2,
         "matched_kf": -1, "matches_inliers": 8, "details": ""},
        {"frame_id": 9, "dataset_time": 0.9, "module": "Tracking",
         "event": "RELOCALIZATION_ATTEMPT", "state": -1,
         "state_name": "N/A", "current_kf": -1,
         "matched_kf": -1, "matches_inliers": -1, "details": ""},
        {"frame_id": 9, "dataset_time": 0.9, "module": "Tracking",
         "event": "RELOCALIZATION_SUCCESS", "state": -1,
         "state_name": "N/A", "current_kf": -1,
         "matched_kf": -1, "matches_inliers": -1,
         "details": f"source={source};candidate_keyframe=1;inliers=42"},
        {"frame_id": 10, "dataset_time": 1.0, "module": "Tracking",
         "event": "TRACKING_STABLE", "state": 2, "state_name": "OK",
         "current_kf": 3, "matched_kf": -1, "matches_inliers": 70,
         "details": ""},
    ])
    write_csv(keyframes, ["keyframe_id", "frame_id"], [
        {"keyframe_id": 1, "frame_id": 0},
    ])
    write_csv(index, ["trajectory_row", "frame_id", "source_frame_id"], [
        {"trajectory_row": frame, "frame_id": frame,
         "source_frame_id": f"camera|dataset_frame={frame}"}
        for frame in range(11)
    ])
    poses = []
    for frame in range(11):
        pose = np.eye(4)[:3]
        pose[0, 3] = 0.1 * frame
        poses.append(" ".join(str(value) for value in pose.reshape(-1)))
    ground_truth.write_text("\n".join(poses) + "\n", encoding="utf-8")

    report = evaluate(events, keyframes, ground_truth, index, 5.0, 30.0)

    assert report["tracking_loss"]["episodes"] == 1
    assert report["tracking_loss"]["recovered_episodes"] == 1
    assert report["tracking_loss"]["frames_to_recovery_mean"] == 2
    assert report["tracking_loss"]["recovery_sources"] == {source: 1}
    assert report["relocalisation"]["success_sources"] == {source: 1}
    assert report["relocalisation"]["ground_truth_correct"] == 1
