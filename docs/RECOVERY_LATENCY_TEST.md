# Forced-loss relocalisation and latency test

## Purpose

This test separates three claims that must not be conflated:

1. the perturbation causes ORB-SLAM3 tracking to become weak or lost;
2. HumanSLAM retrieves earlier ORB keyframes within a useful latency budget;
3. ORB-SLAM3 geometrically accepts a HumanSLAM-supplied keyframe and recovers.

A `Relocalized!!` message alone proves only ORB recovery. A
`HumanSLAM relocalisation successful` message is required to attribute the
accepted candidate to HumanSLAM.

## Scenario

`tools/make_relocalisation_scenario.py` creates a KITTI-like stereo sequence
with:

- a clean mapping phase;
- identical black left/right images for a configurable interval, forcing
  tracking failure; and
- a night-transformed revisit.

For a controlled frame-level relocalisation test, `--revisit-source-start`
replays an earlier mapped stretch under the night transform. Without that
option, the tool perturbs the sequence's natural revisit, which is more
appropriate for testing map merge.

Example:

```bash
python3 tools/make_relocalisation_scenario.py \
  --source /path/to/KITTI/sequences/06 \
  --output /path/to/kitti06_recovery \
  --dropout-start 200 --dropout-end 209 \
  --night-start 210 --night-end 250 \
  --revisit-source-start 0 \
  --brightness -100 --contrast 0.55 --gamma 0.7
```

## Matched runs

Use the same sequence, frame range, startup delay and settings for both runs:

```bash
ros2 launch slam offline_benchmark.launch.py \
  dataset_path:=/path/to/kitti06_recovery \
  settings_file:=/path/to/KITTI04-12.yaml \
  use_human_slam:=false \
  results_dir:=/path/to/results/baseline

ros2 launch slam offline_benchmark.launch.py \
  dataset_path:=/path/to/kitti06_recovery \
  settings_file:=/path/to/KITTI04-12.yaml \
  use_human_slam:=true \
  semantic_threshold:=0.55 \
  candidate_response_count:=25 \
  results_dir:=/path/to/results/assisted
```

Summarise latency:

```bash
python3 tools/summarise_latency.py \
  --human-csv /path/to/results/assisted/human_latency.csv \
  --orb-csv /path/to/results/assisted/orb_latency.csv \
  --output /path/to/results/assisted/latency_summary.json
```

The ORB bridge also publishes `/orbslam3/status` (`std_msgs/String`) and writes
the same state context into `orb_latency.csv`. Each tracking row contains the
numeric state and its name (`OK`, `RECENTLY_LOST`, `LOST`, and so on), active
map ID, reference-keyframe ID, inlier count and reference-keyframe validity.
Discrete rows identify:

- `TRACKING_RECENTLY_LOST`, `TRACKING_LOST` and `TRACKING_RECOVERED`;
- `SEMANTIC_RESPONSE`, `SEMANTIC_CANDIDATE_QUEUED` and
  `SEMANTIC_NO_CANDIDATE`; and
- `MAP_ACTIVE`, `NEW_MAP_OBSERVED` and `MAP_SWITCHED`.

`NEW_MAP_OBSERVED` means the wrapper saw a previously unseen ORB map ID; it is
evidence of a new-map transition, not by itself proof that map fusion succeeded.
The latency summary reports event counts and the number of frames spent in each
tracking state.

## Current measured result (30 July 2026)

On the shortened strong-night test, tracking changed from OK to weak at frame
200. With a semantic threshold of 0.55 and 25 returned candidates, tracking
returned to OK at frame 212 (1,297.6 ms wall time). HumanSLAM supplied 25 valid
candidates before recovery, but ORB printed `Relocalized!!`, not
`HumanSLAM relocalisation successful`. Therefore this run does **not** yet prove
that semantics caused recovery; ORB's BoW candidate won geometric verification.

Measured assisted-run latencies:

| Measurement | Median | P95 |
|---|---:|---:|
| HumanSLAM recovery query | 14.77 ms | 27.82 ms |
| ORB-to-semantic response | 14.55 ms | 25.01 ms |
| ORB `TrackStereo` | 38.12 ms | 50.10 ms |

The semantic worker is asynchronous, so its 14.55 ms median response is not
added directly to every 38.12 ms tracking call. It consumes GPU/CPU capacity
and supplies a candidate for a subsequent lost frame.

## Interpretation

The infrastructure now reproduces tracking failure, candidate delivery,
geometric rejection/acceptance and component latency. The remaining research
task is retrieval calibration: on this KITTI segment the scene-only evidence
does not rank the geometrically correct keyframe highly enough to make
HumanSLAM the winning source. Report this as a negative result until the log
explicitly attributes a verified recovery to HumanSLAM. Suitable next steps
are fine-tuning the stable-object model, learning/calibrating the retrieval
threshold on a separate validation split, and evaluating several perturbation
severities without selecting the final test case post hoc.
