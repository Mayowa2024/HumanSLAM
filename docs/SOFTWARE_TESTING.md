# HumanSLAM Software Testing and Verification

## 1. Where this belongs in the dissertation

Place a concise **Software Verification** subsection in the methodology to show
how implementation correctness was established before experimentation. Place
the test results, coverage table and failures in the **Test and Evaluation**
chapter. This avoids mixing “the code implements the equation” with “the method
improves SLAM”.

Recommended structure:

- Methodology → Implementation → Software verification strategy
- Evaluation → Component tests → Integration tests → SLAM experiments

## 2. Verification objectives

The software test programme answers:

1. Do mathematical functions implement their stated invariants?
2. Does missing evidence behave neutrally?
3. Can every intended layer ablation run?
4. Are ROS messages well-formed and identifiers preserved?
5. Do GPU models load and produce expected shapes?
6. Can ORB accept, verify and reject semantic candidates safely?
7. Does the complete launch terminate and save results?

These are correctness questions. ATE/RPE and recovery-rate comparisons are
research-performance questions.

## 3. Automated unit-test matrix

| Test behaviour | Intended property | Level |
|---|---|---|
| Identical semantics | Unified score equals 1 | Unit |
| Strong mismatch | Score remains near zero | Unit |
| Low classifier confidence | Does not suppress identical embedding | Unit |
| Missing objects/text | Available scene evidence is renormalised | Unit |
| All seven valid ablations | One, two or three layers can run | Unit |
| All layers disabled | Invalid configuration is rejected | Unit |
| Missing OCR | Treated as unavailable, not contradiction | Unit |
| Changed storefront text | Produces bounded penalty | Unit |
| Partial OCR/distinctiveness | Handles OCR truncation and weak words | Unit |
| Temporal scene weighting | Recent context contributes most | Unit |
| Object spatial decay | Displacement reduces similarity | Unit |
| Object class gate | Different classes cannot correspond | Unit |
| Text geometry gate | Unrelated spatial objects cannot share text | Unit |
| Recovery boundary | Requires low inliers and high semantic score | Unit |
| Candidate ranking | Highest unified score is returned | Unit |
| String cleaning | Case/punctuation variation is normalised | Unit |

## 4. Current executed evidence

Command:

```bash
export PYTHONNOUSERSITE=1
source /opt/ros/humble/setup.bash
source /home/teleopbike/orbslam3_ros2_ws/install/setup.bash
cd /home/teleopbike/Documents/Mayowa/ros2_ws/src/slam
python3 -m pytest -q test/test_cognitive_math_model.py
```

Initial recorded result on 29 July 2026:

```text
9 passed in 0.20s
```

This result covered the original mathematical tests. Re-run after every change
and replace/add a dated result below. Store the raw console output in an
experiment artefact directory so the dissertation result is auditable.

Expanded suite recorded result on 29 July 2026:

```text
17 passed in 0.12s
```

The expanded suite adds explicit temporal weighting, spatial decay, class and
text gates, recovery-policy boundaries, ranking, string normalisation and
no-evidence behaviour.

## 5. Build verification

The following builds verify interface compatibility and linking:

```bash
cmake --build /home/teleopbike/ORB_SLAM3/build --target ORB_SLAM3 -j1
```

Recorded result on 29 July 2026:

```text
[100%] Built target ORB_SLAM3
```

```bash
cd /home/teleopbike/orbslam3_ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select \
  human_slam_interfaces orbslam3_zed_stereo \
  --allow-overriding human_slam_interfaces
```

Recorded result on 29 July 2026:

```text
2 packages finished
```

Compiler deprecation/style warnings from the upstream ORB-SLAM3/Eigen code are
not test failures, but should be retained in raw logs and distinguished from
errors introduced by HumanSLAM.

## 6. Required integration tests

### ROS interface test

- Launch ORB and HumanSLAM.
- Confirm `/orbslam3/semantic_frame` and
  `/human_slam/semantic_candidates` exist.
- Record one message from each.
- Assert candidate arrays have equal lengths.
- Assert returned map/keyframe IDs exist in the Atlas.

### Model smoke test

- Deserialize both TensorRT engines.
- Record input/output tensor names, shapes and dtypes.
- Run a known image twice and assert finite, stable-shaped outputs.
- Confirm embeddings have dimension 512 and non-zero norm.
- Confirm YOLO parsing works both with masks and box fallback.
- Record the OCR-selected device and perform a known-text crop test.

### Asynchronous queue test

- Use a deliberately slow fake inference function.
- Submit more tasks than the queue capacity.
- Assert the newest pending frame replaces stale pending work.
- Assert the callback thread remains responsive.

### Relocalisation integration test

- Publish a known stored candidate for a lost query.
- Confirm the semantic response enters Tracking.
- Confirm a correct candidate can reach PnP/pose optimisation.
- Confirm a wrong candidate is rejected by geometry.
- Confirm stale candidate responses are not applied indefinitely.

### Map-fusion integration test

- Create two Atlas maps with a known overlapping region.
- Return an old-map keyframe for a new-map query.
- Confirm same-map candidates are filtered.
- Confirm the cross-map candidate reaches Sim(3) verification.
- Confirm one observation is insufficient.
- Confirm three consistent observations invoke Atlas merge.
- Confirm a wrong semantic candidate never invokes merge.

### Offline launch test

- Run a short KITTI-like stereo sequence.
- Assert both topics publish.
- Assert dataset completion shuts down the launch.
- Assert a non-empty KITTI trajectory file is created.
- Repeat with HumanSLAM disabled to verify the baseline path.

## 7. Test data and reproducibility

Every test/experiment record should include:

- date and source revision identifier;
- machine, GPU, driver, CUDA, TensorRT and ROS versions;
- model filenames and checksums;
- dataset and exact frame range;
- camera/settings file;
- launch arguments and parameter YAML;
- random seed where applicable;
- raw terminal/ROS logs;
- generated trajectories and metric script output.

Use a result layout such as:

```text
results/
  20260729_sequence_condition_run01/
    command.txt
    environment.txt
    parameters.yaml
    console.log
    trajectory_kitti.txt
    metrics.json
```

## 8. Acceptance criteria

Define criteria before viewing final results:

- all deterministic unit tests pass;
- no malformed message reaches ORB;
- no NaN/Inf semantic scores;
- no fusion without geometric and temporal verification;
- zero known false fusions in the controlled negative set;
- trajectory output exists for every completed run;
- HumanSLAM does not reduce processing below the declared real-time target;
- any improvement claim is supported across multiple sequences, not one
  favourable example.

## 9. Interpretation rule

Passing software tests supports the statement:

> “The implemented components behave according to their specified software
> contracts.”

It does not support:

> “HumanSLAM improves localisation accuracy.”

That second statement requires the baseline, ablation and end-to-end trajectory
evaluation described in the methodology.
