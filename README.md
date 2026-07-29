# HumanSLAM

HumanSLAM is a semantic place-recognition extension for ORB-SLAM3. It describes
ORB-SLAM3 keyframes using temporal scene context, stable-object layout and
object-grounded OCR, then returns ranked map/keyframe candidates for
geometrically verified relocalisation and Atlas map fusion.

HumanSLAM does **not** replace ORB-SLAM3 visual odometry, local mapping or
optimisation. It proposes where the camera may have been observed previously;
ORB-SLAM3 remains responsible for proving the hypothesis and estimating the
camera/map transformation.

## Project status

Implemented and build-verified:

- ORB-SLAM3 keyframe, pose, tracking-state and map-ID publication;
- asynchronous scene, object and OCR inference;
- hierarchical semantic keyframe memory and retrieval;
- scene/object/text single- and multi-layer ablations;
- semantic candidate delivery to ORB-SLAM3;
- PnP/geometrically verified relocalisation assistance;
- cross-map semantic proposals using ORB-SLAM3 Sim(3) verification;
- three-keyframe temporal consistency before Atlas map fusion;
- KITTI-like offline stereo playback and trajectory output; and
- 17 deterministic tests for the cognitive scoring model.

The system is operational, but an improvement over the ORB-SLAM3 baseline has
not yet been established. That claim requires the planned baseline, ablation,
relocalisation, map-fusion, ATE/RPE and latency experiments.

## Motivation

Geometry-centred SLAM can lose localisation under illumination, weather,
viewpoint, blur and other appearance changes. Local visual structures can also
look alike in physically different places. HumanSLAM investigates whether cues
similar to those used in human place recognition can complement geometry:

1. **Scene context** identifies the broad visual environment and recent scene
   sequence.
2. **Stable-object layout** compares the classes, positions and areas of
   persistent landmarks.
3. **Object-grounded text** compares OCR only when it is attached to compatible,
   spatially consistent objects.

Semantics narrows the candidate search. Geometry remains the safety check,
particularly before map fusion.

## Architecture

```text
KITTI-like dataset or stereo camera
                 |
                 v
        ORB-SLAM3 ROS 2 node
     tracking, keyframes and Atlas
                 |
      /orbslam3/semantic_frame
                 |
                 v
          HumanSLAM node
   scene -> object -> grounded OCR
        cognitive score/ranking
                 |
   /human_slam/semantic_candidates
                 |
          +------+------+
          |             |
          v             v
   Relocalisation   LoopClosing
   ORB + PnP +      ORB + Sim(3) +
   pose optimise    3-frame consistency
          |             |
          v             v
     tracked pose    Atlas map fusion
```

For relocalisation, HumanSLAM returns an old keyframe as a place hypothesis.
ORB-SLAM3 estimates the current query pose from present ORB correspondences; it
does not copy the old keyframe pose directly.

For map fusion, only candidates belonging to another Atlas map are considered.
Semantic candidates are prioritised, but ORB-SLAM3 feature matching, Sim(3)
optimisation, projection checks and temporal consistency remain mandatory.

## Repository and workspace layout

HumanSLAM is used as an `ament_python` package named `slam`:

```text
ros2_ws/
  src/
    slam/
      config/
        human_slam_params.yaml
      launch/
        human_slam.launch.py
        offline_benchmark.launch.py
      resource/
      slam/
        human_slam_node.py
        cognitive_math_model.py
        kitti_dataset_player.py
        types.py
        weights/
        tools/
        docs/
      test/
      package.xml
      setup.py
```

The integrated system also needs:

- the modified ORB-SLAM3 core;
- the `orbslam3_zed_stereo` ROS 2 wrapper; and
- the `human_slam_interfaces` message package.

Local `external/` links may be used to browse those repositories together, but
they are not portable dependencies and are not required by Python imports.

Generated ROS directories belong at the workspace root:

```text
ros2_ws/build/
ros2_ws/install/
ros2_ws/log/
```

They are build products, not source files.

## Requirements

The current development environment uses:

- Ubuntu with ROS 2 Humble;
- Python 3.10;
- a CUDA-capable NVIDIA GPU;
- a TensorRT version compatible with the exported engines;
- ORB-SLAM3 and its normal C++ dependencies;
- OpenCV and `cv_bridge`;
- NumPy;
- Ultralytics YOLO;
- PaddleOCR/PaddlePaddle with the intended GPU runtime; and
- the custom `human_slam_interfaces` ROS messages.

TensorRT engines are not universally portable. Rebuild them from ONNX when the
GPU architecture, CUDA or TensorRT version is incompatible.

## Model files

Expected under `slam/weights/`:

| File | Runtime purpose |
|---|---|
| `resnet50_places365_embed.engine` | Places365 TensorRT classification and 512-D scene embedding |
| `yolo26n-seg.engine` | YOLO segmentation/object inference |
| `resnet50_places365_embed.onnx` | Reproducible intermediate used to rebuild the scene engine |

The ONNX model is not loaded during a normal run.

The binaries are excluded from the Git repository because TensorRT engines are
machine/runtime-specific and the scene engine exceeds GitHub's normal per-file
limit. See `weights/README.md` for expected filenames, checksums and rebuild
guidance.

The current detector vocabulary is temporary. The planned custom stable-road
dataset will add more useful persistent classes such as buildings, shopfronts,
billboards and road signs. Do not list classes in `stable_classes` unless the
deployed YOLO model actually supports them.

## Build

The paths below are examples. Substitute your own workspace locations.

### 1. Build ORB-SLAM3

```bash
cd /path/to/ORB_SLAM3
cmake -S . -B build
cmake --build build --target ORB_SLAM3 -j1
```

### 2. Build messages and the ORB ROS wrapper

```bash
source /opt/ros/humble/setup.bash
cd /path/to/orbslam3_ros2_ws
colcon build --packages-select \
  human_slam_interfaces orbslam3_zed_stereo \
  --allow-overriding human_slam_interfaces
```

### 3. Build HumanSLAM

```bash
source /opt/ros/humble/setup.bash
source /path/to/orbslam3_ros2_ws/install/setup.bash
cd /path/to/ros2_ws
colcon build --packages-select slam --symlink-install
```

Source both overlays in every new terminal:

```bash
source /opt/ros/humble/setup.bash
source /path/to/orbslam3_ros2_ws/install/setup.bash
source /path/to/ros2_ws/install/setup.bash
```

## Configuration

Edit `config/human_slam_params.yaml` and replace development-machine paths:

```yaml
scene_classifier_path: "/path/to/resnet50_places365_embed.engine"
yolo_model_path: "/path/to/yolo26n-seg.engine"
```

Important parameter groups:

| Parameter | Meaning | Default |
|---|---|---:|
| `use_scene` | Enable temporal scene layer | `true` |
| `use_object` | Enable stable-object layer | `true` |
| `use_text` | Enable grounded OCR layer | `true` |
| `w_scene` | Scene base weight | `0.3` |
| `w_object` | Object base weight | `0.3` |
| `w_text` | Text base weight | `0.4` |
| `candidate_top_k` | Scene-preselected candidates receiving full scoring | `25` |
| `candidate_min_keyframe_separation` | Exclude nearby temporal neighbours | `20` |
| `candidate_response_count` | Candidates returned to ORB-SLAM3 | `5` |
| `semantic_threshold` | Minimum accepted semantic score | `0.75` |
| `semantic_ambiguity_margin` | Top-two margin regarded as ambiguous | `0.05` |
| `sigma_mask` | Object-layout displacement tolerance | `0.25` |
| `text_geom_threshold` | Object geometry gate before text comparison | `0.1` |
| `text_conflict_floor` | Bound on contradictory-text penalty | `0.25` |
| `ocr_keyframe_interval` | Normal keyframe OCR interval | `5` |
| `ocr_max_objects_per_keyframe` | Maximum OCR crops per keyframe | `3` |
| `semantic_queue_size` | Pending asynchronous semantic work | `1` |

At least one of `use_scene`, `use_object` and `use_text` must be enabled.

## Dataset format

The offline player accepts any stereo dataset arranged in a KITTI-like
directory interface:

```text
sequence/
  image_0/
    000000.png
    000001.png
    ...
  image_1/
    000000.png
    000001.png
    ...
  times.txt
```

`times.txt` contains one monotonically increasing timestamp in seconds per
stereo pair. Left images, right images and timestamps must have equal counts.
Directory and timestamp names can be changed with launch arguments.

## Run an offline benchmark

```bash
ros2 launch slam offline_benchmark.launch.py \
  dataset_path:=/path/to/sequence \
  settings_file:=/path/to/orb_stereo_settings.yaml \
  vocabulary_file:=/path/to/ORBvoc.txt \
  human_slam_config:=/path/to/human_slam_params.yaml \
  use_human_slam:=true \
  results_dir:=/path/to/results/humanslam_run
```

Run the matched ORB-SLAM3 baseline with the same dataset, frame range, settings
and playback rate:

```bash
ros2 launch slam offline_benchmark.launch.py \
  dataset_path:=/path/to/sequence \
  settings_file:=/path/to/orb_stereo_settings.yaml \
  vocabulary_file:=/path/to/ORBvoc.txt \
  use_human_slam:=false \
  results_dir:=/path/to/results/orbslam3_baseline
```

Useful offline arguments:

| Argument | Default |
|---|---|
| `left_dir` | `image_0` |
| `right_dir` | `image_1` |
| `times_file` | `times.txt` |
| `playback_rate` | `1.0` |
| `start_frame` | `0` |
| `end_frame` | `-1` (all frames) |
| `startup_delay` | `8.0` seconds |
| `completion_delay` | `3.0` seconds |

The launch shuts down when playback completes and requests a KITTI trajectory
at:

```text
<results_dir>/trajectory_kitti.txt
```

## Ablation examples

Scene only:

```bash
ros2 launch slam offline_benchmark.launch.py ... \
  use_scene:=true use_object:=false use_text:=false
```

Objects and text:

```bash
ros2 launch slam offline_benchmark.launch.py ... \
  use_scene:=false use_object:=true use_text:=true
```

The supported experimental conditions are scene, objects, text, each of the
three possible pairs, and all three layers.

## ROS interfaces

### ORB-SLAM3 to HumanSLAM

Topic:

```text
/orbslam3/semantic_frame
```

Message: `human_slam_interfaces/msg/OrbSlamFrame`

It contains the current image, frame ID, reference keyframe ID, Atlas map ID,
tracking state, tracking inliers and camera-pose metadata.

### HumanSLAM to ORB-SLAM3

Topic:

```text
/human_slam/semantic_candidates
```

Message: `human_slam_interfaces/msg/SemanticCandidates`

It contains the query identity, candidate map/keyframe IDs, candidate poses,
semantic scores, acceptance flag, ambiguity flag and best score. Candidate
arrays must have equal lengths; the wrapper rejects malformed responses.

## Expected runtime messages

Useful messages include:

```text
HumanSLAM active
Tracking lost
HumanSLAM recovery/map-fusion proposal queued
HumanSLAM map-fusion verification
HumanSLAM cross-map Sim3 verified; awaiting temporal consistency (1/3)
HumanSLAM map fusion successful: map X fused with map Y
```

The proposal message does not mean recovery or fusion succeeded. Only subsequent
geometric verification and the final success message establish a completed
operation.

## Tests

Run the deterministic cognitive-model suite:

```bash
cd /path/to/ros2_ws/src/slam
python3 -m pytest -q test/test_cognitive_math_model.py
```

Current recorded result:

```text
17 passed
```

The tests cover identity/mismatch scoring, evidence renormalisation, all valid
ablations, temporal weighting, spatial decay, class/text gates, OCR
normalisation, recovery boundaries, candidate ranking and no-evidence
behaviour.

Passing unit and build tests shows that components obey their software
contracts. It does not demonstrate improved SLAM performance.

## Evaluation plan

The final evaluation compares:

- ORB-SLAM3 baseline;
- full HumanSLAM;
- all seven non-empty scene/object/text ablations;
- nominal and perceptually varied traversals;
- tracking-loss and new-map scenarios; and
- previously unseen places that should be rejected.

Primary metrics:

- absolute trajectory error (ATE);
- relative pose error (RPE);
- tracking-success and lost-frame counts;
- relocalisation success, false-relocalisation rate and recovery time;
- semantic Recall@1/Recall@5 and rejection behaviour;
- number of fragmented maps;
- correct and incorrect map fusions; and
- component/end-to-end latency and throughput.

## Troubleshooting

### TensorRT engine fails to deserialize

The engine may have been built with an incompatible TensorRT/CUDA version or
GPU architecture. Re-export it from the ONNX model on the target machine.

### CUDA device is not detected

Check the NVIDIA driver and container/device visibility first:

```bash
nvidia-smi
```

Then verify that the installed TensorRT, PyTorch or PaddlePaddle build includes
CUDA support. A physical GPU alone does not guarantee that a Python runtime was
installed with GPU support.

### OCR runs on CPU

Confirm the PaddlePaddle package is a compatible GPU build and inspect the
HumanSLAM startup log for the selected OCR device. See
`docs/GPU_OCR.md` for the known local configuration.

### No semantic candidates are accepted

Check:

- whether HumanSLAM receives `/orbslam3/semantic_frame`;
- model-loading messages;
- whether enough separated keyframes exist in memory;
- `candidate_min_keyframe_separation`;
- `semantic_threshold`; and
- whether enabled layers have usable evidence.

### No map fusion occurs

Map fusion requires two Atlas maps containing a genuinely overlapping place,
a cross-map semantic proposal, successful ORB/Sim(3) verification and three
temporally consistent observations. A semantic match alone is intentionally
insufficient.

### Dataset player rejects the sequence

Verify equal left/right/timestamp counts, supported image extensions and
monotonically increasing timestamps.

## Limitations

- End-to-end improvement over ORB-SLAM3 is not yet experimentally proven.
- The stable-object vocabulary is limited by the deployed YOLO model.
- Semantic memory currently grows with stored keyframes.
- OCR can be sparse, viewpoint-sensitive and expensive.
- Scene-first preselection may miss candidates detectable only by objects/text.
- TensorRT engines reduce portability.
- HumanSLAM cannot recover a place that has never been mapped.
- Correct semantic candidates can still fail when geometry is insufficient.

## Documentation

- `docs/METHODOLOGY.md` — detailed method and experimental design.
- `docs/METHODOLOGY.docx` — editable Word version.
- `docs/SOFTWARE_TESTING.md` — verification strategy and recorded evidence.
- `docs/GPU_OCR.md` — OCR/CUDA environment notes.

## Licence and citation

The ROS package declares Apache-2.0, but third-party components and model
weights retain their own licences. Verify dataset, ORB-SLAM3, Places365,
Ultralytics and PaddleOCR terms before redistribution.

A formal project citation has not yet been published. If this repository
supports an academic submission, cite the dissertation and the upstream
ORB-SLAM3, Places365, YOLO and OCR works used by the implementation.
