# HumanSLAM Dissertation Work Log

This is the living record of implementation and experimental decisions made
during HumanSLAM development. It complements the formal methodology and test
documents. Entries should preserve negative results and changes of direction,
not only successful outcomes.

## Recording protocol

For every material task, record:

- date and research purpose;
- hypothesis or engineering objective;
- dataset, split and frame/traversal identifiers;
- exact model, weights, parameters and random seed;
- software revision and relevant environment versions;
- commands or reproducible scripts;
- quantitative results, including median and P95 latency where relevant;
- failures, warnings and rejected alternatives;
- interpretation and the next decision;
- paths to raw logs, trajectories, plots, checkpoints and reports.

Do not claim that HumanSLAM caused a recovery unless ORB geometrically accepted
a candidate attributed to HumanSLAM. Do not treat passing software tests as
evidence of improved localisation accuracy.

The recorded ORB-SLAM3 KITTI odometry runs for sequences 00--10 are stereo-only.
They contain no IMU input or inertial optimisation and must not be described as
stereo-inertial results. Future bike/ZED stereo-inertial experiments are a
separate sensor configuration and require separate reporting.

## 30 July 2026 — Forced-loss and latency instrumentation

### Objective

Create a reproducible offline condition in which ORB-SLAM3 tracking becomes
weak, HumanSLAM can return stored semantic keyframes, and recovery latency can
be measured at component and system boundaries.

### Implementation

- Added a KITTI-like forced-loss/night-revisit scenario generator.
- Added HumanSLAM timings for queueing, scene inference, YOLO/OCR, ranking and
  total response.
- Added ORB-wrapper timings for `TrackStereo`, semantic response age and
  weak/lost-to-recovered duration.
- Added a latency summariser and automatic CSV outputs to the offline launch.
- Changed ambiguous semantic responses to return multiple candidates for ORB
  geometric verification instead of suppressing the entire response.

### Result

- Median HumanSLAM recovery-query latency: 14.77 ms.
- P95 HumanSLAM recovery-query latency: 27.82 ms.
- Median ORB-to-semantic response: 14.55 ms.
- P95 ORB-to-semantic response: 25.01 ms.
- Median ORB `TrackStereo`: 38.12 ms.
- P95 ORB `TrackStereo`: 50.10 ms.
- Weak-to-OK duration in the recorded short experiment: 1,297.6 ms.

HumanSLAM supplied candidates, but ORB reported ordinary `Relocalized!!` rather
than `HumanSLAM relocalisation successful`. The experiment therefore did not
demonstrate causation by the semantic candidate. This negative result isolated
semantic retrieval ranking as the next bottleneck; runtime latency was not the
principal limitation.

Full procedure and interpretation: `docs/RECOVERY_LATENCY_TEST.md`.

## 3 August 2026 — Mapillary Vistas audit and stable-object decision

### Objective

Determine whether the downloaded Mapillary Vistas dataset can supply suitable
segmentation supervision for HumanSLAM's spatially structured stable-object
layer and reduce reliance on the current COCO-only model.

### Dataset audit

Local dataset: `/home/teleopbike/Documents/Mayowa/mapillary vistas dataset`

- Total on-disk size: 31 GB.
- Version selected: v2.0.
- Training: 18,000 images with labels, instances, panoptic masks and polygons.
- Validation: 2,000 labelled images with the same annotation forms.
- Test: 5,000 images without public labels.
- Label taxonomy: 124 classes, including large structural regions,
  instance-specific signs and permanent street furniture.

Selected-class prevalence found by parsing the raw polygon JSON files:

| Original Mapillary class | Training images | Training polygons/objects | Validation images | Validation polygons/objects |
|---|---:|---:|---:|---:|
| Building | 16,489 | 29,169 | 1,810 | 3,123 |
| Bridge | 2,494 | 3,302 | 286 | 398 |
| Advertisement signage | 10,139 | 35,692 | 1,130 | 3,862 |
| Store signage | 7,019 | 28,353 | 765 | 3,039 |
| Traffic-sign front | 15,452 | 84,007 | 1,726 | 9,280 |
| General single traffic light | 682 | 940 | 73 | 102 |

Building polygons contain approximately 53 vertices on average. The raw v2.0
polygons are therefore suitable for conversion to normalised YOLO segmentation
labels, subject to polygon validation and visual overlay checks.

### Model and taxonomy decision

The intended approach is transfer learning from COCO-pretrained
`YOLO26n-seg`, not training from scratch. The retained HumanSLAM taxonomy is:

1. building;
2. bridge;
3. advertisement sign;
4. store sign;
5. information sign;
6. traffic sign;
7. traffic light;
8. wall;
9. fence;
10. guard rail;
11. tunnel;
12. street light;
13. pole;
14. bench;
15. bike rack;
16. CCTV camera;
17. fire hydrant;
18. junction box;
19. mailbox;
20. parking meter; and
21. phone booth.

Fences remain OCR-eligible because they commonly carry notices,
advertisements and construction information. For large structural regions,
the segmentation mask defines a permitted text-search region; detected text
boxes, rather than the complete large crop, should be passed to recognition.

### Domain-adaptation decision

Oxford RobotCar will initially remain an unseen target-domain evaluation set.
Fine-tuning on Oxford is conditional on measured domain shift after Mapillary
training. This preserves the stronger claim of cross-dataset generalisation if
the Mapillary-trained model performs adequately. If adaptation becomes
necessary, training traversals must remain disjoint from final evaluation
traversals.

### Planned training procedure

1. Convert selected Mapillary v2.0 polygons to the 21-class YOLO segmentation
   taxonomy while preserving the official split.
2. Generate statistics and mask-overlay previews.
3. Run a short one-epoch preflight to detect conversion, loader and VRAM
   problems.
4. Fine-tune COCO-pretrained `YOLO26n-seg` on the complete training split,
   using early stopping and periodic checkpoints.
5. Evaluate per-class mask metrics and HumanSLAM retrieval Recall@K.
6. Export the selected checkpoint to TensorRT FP16 and repeat matched
   ORB-SLAM3/HumanSLAM accuracy-versus-latency experiments.

The preflight is a validation step, not a reduced research experiment, and is
retained to avoid wasting a full training run on malformed labels.

## 3 August 2026 — Dataset conversion and CUDA preflight

The official Mapillary splits were converted to the 21-class HumanSLAM YOLO
segmentation format. Automated checks and mask-overlay previews were used to
validate class mapping and polygon geometry.

An initial one-epoch preflight exposed a path defect: resolving the image
symlink caused Ultralytics to search the original Mapillary tree for labels. It
reported zero instances, so the attempt was rejected. The converter was changed
to retain generated dataset-facing paths and sample image-to-label mapping was
verified programmatically.

The corrected CUDA preflight used 200 training and 200 validation images. It
loaded 8,980 validation instances, produced non-zero box and mask losses, found
no corrupt images and saved valid checkpoints on the RTX 4070 Laptop GPU.
Accuracy after one epoch is deliberately not interpreted. Full commands and
reporting criteria are in `docs/MAPILLARY_YOLO_TRAINING.md`.

The first full-run launch at batch size 4 was stopped after 48 of 4,500 batches
because it used only about 1.9 GB of the 8 GB GPU and projected approximately 36
minutes per epoch. This was a throughput calibration, not an accuracy run. The
final configured run uses batch size 8 and a distinct output directory so the
aborted launch cannot be confused with reported results.
# 3 August 2026 — machine-readable ORB tracking status

- Exposed ORB-SLAM3 tracking transitions on `/orbslam3/status` and in the
  benchmark latency CSV, including explicit `RECENTLY_LOST`, `LOST`, recovered,
  semantic-response and semantic-candidate events.
- Added per-frame map ID, reference-keyframe ID, tracking-inlier count and named
  tracking state to support failure/recovery analysis without relying on console
  messages.
- Added observable map transition events and extended the latency summariser
  with event and tracking-state counts. Map-ID changes are treated as observable
  transitions rather than unverified claims of successful fusion.
# 4 August 2026 — YOLO external-domain validation

- Froze and recorded the SHA-256 hash of the completed Mapillary-fine-tuned
  YOLO26n-seg checkpoint.
- Ran a deterministic 30-image qualitative validation on explicitly named
  Mapillary, normal KITTI, darkened KITTI, CARLA and Oxford Radar RobotCar RGB
  sources (240 images total).
- Saved per-image source paths, predicted classes/confidences and visual
  overlays. Initial inspection found strong transfer of large building masks;
  severe synthetic darkening retained many large anchors but degraded or
  changed several smaller pole/sign detections.
- Kept qualitative domain-transfer evidence separate from accuracy claims,
  because the external images were not labelled in the trained YOLO schema.
- Froze the validated best checkpoint as `weights/humanSLAM_YOLO_seg.pt` with
  SHA-256 `efdc5c61d87e78d5d346c2de60bd7264a7e6265fff97068172759d4b42338230`.
- Exported a fixed-batch-1, 640-pixel FP16 TensorRT 11 engine for the RTX 4070
  Laptop GPU. A matched 30-image warm run measured 3.94 ms median preprocessing
  + inference + postprocessing versus 12.83 ms for PyTorch (about 3.3x faster).
- Updated HumanSLAM to load TensorRT explicitly as a segmentation task after a
  smoke test exposed Ultralytics' box-only default for task-ambiguous engine
  filenames. The corrected test returned eight masks for eight detections.
- Replaced obsolete COCO stable/OCR class names with the new Mapillary schema;
  OCR now targets buildings and four text-bearing sign categories while the
  existing interval and three-object cap constrain cost.

## 4 August 2026 — matched detector accuracy--latency selection

- Re-ran COCO PyTorch, Mapillary PyTorch and Mapillary TensorRT FP16 on the
  exact same locked 240-image manifest: 30 images each from Mapillary, KITTI 00,
  normal and darkened KITTI 06, three CARLA sources and Oxford Radar RobotCar.
- Measured median detector pipeline latency of 13.01 ms (COCO PyTorch), 13.30
  ms (Mapillary PyTorch) and 4.84 ms (Mapillary TensorRT), with P95 values of
  15.37, 15.14 and 6.02 ms respectively.
- Evaluated the Mapillary PT and TensorRT models on all 2,000 labelled
  validation images. TensorRT retained segmentation performance: mask mAP50
  was 0.1393 versus 0.1366 and mask mAP50--95 was 0.0582 versus 0.0572, while
  inference decreased from 10.93 to 2.27 ms per image.
- Did not report COCO accuracy against the 21-class labels because its class
  schema is incompatible. External detection counts remain qualitative rather
  than precision/recall evidence.
- Selected `weights/humanSLAM_YOLO_seg.engine` for deployment because it
  preserves task accuracy, provides the stable landmark vocabulary and reduces
  matched median pipeline latency by 63.6% (2.75x faster).

## 4 August 2026 — Places365 contextual grouping

- Added an index-safe grouping of the top three Places365 predictions into
  road-localisation contexts without requiring an external label file.
- Retained the 512-D scene embedding as the primary signal and introduced only
  a bounded 0.10-weight category modulation; category agreement cannot create
  similarity when the embeddings disagree.
- Added ROS parameters `use_scene_category`, `scene_category_weight` and
  `scene_category_top_k`. The category component can therefore be disabled
  independently during ablation.
- Added unit tests covering top-k aggregation, same/related/incompatible group
  ordering, bounded score modulation and disabled-category behaviour. The full
  suite passed 25 tests.

## 4 August 2026 — current-model forced-loss pilot

- Replayed a 261-frame KITTI 06 scenario containing black stereo frames
  200--209 and a strong night-transformed replay during frames 210--250.
- Compared matched ORB-SLAM3-only and current full-HumanSLAM runs after ensuring
  all models were warm before frame zero. Both processed 261 frames and recorded
  214 OK, 45 RECENTLY_LOST and one LOST frame.
- HumanSLAM issued 137 semantic responses and 46 recovery queries, supplied ORB
  keyframe candidates during weak tracking and later supplied cross-map
  candidates for verification. Median recovery-query latency was 15.32 ms.
- The first recovery occurred at frame 212 with HumanSLAM and frame 213 without,
  but both created map 1 at frame 248 and neither emitted an explicit HumanSLAM
  recovery or verified map-fusion success marker. This is therefore an
  operational success but a negative/inconclusive effectiveness result.
- Recorded the complete protocol, exclusions, results and interpretation in
  `test_results/current_yolo_category/PILOT_RECOVERY_REPORT.md`.
