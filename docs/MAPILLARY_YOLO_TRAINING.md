# HumanSLAM Mapillary YOLO Segmentation Training

This is the reproducible record for adapting the HumanSLAM stable-object
segmenter from COCO-pretrained `YOLO26n-seg` to Mapillary Vistas v2.0. Dataset
outputs and checkpoints are ignored by Git; scripts, mappings and experimental
decisions are version controlled.

## Research purpose

The model supplies object identity and pixel masks to HumanSLAM. Masks preserve
spatial layout and restrict OCR to plausible stable surfaces. The hypothesis is
not that segmentation alone improves tracking, but that persistent road-scene
classes provide more invariant place-retrieval evidence when appearance or
local geometric descriptors degrade.

## Retained taxonomy

The 21 classes are building, bridge, advertisement sign, store sign,
information sign, traffic sign, traffic light, wall, fence, guard rail, tunnel,
street light, pole, bench, bike rack, CCTV camera, fire hydrant, junction box,
mailbox, parking meter and phone booth. The exact mapping from 32 retained
Mapillary v2.0 labels is defined in `tools/convert_mapillary_to_yolo_seg.py`.

## Dataset conversion

The converter preserves the official 18,000-image training and 2,000-image
validation split. It clips and normalises polygons, rejects degenerate/tiny
regions, writes YOLO segmentation labels, records class statistics and produces
random mask-overlay previews.

```bash
python3 tools/convert_mapillary_to_yolo_seg.py \
  --mapillary-root "/home/teleopbike/Documents/Mayowa/mapillary vistas dataset" \
  --output training_data/mapillary_humanslam_full \
  --preview-count 100 --seed 42 --overwrite
```

Inspect `training_data/mapillary_humanslam_full/previews/` before training and
confirm that masks follow the correct objects and labels are plausible.

## Preflight validation

A deterministic subset of 200 training and 200 validation images was used for a
one-epoch CUDA preflight. The first attempt revealed that resolved image
symlinks caused Ultralytics to search for annotations inside the original
Mapillary directory. It reported 200 backgrounds and zero instances. This run
is invalid and is retained only as an engineering finding.

The converter was corrected to retain dataset-facing `/images/` paths, allowing
Ultralytics to derive generated `/labels/` paths. The valid preflight found 200
labelled training images and 8,980 validation instances (199 labelled images and
one true background). It completed on the RTX 4070 Laptop GPU with non-zero box
and segmentation losses and no corrupt images. Accuracy is not interpreted
after one epoch; this verified CUDA, loading, masks, backpropagation, validation
and checkpoint output. Outputs are in
`runs/mapillary_humanslam/yolo26n_seg_preflight_valid_labels/`.

## Full fine-tuning protocol

The serious run uses `yolo26n-seg.pt`, image size 640, batch size 8, AMP, seed
42, deterministic training, 75 maximum epochs, early-stopping patience 15 and
checkpoints every five epochs. Dataset caching is disabled to limit RAM use.

```bash
PYTORCH_NVML_BASED_CUDA_CHECK=1 python3 tools/train_mapillary_yolo.py
```

The script records environment and parameter metadata. Ultralytics writes its
resolved arguments, epoch metrics, plots and `best.pt`/`last.pt`. Resume with:

```bash
PYTORCH_NVML_BASED_CUDA_CHECK=1 python3 tools/train_mapillary_yolo.py \
  --resume runs/mapillary_humanslam/yolo26n_seg_full_seed42_b8/weights/last.pt
```

## Evaluation and dissertation reporting

Report per-class and aggregate mask precision, recall, mAP50 and mAP50-95 on the
unchanged validation split, with instance counts for rare classes. Then compare
ORB-SLAM3 and HumanSLAM-enabled ORB-SLAM3 on identical sequences using trajectory
accuracy, recovery success, retrieval Recall@K, false proposals and median/P95
latency. Oxford RobotCar remains unseen target-domain data unless measured
domain shift justifies a separately documented adaptation experiment. Select
the PyTorch checkpoint before TensorRT export and measure exported accuracy and
latency separately.
