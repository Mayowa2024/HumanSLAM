import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def natural_key(path):
    digits = "".join(character for character in path.stem if character.isdigit())
    return int(digits) if digits else path.name


def load_ground_truth(path):
    poses = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            poses[int(row["image_index"])] = np.array(
                [float(row["x"]), float(row["y"]), float(row["z"])],
                dtype=np.float32,
            )
    return poses


def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    offset = (256 - 224) // 2
    image = image[offset:offset + 224, offset:offset + 224]
    tensor = image.astype(np.float32) / 255.0
    tensor = (tensor - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    return np.transpose(tensor, (2, 0, 1))[None]


def extract_embeddings(model_path, images):
    session = ort.InferenceSession(
        str(model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    if "embedding" not in output_names:
        raise RuntimeError("Model does not expose the 'embedding' output")

    embeddings = []
    for index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {image_path}")
        embedding = session.run(["embedding"], {input_name: preprocess(image)})[0]
        vector = embedding.reshape(-1).astype(np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        embeddings.append(vector)
        if index == 1 or index % 100 == 0:
            print(f"Embedding progress: {index}/{len(images)}")
    return np.stack(embeddings)


def choose_threshold(positive_scores, negative_scores):
    candidates = np.unique(
        np.concatenate((positive_scores, negative_scores, [0.0, 1.0]))
    )
    best = None
    for threshold in candidates:
        true_positive = int(np.sum(positive_scores >= threshold))
        false_negative = int(np.sum(positive_scores < threshold))
        false_positive = int(np.sum(negative_scores >= threshold))
        true_negative = int(np.sum(negative_scores < threshold))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        false_positive_rate = false_positive / max(
            false_positive + true_negative, 1
        )
        result = {
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": false_positive_rate,
        }
        if best is None or (result["f1"], -result["false_positive_rate"]) > (
            best["f1"],
            -best["false_positive_rate"],
        ):
            best = result
    return best


def evaluate(embeddings, positions, frame_ids, min_separation, positive_radius,
             negative_radius, threshold):
    recall_at_1 = []
    recall_at_5 = []
    reciprocal_ranks = []
    accepted = []
    accepted_correct = []
    positive_scores = []
    negative_scores = []

    for query_index in range(len(frame_ids)):
        eligible = np.arange(0, query_index - min_separation + 1)
        if eligible.size == 0:
            continue
        distances = np.linalg.norm(
            positions[eligible] - positions[query_index], axis=1
        )
        positives = distances <= positive_radius
        if not np.any(positives):
            continue
        scores = embeddings[eligible] @ embeddings[query_index]
        order = np.argsort(-scores)
        ranked_positive = positives[order]
        first_positive = int(np.flatnonzero(ranked_positive)[0])
        recall_at_1.append(bool(ranked_positive[0]))
        recall_at_5.append(bool(np.any(ranked_positive[:5])))
        reciprocal_ranks.append(1.0 / (first_positive + 1))
        accepted.append(bool(scores[order[0]] >= threshold))
        accepted_correct.append(bool(ranked_positive[0]))
        positive_scores.extend(scores[distances <= positive_radius])
        negative_scores.extend(scores[distances >= negative_radius])

    positive_scores = np.asarray(positive_scores, dtype=np.float32)
    negative_scores = np.asarray(negative_scores, dtype=np.float32)
    calibrated = choose_threshold(positive_scores, negative_scores)
    accepted = np.asarray(accepted, dtype=bool)
    accepted_correct = np.asarray(accepted_correct, dtype=bool)
    false_matches = accepted & ~accepted_correct
    return {
        "queries_with_prior_match": len(recall_at_1),
        "recall_at_1": float(np.mean(recall_at_1)) if recall_at_1 else 0.0,
        "recall_at_5": float(np.mean(recall_at_5)) if recall_at_5 else 0.0,
        "mean_reciprocal_rank": (
            float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
        ),
        "evaluated_threshold": threshold,
        "acceptance_rate": float(np.mean(accepted)) if accepted.size else 0.0,
        "false_match_rate": (
            float(np.mean(false_matches)) if false_matches.size else 0.0
        ),
        "calibrated": calibrated,
        "positive_pairs": int(positive_scores.size),
        "negative_pairs": int(negative_scores.size),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image-dir", default="image_0")
    parser.add_argument("--ground-truth", default="pose_gt.csv")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--min-separation", type=int, default=20)
    parser.add_argument("--positive-radius", type=float, default=5.0)
    parser.add_argument("--negative-radius", type=float, default=20.0)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    all_images = sorted(
        (args.dataset / args.image_dir).glob("*"), key=natural_key
    )
    selected = all_images[::max(1, args.stride)]
    frame_ids = np.asarray([natural_key(path) for path in selected], dtype=int)
    ground_truth = load_ground_truth(args.dataset / args.ground_truth)
    keep = np.asarray([frame_id in ground_truth for frame_id in frame_ids])
    selected = [path for path, valid in zip(selected, keep) if valid]
    frame_ids = frame_ids[keep]
    positions = np.stack([ground_truth[index] for index in frame_ids])
    embeddings = extract_embeddings(args.model, selected)
    report = evaluate(
        embeddings,
        positions,
        frame_ids,
        args.min_separation,
        args.positive_radius,
        args.negative_radius,
        args.threshold,
    )
    report["dataset"] = str(args.dataset)
    report["sampled_frames"] = len(selected)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
