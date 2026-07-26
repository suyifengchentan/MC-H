#!/usr/bin/env python3
"""Estimate an unknown cluster count and audit post-hoc semantic splits.

The estimator only consumes cached frozen image features. Ground-truth labels
are loaded after clustering solely to report K and to audit whether unmatched
clusters are coherent semantic subdivisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.linalg import eigsh
from sklearn.cluster import MiniBatchKMeans

from hypersphere_dpp.config import ExperimentConfig
from hypersphere_dpp.data import build_experiment_dataloaders
from hypersphere_dpp.semantic import load_class_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label-blind kNN eigengap K estimation with post-hoc semantic auditing."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--max-k", type=int, required=True)
    parser.add_argument("--sample-size", type=int, default=4096)
    parser.add_argument("--knn", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-extra-samples", type=int, default=25)
    parser.add_argument("--purity-threshold", type=float, default=0.5)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def load_cached_features_and_labels(dataset: str, dataset_dir: str) -> tuple[np.ndarray, np.ndarray]:
    # class_num is intentionally not used by the estimator. It only satisfies
    # the data-loader configuration contract while cache features are loaded.
    config = ExperimentConfig(
        dataset=dataset,
        dataset_dir=dataset_dir,
        class_num=2,
        feature_dim=512,
        backbone_name="ViT-B-16",
        image_size=224,
        cached_teacher_views_enabled=True,
        cached_teacher_num_views=8,
        cached_teacher_dtype="float16",
        validation_ratio=0.1,
        workers=0,
    )
    bundle = build_experiment_dataloaders(config)
    cached_dataset = bundle.train_loader.dataset
    if not hasattr(cached_dataset, "teacher_cache") or not hasattr(cached_dataset, "base_dataset"):
        raise RuntimeError("Expected cached teacher-feature train dataset.")
    cache = cached_dataset.teacher_cache
    targets = getattr(cached_dataset.base_dataset, "targets", None)
    if targets is None:
        raise RuntimeError("The training dataset must expose targets for post-hoc auditing.")
    labels = np.asarray(targets, dtype=np.int64)
    if len(cache) != labels.size:
        raise RuntimeError("Teacher cache and training labels have inconsistent lengths.")

    features = np.empty((len(cache), cache.feature_dim), dtype=np.float32)
    for start in range(0, len(cache), 2048):
        stop = min(start + 2048, len(cache))
        candidates = np.asarray(cache.features[start:stop], dtype=np.float32)
        averaged = candidates.mean(axis=1)
        norms = np.linalg.norm(averaged, axis=1, keepdims=True).clip(min=1e-12)
        features[start:stop] = averaged / norms
    return features, labels


def estimate_k_eigengap(
    features: np.ndarray,
    sample_indices: np.ndarray,
    max_k: int,
    knn: int,
    device: torch.device,
) -> tuple[int, np.ndarray, np.ndarray]:
    if max_k < 2:
        raise ValueError("max_k must be at least 2.")
    sample = torch.from_numpy(features[sample_indices]).to(device=device, dtype=torch.float32)
    sample = torch.nn.functional.normalize(sample, dim=1)
    neighbor_count = min(knn + 1, sample.size(0))
    similarities = sample @ sample.t()
    values, indices = similarities.topk(neighbor_count, dim=1)
    values, indices = values[:, 1:].cpu().numpy(), indices[:, 1:].cpu().numpy()
    distances = np.clip(1.0 - values, 0.0, None)
    scale = float(np.median(distances[distances > 0])) if np.any(distances > 0) else 1.0
    weights = np.exp(-distances / max(scale, 1e-6)).astype(np.float64, copy=False)

    n = sample_indices.size
    rows = np.repeat(np.arange(n), indices.shape[1])
    cols = indices.reshape(-1)
    graph = coo_matrix((weights.reshape(-1), (rows, cols)), shape=(n, n)).maximum(
        coo_matrix((weights.reshape(-1), (cols, rows)), shape=(n, n))
    ).tocsr()
    degree = np.asarray(graph.sum(axis=1)).reshape(-1).clip(min=1e-12)
    inv_sqrt_degree = diags(1.0 / np.sqrt(degree))
    laplacian = eye(n, format="csr") - inv_sqrt_degree @ graph @ inv_sqrt_degree
    requested = min(max_k + 1, n - 1)
    eigenvalues = np.sort(eigsh(laplacian, k=requested, which="SM", tol=1e-3)[0])
    candidates = np.arange(2, min(max_k, eigenvalues.size - 1) + 1)
    if candidates.size == 0:
        raise RuntimeError("Too few eigenvalues to estimate K.")
    gaps = eigenvalues[candidates] - eigenvalues[candidates - 1]
    estimated_k = int(candidates[int(gaps.argmax())])
    return estimated_k, eigenvalues, gaps


def semantic_audit(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    min_extra_samples: int,
    purity_threshold: float,
) -> dict[str, object]:
    true_ids = np.unique(labels)
    predicted_ids = np.unique(predictions)
    contingency = np.zeros((true_ids.size, predicted_ids.size), dtype=np.int64)
    true_columns = {int(value): index for index, value in enumerate(true_ids)}
    predicted_columns = {int(value): index for index, value in enumerate(predicted_ids)}
    np.add.at(
        contingency,
        (
            np.asarray([true_columns[int(value)] for value in labels]),
            np.asarray([predicted_columns[int(value)] for value in predictions]),
        ),
        1,
    )
    _, matched_columns = linear_sum_assignment(-contingency)
    matched = {int(predicted_ids[column]) for column in matched_columns}
    extras = sorted(set(int(value) for value in predicted_ids) - matched)
    records = []
    for cluster_id in extras:
        members = labels[predictions == cluster_id]
        values, counts = np.unique(members, return_counts=True)
        dominant_index = int(counts.argmax())
        dominant_label = int(values[dominant_index])
        size = int(members.size)
        purity = float(counts[dominant_index] / size)
        records.append(
            {
                "cluster": int(cluster_id),
                "size": size,
                "dominant_class": class_names[dominant_label]
                if dominant_label < len(class_names)
                else str(dominant_label),
                "purity": purity,
                "meaningful": bool(size >= min_extra_samples and purity >= purity_threshold),
            }
        )
    records.sort(key=lambda record: (-record["size"], -record["purity"]))
    return {
        "active_clusters": int(predicted_ids.size),
        "unmatched_clusters": len(records),
        "meaningful_new_splits": sum(int(record["meaningful"]) for record in records),
        "criterion": {
            "minimum_samples": min_extra_samples,
            "dominant_class_purity": purity_threshold,
        },
        "extra_cluster_examples": records[:5],
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the configured kNN feature graph.")
    features, labels = load_cached_features_and_labels(args.dataset, args.dataset_dir)
    generator = np.random.default_rng(args.seed)
    sample_size = min(args.sample_size, features.shape[0])
    sample_indices = np.sort(generator.choice(features.shape[0], size=sample_size, replace=False))
    estimated_k, eigenvalues, gaps = estimate_k_eigengap(
        features,
        sample_indices,
        max_k=min(args.max_k, sample_size - 2),
        knn=args.knn,
        device=torch.device(args.device),
    )
    kmeans = MiniBatchKMeans(
        n_clusters=estimated_k,
        batch_size=min(2048, features.shape[0]),
        n_init=5,
        max_iter=200,
        random_state=args.seed,
    )
    predictions = kmeans.fit_predict(features)
    class_names = load_class_texts(args.dataset, args.dataset_dir)
    audit = semantic_audit(
        labels,
        predictions,
        class_names,
        args.min_extra_samples,
        args.purity_threshold,
    )
    payload = {
        "dataset": args.dataset,
        "feature_source": "mean of eight cached frozen ViT-B/16 teacher views",
        "estimator": {
            "name": "kNN normalized-Laplacian eigengap",
            "sample_size": int(sample_size),
            "knn": int(args.knn),
            "candidate_k_range": [2, int(min(args.max_k, sample_size - 2))],
            "estimated_k": estimated_k,
            "selected_eigengap": float(gaps[estimated_k - 2]),
            "first_eigenvalues": [float(value) for value in eigenvalues[: min(12, eigenvalues.size)]],
        },
        "posthoc_ground_truth_k": int(np.unique(labels).size),
        "semantic_audit": audit,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
