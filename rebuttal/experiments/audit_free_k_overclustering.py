#!/usr/bin/env python3
"""Audit cardinality error and class fragmentation for a precomputed free-K output.

Addresses R1-Q4 and R2-Q3 without leaking labels into fitting or checkpoint
selection. Given predictions in official CIFAR-100 test order, it reports K
error, rectangular Hungarian ACC, per-class splitting, cluster purity, and the
coarse/fine hierarchy evidence needed before claiming semantic substructure.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.clustering_eval_utils import clustering_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-npy", type=Path, required=True)
    parser.add_argument("--cifar-root", type=Path, required=True)
    parser.add_argument("--label-level", choices=["coarse", "fine"], required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_labels(cifar_root: Path) -> tuple[np.ndarray, np.ndarray]:
    test_path = cifar_root / "cifar-100-python" / "test"
    with test_path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    return (
        np.asarray(payload["coarse_labels"], dtype=np.int64),
        np.asarray(payload["fine_labels"], dtype=np.int64),
    )


def cluster_records(labels: np.ndarray, predictions: np.ndarray) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for cluster_id in np.unique(predictions):
        members = labels[predictions == cluster_id]
        label_ids, counts = np.unique(members, return_counts=True)
        majority = int(counts.max())
        records.append(
            {
                "cluster_id": int(cluster_id),
                "size": int(members.size),
                "purity": float(majority / members.size),
                "majority_label": int(label_ids[counts.argmax()]),
                "label_count": int(label_ids.size),
            }
        )
    return records


def class_records(labels: np.ndarray, predictions: np.ndarray) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for label_id in np.unique(labels):
        member_predictions = predictions[labels == label_id]
        cluster_ids, counts = np.unique(member_predictions, return_counts=True)
        records.append(
            {
                "label_id": int(label_id),
                "sample_count": int(member_predictions.size),
                "predicted_cluster_count": int(cluster_ids.size),
                "largest_cluster_share": float(counts.max() / member_predictions.size),
                "predicted_cluster_sizes": {str(int(cluster)): int(size) for cluster, size in zip(cluster_ids, counts)},
            }
        )
    return records


def main() -> None:
    args = parse_args()
    coarse_labels, fine_labels = load_labels(args.cifar_root)
    labels = coarse_labels if args.label_level == "coarse" else fine_labels
    predictions = np.asarray(np.load(args.predictions_npy), dtype=np.int64)
    if predictions.ndim != 1 or predictions.size != labels.size:
        raise ValueError(f"Expected {labels.size} predictions, got shape {predictions.shape}.")

    k_hat = int(np.unique(predictions).size)
    k_gt = int(np.unique(labels).size)
    payload = {
        "protocol": {
            "split": "CIFAR-100 official test",
            "label_level": args.label_level,
            "acc": "rectangular one-to-one Hungarian accuracy",
        },
        "source": str(args.predictions_npy.resolve()),
        "k_gt": k_gt,
        "k_hat": k_hat,
        "absolute_k_error": abs(k_hat - k_gt),
        "relative_k_error": abs(k_hat - k_gt) / k_gt,
        "metrics": clustering_metrics(labels, predictions),
        "predicted_clusters": cluster_records(labels, predictions),
        "ground_truth_classes": class_records(labels, predictions),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("k_gt", "k_hat", "absolute_k_error", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
