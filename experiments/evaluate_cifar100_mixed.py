#!/usr/bin/env python3
"""Evaluate K=60 predictions on balanced CIFAR-100 coarse/fine mixed targets."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import torch

try:  # Supports both `python experiments/...` and `python -m experiments...`.
    from .clustering_eval_utils import clustering_metrics
except ImportError:  # pragma: no cover - exercised by direct script execution.
    from clustering_eval_utils import clustering_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a single CIFAR-100 K=60 clustering against mixed coarse/fine targets."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions-npy", help="1-D test prediction array in official CIFAR-100 test order.")
    source.add_argument("--checkpoint", help="K=60 MC-H checkpoint to run on the official test split.")
    parser.add_argument("--config", help="Required with --checkpoint; must have dataset=CIFAR100 and class_num=60.")
    parser.add_argument("--cifar-root", required=True, help="CIFAR-100 dataset root containing cifar-100-python/test.")
    parser.add_argument("--num-partition-pairs", type=int, default=10)
    parser.add_argument("--partition-seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def load_cifar100_test_hierarchy(cifar_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    test_path = Path(cifar_root) / "cifar-100-python" / "test"
    if not test_path.is_file():
        raise FileNotFoundError(f"CIFAR-100 test file not found: {test_path}")
    with test_path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    fine_labels = np.asarray(payload["fine_labels"], dtype=np.int64)
    coarse_labels = np.asarray(payload["coarse_labels"], dtype=np.int64)
    if fine_labels.shape != coarse_labels.shape:
        raise ValueError("CIFAR-100 fine and coarse label arrays have inconsistent shapes.")
    return fine_labels, coarse_labels


def balanced_coarse_superclass_sets(num_pairs: int, seed: int) -> list[tuple[int, ...]]:
    if num_pairs <= 0:
        raise ValueError("num-partition-pairs must be positive.")
    generator = np.random.default_rng(seed)
    all_ids = set(range(20))
    pairs: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(pairs) < num_pairs:
        selected = tuple(sorted(generator.choice(20, size=10, replace=False).tolist()))
        complement = tuple(sorted(all_ids.difference(selected)))
        if selected in seen or complement in seen:
            continue
        pairs.append(selected)
        seen.add(selected)
        seen.add(complement)

    expanded: list[tuple[int, ...]] = []
    for selected in pairs:
        expanded.append(selected)
        expanded.append(tuple(sorted(all_ids.difference(selected))))
    return expanded


def build_mixed_labels(
    fine_labels: np.ndarray,
    coarse_labels: np.ndarray,
    coarse_superclasses: tuple[int, ...],
) -> np.ndarray:
    selected = set(coarse_superclasses)
    if len(selected) != 10 or not selected.issubset(set(range(20))):
        raise ValueError("coarse_superclasses must contain exactly ten CIFAR-100 superclass IDs.")

    fine_to_coarse: dict[int, int] = {}
    for fine_label, coarse_label in zip(fine_labels.tolist(), coarse_labels.tolist()):
        previous = fine_to_coarse.setdefault(int(fine_label), int(coarse_label))
        if previous != int(coarse_label):
            raise ValueError(f"Fine label {fine_label} maps to multiple superclass labels.")

    coarse_to_mixed = {coarse_id: index for index, coarse_id in enumerate(sorted(selected))}
    fine_to_mixed: dict[int, int] = {}
    next_id = len(coarse_to_mixed)
    for coarse_id in sorted(set(range(20)).difference(selected)):
        for fine_id in sorted(fine for fine, parent in fine_to_coarse.items() if parent == coarse_id):
            fine_to_mixed[fine_id] = next_id
            next_id += 1
    if next_id != 60:
        raise AssertionError(f"Expected 60 mixed targets, built {next_id}.")

    mixed = np.empty_like(fine_labels)
    for index, (fine_label, coarse_label) in enumerate(zip(fine_labels, coarse_labels)):
        mixed[index] = (
            coarse_to_mixed[int(coarse_label)]
            if int(coarse_label) in selected
            else fine_to_mixed[int(fine_label)]
        )
    return mixed


def fragmentation_at_coverage(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    coverage: float = 0.9,
) -> float:
    values: list[int] = []
    for true_id in np.unique(true_labels):
        cluster_counts = np.unique(predictions[true_labels == true_id], return_counts=True)[1]
        sorted_counts = np.sort(cluster_counts)[::-1]
        needed = int(np.searchsorted(np.cumsum(sorted_counts), coverage * sorted_counts.sum()) + 1)
        values.append(needed)
    return float(np.mean(values))


def fine_merge_rate(true_labels: np.ndarray, predictions: np.ndarray) -> float:
    total_pairs = 0
    same_class_pairs = 0
    for predicted_id in np.unique(predictions):
        members = true_labels[predictions == predicted_id]
        total_pairs += members.size * (members.size - 1) // 2
        _, counts = np.unique(members, return_counts=True)
        same_class_pairs += int(np.sum(counts * (counts - 1) // 2))
    return 0.0 if total_pairs == 0 else float((total_pairs - same_class_pairs) / total_pairs)


@torch.inference_mode()
def predict_checkpoint(config, checkpoint: str, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    from hypersphere_dpp.checkpoint import load_checkpoint
    from hypersphere_dpp.data import build_experiment_dataloaders
    from hypersphere_dpp.model import HypersphereDPPModel

    if config.dataset != "CIFAR100":
        raise ValueError("CIFAR-100-Mixed evaluation requires dataset=CIFAR100.")
    if config.class_num != 60:
        raise ValueError("CIFAR-100-Mixed checkpoint evaluation requires class_num=60.")
    datasets = build_experiment_dataloaders(config)
    model = HypersphereDPPModel(
        backbone_name=config.backbone_name,
        pretrained=config.pretrained,
        pretrained_checkpoint=config.pretrained_checkpoint,
        feature_dim=config.feature_dim,
        adapter_hidden_dim=config.adapter_hidden_dim,
        adapter_dropout=config.adapter_dropout,
        num_prototypes=config.class_num,
        device=device,
    ).to(device)
    load_checkpoint(checkpoint, model=model, map_location=device)
    model.eval()
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in datasets.test_loader:
        images = batch["image"].to(device, non_blocking=True)
        teacher = model.encode_teacher(images)
        student = model.encode_student(teacher_features=teacher)
        predictions.append(model.prototype_logits(student).argmax(dim=1).cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
    return (
        np.concatenate(predictions).astype(np.int64, copy=False),
        np.concatenate(labels).astype(np.int64, copy=False),
    )


def summarize(records: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    metric_paths = {
        "global.acc_1to1": ("global", "acc_1to1"),
        "global.nmi": ("global", "nmi"),
        "global.ari": ("global", "ari"),
        "coarse.ari": ("coarse", "ari"),
        "coarse.fragmentation_at_90": ("coarse", "fragmentation_at_90"),
        "fine.ari": ("fine", "ari"),
        "fine.merge_rate": ("fine", "merge_rate"),
    }
    summary: dict[str, dict[str, float]] = {}
    for name, (section, key) in metric_paths.items():
        values = np.asarray([record[section][key] for record in records], dtype=float)
        summary[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    return summary


def main() -> None:
    args = parse_args()
    if args.checkpoint and not args.config:
        raise ValueError("--config is required with --checkpoint.")

    fine_labels, coarse_labels = load_cifar100_test_hierarchy(args.cifar_root)
    if args.predictions_npy:
        predictions = np.load(args.predictions_npy)
        observed_fine_labels = fine_labels
        source = {"predictions_npy": str(Path(args.predictions_npy).resolve())}
    else:
        from hypersphere_dpp.config import load_experiment_config
        from hypersphere_dpp.utils import resolve_device, set_seed

        config = load_experiment_config(args.config)
        if args.workers is not None:
            config.workers = args.workers
        device = resolve_device(args.device or config.device)
        set_seed(config.seed)
        predictions, observed_fine_labels = predict_checkpoint(config, args.checkpoint, device)
        source = {
            "config": str(Path(args.config).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
        }

    predictions = np.asarray(predictions)
    if predictions.ndim != 1 or predictions.size != fine_labels.size:
        raise ValueError(
            f"Expected {fine_labels.size} 1-D test predictions, got shape {predictions.shape}."
        )
    if not np.array_equal(observed_fine_labels, fine_labels):
        raise ValueError("Prediction label order does not match the official CIFAR-100 test order.")

    records: list[dict[str, object]] = []
    for partition_index, coarse_set in enumerate(
        balanced_coarse_superclass_sets(args.num_partition_pairs, args.partition_seed)
    ):
        mixed_labels = build_mixed_labels(fine_labels, coarse_labels, coarse_set)
        coarse_mask = np.isin(coarse_labels, list(coarse_set))
        fine_mask = ~coarse_mask
        coarse_metrics = clustering_metrics(coarse_labels[coarse_mask], predictions[coarse_mask])
        coarse_metrics["fragmentation_at_90"] = fragmentation_at_coverage(
            coarse_labels[coarse_mask], predictions[coarse_mask]
        )
        fine_metrics = clustering_metrics(fine_labels[fine_mask], predictions[fine_mask])
        fine_metrics["merge_rate"] = fine_merge_rate(
            fine_labels[fine_mask], predictions[fine_mask]
        )
        records.append(
            {
                "partition_index": partition_index,
                "coarse_superclasses": list(coarse_set),
                "global": clustering_metrics(mixed_labels, predictions),
                "coarse": coarse_metrics,
                "fine": fine_metrics,
            }
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "protocol": {
            "dataset": "CIFAR-100 official test split",
            "mixed_cluster_count": 60,
            "num_partition_pairs": args.num_partition_pairs,
            "num_partitions": len(records),
            "partition_seed": args.partition_seed,
            "acc": "rectangular one-to-one Hungarian accuracy",
            "coarse_metric": "superclass labels on coarse-designated samples",
            "fine_metric": "subclass labels on fine-designated samples",
        },
        "occupied_prediction_clusters": int(np.unique(predictions).size),
        "summary": summarize(records),
        "partitions": records,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.save_predictions:
        np.save(output_path.with_name("test_predictions.npy"), predictions)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
