#!/usr/bin/env python3
"""Fair frozen/adapted CLIP k-means baselines.

Fits clustering centers on the protocol training split and evaluates them on the
official test split. The output prediction files can be passed directly to
``evaluate_cifar100_mixed.py`` when K=60.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Subset

try:  # Supports both `python experiments/...` and `python -m experiments...`.
    from .clustering_eval_utils import clustering_metrics, fit_spherical_kmeans
except ImportError:  # pragma: no cover - exercised by direct script execution.
    from clustering_eval_utils import clustering_metrics, fit_spherical_kmeans

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit fair frozen/adapted CLIP k-means baselines on a configured dataset."
    )
    parser.add_argument("--config", required=True, help="Training or evaluation YAML.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="MC-H checkpoint required for adapted-feature baselines.",
    )
    parser.add_argument(
        "--features",
        choices=["frozen", "adapted", "both"],
        default="both",
        help="Feature spaces to benchmark.",
    )
    parser.add_argument(
        "--clusterers",
        choices=["euclidean", "spherical", "both"],
        default="both",
        help="Clustering algorithms to benchmark.",
    )
    parser.add_argument(
        "--cluster-count",
        type=int,
        default=None,
        help="Known K. Defaults to class_num in the config.",
    )
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Clustering seed. Defaults to seed from the experiment config.",
    )
    parser.add_argument(
        "--fit-on-all-train",
        action="store_true",
        help="Fit centers on all official training samples instead of the training protocol split.",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def build_feature_loaders(config, fit_on_all_train: bool, workers: int | None):
    from hypersphere_dpp.data import (
        SingleViewDataset,
        _build_base_datasets,
        _extract_dataset_labels,
        safe_default_collate,
        split_train_indices,
    )
    from hypersphere_dpp.transforms import build_eval_transform

    train_base, test_base = _build_base_datasets(config)
    if fit_on_all_train:
        fit_base = train_base
    else:
        labels = _extract_dataset_labels(train_base)
        train_indices, _ = split_train_indices(
            size=len(train_base),
            validation_ratio=config.validation_ratio,
            seed=config.validation_seed,
            labels=labels,
        )
        fit_base = Subset(train_base, train_indices)

    effective_workers = config.workers if workers is None else workers
    loader_kwargs = {
        "batch_size": config.eval_batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": effective_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": safe_default_collate,
    }
    if effective_workers > 0:
        loader_kwargs["persistent_workers"] = True

    transform = build_eval_transform(config.image_size, config.transform_profile)
    fit_loader = DataLoader(SingleViewDataset(fit_base, transform), **loader_kwargs)
    test_loader = DataLoader(SingleViewDataset(test_base, transform), **loader_kwargs)
    return fit_loader, test_loader


@torch.inference_mode()
def collect_features(model, dataloader, device: torch.device, requested_features: set[str]):
    features = {name: [] for name in requested_features}
    labels: list[np.ndarray] = []
    model.eval()
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        teacher = model.encode_teacher(images)
        if "frozen" in requested_features:
            features["frozen"].append(teacher.cpu().numpy())
        if "adapted" in requested_features:
            student = model.encode_student(teacher_features=teacher)
            features["adapted"].append(student.cpu().numpy())
        labels.append(batch["label"].cpu().numpy())

    return (
        {name: np.concatenate(values, axis=0) for name, values in features.items()},
        np.concatenate(labels, axis=0).astype(np.int64, copy=False),
    )


def fit_predict_euclidean(train_features, test_features, k: int, args: argparse.Namespace) -> np.ndarray:
    estimator = KMeans(
        n_clusters=k,
        random_state=args.seed,
        n_init=args.n_init,
        max_iter=args.max_iter,
        algorithm="lloyd",
    )
    estimator.fit(train_features)
    return estimator.predict(test_features).astype(np.int64, copy=False)


def fit_predict_spherical(train_features, test_features, k: int, args: argparse.Namespace) -> np.ndarray:
    model = fit_spherical_kmeans(
        train_features,
        k,
        seed=args.seed,
        n_init=args.n_init,
        max_iter=args.max_iter,
    )
    return model.predict(test_features)


def main() -> None:
    args = parse_args()

    from hypersphere_dpp.checkpoint import load_checkpoint
    from hypersphere_dpp.config import load_experiment_config
    from hypersphere_dpp.model import HypersphereDPPModel
    from hypersphere_dpp.utils import resolve_device, set_seed

    if args.features in {"adapted", "both"} and not args.checkpoint:
        raise ValueError("--checkpoint is required for adapted or both feature baselines.")

    config = load_experiment_config(args.config)
    args.seed = config.seed if args.seed is None else args.seed
    if args.workers is not None:
        config.workers = args.workers
    device = resolve_device(args.device or config.device)
    set_seed(args.seed)
    k = args.cluster_count or config.class_num
    if k <= 1:
        raise ValueError("cluster-count must be greater than one.")

    requested_features = {"frozen", "adapted"} if args.features == "both" else {args.features}
    requested_clusterers = {"euclidean", "spherical"} if args.clusterers == "both" else {args.clusterers}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = HypersphereDPPModel(
        backbone_name=config.backbone_name,
        pretrained=config.pretrained,
        pretrained_checkpoint=config.pretrained_checkpoint,
        feature_dim=config.feature_dim,
        adapter_hidden_dim=config.adapter_hidden_dim,
        adapter_dropout=config.adapter_dropout,
        num_prototypes=config.class_num,
        device=device,
        representation_dim=config.representation_dim,
    ).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model=model, map_location=device)

    fit_loader, test_loader = build_feature_loaders(
        config, fit_on_all_train=args.fit_on_all_train, workers=args.workers
    )
    train_features, _ = collect_features(model, fit_loader, device, requested_features)
    test_features, test_labels = collect_features(model, test_loader, device, requested_features)

    results: dict[str, dict[str, float]] = {}
    for feature_name in sorted(requested_features):
        for clusterer_name in sorted(requested_clusterers):
            run_name = f"{feature_name}_{clusterer_name}"
            if clusterer_name == "euclidean":
                predictions = fit_predict_euclidean(
                    train_features[feature_name], test_features[feature_name], k, args
                )
            else:
                predictions = fit_predict_spherical(
                    train_features[feature_name], test_features[feature_name], k, args
                )
            np.save(output_dir / f"{run_name}_test_predictions.npy", predictions)
            results[run_name] = clustering_metrics(test_labels, predictions)
            results[run_name]["occupied_clusters"] = float(np.unique(predictions).size)
            print(f"{run_name}={results[run_name]}")

    np.save(output_dir / "test_labels.npy", test_labels)
    payload = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "dataset": config.dataset,
        "cluster_count": k,
        "fit_on_all_train": args.fit_on_all_train,
        "fit_samples": int(next(iter(train_features.values())).shape[0]),
        "test_samples": int(test_labels.size),
        "seed": args.seed,
        "n_init": args.n_init,
        "max_iter": args.max_iter,
        "results": results,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
