#!/usr/bin/env python3
"""Evaluate frozen CLIP and adapted-feature k-means baselines on CIFAR-100-Mixed.

Addresses R2-Q2: it keeps the ViT-B/16 checkpoint, training split, K=60, and
official-test evaluator fixed while separating pretrained frozen features from
the trained adapter and prototype-assignment method.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import REPO_ROOT, add_cuda_arguments, append_command, default_python, resolve_cuda_devices, run_or_print, write_manifest

EXPERIMENT_ID = "cifar100_mixed_kmeans_baselines"
BASE_CONFIG = Path("config/train/rebuttal/cifar100_mixed_vitb16_hypersphere_dpp_config.yaml")
DEFAULT_CHECKPOINT = Path("save/experiments/cifar100_mixed/adaptive/seed_42/last.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", default=default_python())
    add_cuda_arguments(parser)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = resolve_cuda_devices(args.devices)
    base = REPO_ROOT / BASE_CONFIG
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    output_root = REPO_ROOT / "save/experiments/rebuttal" / EXPERIMENT_ID / f"seed_{args.seed}"
    frozen_dir = output_root / "frozen"
    adapted_dir = output_root / "adapted"
    records: list[dict[str, object]] = []

    frozen_command = [
        args.python,
        "experiments/run_kmeans_baselines.py",
        "--config",
        str(base.relative_to(REPO_ROOT)),
        "--features",
        "frozen",
        "--clusterers",
        "both",
        "--cluster-count",
        "60",
        "--output-dir",
        str(frozen_dir.relative_to(REPO_ROOT)),
    ]
    adapted_command = [
        args.python,
        "experiments/run_kmeans_baselines.py",
        "--config",
        str(base.relative_to(REPO_ROOT)),
        "--checkpoint",
        str(checkpoint.relative_to(REPO_ROOT)),
        "--features",
        "adapted",
        "--clusterers",
        "both",
        "--cluster-count",
        "60",
        "--output-dir",
        str(adapted_dir.relative_to(REPO_ROOT)),
    ]
    append_command(
        records,
        name="fit-and-predict:frozen",
        command=frozen_command,
        job_id="frozen",
    )
    append_command(
        records,
        name="fit-and-predict:adapted",
        command=adapted_command,
        job_id="adapted",
    )

    for feature_dir, feature_name in ((frozen_dir, "frozen"), (adapted_dir, "adapted")):
        for clusterer in ("euclidean", "spherical"):
            evaluate_command = [
                args.python,
                "experiments/evaluate_cifar100_mixed.py",
                "--predictions-npy",
                str((feature_dir / f"{feature_name}_{clusterer}_test_predictions.npy").relative_to(REPO_ROOT)),
                "--cifar-root",
                str(Path.home() / "hypersphere-data" / "CIFAR100"),
                "--num-partition-pairs",
                "10",
                "--partition-seed",
                "2026",
                "--output-json",
                str((feature_dir / f"{clusterer}_mixed_metrics.json").relative_to(REPO_ROOT)),
            ]
            append_command(
                records,
                name=f"mixed-eval:{feature_name}:{clusterer}",
                command=evaluate_command,
                job_id=feature_name,
            )

    manifest = write_manifest(EXPERIMENT_ID, records)
    print(f"Wrote {len(records)} commands to {manifest.relative_to(REPO_ROOT)}")
    run_or_print(records, args.execute, devices=devices, max_parallel=args.max_parallel)


if __name__ == "__main__":
    main()
