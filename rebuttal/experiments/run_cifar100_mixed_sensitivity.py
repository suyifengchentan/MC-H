#!/usr/bin/env python3
"""Run minimal CIFAR-100-Mixed batch-size and feature-dimension controls.

Addresses R2-Q6 with a fixed K=60 protocol. A batch control changes only the
batch size at the base projection dimension; a feature control changes only
the projection dimension at the base batch size.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    GENERATED_ROOT,
    REPO_ROOT,
    add_cuda_arguments,
    append_command,
    append_train_and_eval,
    default_python,
    resolve_cuda_devices,
    run_or_print,
    write_manifest,
)

EXPERIMENT_ID = "cifar100_mixed_sensitivity_adaptive_semantic"
BASE_CONFIG = Path("config/train/rebuttal/cifar100_mixed_vitb16_hypersphere_dpp_config.yaml")
BASE_BATCH_SIZE = 192
BASE_FEATURE_DIM = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=["batch", "feature"], required=True)
    parser.add_argument("--values", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--python", default=default_python())
    add_cuda_arguments(parser)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(value <= 1 for value in args.values):
        raise ValueError("Every controlled value must be greater than one.")
    devices = resolve_cuda_devices(args.devices)
    records: list[dict[str, object]] = []
    for value in args.values:
        if args.control == "batch":
            variant_id = f"adaptive_hdpp_batch_{value}_dim_{BASE_FEATURE_DIM}"
            overrides = {"batch_size": value, "feature_dim": BASE_FEATURE_DIM}
        else:
            variant_id = f"adaptive_hdpp_batch_{BASE_BATCH_SIZE}_dim_{value}"
            overrides = {"batch_size": BASE_BATCH_SIZE, "feature_dim": value}
        for seed in args.seeds:
            config, run_dir = append_train_and_eval(
                records,
                experiment_id=EXPERIMENT_ID,
                variant_id=variant_id,
                base_config=BASE_CONFIG,
                seed=seed,
                overrides=overrides,
                python_bin=args.python,
                device=None,
            )
            config_path = GENERATED_ROOT / EXPERIMENT_ID / variant_id / f"seed_{seed}_train.yaml"
            append_command(
                records,
                name=f"mixed-eval:{variant_id}:seed_{seed}",
                command=[
                    args.python,
                    "experiments/evaluate_cifar100_mixed.py",
                    "--config",
                    str(config_path.relative_to(REPO_ROOT)),
                    "--checkpoint",
                    str((run_dir / "last.pt").relative_to(REPO_ROOT)),
                    "--cifar-root",
                    str(config["dataset_dir"]),
                    "--num-partition-pairs",
                    "10",
                    "--partition-seed",
                    "2026",
                    "--output-json",
                    str((run_dir / "mixed_metrics.json").relative_to(REPO_ROOT)),
                ],
                job_id=f"{EXPERIMENT_ID}:{variant_id}:seed_{seed}",
            )
    manifest = write_manifest(EXPERIMENT_ID, records)
    print(f"Wrote {len(records)} commands to {manifest.relative_to(REPO_ROOT)}")
    run_or_print(records, args.execute, devices=devices, max_parallel=args.max_parallel)


if __name__ == "__main__":
    main()
