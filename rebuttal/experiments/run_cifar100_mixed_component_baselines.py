#!/usr/bin/env python3
"""Run the instance-only, DPCA-only, and DPCA+SSD component baselines.

Addresses R2-Q2 and R1's observation that DPCA/SSD may explain most gains.
The script fixes the K=60 data protocol and semantic prior while removing
components one at a time; the instance-only adapter is then evaluated with
spherical k-means rather than the prototype head.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import GENERATED_ROOT, REPO_ROOT, add_cuda_arguments, append_command, append_train_and_eval, default_python, resolve_cuda_devices, run_or_print, write_manifest

EXPERIMENT_ID = "cifar100_mixed_component_baselines_semantic"
BASE_CONFIG = Path("config/train/rebuttal/cifar100_mixed_vitb16_hypersphere_dpp_config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--python", default=default_python())
    add_cuda_arguments(parser)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = resolve_cuda_devices(args.devices)
    variants = {
        "instance_only": {
            "lambda_cluster": 0.0,
            "lambda_subspace": 0.0,
            "repulsion_mode": "none",
            "alpha_div_base": 0.0,
        },
        "dpca_only": {
            "lambda_subspace": 0.0,
            "repulsion_mode": "none",
            "alpha_div_base": 0.0,
        },
        "dpca_ssd": {"repulsion_mode": "none", "alpha_div_base": 0.0},
    }
    records: list[dict[str, object]] = []
    for variant_id, overrides in variants.items():
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
            mixed_eval = [
                args.python,
                "experiments/evaluate_cifar100_mixed.py",
                "--config",
                str(config_path.relative_to(REPO_ROOT)),
                "--checkpoint",
                str((run_dir / "last.pt").relative_to(REPO_ROOT)),
                "--cifar-root",
                str(config["dataset_dir"]),
                "--output-json",
                str((run_dir / "mixed_metrics.json").relative_to(REPO_ROOT)),
            ]
            job_id = f"{EXPERIMENT_ID}:{variant_id}:seed_{seed}"
            append_command(
                records,
                name=f"mixed-eval:{variant_id}:seed_{seed}",
                command=mixed_eval,
                job_id=job_id,
            )

            if variant_id == "instance_only":
                kmeans_dir = run_dir / "adapted_spherical_kmeans"
                kmeans = [
                    args.python,
                    "experiments/run_kmeans_baselines.py",
                    "--config",
                    str(config_path.relative_to(REPO_ROOT)),
                    "--checkpoint",
                    str((run_dir / "last.pt").relative_to(REPO_ROOT)),
                    "--features",
                    "adapted",
                    "--clusterers",
                    "spherical",
                    "--cluster-count",
                    "60",
                    "--output-dir",
                    str(kmeans_dir.relative_to(REPO_ROOT)),
                ]
                append_command(
                    records,
                    name=f"kmeans:instance-only:seed_{seed}",
                    command=kmeans,
                    job_id=job_id,
                )
                kmeans_eval = [
                    args.python,
                    "experiments/evaluate_cifar100_mixed.py",
                    "--predictions-npy",
                    str((kmeans_dir / "adapted_spherical_test_predictions.npy").relative_to(REPO_ROOT)),
                    "--cifar-root",
                    str(config["dataset_dir"]),
                    "--output-json",
                    str((kmeans_dir / "mixed_metrics.json").relative_to(REPO_ROOT)),
                ]
                append_command(
                    records,
                    name=f"mixed-eval:instance-only-kmeans:seed_{seed}",
                    command=kmeans_eval,
                    job_id=job_id,
                )
    manifest = write_manifest(EXPERIMENT_ID, records)
    print(f"Wrote {len(records)} commands to {manifest.relative_to(REPO_ROOT)}")
    run_or_print(records, args.execute, devices=devices, max_parallel=args.max_parallel)


if __name__ == "__main__":
    main()
