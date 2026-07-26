#!/usr/bin/env python3
"""Run the CIFAR-100-Mixed controlled factorial experiment.

Addresses R1-Q1 and R2's central concern: it tests coarse/fine coexistence in
one K=60 task while holding DPCA and SSD fixed, and separates global
uniformity, constant H-DPP, and density-adaptive H-DPP.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    GENERATED_ROOT,
    REPO_ROOT,
    SEMANTIC_GUIDANCE_DISABLED,
    add_cuda_arguments,
    append_command,
    build_train_plan,
    default_python,
    resolve_cuda_devices,
    run_or_print,
    write_manifest,
)

EXPERIMENT_ID = "cifar100_mixed_controlled_adaptive_semantic"
BASE_CONFIG = Path("config/train/rebuttal/cifar100_mixed_vitb16_hypersphere_dpp_config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--python", default=default_python())
    add_cuda_arguments(parser)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Planning is deliberately hardware-independent; CUDA is needed only when
    # --execute launches the training and evaluation subprocesses.
    devices = resolve_cuda_devices(args.devices) if args.execute else []
    variants = {
        "uniformity_010": SEMANTIC_GUIDANCE_DISABLED
        | {"repulsion_mode": "uniformity", "alpha_div_base": 0.10},
        "constant_hdpp": SEMANTIC_GUIDANCE_DISABLED
        | {"repulsion_mode": "constant_hdpp", "alpha_div_base": 0.10},
        "constant_hdpp_matched": SEMANTIC_GUIDANCE_DISABLED
        | {
            "repulsion_mode": "constant_hdpp",
            "alpha_div_base": 0.0012,
        },
        "adaptive_hdpp": {"repulsion_mode": "adaptive_hdpp", "alpha_div_base": 0.30},
    }
    records: list[dict[str, object]] = []
    for variant_id, overrides in variants.items():
        for seed in args.seeds:
            config, config_path, run_dir, train_command = build_train_plan(
                experiment_id=EXPERIMENT_ID,
                variant_id=variant_id,
                base_config=BASE_CONFIG,
                seed=seed,
                overrides=overrides,
                python_bin=args.python,
                device=None,
            )
            job_id = f"{EXPERIMENT_ID}:{variant_id}:seed_{seed}"
            append_command(
                records,
                name=f"train:{variant_id}:seed_{seed}",
                command=train_command,
                job_id=job_id,
            )
            mixed_eval = [
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
                "--save-predictions",
            ]
            append_command(
                records,
                name=f"mixed-eval:{variant_id}:seed_{seed}",
                command=mixed_eval,
                job_id=job_id,
            )

    manifest = write_manifest(EXPERIMENT_ID, records)
    print(f"Wrote {len(records)} commands to {manifest.relative_to(REPO_ROOT)}")
    run_or_print(records, args.execute, devices=devices, max_parallel=args.max_parallel)


if __name__ == "__main__":
    main()
