#!/usr/bin/env python3
"""Run the CIFAR-100-Mixed batch-size sensitivity experiment.

Addresses the meta-review reproducibility concern and the stated limitation
that H-DPP uses a batch kernel. It measures whether the K=60 conclusion holds
for batch sizes 128, 256, and 512 with the same adaptive protocol.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import GENERATED_ROOT, REPO_ROOT, add_cuda_arguments, append_command, append_train_and_eval, default_python, resolve_cuda_devices, run_or_print, write_manifest

EXPERIMENT_ID = "cifar100_mixed_batch_size_adaptive_semantic"
BASE_CONFIG = Path("config/train/rebuttal/cifar100_mixed_vitb16_hypersphere_dpp_config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--python", default=default_python())
    add_cuda_arguments(parser)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = resolve_cuda_devices(args.devices)
    records: list[dict[str, object]] = []
    for batch_size in args.batch_sizes:
        if batch_size <= 1:
            raise ValueError("batch size must be greater than one")
        for seed in args.seeds:
            variant_id = f"adaptive_hdpp_batch_{batch_size}"
            config, run_dir = append_train_and_eval(
                records,
                experiment_id=EXPERIMENT_ID,
                variant_id=variant_id,
                base_config=BASE_CONFIG,
                seed=seed,
                overrides={"batch_size": batch_size, "repulsion_mode": "adaptive_hdpp"},
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
            append_command(
                records,
                name=f"mixed-eval:{variant_id}:seed_{seed}",
                command=mixed_eval,
                job_id=f"{EXPERIMENT_ID}:{variant_id}:seed_{seed}",
            )
    manifest = write_manifest(EXPERIMENT_ID, records)
    print(f"Wrote {len(records)} commands to {manifest}")
    run_or_print(records, args.execute, devices=devices, max_parallel=args.max_parallel)


if __name__ == "__main__":
    main()
