#!/usr/bin/env python3
"""Re-run the Table 7 geometry ablation with DPCA and SSD held fixed.

Addresses R1-Q2 and R2's Table 7 objection. Every row keeps DPCA and SSD fixed,
while semantic guidance is enabled only for adaptive H-DPP. The comparison
isolates no repulsion, uniformity, constant H-DPP, and adaptive H-DPP under K.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    SEMANTIC_GUIDANCE_DISABLED,
    add_cuda_arguments,
    append_train_and_eval,
    default_python,
    resolve_cuda_devices,
    run_or_print,
    write_manifest,
)

EXPERIMENT_ID = "table7_fixed_ssd_adaptive_semantic"
DATASETS = {
    "cifar10": Path("config/train/coarse_grained/cifar10_vitb16_hypersphere_dpp_config.yaml"),
    "image50": Path("config/train/medium_grained/image50_vitb16_hypersphere_dpp_config.yaml"),
    "oxford_flowers": Path("config/train/fine_grained/oxford_flowers_vitb16_hypersphere_dpp_config.yaml"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
        help="Dataset rows to run; use this to fill missing Table 7 entries.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["dpca_ssd", "uniformity_010", "constant_hdpp", "adaptive_hdpp"],
        default=["dpca_ssd", "uniformity_010", "constant_hdpp", "adaptive_hdpp"],
        help="Ablation rows to run; use this to fill missing entries without rerunning completed ones.",
    )
    parser.add_argument("--python", default=default_python())
    add_cuda_arguments(parser)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = resolve_cuda_devices(args.devices)
    variants = {
        "dpca_ssd": SEMANTIC_GUIDANCE_DISABLED
        | {"repulsion_mode": "none", "alpha_div_base": 0.0},
        "uniformity_010": SEMANTIC_GUIDANCE_DISABLED
        | {"repulsion_mode": "uniformity", "alpha_div_base": 0.10},
        "constant_hdpp": SEMANTIC_GUIDANCE_DISABLED
        | {"repulsion_mode": "constant_hdpp", "alpha_div_base": 0.10},
        "adaptive_hdpp": {"repulsion_mode": "adaptive_hdpp"},
    }
    shared_overrides = {
        "best_checkpoint_patience": 0,
    }
    records: list[dict[str, object]] = []
    for dataset_id in args.datasets:
        base_config = DATASETS[dataset_id]
        for variant_id in args.variants:
            variant = variants[variant_id]
            for seed in args.seeds:
                append_train_and_eval(
                    records,
                    experiment_id=EXPERIMENT_ID,
                    variant_id=f"{dataset_id}_{variant_id}",
                    base_config=base_config,
                    seed=seed,
                    overrides=shared_overrides | variant,
                    python_bin=args.python,
                    device=None,
                )
    manifest = write_manifest(EXPERIMENT_ID, records)
    print(f"Wrote {len(records)} commands to {manifest}")
    run_or_print(records, args.execute, devices=devices, max_parallel=args.max_parallel)


if __name__ == "__main__":
    main()
