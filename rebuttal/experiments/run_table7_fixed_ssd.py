#!/usr/bin/env python3
"""Re-run the Table 7 geometry ablation with DPCA and SSD held fixed.

Addresses R1-Q2 and R2's Table 7 objection. Every row disables text-derived
semantic signals and keeps the same DPCA/SSD setup, isolating no repulsion,
uniformity, constant H-DPP, and adaptive H-DPP under the true known K.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import add_cuda_arguments, append_train_and_eval, default_python, resolve_cuda_devices, run_or_print, write_manifest

EXPERIMENT_ID = "table7_fixed_ssd"
DATASETS = {
    "cifar10": Path("config/train/coarse_grained/cifar10_vitb16_hypersphere_dpp_config.yaml"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--python", default=default_python())
    add_cuda_arguments(parser)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = resolve_cuda_devices(args.devices)
    variants = {
        "dpca_ssd": {"repulsion_mode": "none", "alpha_div_base": 0.0},
        "uniformity_010": {"repulsion_mode": "uniformity", "alpha_div_base": 0.10},
        "constant_hdpp": {"repulsion_mode": "constant_hdpp", "alpha_div_base": 0.10},
        "adaptive_hdpp": {"repulsion_mode": "adaptive_hdpp"},
    }
    visual_only = {
        "semantic_prototype_enabled": False,
        "lambda_semantic": 0.0,
        "lambda_semantic_retrieval": 0.0,
        "lambda_prototype_anchor": 0.0,
        "best_checkpoint_patience": 0,
    }
    records: list[dict[str, object]] = []
    for dataset_id, base_config in DATASETS.items():
        for variant_id, variant in variants.items():
            for seed in args.seeds:
                append_train_and_eval(
                    records,
                    experiment_id=EXPERIMENT_ID,
                    variant_id=f"{dataset_id}_{variant_id}",
                    base_config=base_config,
                    seed=seed,
                    overrides=visual_only | variant,
                    python_bin=args.python,
                    device=None,
                )
    manifest = write_manifest(EXPERIMENT_ID, records)
    print(f"Wrote {len(records)} commands to {manifest}")
    run_or_print(records, args.execute, devices=devices, max_parallel=args.max_parallel)


if __name__ == "__main__":
    main()
