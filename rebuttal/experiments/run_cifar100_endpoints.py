#!/usr/bin/env python3
"""Run CIFAR-100 K=20/K=100 endpoint comparisons with shared training data.

Addresses the supporting part of R1-Q1: it shows whether constant and adaptive
H-DPP remain stable at the coarse and fine endpoints. It is explicitly
auxiliary evidence, not a replacement for the K=60 mixed-granularity test.
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

EXPERIMENT_ID = "cifar100_endpoints_adaptive_semantic"
BASE_CONFIG = Path("config/train/rebuttal/cifar100_mixed_vitb16_hypersphere_dpp_config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument(
        "--repulsions",
        nargs="+",
        choices=["uniformity_010", "constant_hdpp", "adaptive_hdpp"],
        default=["uniformity_010", "constant_hdpp", "adaptive_hdpp"],
        help="Repulsion variants to run; use this to fill missing rows without rerunning completed ones.",
    )
    parser.add_argument("--python", default=default_python())
    add_cuda_arguments(parser)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = resolve_cuda_devices(args.devices)
    endpoint_overrides = {
        "coarse_k20": {
            "class_num": 20,
            "cifar100_label_mode": "coarse",
            "cifar100_allow_noncanonical_class_num": False,
        },
        "fine_k100": {
            "class_num": 100,
            "cifar100_label_mode": "fine",
            "cifar100_allow_noncanonical_class_num": False,
        },
    }
    repulsion_overrides = {
        "uniformity_010": SEMANTIC_GUIDANCE_DISABLED
        | {"repulsion_mode": "uniformity", "alpha_div_base": 0.10},
        "constant_hdpp": SEMANTIC_GUIDANCE_DISABLED
        | {"repulsion_mode": "constant_hdpp", "alpha_div_base": 0.10},
        "adaptive_hdpp": {"repulsion_mode": "adaptive_hdpp", "alpha_div_base": 0.30},
    }
    records: list[dict[str, object]] = []
    for endpoint_id, endpoint in endpoint_overrides.items():
        for repulsion_id in args.repulsions:
            repulsion = repulsion_overrides[repulsion_id]
            for seed in args.seeds:
                append_train_and_eval(
                    records,
                    experiment_id=EXPERIMENT_ID,
                    variant_id=f"{endpoint_id}_{repulsion_id}",
                    base_config=BASE_CONFIG,
                    seed=seed,
                    overrides=endpoint | repulsion,
                    python_bin=args.python,
                    device=None,
                )
    manifest = write_manifest(EXPERIMENT_ID, records)
    print(f"Wrote {len(records)} commands to {manifest}")
    run_or_print(records, args.execute, devices=devices, max_parallel=args.max_parallel)


if __name__ == "__main__":
    main()
