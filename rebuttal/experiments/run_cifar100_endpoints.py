#!/usr/bin/env python3
"""Run CIFAR-100 K=20/K=100 endpoint comparisons with shared training data.

Addresses the supporting part of R1-Q1: it shows whether constant and adaptive
H-DPP remain stable at the coarse and fine endpoints. It is explicitly
auxiliary evidence, not a replacement for the K=60 mixed-granularity test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import append_train_and_eval, default_python, run_or_print, write_manifest

EXPERIMENT_ID = "cifar100_endpoints"
BASE_CONFIG = Path("config/train/rebuttal/cifar100_mixed_vitb16_hypersphere_dpp_config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--python", default=default_python())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        "constant_hdpp": {"repulsion_mode": "constant_hdpp", "alpha_div_base": 0.10},
        "adaptive_hdpp": {"repulsion_mode": "adaptive_hdpp", "alpha_div_base": 0.30},
    }
    records: list[dict[str, object]] = []
    for endpoint_id, endpoint in endpoint_overrides.items():
        for repulsion_id, repulsion in repulsion_overrides.items():
            for seed in args.seeds:
                append_train_and_eval(
                    records,
                    experiment_id=EXPERIMENT_ID,
                    variant_id=f"{endpoint_id}_{repulsion_id}",
                    base_config=BASE_CONFIG,
                    seed=seed,
                    overrides=endpoint | repulsion,
                    python_bin=args.python,
                    device=args.device,
                )
    manifest = write_manifest(EXPERIMENT_ID, records)
    print(f"Wrote {len(records)} commands to {manifest}")
    run_or_print(records, args.execute)


if __name__ == "__main__":
    main()
