#!/usr/bin/env python3
"""Unified entry for the rebuttal experiment runners.

It delegates to one experiment-specific runner. That runner discovers CUDA
devices: one visible GPU uses a direct single-GPU train command; multiple GPUs
use torchrun and DDP for every training workflow.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

RUNNERS = {
    "mixed-controlled": "run_cifar100_mixed_controlled.py",
    "endpoints": "run_cifar100_endpoints.py",
    "table7-fixed-ssd": "run_table7_fixed_ssd.py",
    "kmeans": "run_cifar100_mixed_kmeans_baselines.py",
    "components": "run_cifar100_mixed_component_baselines.py",
    "batch-size": "run_cifar100_mixed_batch_size.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=sorted(RUNNERS))
    args, runner_args = parser.parse_known_args()
    runner_path = Path(__file__).resolve().parent / RUNNERS[args.experiment]
    raise SystemExit(subprocess.call([sys.executable, str(runner_path), *runner_args]))


if __name__ == "__main__":
    main()
