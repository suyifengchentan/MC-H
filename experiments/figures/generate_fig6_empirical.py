#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ieee_style import apply_ieee_style


def _load_metrics(path: Path) -> tuple[list[dict], list[dict]]:
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split") == "train":
                train_rows.append(row)
            elif row.get("split") == "val":
                val_rows.append(row)
    return train_rows, val_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a training-dynamics figure from metrics.jsonl.")
    parser.add_argument("--metrics", required=True, help="Path to metrics.jsonl.")
    parser.add_argument(
        "--output",
        default="experiments/generated/figures/fig6_training_dynamics_empirical.png",
        help="Output image path.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_ieee_style()
    train_rows, val_rows = _load_metrics(Path(args.metrics))
    if not train_rows:
        raise ValueError(f"No train rows found in {args.metrics}")

    train_epochs = np.asarray([int(row["epoch"]) for row in train_rows], dtype=np.int64)
    alpha_values = np.asarray([float(row.get("alpha_div", 0.0)) for row in train_rows], dtype=np.float64)
    rho_values = np.asarray([float(row.get("rho", 0.0)) for row in train_rows], dtype=np.float64)
    estimated_k_values = np.asarray([float(row.get("estimated_k", 0.0)) for row in train_rows], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=True, constrained_layout=True)

    axes[0].plot(train_epochs, alpha_values, color="#1f77b4", linewidth=1.8)
    axes[0].set_ylabel(r"$\alpha_{\mathrm{div}}$")
    axes[0].set_title("Empirical Training Dynamics")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].plot(train_epochs, rho_values, color="#ff7f0e", linewidth=1.8)
    axes[1].set_ylabel(r"$\rho$")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    axes[2].plot(train_epochs, estimated_k_values, color="#2a9d5b", linewidth=1.8, label=r"Train $\hat{K}$")
    if val_rows:
        val_epochs = np.asarray([int(row["epoch"]) for row in val_rows], dtype=np.int64)
        selection_scores = np.asarray([float(row.get("selection_score", 0.0)) for row in val_rows], dtype=np.float64)
        ax2b = axes[2].twinx()
        ax2b.plot(val_epochs, selection_scores, color="#c43e3e", linestyle="--", linewidth=1.4, label="Val score")
        ax2b.set_ylabel("Val score")
    axes[2].set_ylabel(r"$\hat{K}$")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, linestyle="--", alpha=0.35)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {output_path}")


if __name__ == "__main__":
    main()
