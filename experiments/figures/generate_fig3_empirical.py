#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ieee_style import apply_ieee_style


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in {"", None}:
        return 0.0
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an empirical density-analysis figure from suite summaries.")
    parser.add_argument(
        "--summary",
        default="experiments/generated/methodology_validation/summary_grouped.csv",
        help="Grouped summary CSV produced by experiments/collect_results.py.",
    )
    parser.add_argument(
        "--output",
        default="experiments/generated/figures/fig3_density_adaptive_empirical.png",
        help="Output image path.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_ieee_style()
    rows = _load_rows(Path(args.summary))
    rows = [row for row in rows if row.get("case_id")]
    if not rows:
        raise ValueError(f"No rows found in {args.summary}")

    labels = [str(row["case_id"]).replace("_full", "") for row in rows]
    rho_values = np.asarray([_float(row, "mean_train_rho_mean") for row in rows], dtype=np.float64)
    alpha_values = np.asarray([_float(row, "mean_train_alpha_div_mean") for row in rows], dtype=np.float64)
    acc_values = np.asarray([_float(row, "eval_test_acc_mean") for row in rows], dtype=np.float64)

    x = np.arange(len(rows))
    width = 0.36

    fig, ax_left = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    ax_right = ax_left.twinx()

    bars_rho = ax_left.bar(x - width / 2, rho_values, width=width, color="#3f72af", label=r"Mean $\rho$")
    bars_alpha = ax_left.bar(
        x + width / 2, alpha_values, width=width, color="#dd8452", label=r"Mean $\alpha_{\mathrm{div}}$"
    )
    line_acc = ax_right.plot(
        x,
        acc_values,
        color="#2a9d5b",
        marker="o",
        linewidth=1.8,
        label="Test ACC",
    )[0]

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, rotation=20, ha="right")
    ax_left.set_ylabel(r"Train-Time Density / Repulsion Statistics")
    ax_right.set_ylabel("Test ACC")
    ax_left.set_title("Empirical Density-Adaptive Analysis Across Granularity")
    ax_left.grid(True, axis="y", linestyle="--", alpha=0.35)

    handles = [bars_rho, bars_alpha, line_acc]
    labels = [handle.get_label() for handle in handles]
    ax_left.legend(handles, labels, loc="upper left")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {output_path}")


if __name__ == "__main__":
    main()
