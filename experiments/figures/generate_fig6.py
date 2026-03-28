#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ieee_style import apply_ieee_style


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 6: Curriculum Scheduling and Training Dynamics."
    )
    parser.add_argument(
        "--output",
        default="experiments/generated/figures/fig6_curriculum_scheduling.png",
        help="Output image path.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Total training epochs (T).")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_ieee_style()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t = np.arange(args.epochs + 1, dtype=np.float64)
    t_norm = t / max(args.epochs, 1)

    # Simulated density signal sg(rho): mild increase and then saturation.
    rho = 0.28 + 0.42 * _sigmoid((t_norm - 0.40) / 0.10)
    alpha_base = 1.6
    gamma_t = _sigmoid((t_norm - 0.38) / 0.08)
    alpha_div = gamma_t * np.minimum(alpha_base * rho, 1.0)

    tau_max = 0.85
    tau_min = 0.12
    tau = tau_min + (tau_max - tau_min) * np.exp(-3.0 * t_norm)

    phase_split_1 = int(0.35 * args.epochs)
    phase_split_2 = int(0.55 * args.epochs)

    # IEEE two-column friendly width.
    fig, ax_left = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    ax_right = ax_left.twinx()

    ax_left.set_facecolor("white")
    ax_left.grid(True, which="major")

    ax_left.axvspan(0, phase_split_1, color="#7BC47F", alpha=0.18, label="Cluster formation phase")
    ax_left.axvspan(
        phase_split_2,
        args.epochs,
        color="#8E6BBE",
        alpha=0.16,
        label="DPP expansion phase",
    )

    alpha_line = ax_left.plot(
        t,
        alpha_div,
        color="#1f77b4",
        linewidth=1.9,
        label=r"$\gamma(t)\cdot\alpha_{\mathrm{base}}\cdot\mathrm{sg}(\rho)$ (effective)",
    )[0]
    tau_line = ax_right.plot(
        t,
        tau,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.8,
        label=r"$\tau(t)$ annealing",
    )[0]

    ax_left.set_xlim(0, args.epochs)
    ax_left.set_ylim(0.0, 1.05)
    ax_right.set_ylim(0.0, 0.95)

    ax_left.set_xlabel(r"Training Epoch $t$")
    ax_left.set_ylabel(r"Effective DPP Strength $\alpha_{\mathrm{div}}^{(t)}$")
    ax_right.set_ylabel(r"Softmax Temperature $\tau$")
    ax_left.tick_params(axis="y")
    ax_right.tick_params(axis="y")

    ax_left.annotate(
        "Early: contrastive structure first",
        xy=(phase_split_1 * 0.55, alpha_div[max(phase_split_1 // 2, 1)]),
        xytext=(phase_split_1 * 0.12, 0.80),
        arrowprops={"arrowstyle": "->", "lw": 0.95, "color": "#2e7d32"},
        fontsize=8.2,
        color="#2e7d32",
    )
    ax_left.annotate(
        "Late: density-aware repulsion kicks in",
        xy=(phase_split_2 + 0.18 * (args.epochs - phase_split_2), alpha_div[phase_split_2]),
        xytext=(phase_split_2 - 0.18 * args.epochs, 0.33),
        arrowprops={"arrowstyle": "->", "lw": 0.95, "color": "#5e3ea1"},
        fontsize=8.2,
        color="#5e3ea1",
    )

    formula_text = (
        r"$\alpha_{\mathrm{div}}^{(t)}=\gamma(t)\cdot\min("
        r"\alpha_{\mathrm{base}}\cdot\mathrm{sg}(\rho),\,1)$"
    )
    ax_left.text(
        0.52,
        0.90,
        formula_text,
        transform=ax_left.transAxes,
        fontsize=8.1,
        bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round,pad=0.22", "alpha": 0.98},
    )

    handles = [alpha_line, tau_line]
    labels = [h.get_label() for h in handles]
    phase_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#7BC47F", alpha=0.18),
        plt.Rectangle((0, 0), 1, 1, color="#8E6BBE", alpha=0.16),
    ]
    handles.extend(phase_handles)
    labels.extend(["Cluster formation phase", "DPP expansion phase"])
    legend = ax_left.legend(handles, labels, loc="lower right")
    legend.get_frame().set_alpha(0.94)

    ax_left.set_title("Figure 6. Curriculum Scheduling and Training Dynamics", pad=6)
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)
    print(f"[ok] wrote {output_path}")


if __name__ == "__main__":
    main()
