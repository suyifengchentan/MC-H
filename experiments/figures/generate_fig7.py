#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ieee_style import apply_ieee_style


def _ellipsoid_from_covariance(
    cov: np.ndarray,
    center: np.ndarray,
    num_u: int = 60,
    num_v: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * np.pi, num_u)
    v = np.linspace(0.0, np.pi, num_v)
    uu, vv = np.meshgrid(u, v)
    sphere = np.stack(
        [np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv)],
        axis=-1,
    )

    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-6, None)
    transform = evecs @ np.diag(np.sqrt(evals))
    points = sphere.reshape(-1, 3) @ transform.T + center
    xx = points[:, 0].reshape(v.size, u.size)
    yy = points[:, 1].reshape(v.size, u.size)
    zz = points[:, 2].reshape(v.size, u.size)
    return xx, yy, zz


def _draw_anchor_plane(ax, color: str = "#6f8fb8") -> None:
    grid = np.linspace(-1.5, 1.5, 2)
    xx, yy = np.meshgrid(grid, grid)
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color=color, alpha=0.18, linewidth=0)
    ax.text(0.15, 1.05, 0.02, r"Anchor subspace $V_k$", color=color, fontsize=7.8)


def _style_axis(ax, title: str) -> None:
    ax.set_title(title, pad=6)
    ax.set_xlabel(r"$d_1$", labelpad=4)
    ax.set_ylabel(r"$d_2$", labelpad=4)
    ax.set_zlabel(r"$d_3$", labelpad=4)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_zlim(-1.3, 1.3)
    ax.view_init(elev=22, azim=38)
    ax.grid(True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 7: Subspace Regularization Visualization."
    )
    parser.add_argument(
        "--output",
        default="experiments/generated/figures/fig7_subspace_regularization.png",
        help="Output image path.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_ieee_style()
    np.random.seed(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # LEFT: without L_sub (misaligned, leakage into non-discriminative directions)
    cov_wo = np.array(
        [
            [1.10, 0.62, 0.48],
            [0.62, 0.95, 0.44],
            [0.48, 0.44, 0.78],
        ]
    )
    # RIGHT: with L_sub (aligned to anchor subspace d1-d2, reduced residual on d3)
    cov_w = np.array(
        [
            [0.82, 0.35, 0.03],
            [0.35, 0.66, 0.02],
            [0.03, 0.02, 0.07],
        ]
    )
    center = np.zeros(3, dtype=np.float64)

    x1, y1, z1 = _ellipsoid_from_covariance(cov_wo, center=center)
    x2, y2, z2 = _ellipsoid_from_covariance(cov_w, center=center)

    fig = plt.figure(figsize=(7.2, 3.7), constrained_layout=True)
    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2, projection="3d")

    ax_left.plot_surface(x1, y1, z1, color="#d16b6b", alpha=0.46, linewidth=0)
    _draw_anchor_plane(ax_left, color="#6c8fb3")
    _style_axis(ax_left, "Without $L_{sub}$")
    ax_left.quiver(0, 0, 0, 0.2, 0.1, 0.95, color="red", alpha=0.75, linewidth=1.5)
    ax_left.quiver(0, 0, 0, -0.2, -0.1, 0.85, color="red", alpha=0.55, linewidth=1.2)
    ax_left.text(
        -1.35,
        -1.2,
        1.00,
        "Spurious variance leakage",
        color="red",
        fontsize=7.8,
    )
    ax_left.text(
        -1.45,
        1.05,
        -1.1,
        "Residual variance outside\nanchor subspace",
        color="darkred",
        fontsize=7.6,
    )

    ax_right.plot_surface(x2, y2, z2, color="#66b37a", alpha=0.50, linewidth=0)
    _draw_anchor_plane(ax_right, color="#5f87b9")
    _style_axis(ax_right, "With $L_{sub}$")
    ax_right.quiver(0, 0, 0, 0.8, 0.45, 0.05, color="green", alpha=0.75, linewidth=1.5)
    ax_right.quiver(0, 0, 0, -0.7, -0.35, -0.03, color="green", alpha=0.75, linewidth=1.5)
    ax_right.text(
        -1.35,
        -1.2,
        0.98,
        r"$\Sigma_k \approx V_kV_k^\top\Sigma_kV_kV_k^\top$",
        color="darkgreen",
        fontsize=7.8,
    )

    fig.suptitle("Figure 7. Subspace Regularization Visualization", y=1.02)
    fig.text(
        0.5,
        -0.02,
        r"$\|\Sigma_k - V_kV_k^\top\Sigma_kV_kV_k^\top\|_F \rightarrow 0$",
        ha="center",
        va="top",
        fontsize=8.3,
        bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round,pad=0.20", "alpha": 0.96},
    )

    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {output_path}")


if __name__ == "__main__":
    main()
