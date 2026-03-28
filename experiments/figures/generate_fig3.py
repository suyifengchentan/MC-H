#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from ieee_style import apply_ieee_style


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def _build_dense_features(num_points: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    center = _normalize_rows(rng.normal(size=(1, dim)))[0]
    points = center[None, :] + 0.20 * rng.normal(size=(num_points, dim))
    return _normalize_rows(points)


def _build_sparse_features(num_points: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 999)
    points = rng.normal(size=(num_points, dim))
    return _normalize_rows(points)


def _cosine_matrix(features: np.ndarray) -> np.ndarray:
    return features @ features.T


def _top_eigenvalues(matrix: np.ndarray, k: int = 12) -> np.ndarray:
    vals = np.linalg.eigvalsh(matrix)
    vals = np.sort(vals)[::-1]
    return vals[:k]


def _draw_hypersphere_panel(
    ax,
    features: np.ndarray,
    mode: str,
    point_color: str,
    arrow_color: str,
    annotation: str,
) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 40)
    v = np.linspace(0.0, np.pi, 20)
    uu, vv = np.meshgrid(u, v)
    x = np.cos(uu) * np.sin(vv)
    y = np.sin(uu) * np.sin(vv)
    z = np.cos(vv)
    ax.plot_surface(x, y, z, color="#d9d9d9", alpha=0.12, linewidth=0)

    p3 = _normalize_rows(features[:, :3])
    ax.scatter(p3[:, 0], p3[:, 1], p3[:, 2], s=22, c=point_color, alpha=0.85)

    if mode == "dense":
        target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        for i in range(min(8, p3.shape[0])):
            start = p3[i]
            delta = target - start
            delta = delta / (np.linalg.norm(delta) + 1e-12) * 0.55
            ax.quiver(
                start[0],
                start[1],
                start[2],
                delta[0],
                delta[1],
                delta[2],
                color=arrow_color,
                linewidth=1.6,
                alpha=0.9,
            )
    else:
        rng = np.random.default_rng(123)
        for i in range(min(8, p3.shape[0])):
            start = p3[i]
            delta = _normalize_rows(rng.normal(size=(1, 3)))[0] * 0.20
            ax.quiver(
                start[0],
                start[1],
                start[2],
                delta[0],
                delta[1],
                delta[2],
                color=arrow_color,
                linewidth=1.1,
                alpha=0.75,
            )

    ax.text2D(
        0.02,
        0.01,
        annotation,
        transform=ax.transAxes,
        fontsize=7.6,
        color=arrow_color,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90},
    )
    ax.set_xlabel(r"$d_1$", labelpad=2)
    ax.set_ylabel(r"$d_2$", labelpad=2)
    ax.set_zlabel(r"$d_3$", labelpad=2)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    ax.view_init(elev=18, azim=36)
    ax.grid(False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 3: DPP Density-Adaptive Repulsion."
    )
    parser.add_argument(
        "--output",
        default="experiments/generated/figures/fig3_density_adaptive_repulsion.png",
        help="Output image path.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_ieee_style()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_points = 24
    dim = 64
    dense_features = _build_dense_features(num_points=num_points, dim=dim, seed=args.seed)
    sparse_features = _build_sparse_features(num_points=num_points, dim=dim, seed=args.seed)
    dense_sim = _cosine_matrix(dense_features)
    sparse_sim = _cosine_matrix(sparse_features)

    dense_eigs = _top_eigenvalues(dense_sim, k=12)
    sparse_eigs = _top_eigenvalues(sparse_sim, k=12)

    # IEEE two-column-width composition.
    fig = plt.figure(figsize=(7.2, 4.45), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.0, 1.3], hspace=0.08, wspace=0.22)

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2], projection="3d")
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2], projection="3d")

    im_dense = ax00.imshow(dense_sim, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax00.set_title("Dense Regime: Feature Cosine Matrix")
    ax00.set_xlabel("Sample Index")
    ax00.set_ylabel("Sample Index")

    bars_x = np.arange(1, dense_eigs.size + 1)
    ax01.bar(bars_x, dense_eigs, color="#c43e3e", edgecolor="black", linewidth=0.4)
    ax01.set_title("Dense Regime: Eigenspectrum")
    ax01.set_xlabel(r"Eigenvalue Index $k$")
    ax01.set_ylabel(r"$\mu_k$")
    ax01.set_ylim(0.0, max(1.0, dense_eigs.max() * 1.15))
    ax01.grid(True, linestyle="--", alpha=0.4)

    _draw_hypersphere_panel(
        ax02,
        dense_features,
        mode="dense",
        point_color="#b22222",
        arrow_color="#b22222",
        annotation=r"$\mu_k \gg 1 \rightarrow$ strong repulsion"
        "\n"
        r"$\rightarrow$ expansion into sparse subspaces",
    )
    ax02.set_title("Dense Regime: Hypersphere Gradient Field")

    im_sparse = ax10.imshow(sparse_sim, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax10.set_title("Sparse Regime: Feature Cosine Matrix")
    ax10.set_xlabel("Sample Index")
    ax10.set_ylabel("Sample Index")

    ax11.bar(bars_x, sparse_eigs, color="#3f6fb6", edgecolor="black", linewidth=0.4)
    ax11.set_title("Sparse Regime: Eigenspectrum")
    ax11.set_xlabel(r"Eigenvalue Index $k$")
    ax11.set_ylabel(r"$\mu_k$")
    ax11.set_ylim(0.0, max(0.8, sparse_eigs.max() * 1.15))
    ax11.grid(True, linestyle="--", alpha=0.4)

    _draw_hypersphere_panel(
        ax12,
        sparse_features,
        mode="sparse",
        point_color="#1f4e9e",
        arrow_color="#1f4e9e",
        annotation=r"$\mu_k \ll 1 \rightarrow$ isotropic weak gradient"
        "\n"
        r"$\rightarrow$ negligible restructuring",
    )
    ax12.set_title("Sparse Regime: Hypersphere Gradient Field")

    # Row labels
    fig.text(
        0.012,
        0.77,
        "Fine-grained / Dense",
        fontsize=8.7,
        color="#b22222",
        rotation=90,
        va="center",
        fontweight="bold",
    )
    fig.text(
        0.012,
        0.28,
        "Coarse-grained / Sparse",
        fontsize=8.7,
        color="#1f4e9e",
        rotation=90,
        va="center",
        fontweight="bold",
    )

    # Connecting arrows between panels
    for left_ax, right_ax in [(ax00, ax01), (ax01, ax02), (ax10, ax11), (ax11, ax12)]:
        con = ConnectionPatch(
            xyA=(1.02, 0.50),
            coordsA=left_ax.transAxes,
            xyB=(-0.03, 0.50),
            coordsB=right_ax.transAxes,
            arrowstyle="->",
            linewidth=0.9,
            color="black",
            mutation_scale=8,
        )
        fig.add_artist(con)

    cbar = fig.colorbar(im_dense, ax=[ax00, ax10], fraction=0.027, pad=0.03)
    cbar.set_label("Cosine Similarity")

    fig.suptitle("Figure 3. DPP Density-Adaptive Repulsion Mechanism", y=1.02)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {output_path}")


if __name__ == "__main__":
    main()
