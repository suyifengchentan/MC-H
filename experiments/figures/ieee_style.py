#!/usr/bin/env python3
from __future__ import annotations

import matplotlib as mpl


def apply_ieee_style() -> None:
    """Apply IEEE-like publication defaults for Matplotlib figures."""
    mpl.rcParams.update(
        {
            # Typography
            "font.family": "serif",
            "font.serif": [
                "Computer Modern Roman",
                "CMU Serif",
                "cmr10",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "mathtext.rm": "cmr10",
            "axes.formatter.use_mathtext": True,
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.8,
            "figure.titlesize": 9.5,
            # Line/axes weight
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            # Grid and legend appearance
            "grid.linestyle": "--",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.55,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "black",
            # Export quality
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
