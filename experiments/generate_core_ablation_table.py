#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


_VARIANT_LABELS = {
    "full": "Full",
    "no_dpp": "w/o DPP",
    "no_density_weighting": "w/o Density Weighting",
    "fixed_k": "w/o Free-K",
    "no_stabilization": "w/o Proto. Stabilization",
    "no_subspace": "w/o Subspace Reg.",
    "pca_anchor": "PCA Anchor",
}

_DATASET_LABELS = {
    "stl10": "STL-10",
    "image200": "ImageNet-200",
    "cub": "CUB-200-2011",
}


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _metric(row: dict[str, str], key: str) -> str:
    value = row.get(key, "")
    if value in {"", None}:
        return "--"
    return f"{100.0 * float(value):.1f}"


def _parse_case(case_id: str) -> tuple[str, str]:
    for suffix in _VARIANT_LABELS:
        if case_id.endswith("_" + suffix):
            dataset_key = case_id[: -(len(suffix) + 1)].split("_", 1)[1]
            return dataset_key, suffix
    raise ValueError(f"Unsupported case id format: {case_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the core ablation LaTeX table.")
    parser.add_argument(
        "--summary",
        default="experiments/generated/core_ablation_matrix/summary_grouped.csv",
        help="Grouped summary CSV produced by experiments/collect_results.py.",
    )
    parser.add_argument(
        "--output",
        default="experiments/generated/tables/core_ablation_matrix_table.tex",
        help="Output LaTeX table path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    output_path = Path(args.output)
    rows = _load_rows(summary_path)

    grouped: dict[str, list[tuple[str, dict[str, str]]]] = {}
    for row in rows:
        dataset_key, variant_key = _parse_case(str(row["case_id"]))
        grouped.setdefault(dataset_key, []).append((variant_key, row))

    variant_order = list(_VARIANT_LABELS)
    dataset_order = ["stl10", "image200", "cub"]

    lines = [
        "% Auto-generated from core_ablation_matrix/summary_grouped.csv.",
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Cross-granularity core ablation results. Metrics are reported on the official evaluation split. Higher is better for ACC/NMI/ARI.}",
        "\\label{tab:core_ablation_matrix}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Dataset & Variant & ACC & NMI & ARI & $\\hat{K}$ \\\\",
        "\\midrule",
    ]

    for dataset_index, dataset_key in enumerate(dataset_order):
        dataset_rows = {variant_key: row for variant_key, row in grouped.get(dataset_key, [])}
        for variant_index, variant_key in enumerate(variant_order):
            row = dataset_rows.get(variant_key)
            dataset_label = _DATASET_LABELS[dataset_key] if variant_index == 0 else ""
            if row is None:
                metrics = ["--", "--", "--", "--"]
            else:
                metrics = [
                    _metric(row, "eval_test_acc_mean"),
                    _metric(row, "eval_test_nmi_mean"),
                    _metric(row, "eval_test_ari_mean"),
                    row.get("last_train_estimated_k_mean", "--") or "--",
                ]
            lines.append(
                f"{dataset_label} & {_VARIANT_LABELS[variant_key]} & "
                f"{metrics[0]} & {metrics[1]} & {metrics[2]} & {metrics[3]} \\\\"
            )
        if dataset_index != len(dataset_order) - 1:
            lines.append("\\midrule")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote {output_path}")


if __name__ == "__main__":
    main()
