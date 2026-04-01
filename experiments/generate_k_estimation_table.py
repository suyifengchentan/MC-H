#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


_CASE_LABELS = {
    "coarse_cifar10_freek": "CIFAR-10",
    "medium_image50_freek": "ImageNet-50",
    "fine_cub_freek": "CUB-200-2011",
}


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_text(value: str | None, precision: int = 2) -> str:
    if value in {"", None}:
        return "--"
    return f"{float(value):.{precision}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the K-estimation LaTeX table.")
    parser.add_argument(
        "--summary",
        default="experiments/generated/k_estimation_benchmark/summary_grouped.csv",
        help="Grouped summary CSV produced by experiments/collect_results.py.",
    )
    parser.add_argument(
        "--output",
        default="experiments/generated/tables/k_estimation_results_table.tex",
        help="Output LaTeX table path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_rows(Path(args.summary))
    row_map = {str(row["case_id"]): row for row in rows}
    output_path = Path(args.output)

    ordered_case_ids = [
        "coarse_cifar10_freek",
        "medium_image50_freek",
        "fine_cub_freek",
    ]
    lines = [
        "% Auto-generated from k_estimation_benchmark/summary_grouped.csv.",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Automatic cluster-count discovery benchmark. Lower is better for the error columns.}",
        "\\label{tab:k_estimation_results}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Dataset & True $K$ & Predicted $\\hat{K}$ & $|\\hat{K} - K|$ & Relative Error \\\\",
        "\\midrule",
    ]
    for case_id in ordered_case_ids:
        row = row_map.get(case_id)
        if row is None:
            lines.append(f"{_CASE_LABELS[case_id]} & -- & -- & -- & -- \\\\")
            continue
        true_k = _float_text(row.get("class_num"), precision=0)
        predicted_k = _float_text(row.get("last_train_estimated_k_mean"), precision=2)
        abs_error = _float_text(row.get("abs_last_k_error_mean"), precision=2)
        relative_error = _float_text(row.get("relative_last_k_error_mean"), precision=3)
        lines.append(
            f"{_CASE_LABELS[case_id]} & {true_k} & {predicted_k} & {abs_error} & {relative_error} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote {output_path}")


if __name__ == "__main__":
    main()
