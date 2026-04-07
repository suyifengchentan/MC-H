#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = ["case_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    if len(values) <= 1:
        return 0.0 if values else None
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    if not rows:
        return []

    numeric_fields = [
        key
        for key in rows[0].keys()
        if key not in {"suite_name", "case_id", "seed", "granularity", "tags", "run_dir"}
    ]
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["case_id"], []).append(row)

    aggregated_rows: list[dict[str, Any]] = []
    for case_id in sorted(grouped):
        case_rows = grouped[case_id]
        base_row = case_rows[0]
        aggregated: dict[str, Any] = {
            "suite_name": base_row.get("suite_name", ""),
            "case_id": case_id,
            "granularity": base_row.get("granularity", ""),
            "tags": base_row.get("tags", ""),
            "num_seeds": len(case_rows),
        }
        for field in numeric_fields:
            values = [
                parsed
                for parsed in (_safe_float(row.get(field)) for row in case_rows)
                if parsed is not None
            ]
            aggregated[f"{field}_mean"] = _mean(values)
            aggregated[f"{field}_std"] = _std(values)
        aggregated_rows.append(aggregated)
    return aggregated_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split one suite summary into main-text and appendix outputs."
    )
    parser.add_argument(
        "--suite",
        required=True,
        help="Path to suite YAML, e.g. experiments/suites/ablation_paper_three_datasets.yaml",
    )
    parser.add_argument(
        "--groups",
        required=True,
        help="Path to the groups YAML, e.g. experiments/suites/ablation_paper_three_datasets_groups.yaml",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional summary CSV path. Defaults to experiments/generated/<suite_name>/summary.csv",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional split-output root. Defaults to experiments/generated/<suite_name>",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    suite_path = (repo_root / args.suite).resolve()
    groups_path = (repo_root / args.groups).resolve()

    suite = _load_yaml(suite_path)
    groups_spec = _load_yaml(groups_path)
    suite_name = str(suite.get("suite_name") or suite_path.stem)
    groups = groups_spec.get("groups") or {}
    if not isinstance(groups, dict) or not groups:
        raise ValueError(f"groups must be a non-empty mapping in {groups_path}")

    summary_path = (
        (repo_root / args.summary).resolve()
        if args.summary
        else (repo_root / "experiments" / "generated" / suite_name / "summary.csv")
    )
    split_root = (
        (repo_root / args.output_root).resolve()
        if args.output_root
        else (repo_root / "experiments" / "generated" / suite_name)
    )

    summary_rows = _read_csv(summary_path)
    all_case_ids = {str(case["id"]) for case in suite.get("cases") or [] if isinstance(case, dict)}

    for group_name, group_payload in groups.items():
        if not isinstance(group_payload, dict):
            raise ValueError(f"group {group_name!r} must be a mapping in {groups_path}")
        case_ids = group_payload.get("case_ids") or []
        if not isinstance(case_ids, list) or not case_ids:
            raise ValueError(f"group {group_name!r} must define a non-empty case_ids list")

        unknown = sorted(set(case_ids) - all_case_ids)
        if unknown:
            raise ValueError(
                f"group {group_name!r} contains unknown case ids not found in suite {suite_name}: {unknown}"
            )

        filtered_rows = [row for row in summary_rows if row.get("case_id") in set(case_ids)]
        aggregated_rows = _aggregate_rows(filtered_rows)
        group_dir = split_root / group_name

        _write_csv(group_dir / "summary_per_seed.csv", filtered_rows)
        _write_csv(group_dir / "summary_aggregated.csv", aggregated_rows)
        (group_dir / "cases.txt").write_text(
            "\n".join(case_ids) + "\n",
            encoding="utf-8",
        )
        (group_dir / "README.txt").write_text(
            (
                f"group={group_name}\n"
                f"description={group_payload.get('description', '')}\n"
                f"source_summary={summary_path}\n"
                f"per_seed_csv={group_dir / 'summary_per_seed.csv'}\n"
                f"aggregated_csv={group_dir / 'summary_aggregated.csv'}\n"
            ),
            encoding="utf-8",
        )
        print(
            f"[split] group={group_name} per_seed_rows={len(filtered_rows)} "
            f"aggregated_rows={len(aggregated_rows)} output_dir={group_dir}"
        )


if __name__ == "__main__":
    main()
