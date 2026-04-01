#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return None
    return payload


def _load_metrics_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_generated_train_config(
    repo_root: Path,
    suite_name: str,
    case_id: str,
    seed: int,
) -> dict[str, Any]:
    config_path = (
        repo_root / "experiments" / "generated" / suite_name / case_id / f"seed_{seed}_train.yaml"
    )
    if not config_path.exists():
        return {}
    return _load_yaml(config_path)


def _summarize_run(run_dir: Path, train_config: dict[str, Any]) -> dict[str, Any]:
    metrics_rows = _load_metrics_jsonl(run_dir / "metrics.jsonl")
    train_rows = [row for row in metrics_rows if row.get("split") == "train"]
    val_rows = [row for row in metrics_rows if row.get("split") == "val"]

    best_val_row = None
    if val_rows:
        best_val_row = max(
            val_rows,
            key=lambda row: _safe_float(row.get("selection_score"))
            if _safe_float(row.get("selection_score")) is not None
            else float("-inf"),
        )
    last_train_row = train_rows[-1] if train_rows else None

    mean_train_rho = None
    rho_values = [_safe_float(row.get("rho")) for row in train_rows]
    rho_values = [value for value in rho_values if value is not None]
    if rho_values:
        mean_train_rho = sum(rho_values) / len(rho_values)

    mean_train_alpha_div = None
    alpha_values = [_safe_float(row.get("alpha_div")) for row in train_rows]
    alpha_values = [value for value in alpha_values if value is not None]
    if alpha_values:
        mean_train_alpha_div = sum(alpha_values) / len(alpha_values)

    mean_train_estimated_k = None
    estimated_k_values = [_safe_float(row.get("estimated_k")) for row in train_rows]
    estimated_k_values = [value for value in estimated_k_values if value is not None]
    if estimated_k_values:
        mean_train_estimated_k = sum(estimated_k_values) / len(estimated_k_values)

    eval_val = _load_json(run_dir / "eval_val.json")
    eval_test = _load_json(run_dir / "eval_test.json")
    true_k = _safe_float(train_config.get("class_num"))
    last_train_estimated_k = _safe_float(
        last_train_row.get("estimated_k") if last_train_row else None
    )
    abs_last_k_error = None
    relative_last_k_error = None
    if true_k is not None and last_train_estimated_k is not None and true_k > 0:
        abs_last_k_error = abs(last_train_estimated_k - true_k)
        relative_last_k_error = abs_last_k_error / true_k

    return {
        "has_metrics": bool(metrics_rows),
        "best_checkpoint_exists": (run_dir / "best.pt").exists(),
        "last_checkpoint_exists": (run_dir / "last.pt").exists(),
        "class_num": true_k,
        "clustering_mode": train_config.get("clustering_mode", ""),
        "diversity_weight_mode": train_config.get("diversity_weight_mode", ""),
        "subspace_anchor_mode": train_config.get("subspace_anchor_mode", ""),
        "best_val_selection_score": _safe_float(
            best_val_row.get("selection_score") if best_val_row else None
        ),
        "best_val_acc": _safe_float(best_val_row.get("acc") if best_val_row else None),
        "best_val_nmi": _safe_float(best_val_row.get("nmi") if best_val_row else None),
        "best_val_ari": _safe_float(best_val_row.get("ari") if best_val_row else None),
        "last_train_loss": _safe_float(
            last_train_row.get("loss") if last_train_row else None
        ),
        "last_train_inst_loss": _safe_float(
            last_train_row.get("inst_loss") if last_train_row else None
        ),
        "last_train_cluster_loss": _safe_float(
            last_train_row.get("cluster_loss") if last_train_row else None
        ),
        "last_train_subspace_loss": _safe_float(
            last_train_row.get("subspace_loss") if last_train_row else None
        ),
        "last_train_dpp_loss": _safe_float(
            last_train_row.get("dpp_loss") if last_train_row else None
        ),
        "last_train_estimated_k": last_train_estimated_k,
        "mean_train_rho": mean_train_rho,
        "mean_train_alpha_div": mean_train_alpha_div,
        "mean_train_estimated_k": mean_train_estimated_k,
        "abs_last_k_error": abs_last_k_error,
        "relative_last_k_error": relative_last_k_error,
        "eval_val_acc": _safe_float(eval_val.get("acc") if eval_val else None),
        "eval_val_nmi": _safe_float(eval_val.get("nmi") if eval_val else None),
        "eval_val_ari": _safe_float(eval_val.get("ari") if eval_val else None),
        "eval_test_acc": _safe_float(eval_test.get("acc") if eval_test else None),
        "eval_test_nmi": _safe_float(eval_test.get("nmi") if eval_test else None),
        "eval_test_ari": _safe_float(eval_test.get("ari") if eval_test else None),
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mean_value = _mean(values)
    if mean_value is None:
        return None
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["case_id"]), []).append(row)

    aggregated_rows: list[dict[str, Any]] = []
    for case_id, group_rows in sorted(groups.items()):
        base = group_rows[0]
        aggregate: dict[str, Any] = {
            "suite_name": base["suite_name"],
            "case_id": case_id,
            "granularity": base.get("granularity", ""),
            "tags": base.get("tags", ""),
            "num_seeds": len(group_rows),
            "class_num": base.get("class_num"),
            "clustering_mode": base.get("clustering_mode", ""),
            "diversity_weight_mode": base.get("diversity_weight_mode", ""),
            "subspace_anchor_mode": base.get("subspace_anchor_mode", ""),
        }

        numeric_keys = sorted(
            {
                key
                for row in group_rows
                for key, value in row.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            - {"seed"}
        )
        for key in numeric_keys:
            values = [
                float(row[key])
                for row in group_rows
                if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
            ]
            aggregate[f"{key}_mean"] = _mean(values)
            aggregate[f"{key}_std"] = _std(values)
        aggregated_rows.append(aggregate)

    return aggregated_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect suite results into one CSV table."
    )
    parser.add_argument(
        "--suite",
        required=True,
        help="Path to suite YAML, e.g. experiments/suites/ablation_methodology.yaml",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override output root; defaults to suite defaults.output_root or save/experiments.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path. Defaults to experiments/generated/<suite_name>/summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    suite_path = (repo_root / args.suite).resolve()
    suite = _load_yaml(suite_path)
    suite_name = str(suite.get("suite_name") or suite_path.stem)
    defaults = suite.get("defaults") or {}
    cases = suite.get("cases") or []

    output_root = (
        Path(args.output_root)
        if args.output_root
        else Path(defaults.get("output_root", "save/experiments"))
    )
    seeds = defaults.get("seeds", [42])
    generated_summary_path = (
        Path(args.output)
        if args.output
        else Path("experiments/generated") / suite_name / "summary.csv"
    )

    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = case["id"]
        case_seeds = case.get("seeds") or seeds
        for seed in case_seeds:
            run_dir = output_root / suite_name / case_id / f"seed_{seed}"
            train_config = _load_generated_train_config(
                repo_root=repo_root,
                suite_name=suite_name,
                case_id=case_id,
                seed=int(seed),
            )
            summary = _summarize_run(run_dir, train_config)
            row = {
                "suite_name": suite_name,
                "case_id": case_id,
                "seed": int(seed),
                "granularity": case.get("granularity", ""),
                "tags": ",".join(case.get("tags", [])),
                "run_dir": str(run_dir),
                **summary,
            }
            rows.append(row)

    generated_summary_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = ["suite_name", "case_id", "seed", "run_dir"]
    with generated_summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[summary] wrote {generated_summary_path}")
    print(f"[summary] rows={len(rows)}")

    grouped_rows = _aggregate_rows(rows)
    grouped_summary_path = generated_summary_path.with_name(
        generated_summary_path.stem + "_grouped.csv"
    )
    if grouped_rows:
        grouped_fieldnames = list(grouped_rows[0].keys())
    else:
        grouped_fieldnames = ["suite_name", "case_id", "num_seeds"]
    with grouped_summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=grouped_fieldnames)
        writer.writeheader()
        for row in grouped_rows:
            writer.writerow(row)

    print(f"[summary] wrote {grouped_summary_path}")
    print(f"[summary] grouped_rows={len(grouped_rows)}")


if __name__ == "__main__":
    main()
