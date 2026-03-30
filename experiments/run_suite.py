#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def _merge_overrides(*items: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        if item:
            merged.update(item)
    return merged


def _apply_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overrides.items():
        result[key] = value
    return result


def _parse_cli_overrides(raw_items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for raw in raw_items:
        if "=" not in raw:
            raise ValueError(f"Invalid --override value: {raw!r}. Expected KEY=VALUE.")
        key, value_text = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --override key in {raw!r}.")
        overrides[key] = yaml.safe_load(value_text)
    return overrides


def _resolve_eval_config(
    train_config: dict[str, Any],
    base_eval_config_path: Path | None,
) -> dict[str, Any]:
    if base_eval_config_path is None:
        return deepcopy(train_config)

    eval_config = _load_yaml(base_eval_config_path)
    keys_to_sync = {
        "dataset",
        "dataset_dir",
        "class_num",
        "validation_ratio",
        "validation_seed",
        "official_eval_split",
        "image_size",
        "feature_dim",
        "backbone_name",
        "pretrained",
        "pretrained_checkpoint",
        "adapter_hidden_dim",
        "adapter_dropout",
        "transform_profile",
        "clustering_mode",
        "min_spectral_clusters",
        "max_spectral_clusters",
        "spectral_assignment_temperature",
        "spectral_sharpen_temperature",
        "spectral_proto_momentum",
        "semantic_prototype_enabled",
        "semantic_prompt_style",
        "semantic_prototype_aggregation",
        "lambda_semantic",
        "semantic_temperature",
        "eval_batch_size",
        "workers",
        "seed",
        "device",
    }
    for key in keys_to_sync:
        if key in train_config:
            eval_config[key] = train_config[key]
    return eval_config


def _run_command(command: list[str], cwd: Path, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    printable = " ".join(command)
    print(f"$ {printable}")
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=capture_output,
    )


def _parse_eval_stdout(stdout_text: str) -> dict[str, dict[str, float]]:
    parsed: dict[str, dict[str, float]] = {}
    for line in stdout_text.splitlines():
        line = line.strip()
        for split in ("val", "test"):
            prefix = f"{split}="
            if not line.startswith(prefix):
                continue
            payload = line[len(prefix) :].strip()
            value = ast.literal_eval(payload)
            if isinstance(value, dict):
                parsed[split] = value
    return parsed


def _run_single_case(
    *,
    repo_root: Path,
    generated_root: Path,
    output_root: Path,
    suite_name: str,
    case: dict[str, Any],
    seed: int,
    python_bin: str,
    device: str | None,
    eval_split: str,
    execute: bool,
    skip_train: bool,
    skip_eval: bool,
    force_retrain: bool,
    global_train_overrides: dict[str, Any],
    global_eval_overrides: dict[str, Any],
    fail_fast: bool,
) -> dict[str, Any]:
    case_id = case["id"]
    case_generated_dir = generated_root / suite_name / case_id
    run_dir = output_root / suite_name / case_id / f"seed_{seed}"
    generated_train_config_path = case_generated_dir / f"seed_{seed}_train.yaml"
    generated_eval_config_path = case_generated_dir / f"seed_{seed}_eval.yaml"

    base_train_config_path = repo_root / case["base_train_config"]
    base_eval_config_rel = case.get("base_eval_config")
    base_eval_config_path = (
        repo_root / base_eval_config_rel if base_eval_config_rel else None
    )

    train_config = _load_yaml(base_train_config_path)
    train_overrides = _merge_overrides(
        global_train_overrides,
        case.get("train_overrides"),
    )
    train_config = _apply_overrides(train_config, train_overrides)
    train_config["seed"] = seed
    if case.get("set_validation_seed_from_seed", False):
        train_config["validation_seed"] = seed
    train_config["model_path"] = str(run_dir)
    _dump_yaml(generated_train_config_path, train_config)

    eval_config = _resolve_eval_config(train_config, base_eval_config_path)
    eval_overrides = _merge_overrides(
        global_eval_overrides,
        case.get("eval_overrides"),
    )
    eval_config = _apply_overrides(eval_config, eval_overrides)
    eval_config["seed"] = seed
    _dump_yaml(generated_eval_config_path, eval_config)

    train_cmd = [python_bin, "train.py", "--config", str(generated_train_config_path)]
    eval_cmd = [
        python_bin,
        "eval.py",
        "--config",
        str(generated_eval_config_path),
        "--checkpoint",
        str(run_dir / "best.pt"),
        "--split",
        eval_split,
    ]
    if device:
        train_cmd.extend(["--device", device])
        eval_cmd.extend(["--device", device])

    record: dict[str, Any] = {
        "case_id": case_id,
        "seed": seed,
        "run_dir": str(run_dir),
        "train_config": str(generated_train_config_path),
        "eval_config": str(generated_eval_config_path),
        "train_command": train_cmd,
        "eval_command": eval_cmd,
        "status": "planned",
    }

    if not execute:
        return record

    run_dir.mkdir(parents=True, exist_ok=True)

    if not skip_train:
        best_checkpoint_path = run_dir / "best.pt"
        if best_checkpoint_path.exists() and not force_retrain:
            print(
                f"[skip-train] checkpoint exists: {best_checkpoint_path}. "
                "Use --force-retrain to rerun."
            )
        else:
            train_result = _run_command(train_cmd, cwd=repo_root, capture_output=False)
            if train_result.returncode != 0:
                record["status"] = "train_failed"
                record["train_returncode"] = train_result.returncode
                if fail_fast:
                    raise RuntimeError(
                        f"Training failed for case={case_id} seed={seed} "
                        f"with return code {train_result.returncode}"
                    )
                return record

    if skip_eval:
        record["status"] = "train_only_done"
        return record

    best_checkpoint_path = run_dir / "best.pt"
    if not best_checkpoint_path.exists():
        fallback_checkpoint = run_dir / "last.pt"
        if fallback_checkpoint.exists():
            eval_cmd[eval_cmd.index(str(best_checkpoint_path))] = str(fallback_checkpoint)
            best_checkpoint_path = fallback_checkpoint
            print(f"[eval] best.pt missing, fallback to {fallback_checkpoint}")
        else:
            print(
                f"[skip-eval] checkpoint missing for case={case_id} seed={seed}: "
                f"{run_dir / 'best.pt'}"
            )
            record["status"] = "checkpoint_missing"
            return record

    eval_result = _run_command(eval_cmd, cwd=repo_root, capture_output=True)
    eval_stdout_path = run_dir / "eval.stdout.log"
    eval_stderr_path = run_dir / "eval.stderr.log"
    eval_stdout_path.write_text(eval_result.stdout, encoding="utf-8")
    eval_stderr_path.write_text(eval_result.stderr, encoding="utf-8")
    print(eval_result.stdout, end="")
    if eval_result.stderr.strip():
        print(eval_result.stderr, end="", file=sys.stderr)

    if eval_result.returncode != 0:
        record["status"] = "eval_failed"
        record["eval_returncode"] = eval_result.returncode
        if fail_fast:
            raise RuntimeError(
                f"Evaluation failed for case={case_id} seed={seed} "
                f"with return code {eval_result.returncode}"
            )
        return record

    parsed_eval = _parse_eval_stdout(eval_result.stdout)
    if "val" in parsed_eval:
        (run_dir / "eval_val.json").write_text(
            json.dumps(parsed_eval["val"], ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if "test" in parsed_eval:
        (run_dir / "eval_test.json").write_text(
            json.dumps(parsed_eval["test"], ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    record["status"] = "done"
    record["checkpoint_used"] = str(best_checkpoint_path)
    record["parsed_eval_splits"] = sorted(parsed_eval.keys())
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and optionally run experiment suites defined in YAML."
    )
    parser.add_argument(
        "--suite",
        required=True,
        help="Path to suite YAML, e.g. experiments/suites/validation_methodology.yaml",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run train/eval commands. Without this flag, only generate plans/configs.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used for train/eval commands.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override passed to train.py/eval.py, e.g. cuda:0.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case id filter. Can be used multiple times.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        default=[],
        help="Seed override. Can be used multiple times.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Global train override in KEY=VALUE format, e.g. epochs=80",
    )
    parser.add_argument(
        "--eval-override",
        action="append",
        default=[],
        help="Global eval override in KEY=VALUE format, e.g. eval_batch_size=1024",
    )
    parser.add_argument(
        "--eval-split",
        choices=["val", "test", "both"],
        default=None,
        help="Override eval split passed to eval.py.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training stage.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation stage.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Rerun training even if best.pt already exists.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when one run fails.",
    )
    parser.add_argument(
        "--generated-root",
        default="experiments/generated",
        help="Directory for generated run configs and plans.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override output root for model_path. Defaults to suite defaults.output_root or save/experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    suite_path = (repo_root / args.suite).resolve()
    suite = _load_yaml(suite_path)
    suite_name = str(suite.get("suite_name") or suite_path.stem)
    defaults = suite.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError(f"defaults must be a mapping in suite: {suite_path}")
    cases = suite.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"cases must be a non-empty list in suite: {suite_path}")

    selected_cases = set(args.case)
    global_train_overrides = _parse_cli_overrides(args.override)
    global_eval_overrides = _parse_cli_overrides(args.eval_override)
    default_seeds = defaults.get("seeds", [42])
    if not isinstance(default_seeds, list) or not default_seeds:
        raise ValueError(f"defaults.seeds must be a non-empty list in suite: {suite_path}")
    seed_override = args.seed if args.seed else None
    eval_split = args.eval_split or defaults.get("eval_split", "both")
    output_root = (
        Path(args.output_root)
        if args.output_root
        else Path(defaults.get("output_root", "save/experiments"))
    )
    generated_root = Path(args.generated_root)

    run_records: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError(f"Each case must be a mapping in suite: {suite_path}")
        case_id = case.get("id")
        if not case_id:
            raise ValueError(f"Case missing id in suite: {suite_path}")
        if selected_cases and case_id not in selected_cases:
            continue

        case_seeds = seed_override or case.get("seeds") or default_seeds
        for seed in case_seeds:
            record = _run_single_case(
                repo_root=repo_root,
                generated_root=generated_root,
                output_root=output_root,
                suite_name=suite_name,
                case=case,
                seed=int(seed),
                python_bin=args.python_bin,
                device=args.device,
                eval_split=eval_split,
                execute=args.execute,
                skip_train=args.skip_train,
                skip_eval=args.skip_eval,
                force_retrain=args.force_retrain,
                global_train_overrides=_merge_overrides(
                    defaults.get("train_overrides"),
                    global_train_overrides,
                ),
                global_eval_overrides=_merge_overrides(
                    defaults.get("eval_overrides"),
                    global_eval_overrides,
                ),
                fail_fast=args.fail_fast,
            )
            run_records.append(record)

    plan_dir = generated_root / suite_name
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plan_dir / "plan.json"
    plan_payload = {
        "suite_path": str(suite_path),
        "suite_name": suite_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execute": args.execute,
        "run_count": len(run_records),
        "runs": run_records,
    }
    plan_path.write_text(
        json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"[plan] wrote {plan_path}")
    print(f"[plan] runs={len(run_records)} execute={args.execute}")
    if not run_records:
        print("[plan] no runs selected. Check --case filters.")


if __name__ == "__main__":
    main()
