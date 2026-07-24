"""Shared planning helpers for standalone rebuttal experiments.

Each rebuttal runner is dry-run by default: it writes the fully resolved YAML
files and a JSON manifest, then prints commands. Passing ``--execute`` is the
only path that starts a subprocess.
"""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATED_ROOT = REPO_ROOT / "rebuttal" / "generated"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def build_train_plan(
    *,
    experiment_id: str,
    variant_id: str,
    base_config: Path,
    seed: int,
    overrides: dict[str, Any],
    python_bin: str,
    device: str | None,
) -> tuple[dict[str, Any], Path, Path, list[str]]:
    config = deepcopy(load_yaml(REPO_ROOT / base_config))
    run_dir = REPO_ROOT / "save" / "experiments" / "rebuttal" / experiment_id / variant_id / f"seed_{seed}"
    config.update(overrides)
    config["seed"] = seed
    config["validation_seed"] = seed
    config["model_path"] = str(run_dir.relative_to(REPO_ROOT))
    config_path = GENERATED_ROOT / experiment_id / variant_id / f"seed_{seed}_train.yaml"
    write_yaml(config_path, config)

    command = [python_bin, "train.py", "--config", str(config_path.relative_to(REPO_ROOT))]
    if device:
        command.extend(["--device", device])
    return config, config_path, run_dir, command


def append_command(
    records: list[dict[str, Any]],
    *,
    name: str,
    command: list[str],
    cwd: Path = REPO_ROOT,
) -> None:
    records.append({"name": name, "cwd": str(cwd), "command": command})


def append_train_and_eval(
    records: list[dict[str, Any]],
    *,
    experiment_id: str,
    variant_id: str,
    base_config: Path,
    seed: int,
    overrides: dict[str, Any],
    python_bin: str,
    device: str | None,
) -> tuple[dict[str, Any], Path]:
    config, config_path, run_dir, train_command = build_train_plan(
        experiment_id=experiment_id,
        variant_id=variant_id,
        base_config=base_config,
        seed=seed,
        overrides=overrides,
        python_bin=python_bin,
        device=device,
    )
    append_command(records, name=f"train:{variant_id}:seed_{seed}", command=train_command)
    eval_command = [
        python_bin,
        "eval.py",
        "--config",
        str(config_path.relative_to(REPO_ROOT)),
        "--checkpoint",
        str((run_dir / "last.pt").relative_to(REPO_ROOT)),
        "--split",
        "test",
    ]
    if device:
        eval_command.extend(["--device", device])
    append_command(records, name=f"eval:{variant_id}:seed_{seed}", command=eval_command)
    return config, run_dir


def write_manifest(experiment_id: str, records: Iterable[dict[str, Any]]) -> Path:
    path = GENERATED_ROOT / experiment_id / "plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(records), indent=2) + "\n", encoding="utf-8")
    return path


def run_or_print(records: list[dict[str, Any]], execute: bool) -> None:
    for record in records:
        command = record["command"]
        printable = " ".join(command)
        print(f"$ {printable}")
        if execute:
            subprocess.run(command, cwd=record["cwd"], check=True)
    if not execute:
        print("Dry run only. Re-run with --execute to start these commands.")


def default_python() -> str:
    return sys.executable
