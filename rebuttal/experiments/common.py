"""Shared planning helpers for standalone rebuttal experiments.

Each rebuttal runner is dry-run by default: it writes the fully resolved YAML
files and a JSON manifest, then prints commands. Passing ``--execute`` is the
only path that starts a subprocess.
"""

from __future__ import annotations

import json
import os
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
    job_id: str | None = None,
) -> None:
    records.append(
        {
            "name": name,
            "cwd": str(cwd),
            "command": command,
            "job_id": job_id or name,
        }
    )


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
    job_id = f"{experiment_id}:{variant_id}:seed_{seed}"
    append_command(
        records,
        name=f"train:{variant_id}:seed_{seed}",
        command=train_command,
        job_id=job_id,
    )
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
    append_command(
        records,
        name=f"eval:{variant_id}:seed_{seed}",
        command=eval_command,
        job_id=job_id,
    )
    return config, run_dir


def write_manifest(experiment_id: str, records: Iterable[dict[str, Any]]) -> Path:
    path = GENERATED_ROOT / experiment_id / "plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(records), indent=2) + "\n", encoding="utf-8")
    return path


def add_cuda_arguments(parser: Any) -> None:
    parser.add_argument(
        "--devices",
        default="auto",
        help="CUDA device list such as 0,2, or auto for every visible CUDA device.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=0,
        help="Deprecated compatibility option; DDP runs one workflow across all selected GPUs.",
    )


def resolve_cuda_devices(spec: str) -> list[str]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime environment failure.
        raise RuntimeError("PyTorch is required to discover CUDA devices.") from error

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device is available. Pass a valid CUDA environment before running.")
    count = torch.cuda.device_count()
    normalized = spec.strip().lower()
    if normalized in {"auto", "all"}:
        return [f"cuda:{index}" for index in range(count)]

    device_ids: list[int] = []
    for value in spec.split(","):
        value = value.strip()
        if not value:
            continue
        if value.startswith("cuda:"):
            value = value.split(":", 1)[1]
        index = int(value)
        if index < 0 or index >= count:
            raise ValueError(f"CUDA device {index} is outside the visible range [0, {count - 1}].")
        if index not in device_ids:
            device_ids.append(index)
    if not device_ids:
        raise ValueError("--devices must name at least one CUDA device.")
    return [f"cuda:{index}" for index in device_ids]


def _command_with_device(command: list[str], device: str) -> list[str]:
    if "--device" in command:
        return command
    return [*command, "--device", device]


def _is_train_command(command: list[str]) -> bool:
    return len(command) >= 2 and command[1] == "train.py"


def _distributed_train_command(command: list[str], device_count: int) -> list[str]:
    return [
        command[0],
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(device_count),
        *command[1:],
    ]


def _group_records(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record["job_id"]), []).append(record)
    return list(groups.values())


def run_or_print(
    records: list[dict[str, Any]],
    execute: bool,
    devices: list[str] | None = None,
    max_parallel: int = 0,
) -> None:
    selected_devices = devices or []
    groups = _group_records(records)
    distributed = len(selected_devices) > 1
    visible_ids = ",".join(device.split(":", 1)[1] for device in selected_devices)
    for group in groups:
        device = selected_devices[0] if selected_devices else None
        print(f"# workflow={group[0]['job_id']} device={'ddp:' + visible_ids if distributed else device or 'default'}")
        for record in group:
            command = list(record["command"])
            if distributed and _is_train_command(command):
                command = _distributed_train_command(command, len(selected_devices))
                print(f"$ CUDA_VISIBLE_DEVICES={visible_ids} {' '.join(command)}")
                continue
            if device:
                command = _command_with_device(command, device)
            print(f"$ {' '.join(command)}")

    if not execute:
        print("Dry run only. Re-run with --execute to start these commands.")
        return

    if not selected_devices:
        raise RuntimeError("A CUDA device list is required when --execute is used.")
    if max_parallel not in {0, 1}:
        raise ValueError("DDP uses every selected GPU for one workflow; --max-parallel must be 0 or 1.")

    for group in groups:
        for record in group:
            command = list(record["command"])
            environment = None
            if distributed and _is_train_command(command):
                command = _distributed_train_command(command, len(selected_devices))
                environment = os.environ | {"CUDA_VISIBLE_DEVICES": visible_ids}
            else:
                command = _command_with_device(command, selected_devices[0])
            subprocess.run(command, cwd=record["cwd"], check=True, env=environment)
    if not execute:
        print("Dry run only. Re-run with --execute to start these commands.")


def default_python() -> str:
    return sys.executable
