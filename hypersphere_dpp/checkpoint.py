from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _move_optimizer_state_to_device(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def save_checkpoint(
    checkpoint_path: str | Path,
    model,
    optimizer,
    scaler,
    scheduler,
    queue_state: dict,
    epoch: int,
    global_step: int,
    best_metric: float,
    config: dict,
    training_state: dict[str, Any] | None = None,
) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": _to_cpu(model.trainable_state_dict()),
        "optimizer": _to_cpu(optimizer.state_dict()),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler": _to_cpu(scheduler.state_dict()) if scheduler is not None else None,
        "queue": _to_cpu(queue_state),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "config": config,
        "training_state": _to_cpu(training_state or {}),
    }
    torch.save(payload, path)


def load_checkpoint(
    checkpoint_path: str | Path,
    model,
    optimizer=None,
    scaler=None,
    scheduler=None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_trainable_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        target_device = (
            map_location
            if isinstance(map_location, torch.device)
            else torch.device(map_location)
        )
        _move_optimizer_state_to_device(optimizer, target_device)
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint
