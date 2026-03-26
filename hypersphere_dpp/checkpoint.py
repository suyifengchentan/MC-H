from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    checkpoint_path: str | Path,
    model,
    optimizer,
    scaler,
    queue_state: dict,
    epoch: int,
    global_step: int,
    best_metric: float,
    config: dict,
) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "queue": queue_state,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "config": config,
    }
    torch.save(payload, path)


def load_checkpoint(
    checkpoint_path: str | Path,
    model,
    optimizer=None,
    scaler=None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint

