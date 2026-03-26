from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable
import warnings

import yaml


@dataclass
class ExperimentConfig:
    dataset: str = "CUB_200_2011"
    dataset_dir: str = ""
    class_num: int = 200
    validation_ratio: float = 0.1
    validation_seed: int = 42

    image_size: int = 224
    feature_dim: int = 768
    batch_size: int = 128
    eval_batch_size: int = 256
    epochs: int = 100
    start_epoch: int = 0

    learning_rate: float = 2.5e-4
    weight_decay: float = 1e-6
    precision: str = "fp16"
    instance_temperature: float = 0.2
    cluster_temperature: float = 0.1
    lambda_instance: float = 1.0
    lambda_cluster: float = 1.0
    lambda_subspace: float = 0.15
    alpha_div_base: float = 0.6

    backbone_name: str = "ViT-L-14"
    pretrained: str = "laion2b_s32b_b82k"
    pretrained_checkpoint: str | None = None
    adapter_hidden_dim: int = 1024
    adapter_dropout: float = 0.1

    dpp_warmup_epochs: int = 20
    dpp_every_n_steps: int = 1
    dpp_jitter: float = 1e-5

    sinkhorn_iters: int = 3
    sinkhorn_epsilon: float = 0.05

    subspace_rank: int = 8
    subspace_queue_size: int = 64
    subspace_min_mass: float = 2.0

    eval_interval: int = 1
    workers: int = 8
    reload: bool = False
    save_best_only: bool = True
    seed: int = 42
    best_checkpoint_patience: int = 50
    model_path: str = "save/hypersphere_dpp/cub_vitl14"
    transform_profile: str = "fine_grained_two_view"

    device: str | None = None

    @property
    def output_dir(self) -> Path:
        return Path(self.model_path)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _field_names() -> set[str]:
    return {item.name for item in fields(ExperimentConfig)}


def _warn_unknown_fields(values: Dict[str, Any], supported_fields: Iterable[str]) -> None:
    supported = set(supported_fields)
    unknown = sorted(set(values) - supported)
    for key in unknown:
        warnings.warn(
            f"Unsupported config field '{key}' will be ignored in the ViT-L/14 prototype.",
            stacklevel=2,
        )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")

    _warn_unknown_fields(raw, _field_names())
    usable = {key: value for key, value in raw.items() if key in _field_names()}
    config = ExperimentConfig(**usable)

    if config.dataset != "CUB_200_2011":
        raise ValueError(
            f"This prototype only supports dataset='CUB_200_2011', got '{config.dataset}'."
        )

    if config.feature_dim != 768:
        raise ValueError(f"feature_dim must be 768 for ViT-L/14 CLIP, got {config.feature_dim}.")

    if not 0.0 <= config.validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in [0, 1).")

    if config.class_num <= 1:
        raise ValueError("class_num must be greater than 1.")

    return config

