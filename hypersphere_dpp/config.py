from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

SUPPORTED_PRECISIONS = {"fp16", "bf16", "fp32"}
SUPPORTED_DATASETS = {
    "CIFAR10",
    "CIFAR100",
    "CUB_200_2011",
    "FGVC_Aircraft",
    "Image10",
    "Image50",
    "Image100",
    "Image200",
    "Oxford_Flowers",
    "Oxford_Pet",
    "STL10",
    "Stanford_Dogs",
}
SUPPORTED_EVAL_SPLITS = {"val", "valid", "test"}
SUPPORTED_EVAL_SPLITS_BY_DATASET = {
    "CIFAR10": {"test"},
    "CIFAR100": {"test"},
    "CUB_200_2011": {"test"},
    "FGVC_Aircraft": {"val", "test"},
    "Image10": {"test"},
    "Image50": {"val"},
    "Image100": {"val"},
    "Image200": {"val"},
    "Oxford_Flowers": {"valid"},
    "Oxford_Pet": {"test"},
    "STL10": {"test"},
    "Stanford_Dogs": {"test"},
}
SUPPORTED_LR_SCHEDULERS = {"none", "cosine"}
SUPPORTED_BACKBONE_FEATURE_DIMS = {
    "ViT-L-14": 768,
    "ViT-B-16": 512,
}


@dataclass
class ExperimentConfig:
    # Data
    dataset: str = "CUB_200_2011"
    dataset_dir: str = ""
    class_num: int = 200
    validation_ratio: float = 0.1
    validation_seed: int = 42
    official_eval_split: str | None = None

    # Model
    image_size: int = 224
    feature_dim: int = 768
    backbone_name: str = "ViT-L-14"
    pretrained: str = "laion2b_s32b_b82k"
    pretrained_checkpoint: str | None = None
    adapter_hidden_dim: int = 1024
    adapter_dropout: float = 0.1
    transform_profile: str = "fine_grained_two_view"

    # Optimization
    batch_size: int = 128
    eval_batch_size: int = 256
    epochs: int = 100
    learning_rate: float = 2.5e-4
    weight_decay: float = 1e-6
    precision: str = "fp16"
    lr_scheduler: str = "none"
    min_learning_rate: float = 0.0

    # Compact / clustering
    instance_temperature: float = 0.2
    cluster_temperature: float = 0.1
    lambda_instance: float = 1.0
    lambda_cluster: float = 1.0
    sinkhorn_iters: int = 3
    sinkhorn_epsilon: float = 0.05

    # Subspace
    lambda_subspace: float = 0.15
    subspace_rank: int = 8
    subspace_queue_size: int = 64
    subspace_min_mass: float = 4.0
    subspace_every_n_steps: int = 2
    subspace_pca_q_offset: int = 4
    subspace_topk_clusters: int | None = None

    # DPP
    alpha_div_base: float = 0.6
    dpp_warmup_epochs: int = 20
    dpp_every_n_steps: int = 4
    dpp_jitter: float = 1e-5

    # Runtime / output
    eval_interval: int = 1
    workers: int = 8
    reload: bool = False
    seed: int = 42
    best_checkpoint_patience: int = 50
    model_path: str = "save/hypersphere_dpp/cub_vitl14"
    device: str | None = None

    @property
    def output_dir(self) -> Path:
        return Path(self.model_path)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _field_names() -> set[str]:
    return {item.name for item in fields(ExperimentConfig)}


def _warn_unknown_fields(
    values: Dict[str, Any], supported_fields: Iterable[str]
) -> None:
    supported = set(supported_fields)
    unknown = sorted(set(values) - supported)
    for key in unknown:
        warnings.warn(
            f"Unsupported config field '{key}' will be ignored in the CLIP prototype.",
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

    if config.dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"dataset must be one of {sorted(SUPPORTED_DATASETS)}, got {config.dataset!r}."
        )

    if config.backbone_name not in SUPPORTED_BACKBONE_FEATURE_DIMS:
        raise ValueError(
            f"backbone_name must be one of {sorted(SUPPORTED_BACKBONE_FEATURE_DIMS)}, got {config.backbone_name!r}."
        )

    expected_feature_dim = SUPPORTED_BACKBONE_FEATURE_DIMS[config.backbone_name]
    if config.feature_dim != expected_feature_dim:
        raise ValueError(
            f"feature_dim must be {expected_feature_dim} for {config.backbone_name} CLIP, got {config.feature_dim}."
        )

    if not 0.0 < config.validation_ratio < 1.0:
        raise ValueError(
            "validation_ratio must be in (0, 1) so training always has a validation split."
        )

    if config.class_num <= 1:
        raise ValueError("class_num must be greater than 1.")

    if config.eval_interval <= 0:
        raise ValueError("eval_interval must be a positive integer.")

    if config.subspace_every_n_steps <= 0:
        raise ValueError("subspace_every_n_steps must be a positive integer.")

    if config.subspace_pca_q_offset < 0:
        raise ValueError("subspace_pca_q_offset must be non-negative.")

    if config.subspace_topk_clusters is not None and config.subspace_topk_clusters <= 0:
        raise ValueError(
            "subspace_topk_clusters must be a positive integer when provided."
        )

    if config.dpp_every_n_steps <= 0:
        raise ValueError("dpp_every_n_steps must be a positive integer.")

    if config.subspace_min_mass <= 0:
        raise ValueError("subspace_min_mass must be positive.")

    if (
        config.official_eval_split is not None
        and config.official_eval_split not in SUPPORTED_EVAL_SPLITS
    ):
        raise ValueError(
            f"official_eval_split must be one of {sorted(SUPPORTED_EVAL_SPLITS)}, got {config.official_eval_split!r}."
        )
    if config.official_eval_split is not None:
        allowed_eval_splits = SUPPORTED_EVAL_SPLITS_BY_DATASET[config.dataset]
        if config.official_eval_split not in allowed_eval_splits:
            raise ValueError(
                f"official_eval_split for {config.dataset} must be one of {sorted(allowed_eval_splits)}, "
                f"got {config.official_eval_split!r}."
            )

    if config.precision not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"precision must be one of {sorted(SUPPORTED_PRECISIONS)}, got {config.precision!r}."
        )

    if config.lr_scheduler not in SUPPORTED_LR_SCHEDULERS:
        raise ValueError(
            f"lr_scheduler must be one of {sorted(SUPPORTED_LR_SCHEDULERS)}, got {config.lr_scheduler!r}."
        )

    if config.min_learning_rate < 0:
        raise ValueError("min_learning_rate must be non-negative.")

    if config.min_learning_rate > config.learning_rate:
        raise ValueError(
            "min_learning_rate must be less than or equal to learning_rate."
        )

    return config
