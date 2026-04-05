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
    "ImageNet1K",
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
    "ImageNet1K": {"val"},
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
SUPPORTED_SEMANTIC_AGGREGATIONS = {"mean", "max", "first"}
SUPPORTED_SEMANTIC_PROMPT_STYLES = {"auto", "single", "clip_ensemble"}
SUPPORTED_CLUSTERING_MODES = {"fixed_k", "spectral_free_k"}
SUPPORTED_DIVERSITY_WEIGHT_MODES = {"adaptive", "constant"}
SUPPORTED_SUBSPACE_ANCHOR_MODES = {"auto", "teacher_pca", "spectral_prototype"}


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
    clustering_mode: str = "fixed_k"
    min_spectral_clusters: int = 2
    max_spectral_clusters: int = 0
    initial_spectral_clusters: int = 0
    spectral_assignment_temperature: float = 0.2
    spectral_sharpen_temperature: float = 0.1
    spectral_proto_momentum: float = 0.9
    spectral_eval_refresh_enabled: bool = True
    spectral_eval_refresh_passes: int = 1
    spectral_eval_refresh_temperature: float = 0.2
    spectral_eval_refresh_min_mass: float = 1.0
    diversity_weight_mode: str = "adaptive"
    subspace_anchor_mode: str = "auto"

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
    semantic_prototype_enabled: bool = False
    semantic_prompt_style: str = "auto"
    semantic_prototype_aggregation: str = "mean"
    lambda_semantic: float = 0.0
    lambda_semantic_retrieval: float = 0.0
    semantic_temperature: float = 0.1
    semantic_retrieval_topk: int = 8

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
    class_aware_dpp: bool = False
    class_aware_dpp_min_mass: float = 1.0

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

    if config.clustering_mode not in SUPPORTED_CLUSTERING_MODES:
        raise ValueError(
            "clustering_mode must be one of "
            f"{sorted(SUPPORTED_CLUSTERING_MODES)}, got {config.clustering_mode!r}."
        )

    if config.diversity_weight_mode not in SUPPORTED_DIVERSITY_WEIGHT_MODES:
        raise ValueError(
            "diversity_weight_mode must be one of "
            f"{sorted(SUPPORTED_DIVERSITY_WEIGHT_MODES)}, got {config.diversity_weight_mode!r}."
        )

    if config.subspace_anchor_mode not in SUPPORTED_SUBSPACE_ANCHOR_MODES:
        raise ValueError(
            "subspace_anchor_mode must be one of "
            f"{sorted(SUPPORTED_SUBSPACE_ANCHOR_MODES)}, got {config.subspace_anchor_mode!r}."
        )
    if config.clustering_mode == "fixed_k" and config.subspace_anchor_mode == "spectral_prototype":
        raise ValueError(
            "subspace_anchor_mode='spectral_prototype' requires clustering_mode='spectral_free_k'."
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

    if config.min_spectral_clusters <= 0:
        raise ValueError("min_spectral_clusters must be a positive integer.")

    if config.max_spectral_clusters < 0:
        raise ValueError("max_spectral_clusters must be non-negative.")

    if config.initial_spectral_clusters < 0:
        raise ValueError("initial_spectral_clusters must be non-negative.")

    if (
        config.max_spectral_clusters > 0
        and config.max_spectral_clusters < config.min_spectral_clusters
    ):
        raise ValueError(
            "max_spectral_clusters must be >= min_spectral_clusters when provided."
        )

    if (
        config.initial_spectral_clusters > 0
        and config.max_spectral_clusters > 0
        and config.initial_spectral_clusters < config.max_spectral_clusters
    ):
        raise ValueError(
            "initial_spectral_clusters must be >= max_spectral_clusters when both are provided."
        )

    if config.spectral_assignment_temperature <= 0:
        raise ValueError("spectral_assignment_temperature must be positive.")

    if config.spectral_sharpen_temperature <= 0:
        raise ValueError("spectral_sharpen_temperature must be positive.")

    if not 0.0 <= config.spectral_proto_momentum < 1.0:
        raise ValueError("spectral_proto_momentum must be in [0, 1).")

    if config.spectral_eval_refresh_passes <= 0:
        raise ValueError("spectral_eval_refresh_passes must be a positive integer.")

    if config.spectral_eval_refresh_temperature <= 0:
        raise ValueError("spectral_eval_refresh_temperature must be positive.")

    if config.spectral_eval_refresh_min_mass <= 0:
        raise ValueError("spectral_eval_refresh_min_mass must be positive.")

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

    if config.class_aware_dpp_min_mass <= 0:
        raise ValueError("class_aware_dpp_min_mass must be positive.")

    if config.semantic_prompt_style not in SUPPORTED_SEMANTIC_PROMPT_STYLES:
        raise ValueError(
            "semantic_prompt_style must be one of "
            f"{sorted(SUPPORTED_SEMANTIC_PROMPT_STYLES)}, got {config.semantic_prompt_style!r}."
        )

    if (
        config.semantic_prototype_aggregation
        not in SUPPORTED_SEMANTIC_AGGREGATIONS
    ):
        raise ValueError(
            "semantic_prototype_aggregation must be one of "
            f"{sorted(SUPPORTED_SEMANTIC_AGGREGATIONS)}, got {config.semantic_prototype_aggregation!r}."
        )

    if config.lambda_semantic < 0:
        raise ValueError("lambda_semantic must be non-negative.")

    if config.lambda_semantic_retrieval < 0:
        raise ValueError("lambda_semantic_retrieval must be non-negative.")

    if config.semantic_temperature <= 0:
        raise ValueError("semantic_temperature must be positive.")

    if config.semantic_retrieval_topk <= 0:
        raise ValueError("semantic_retrieval_topk must be a positive integer.")

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
