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
    "ImageNet1K",
    "Image50",
    "Image100",
    "Image200",
    "TinyImageNet200",
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
    "ImageNet1K": {"val", "test"},
    "Image50": {"val"},
    "Image100": {"val"},
    "Image200": {"val"},
    "TinyImageNet200": {"val"},
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
SUPPORTED_SEMANTIC_IMAGENET_SYNONYM_MODES = {"mean_all", "primary_weighted"}
SUPPORTED_SEMANTIC_MODES = {"legacy", "tac_imagenet1k"}
SUPPORTED_DISTRIBUTED_BATCH_SIZE_MODES = {"global", "per_rank"}
SUPPORTED_IMAGENET1K_CACHED_TEACHER_DTYPES = {"float16", "float32"}
SUPPORTED_ALPHA_DIV_MODES = {"clamp_positive", "sigmoid_floor"}


@dataclass
class ExperimentConfig:
    # Data
    dataset: str = "CUB_200_2011"
    dataset_dir: str = ""
    class_num: int = 200
    validation_ratio: float = 0.1
    validation_seed: int = 42
    official_eval_split: str | None = None
    imagenet1k_cached_teacher_views_enabled: bool = False
    imagenet1k_cached_teacher_views_dir: str | None = None
    imagenet1k_cached_teacher_num_views: int = 6
    imagenet1k_cached_teacher_dtype: str = "float16"

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
    semantic_prototype_enabled: bool = False
    semantic_mode: str = "legacy"
    semantic_prompt_style: str = "auto"
    semantic_prototype_aggregation: str = "mean"
    lambda_semantic: float = 0.0
    lambda_semantic_retrieval: float = 0.0
    semantic_temperature: float = 0.1
    semantic_imagenet_synonym_mode: str = "mean_all"
    semantic_primary_weight: float = 0.8
    semantic_template_topk: int = 0
    semantic_target_topk: int = 0
    semantic_retrieval_topk: int = 0
    semantic_tac_class_blend: float = 0.5
    prototype_freeze_epochs: int = 0
    prototype_lr_multiplier: float = 1.0
    lambda_prototype_anchor: float = 0.0

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
    alpha_div_mode: str = "clamp_positive"
    alpha_div_floor: float = 0.0
    alpha_div_rho_center: float = 0.0
    alpha_div_rho_scale: float = 0.01
    alpha_div_max: float = 1.0
    alpha_div_decay_start_epoch: float = -1.0
    alpha_div_decay_end_epoch: float = -1.0
    alpha_div_decay_target: float = 1.0
    dpp_warmup_epochs: int = 20
    dpp_every_n_steps: int = 4
    dpp_jitter: float = 1e-5
    class_aware_dpp: bool = False
    class_aware_dpp_min_mass: float = 1.0

    # Runtime / output
    eval_interval: int = 1
    workers: int = 8
    train_prefetch_factor: int | None = None
    eval_prefetch_factor: int | None = None
    persistent_workers: bool | None = None
    step_profile_enabled: bool = False
    step_profile_log_interval: int = 1
    step_profile_cuda_sync: bool = True
    distributed_batch_size_mode: str = "per_rank"
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

    @property
    def uses_imagenet1k_cached_teacher_views(self) -> bool:
        return self.dataset == "ImageNet1K" and self.imagenet1k_cached_teacher_views_enabled

    @property
    def resolved_imagenet1k_cached_teacher_views_dir(self) -> Path | None:
        if not self.uses_imagenet1k_cached_teacher_views:
            return None
        if self.imagenet1k_cached_teacher_views_dir:
            return Path(self.imagenet1k_cached_teacher_views_dir)

        backbone_tag = self.backbone_name.lower().replace("-", "").replace("/", "")
        cache_dir_name = (
            f"teacher_cache_{backbone_tag}_{self.transform_profile}_"
            f"views{self.imagenet1k_cached_teacher_num_views}"
        )
        return Path(self.dataset_dir) / cache_dir_name


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

    if config.imagenet1k_cached_teacher_views_enabled and config.dataset != "ImageNet1K":
        raise ValueError(
            "imagenet1k_cached_teacher_views_enabled is only supported when dataset='ImageNet1K'."
        )

    if config.imagenet1k_cached_teacher_num_views < 2:
        raise ValueError("imagenet1k_cached_teacher_num_views must be at least 2.")

    if (
        config.imagenet1k_cached_teacher_dtype
        not in SUPPORTED_IMAGENET1K_CACHED_TEACHER_DTYPES
    ):
        raise ValueError(
            "imagenet1k_cached_teacher_dtype must be one of "
            f"{sorted(SUPPORTED_IMAGENET1K_CACHED_TEACHER_DTYPES)}, "
            f"got {config.imagenet1k_cached_teacher_dtype!r}."
        )

    if config.eval_interval <= 0:
        raise ValueError("eval_interval must be a positive integer.")

    if config.workers < 0:
        raise ValueError("workers must be non-negative.")

    if config.train_prefetch_factor is not None and config.train_prefetch_factor <= 0:
        raise ValueError("train_prefetch_factor must be positive when provided.")

    if config.eval_prefetch_factor is not None and config.eval_prefetch_factor <= 0:
        raise ValueError("eval_prefetch_factor must be positive when provided.")

    if config.step_profile_log_interval <= 0:
        raise ValueError("step_profile_log_interval must be a positive integer.")

    if config.persistent_workers and config.workers == 0:
        raise ValueError(
            "persistent_workers cannot be true when workers=0."
        )

    if config.distributed_batch_size_mode not in SUPPORTED_DISTRIBUTED_BATCH_SIZE_MODES:
        raise ValueError(
            "distributed_batch_size_mode must be one of "
            f"{sorted(SUPPORTED_DISTRIBUTED_BATCH_SIZE_MODES)}, got {config.distributed_batch_size_mode!r}."
        )

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

    if config.alpha_div_mode not in SUPPORTED_ALPHA_DIV_MODES:
        raise ValueError(
            "alpha_div_mode must be one of "
            f"{sorted(SUPPORTED_ALPHA_DIV_MODES)}, got {config.alpha_div_mode!r}."
        )

    if config.alpha_div_base < 0:
        raise ValueError("alpha_div_base must be non-negative.")

    if not 0.0 <= config.alpha_div_floor <= 1.0:
        raise ValueError("alpha_div_floor must be in [0, 1].")

    if config.alpha_div_rho_scale <= 0:
        raise ValueError("alpha_div_rho_scale must be positive.")

    if config.alpha_div_max <= 0:
        raise ValueError("alpha_div_max must be positive.")

    if (
        config.alpha_div_decay_end_epoch > 0
        and config.alpha_div_decay_start_epoch >= config.alpha_div_decay_end_epoch
    ):
        raise ValueError(
            "alpha_div_decay_start_epoch must be smaller than alpha_div_decay_end_epoch."
        )

    if not 0.0 <= config.alpha_div_decay_target <= 1.0:
        raise ValueError("alpha_div_decay_target must be in [0, 1].")

    if config.subspace_min_mass <= 0:
        raise ValueError("subspace_min_mass must be positive.")

    if config.class_aware_dpp_min_mass <= 0:
        raise ValueError("class_aware_dpp_min_mass must be positive.")

    if config.semantic_prompt_style not in SUPPORTED_SEMANTIC_PROMPT_STYLES:
        raise ValueError(
            "semantic_prompt_style must be one of "
            f"{sorted(SUPPORTED_SEMANTIC_PROMPT_STYLES)}, got {config.semantic_prompt_style!r}."
        )

    if config.semantic_mode not in SUPPORTED_SEMANTIC_MODES:
        raise ValueError(
            "semantic_mode must be one of "
            f"{sorted(SUPPORTED_SEMANTIC_MODES)}, got {config.semantic_mode!r}."
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

    if (
        config.semantic_imagenet_synonym_mode
        not in SUPPORTED_SEMANTIC_IMAGENET_SYNONYM_MODES
    ):
        raise ValueError(
            "semantic_imagenet_synonym_mode must be one of "
            f"{sorted(SUPPORTED_SEMANTIC_IMAGENET_SYNONYM_MODES)}, "
            f"got {config.semantic_imagenet_synonym_mode!r}."
        )

    if not 0.0 <= config.semantic_primary_weight <= 1.0:
        raise ValueError("semantic_primary_weight must be in [0, 1].")

    if config.semantic_template_topk < 0:
        raise ValueError("semantic_template_topk must be non-negative.")

    if config.semantic_target_topk < 0:
        raise ValueError("semantic_target_topk must be non-negative.")

    if config.semantic_retrieval_topk < 0:
        raise ValueError("semantic_retrieval_topk must be non-negative.")

    if not 0.0 <= config.semantic_tac_class_blend <= 1.0:
        raise ValueError("semantic_tac_class_blend must be in [0, 1].")

    if config.prototype_freeze_epochs < 0:
        raise ValueError("prototype_freeze_epochs must be non-negative.")

    if config.prototype_lr_multiplier <= 0:
        raise ValueError("prototype_lr_multiplier must be positive.")

    if config.lambda_prototype_anchor < 0:
        raise ValueError("lambda_prototype_anchor must be non-negative.")

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
