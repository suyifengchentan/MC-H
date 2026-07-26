from __future__ import annotations

import argparse
import logging
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import amp
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from hypersphere_dpp.checkpoint import load_checkpoint, save_checkpoint
from hypersphere_dpp.config import load_experiment_config
from hypersphere_dpp.data import (
    TEACHER_FEATURE_CANDIDATES_KEY,
    build_experiment_dataloaders,
    sample_two_teacher_feature_views,
)
from hypersphere_dpp.logging_utils import append_metrics_record, setup_training_logger
from hypersphere_dpp.losses import (
    adaptive_diversity_weight,
    dpp_diversity_loss,
    nt_xent_loss,
    spectral_cluster_loss,
    swav_cluster_loss,
)
from hypersphere_dpp.metrics import (
    clustering_selection_score,
    evaluate_clustering,
    format_clustering_metrics,
)
from hypersphere_dpp.model import build_model_from_config
from hypersphere_dpp.semantic import (
    build_semantic_text_features,
    semantic_cluster_alignment_loss,
    semantic_retrieval_alignment_loss,
    semantic_spectral_alignment_loss,
)
from hypersphere_dpp.subspace import (
    ClusterFeatureQueue,
    spectral_anchor_subspace_loss,
    subspace_regularization_loss,
    teacher_pca_subspace_loss,
)
from hypersphere_dpp.transforms import build_gpu_train_transform
from hypersphere_dpp.utils import ensure_dir, resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the CUB ViT-L/14 hypersphere-DPP prototype."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the training config YAML."
    )
    parser.add_argument(
        "--device", default=None, help="Override device string, e.g. cuda:0."
    )
    return parser.parse_args()


def resolve_amp_settings(
    precision: str, device: torch.device
) -> tuple[bool, str, torch.dtype | None, bool]:
    device_type = device.type
    if precision == "fp32":
        return False, device_type, None, False
    if precision == "fp16":
        enabled = device_type == "cuda"
        dtype = torch.float16 if enabled else None
        return enabled, device_type, dtype, enabled
    if precision == "bf16":
        enabled = device_type in {"cuda", "cpu"}
        dtype = torch.bfloat16 if enabled else None
        return enabled, device_type, dtype, False
    raise ValueError(f"Unsupported precision mode: {precision}")


def distributed_env() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def resolve_parallel_device(
    requested_device: str | None, world_size: int, local_rank: int
) -> torch.device:
    if world_size == 1:
        return resolve_device(requested_device)
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA.")
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def is_main_process(rank: int) -> bool:
    return rank == 0


def build_distributed_train_loader(
    train_loader: DataLoader, rank: int, world_size: int, batch_size_mode: str
) -> tuple[DataLoader, DistributedSampler]:
    requested_batch_size = int(train_loader.batch_size)
    if batch_size_mode == "global":
        if requested_batch_size % world_size:
            raise ValueError("Global batch_size must be divisible by world_size.")
        per_rank_batch_size = requested_batch_size // world_size
    else:
        per_rank_batch_size = requested_batch_size
    sampler = DistributedSampler(
        train_loader.dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=train_loader.drop_last,
    )
    kwargs = {
        "dataset": train_loader.dataset,
        "batch_size": per_rank_batch_size,
        "sampler": sampler,
        "shuffle": False,
        "drop_last": train_loader.drop_last,
        "num_workers": train_loader.num_workers,
        "pin_memory": train_loader.pin_memory,
        "persistent_workers": train_loader.persistent_workers,
        "collate_fn": train_loader.collate_fn,
    }
    if train_loader.num_workers > 0 and train_loader.prefetch_factor is not None:
        kwargs["prefetch_factor"] = train_loader.prefetch_factor
    return DataLoader(**kwargs), sampler


def gather_with_local_gradient(
    features: torch.Tensor, rank: int, world_size: int
) -> torch.Tensor:
    if world_size == 1:
        return features
    features = features.contiguous()
    gathered = [torch.empty_like(features) for _ in range(world_size)]
    dist.all_gather(gathered, features.detach())
    gathered[rank] = features
    return torch.cat(gathered, dim=0)


@torch.no_grad()
def gather_no_gradient(features: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return features
    features = features.contiguous()
    gathered = [torch.empty_like(features) for _ in range(world_size)]
    dist.all_gather(gathered, features)
    return torch.cat(gathered, dim=0)


def reduce_metrics(
    metrics: dict[str, float], device: torch.device, world_size: int
) -> dict[str, float]:
    if world_size == 1:
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([metrics[key] for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= world_size
    return {key: float(values[index]) for index, key in enumerate(keys)}


def resolve_spectral_cluster_cap(config, epoch: int) -> int:
    target_cap = config.max_spectral_clusters if config.max_spectral_clusters > 0 else config.class_num
    target_cap = max(config.min_spectral_clusters, int(target_cap))
    if config.initial_spectral_clusters > 0:
        initial_cap = int(config.initial_spectral_clusters)
    else:
        initial_cap = int(round(target_cap * 1.5))
    initial_cap = max(target_cap, initial_cap)
    if config.epochs <= 1 or initial_cap == target_cap:
        return target_cap

    progress = min(max(epoch / max(config.epochs - 1, 1), 0.0), 1.0)
    scheduled_cap = int(round(initial_cap + (target_cap - initial_cap) * progress))
    return max(target_cap, scheduled_cap)


def train_one_epoch(
    model,
    dataloader,
    gpu_train_transform,
    optimizer,
    scaler,
    queue: ClusterFeatureQueue | None,
    semantic_text_features: torch.Tensor | None,
    config,
    device: torch.device,
    epoch: int,
    logger,
    rank: int = 0,
    world_size: int = 1,
    sampler: DistributedSampler | None = None,
) -> dict[str, float]:
    if sampler is not None:
        sampler.set_epoch(epoch)
    model.train()
    module = model.module if isinstance(model, DistributedDataParallel) else model
    use_amp, amp_device_type, amp_dtype, _ = resolve_amp_settings(
        config.precision, device
    )
    totals = {
        "loss": 0.0,
        "inst_loss": 0.0,
        "cluster_loss": 0.0,
        "subspace_loss": 0.0,
        "dpp_loss": 0.0,
        "semantic_loss": 0.0,
        "semantic_retrieval_loss": 0.0,
        "alpha_div": 0.0,
        "rho": 0.0,
        "raw_estimated_k": 0.0,
        "estimated_k": 0.0,
    }
    num_steps = len(dataloader)
    log_interval = max(1, num_steps // 5) if num_steps > 0 else 1
    resolved_anchor_mode = config.subspace_anchor_mode
    if resolved_anchor_mode == "auto":
        resolved_anchor_mode = (
            "spectral_prototype" if config.clustering_mode == "spectral_free_k" else "teacher_pca"
        )
    spectral_cluster_cap = None
    if config.clustering_mode == "spectral_free_k":
        spectral_cluster_cap = resolve_spectral_cluster_cap(config, epoch)

    for step, batch in enumerate(dataloader):
        cached_candidates = None
        if TEACHER_FEATURE_CANDIDATES_KEY in batch:
            cached_candidates = batch[TEACHER_FEATURE_CANDIDATES_KEY].to(
                device, non_blocking=True
            )
        else:
            if gpu_train_transform is None:
                raise ValueError("Image batches require a GPU training transform.")
            images = batch["image"].to(device, non_blocking=True)
            view1, view2 = gpu_train_transform(images)

        optimizer.zero_grad(set_to_none=True)
        autocast_kwargs = {"device_type": amp_device_type, "enabled": use_amp}
        if amp_dtype is not None:
            autocast_kwargs["dtype"] = amp_dtype
        amp_context = amp.autocast(**autocast_kwargs) if use_amp else nullcontext()
        with amp_context:
            if cached_candidates is not None:
                teacher1, teacher2 = sample_two_teacher_feature_views(cached_candidates)
            else:
                teacher1 = module.encode_teacher(view1)
                teacher2 = module.encode_teacher(view2)
            local_teacher_cat = torch.cat([teacher1, teacher2], dim=0)
            local_student_cat = model(local_teacher_cat)
            local_student1, local_student2 = local_student_cat.chunk(2)
            teacher1 = gather_no_gradient(teacher1, world_size)
            teacher2 = gather_no_gradient(teacher2, world_size)
            student1 = gather_with_local_gradient(local_student1, rank, world_size)
            student2 = gather_with_local_gradient(local_student2, rank, world_size)
            teacher_cat = torch.cat([teacher1, teacher2], dim=0)
            student_cat = torch.cat([student1, student2], dim=0)

            if config.clustering_mode == "spectral_free_k":
                spectral_output = module.spectral_cluster(
                    student_features=student_cat,
                    temperature=config.spectral_assignment_temperature,
                    min_clusters=config.min_spectral_clusters,
                    max_clusters=spectral_cluster_cap,
                    momentum=config.spectral_proto_momentum,
                    cardinality_ema_momentum=config.spectral_cardinality_ema_momentum,
                    cardinality_hysteresis=config.spectral_cardinality_hysteresis,
                    update_state=True,
                )
                logits_cat = spectral_output["logits"]
                assignment_cat = spectral_output["assignments"]
                estimated_k = float(spectral_output["estimated_k"])
                raw_estimated_k = float(spectral_output["raw_estimated_k"])
                logits1, logits2 = logits_cat.chunk(2)
                assign1, assign2 = assignment_cat.chunk(2)
            else:
                logits_cat = module.prototype_logits(student_cat)
                logits1, logits2 = logits_cat.chunk(2)
                estimated_k = float(config.class_num)
                raw_estimated_k = estimated_k

            inst_loss = nt_xent_loss(
                student1.float(), student2.float(), config.instance_temperature
            )
            if config.clustering_mode == "spectral_free_k":
                cluster_loss, assign1, assign2 = spectral_cluster_loss(
                    logits_a=logits1.float(),
                    logits_b=logits2.float(),
                    prediction_temperature=config.cluster_temperature,
                    sharpen_temperature=config.spectral_sharpen_temperature,
                )
            else:
                cluster_loss, assign1, assign2 = swav_cluster_loss(
                    logits_a=logits1.float(),
                    logits_b=logits2.float(),
                    temperature=config.cluster_temperature,
                    sinkhorn_epsilon=config.sinkhorn_epsilon,
                    sinkhorn_iters=config.sinkhorn_iters,
                )
            assignment_cat = torch.cat([assign1, assign2], dim=0)

            if semantic_text_features is not None and config.lambda_semantic > 0:
                if config.clustering_mode == "spectral_free_k":
                    semantic_loss = semantic_spectral_alignment_loss(
                        prototypes=spectral_output["prototypes"],
                        assignments=assignment_cat,
                        teacher_features=teacher_cat,
                        text_features=semantic_text_features,
                        teacher_logit_scale=module.teacher_logit_scale(),
                        temperature=config.semantic_temperature,
                    )
                else:
                    semantic_loss = semantic_cluster_alignment_loss(
                        logits=logits_cat,
                        teacher_features=teacher_cat,
                        text_features=semantic_text_features,
                        teacher_logit_scale=module.teacher_logit_scale(),
                        temperature=config.semantic_temperature,
                    )
            else:
                semantic_loss = student_cat.new_zeros(())

            if semantic_text_features is not None and config.lambda_semantic_retrieval > 0:
                semantic_retrieval_loss = semantic_retrieval_alignment_loss(
                    student_features=student_cat,
                    teacher_features=teacher_cat,
                    text_features=semantic_text_features,
                    teacher_logit_scale=module.teacher_logit_scale(),
                    topk=config.semantic_retrieval_topk,
                )
            else:
                semantic_retrieval_loss = student_cat.new_zeros(())

            if step % config.subspace_every_n_steps == 0:
                if config.clustering_mode == "spectral_free_k" and resolved_anchor_mode == "spectral_prototype":
                    subspace_loss, _ = spectral_anchor_subspace_loss(
                        student_features=student_cat,
                        assignments=assignment_cat,
                        anchor_prototypes=spectral_output["prototypes"],
                        min_mass=config.subspace_min_mass,
                    )
                elif resolved_anchor_mode == "teacher_pca":
                    subspace_loss, _ = teacher_pca_subspace_loss(
                        student_features=student_cat,
                        teacher_features=teacher_cat,
                        assignments=assignment_cat,
                        rank=config.subspace_rank,
                        min_mass=config.subspace_min_mass,
                        pca_q_offset=config.subspace_pca_q_offset,
                        topk_clusters=config.subspace_topk_clusters,
                    )
                else:
                    subspace_loss, _ = subspace_regularization_loss(
                        student_features=student_cat,
                        teacher_features=teacher_cat,
                        assignments=assignment_cat,
                        queue=queue,
                        rank=config.subspace_rank,
                        min_mass=config.subspace_min_mass,
                        pca_q_offset=config.subspace_pca_q_offset,
                        topk_clusters=config.subspace_topk_clusters,
                    )
            else:
                subspace_loss = student_cat.new_zeros(())

            if step % config.dpp_every_n_steps == 0:
                if config.clustering_mode == "spectral_free_k":
                    dpp_loss, rho, _ = dpp_diversity_loss(
                        features=student_cat,
                        assignments=assignment_cat,
                        jitter=config.dpp_jitter,
                        relative_quality=spectral_output["relative_quality"],
                    )
                else:
                    dpp_loss, rho, _ = dpp_diversity_loss(
                        features=student_cat,
                        assignments=assignment_cat,
                        jitter=config.dpp_jitter,
                        class_aware=config.class_aware_dpp,
                        class_aware_min_mass=config.class_aware_dpp_min_mass,
                    )
                epoch_progress = epoch + step / max(num_steps, 1)
                if config.diversity_weight_mode == "constant":
                    alpha_div = rho.new_tensor(max(config.alpha_div_base, 0.0))
                else:
                    alpha_div = adaptive_diversity_weight(
                        rho=rho,
                        alpha_div_base=config.alpha_div_base,
                        epoch_progress=epoch_progress,
                        warmup_epochs=config.dpp_warmup_epochs,
                    )
            else:
                dpp_loss = student_cat.new_zeros(())
                rho = student_cat.new_zeros(())
                alpha_div = student_cat.new_zeros(())

            loss = (
                config.lambda_instance * inst_loss
                + config.lambda_cluster * cluster_loss
                + config.lambda_semantic * semantic_loss
                + config.lambda_semantic_retrieval * semantic_retrieval_loss
                + config.lambda_subspace * subspace_loss
                + alpha_div * dpp_loss
            )

        # Every rank evaluates the same global-batch objective, but only its
        # local student chunk retains gradients. DDP averages those gradients.
        scaler.scale(loss * world_size).backward()
        scaler.step(optimizer)
        scaler.update()
        module.normalize_prototypes()
        if queue is not None:
            queue.enqueue(teacher_cat, assignment_cat)

        totals["loss"] += float(loss.item())
        totals["inst_loss"] += float(inst_loss.item())
        totals["cluster_loss"] += float(cluster_loss.item())
        totals["semantic_loss"] += float(semantic_loss.item())
        totals["semantic_retrieval_loss"] += float(semantic_retrieval_loss.item())
        totals["subspace_loss"] += float(subspace_loss.item())
        totals["dpp_loss"] += float(dpp_loss.item())
        totals["alpha_div"] += float(alpha_div.item())
        totals["rho"] += float(rho.item())
        totals["raw_estimated_k"] += raw_estimated_k
        totals["estimated_k"] += estimated_k

        if is_main_process(rank) and (
            (step + 1) % log_interval == 0 or step == 0 or (step + 1) == num_steps
        ):
            running_denom = step + 1
            logger.info(
                (
                    "epoch=%03d step=%04d/%04d "
                    "loss=%.4f inst=%.4f cluster=%.4f semantic=%.4f sem_ret=%.4f sub=%.4f dpp=%.4f alpha_div=%.4f rho=%.4f raw_k=%.2f stable_k=%.2f"
                ),
                epoch,
                step + 1,
                num_steps,
                totals["loss"] / running_denom,
                totals["inst_loss"] / running_denom,
                totals["cluster_loss"] / running_denom,
                totals["semantic_loss"] / running_denom,
                totals["semantic_retrieval_loss"] / running_denom,
                totals["subspace_loss"] / running_denom,
                totals["dpp_loss"] / running_denom,
                totals["alpha_div"] / running_denom,
                totals["rho"] / running_denom,
                totals["raw_estimated_k"] / running_denom,
                totals["estimated_k"] / running_denom,
            )

    denom = max(len(dataloader), 1)
    return reduce_metrics({key: value / denom for key, value in totals.items()}, device, world_size)


def build_lr_scheduler(optimizer, config):
    if config.lr_scheduler == "none":
        return None
    if config.lr_scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.min_learning_rate,
        )
    raise ValueError(f"Unsupported lr scheduler: {config.lr_scheduler}")


def current_learning_rate(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def evaluate_splits(
    model, val_loader, device: torch.device
) -> dict[str, dict[str, float]]:
    return {"val": evaluate_clustering(model, val_loader, device)}


def maybe_refresh_free_k_prototypes(
    model,
    reference_loader,
    config,
    device: torch.device,
    logger,
) -> None:
    if config.clustering_mode != "spectral_free_k":
        return
    if not config.spectral_eval_refresh_enabled:
        return
    refresh_stats = model.refresh_spectral_prototypes(
        dataloader=reference_loader,
        device=device,
        assignment_temperature=config.spectral_eval_refresh_temperature,
        passes=config.spectral_eval_refresh_passes,
        min_cluster_mass=config.spectral_eval_refresh_min_mass,
    )
    logger.info(
        "spectral_eval_refresh: passes=%d estimated_k=%.2f mean_cluster_mass=%.4f",
        config.spectral_eval_refresh_passes,
        refresh_stats["estimated_k"],
        refresh_stats["mean_cluster_mass"],
    )


def select_checkpoint_score(
    eval_metrics: dict[str, dict[str, float]],
    logger,
) -> tuple[str, dict[str, float], float]:
    split_name = "val"
    metrics = eval_metrics[split_name]
    score = clustering_selection_score(metrics)
    return split_name, metrics, score


def maybe_resume_training(model, optimizer, scaler, scheduler, queue, config, device):
    last_checkpoint = config.output_dir / "last.pt"
    if not config.reload or not last_checkpoint.exists():
        return 0, 0, float("-inf")

    checkpoint = load_checkpoint(
        checkpoint_path=last_checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        map_location=device,
    )
    if queue is not None:
        queue.load_state_dict(checkpoint.get("queue", {}))
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_metric = float(checkpoint.get("best_metric", float("-inf")))
    return start_epoch, global_step, best_metric


def main():
    args = parse_args()
    rank, world_size, local_rank = distributed_env()
    device = resolve_parallel_device(args.device, world_size, local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    config = load_experiment_config(args.config)
    set_seed(config.seed + rank)
    if is_main_process(rank):
        ensure_dir(config.output_dir)
    if world_size > 1:
        dist.barrier()
    if is_main_process(rank):
        logger = setup_training_logger(config.output_dir)
    else:
        logger = logging.getLogger(f"hypersphere_dpp.rank{rank}")
        logger.handlers = [logging.NullHandler()]
        logger.propagate = False
    use_amp, amp_device_type, amp_dtype, use_grad_scaler = resolve_amp_settings(
        config.precision, device
    )
    logger.info("starting training")
    logger.info("rank=%d world_size=%d device=%s", rank, world_size, device)
    logger.info("device=%s output_dir=%s", device, config.output_dir)
    logger.info("config=%s", config.as_dict())
    logger.info(
        "validation cadence: run val every %d epoch(s) based on config.eval_interval",
        config.eval_interval,
    )
    logger.info(
        "validation coverage: evaluate full val split every eval round (eval_batch_size=%d)",
        config.eval_batch_size,
    )
    logger.info(
        "precision=%s amp_enabled=%s amp_device_type=%s amp_dtype=%s grad_scaler=%s",
        config.precision,
        use_amp,
        amp_device_type,
        amp_dtype,
        use_grad_scaler,
    )
    if config.precision in {"fp16", "bf16"} and not use_amp:
        logger.warning(
            "precision=%s requested but AMP is unsupported on device=%s; falling back to fp32.",
            config.precision,
            device,
        )

    datasets = build_experiment_dataloaders(config)
    train_sampler = None
    if world_size > 1:
        datasets.train_loader, train_sampler = build_distributed_train_loader(
            datasets.train_loader,
            rank,
            world_size,
            config.distributed_batch_size_mode,
        )
    train_prefetch_factor = getattr(datasets.train_loader, "prefetch_factor", None)
    logger.info("validation split: stratified=%s", datasets.val_is_stratified)
    logger.info(
        "train dataloader: workers=%d pin_memory=%s persistent_workers=%s prefetch_factor=%s",
        datasets.train_loader.num_workers,
        datasets.train_loader.pin_memory,
        datasets.train_loader.persistent_workers,
        train_prefetch_factor,
    )
    gpu_train_transform = None
    if not config.uses_cached_teacher_views:
        gpu_train_transform = build_gpu_train_transform(
            config.image_size, config.transform_profile
        ).to(device)
    model = build_model_from_config(config, device=device).to(device)
    semantic_text_features = None
    if config.clustering_mode == "fixed_k" and config.semantic_prototype_enabled:
        logger.info(
            "semantic prototypes requested but disabled for clustering_mode=%s to keep fixed-K baselines comparable",
            config.clustering_mode,
        )
    elif config.semantic_prototype_enabled:
        semantic_text_features, semantic_info = build_semantic_text_features(
            backbone_model=model.backbone.model,
            backbone_name=config.backbone_name,
            dataset_name=config.dataset,
            dataset_dir=config.dataset_dir,
            prompt_style=config.semantic_prompt_style,
            aggregation=config.semantic_prototype_aggregation,
            device=device,
        )
        logger.info(
            "semantic prototypes: enabled=%s lambda_semantic=%.4f lambda_semantic_retrieval=%.4f retrieval_topk=%d prompt_style=%s aggregation=%s classes=%d templates_per_class=%d",
            True,
            config.lambda_semantic,
            config.lambda_semantic_retrieval,
            config.semantic_retrieval_topk,
            semantic_info["prompt_style"],
            semantic_info["aggregation"],
            semantic_info["num_classes"],
            semantic_info["num_templates"],
        )
    logger.info(
        "class-aware dpp: enabled=%s min_mass=%.4f",
        config.class_aware_dpp,
        config.class_aware_dpp_min_mass,
    )
    logger.info(
        "clustering_mode=%s min_spectral_clusters=%d max_spectral_clusters=%d cardinality_ema_momentum=%.3f cardinality_hysteresis=%d",
        config.clustering_mode,
        config.min_spectral_clusters,
        config.max_spectral_clusters,
        config.spectral_cardinality_ema_momentum,
        config.spectral_cardinality_hysteresis,
    )
    if config.clustering_mode == "spectral_free_k":
        target_spectral_cap = config.max_spectral_clusters if config.max_spectral_clusters > 0 else config.class_num
        initial_spectral_cap = (
            config.initial_spectral_clusters
            if config.initial_spectral_clusters > 0
            else int(round(target_spectral_cap * 1.5))
        )
        logger.info(
            "spectral cluster cap schedule: initial=%d target=%d epochs=%d",
            initial_spectral_cap,
            target_spectral_cap,
            config.epochs,
        )
    logger.info(
        "diversity_weight_mode=%s subspace_anchor_mode=%s",
        config.diversity_weight_mode,
        config.subspace_anchor_mode,
    )
    logger.info(
        "spectral_eval_refresh: enabled=%s passes=%d temperature=%.4f min_mass=%.4f",
        config.spectral_eval_refresh_enabled,
        config.spectral_eval_refresh_passes,
        config.spectral_eval_refresh_temperature,
        config.spectral_eval_refresh_min_mass,
    )
    optimizer = optim.AdamW(
        model.trainable_parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = build_lr_scheduler(optimizer, config)
    scaler = amp.GradScaler(
        device="cuda" if device.type == "cuda" else "cpu",
        enabled=use_grad_scaler,
    )
    queue = None
    resolved_anchor_mode = config.subspace_anchor_mode
    if resolved_anchor_mode == "auto":
        resolved_anchor_mode = (
            "spectral_prototype" if config.clustering_mode == "spectral_free_k" else "teacher_pca"
        )
    if config.clustering_mode == "fixed_k" and resolved_anchor_mode == "teacher_pca":
        queue = ClusterFeatureQueue(
            num_clusters=config.class_num,
            feature_dim=config.feature_dim,
            queue_size=config.subspace_queue_size,
            device=device,
        )

    start_epoch, global_step, best_metric = maybe_resume_training(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        queue=queue,
        config=config,
        device=device,
    )
    if start_epoch > 0:
        logger.info(
            "resumed_from=%s epoch=%d global_step=%d best_metric=%.6f",
            config.output_dir / "last.pt",
            start_epoch,
            global_step,
            best_metric,
        )
    logger.info(
        "optimizer: lr=%.8f weight_decay=%.8f lr_scheduler=%s min_lr=%.8f",
        current_learning_rate(optimizer),
        config.weight_decay,
        config.lr_scheduler,
        config.min_learning_rate,
    )

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    epochs_without_improvement = 0
    for epoch in range(start_epoch, config.epochs):
        epoch_lr = current_learning_rate(optimizer)
        if config.clustering_mode == "spectral_free_k":
            logger.info(
                "epoch=%03d spectral_max_clusters=%d",
                epoch,
                resolve_spectral_cluster_cap(config, epoch),
            )
        train_metrics = train_one_epoch(
            model=model,
            dataloader=datasets.train_loader,
            gpu_train_transform=gpu_train_transform,
            optimizer=optimizer,
            scaler=scaler,
            queue=queue,
            semantic_text_features=semantic_text_features,
            config=config,
            device=device,
            epoch=epoch,
            logger=logger,
            rank=rank,
            world_size=world_size,
            sampler=train_sampler,
        )
        global_step += len(datasets.train_loader)
        if is_main_process(rank):
            logger.info("epoch=%03d split=train metrics=%s", epoch, train_metrics)
            append_metrics_record(
                config.output_dir,
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "lr": epoch_lr,
                    "split": "train",
                    **train_metrics,
                },
            )
        if scheduler is not None:
            scheduler.step()
            if is_main_process(rank):
                logger.info(
                    "epoch=%03d lr=%.8f next_lr=%.8f scheduler=%s",
                    epoch,
                    epoch_lr,
                    current_learning_rate(optimizer),
                    config.lr_scheduler,
                )

        if (epoch + 1) % config.eval_interval != 0:
            continue

        if world_size > 1 and not is_main_process(rank):
            dist.barrier()
            continue

        eval_model = model.module if isinstance(model, DistributedDataParallel) else model

        maybe_refresh_free_k_prototypes(
            model=eval_model,
            reference_loader=datasets.train_eval_loader,
            config=config,
            device=device,
            logger=logger,
        )
        eval_metrics = evaluate_splits(eval_model, datasets.val_loader, device)
        for split_name, metrics in eval_metrics.items():
            selection_score = clustering_selection_score(metrics)
            logger.info(
                "epoch=%03d split=%s %s selection_score=%.6f",
                epoch,
                split_name,
                format_clustering_metrics(metrics),
                selection_score,
            )
            append_metrics_record(
                config.output_dir,
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "split": split_name,
                    "selection_score": selection_score,
                    **metrics,
                },
            )

        selected_split, selected_metrics, tracked_metric = select_checkpoint_score(
            eval_metrics, logger
        )
        is_best = tracked_metric > best_metric
        if is_best:
            best_metric = tracked_metric
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path=config.output_dir / "best.pt",
                model=eval_model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                queue_state=queue.state_dict() if queue is not None else {},
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                config=config.as_dict(),
            )
            logger.info(
                "saved checkpoint=%s split=%s %s selection_score=%.6f",
                config.output_dir / "best.pt",
                selected_split,
                format_clustering_metrics(selected_metrics),
                best_metric,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            checkpoint_path=config.output_dir / "last.pt",
            model=eval_model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            queue_state=queue.state_dict() if queue is not None else {},
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
            config=config.as_dict(),
        )
        logger.info(
            "saved checkpoint=%s current_best_selection_score=%.6f",
            config.output_dir / "last.pt",
            best_metric,
        )

        if epochs_without_improvement >= config.best_checkpoint_patience:
            logger.info(
                "stopping early after %d eval rounds without improvement",
                config.best_checkpoint_patience,
            )
            if world_size > 1:
                dist.barrier()
            break

        if world_size > 1:
            dist.barrier()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
