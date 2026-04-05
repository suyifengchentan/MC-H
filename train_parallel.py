from __future__ import annotations

import argparse
import logging
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import amp
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from hypersphere_dpp.checkpoint import save_checkpoint
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
    swav_cluster_loss,
)
from hypersphere_dpp.metrics import (
    clustering_selection_score,
    format_clustering_metrics,
)
from hypersphere_dpp.model import HypersphereDPPModel
from hypersphere_dpp.semantic import (
    build_semantic_prototype_state,
    semantic_cluster_alignment_loss,
    semantic_retrieval_alignment_loss,
    semantic_tac_alignment_losses,
)
from hypersphere_dpp.subspace import ClusterFeatureQueue, subspace_regularization_loss
from hypersphere_dpp.transforms import build_gpu_train_transform
from hypersphere_dpp.utils import ensure_dir, resolve_device, set_seed
from train import (
    build_optimizer,
    build_lr_scheduler,
    current_learning_rate,
    evaluate_splits,
    maybe_profile_sync,
    maybe_resume_training,
    prototype_anchor_loss,
    resolve_amp_settings,
    select_checkpoint_score,
    set_prototype_trainability,
    should_log_step_profile,
    use_cached_teacher_feature_candidates,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the hypersphere-DPP model with multi-GPU data parallelism."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the training config YAML."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override for direct single-process runs. "
        "Under torchrun multi-GPU mode, LOCAL_RANK decides the CUDA device.",
    )
    return parser.parse_args()


def distributed_env() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def resolve_parallel_device(
    requested_device: str | None, world_size: int, local_rank: int
) -> torch.device:
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Distributed multi-GPU training requires CUDA, but no CUDA device is available."
            )
        if requested_device and not requested_device.startswith("cuda"):
            raise ValueError(
                "Distributed multi-GPU training only supports CUDA devices. "
                "Omit --device or provide a CUDA device spec."
            )
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return resolve_device(requested_device)


def init_process_group_if_needed(device: torch.device, world_size: int) -> None:
    if world_size <= 1 or dist.is_initialized():
        return
    backend = "nccl" if device.type == "cuda" else "gloo"
    dist.init_process_group(backend=backend)


def is_main_process(rank: int) -> bool:
    return rank == 0


def setup_rank_logger(rank: int, output_dir) -> logging.Logger:
    if is_main_process(rank):
        return setup_training_logger(output_dir)

    logger = logging.getLogger(f"hypersphere_dpp.train.rank{rank}")
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.addHandler(logging.NullHandler())
    return logger


def build_distributed_train_loader(
    train_loader: DataLoader,
    rank: int,
    world_size: int,
    batch_size_mode: str,
) -> tuple[DataLoader, DistributedSampler, dict[str, int]]:
    requested_batch_size = int(train_loader.batch_size)
    if batch_size_mode == "global":
        per_rank_batch_size = requested_batch_size // world_size
        if per_rank_batch_size <= 0:
            raise ValueError(
                "batch_size is smaller than world_size under distributed_batch_size_mode='global'. "
                "Increase batch_size or switch distributed_batch_size_mode to 'per_rank'."
            )
    elif batch_size_mode == "per_rank":
        per_rank_batch_size = requested_batch_size
    else:
        raise ValueError(
            f"Unsupported distributed batch size mode: {batch_size_mode!r}"
        )

    sampler = DistributedSampler(
        train_loader.dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=train_loader.drop_last,
    )
    loader_kwargs = {
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
        loader_kwargs["prefetch_factor"] = train_loader.prefetch_factor
    effective_global_batch_size = per_rank_batch_size * world_size
    return DataLoader(**loader_kwargs), sampler, {
        "requested_batch_size": requested_batch_size,
        "per_rank_batch_size": per_rank_batch_size,
        "effective_global_batch_size": effective_global_batch_size,
    }


def reduce_metrics_across_ranks(
    metrics: dict[str, float], device: torch.device, world_size: int
) -> dict[str, float]:
    if world_size <= 1:
        return metrics

    keys = sorted(metrics)
    values = torch.tensor(
        [metrics[key] for key in keys], device=device, dtype=torch.float64
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= world_size
    return {key: float(values[index].item()) for index, key in enumerate(keys)}


@torch.no_grad()
def broadcast_training_control(
    *,
    best_metric: float,
    epochs_without_improvement: int,
    stop_training: bool,
    device: torch.device,
    world_size: int,
) -> tuple[float, int, bool]:
    if world_size <= 1:
        return best_metric, epochs_without_improvement, stop_training

    control = torch.tensor(
        [
            best_metric,
            float(epochs_without_improvement),
            float(int(stop_training)),
        ],
        device=device,
        dtype=torch.float64,
    )
    dist.broadcast(control, src=0)
    return (
        float(control[0].item()),
        int(control[1].item()),
        bool(int(control[2].item())),
    )


def train_one_epoch_distributed(
    model: DistributedDataParallel | HypersphereDPPModel,
    dataloader,
    sampler: DistributedSampler | None,
    gpu_train_transform,
    optimizer,
    scaler,
    queue: ClusterFeatureQueue,
    semantic_state,
    config,
    device: torch.device,
    epoch: int,
    logger,
    rank: int,
    world_size: int,
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
        "prototype_anchor_loss": 0.0,
        "alpha_div": 0.0,
        "rho": 0.0,
    }
    num_steps = len(dataloader)
    log_interval = max(1, num_steps // 5) if num_steps > 0 else 1
    step_profile_enabled = config.step_profile_enabled
    step_profile_cuda_sync = step_profile_enabled and config.step_profile_cuda_sync
    data_iter = iter(dataloader)
    previous_step_end = time.perf_counter()

    for step in range(num_steps):
        batch = next(data_iter)
        if step_profile_enabled:
            data_wait_ms = (time.perf_counter() - previous_step_end) * 1000.0
            maybe_profile_sync(device, step_profile_cuda_sync)
            transfer_start = time.perf_counter()

        teacher_feature_candidates = None
        if use_cached_teacher_feature_candidates(batch):
            teacher_feature_candidates = batch[TEACHER_FEATURE_CANDIDATES_KEY].to(
                device,
                non_blocking=True,
            )
        else:
            if gpu_train_transform is None:
                raise ValueError("gpu_train_transform must be provided for image-based training batches.")
            images = batch["image"].to(device, non_blocking=True)
            view1, view2 = gpu_train_transform(images)

        if step_profile_enabled:
            maybe_profile_sync(device, step_profile_cuda_sync)
            forward_start = time.perf_counter()
            batch_to_device_ms = (forward_start - transfer_start) * 1000.0

        optimizer.zero_grad(set_to_none=True)
        autocast_kwargs = {"device_type": amp_device_type, "enabled": use_amp}
        if amp_dtype is not None:
            autocast_kwargs["dtype"] = amp_dtype
        amp_context = amp.autocast(**autocast_kwargs) if use_amp else nullcontext()
        with amp_context:
            if teacher_feature_candidates is not None:
                teacher1, teacher2 = sample_two_teacher_feature_views(
                    teacher_feature_candidates
                )
                teacher_cat = torch.cat([teacher1, teacher2], dim=0)
                student_cat = module.encode_student(teacher_features=teacher_cat)
                logits_cat = module.prototype_logits(student_cat)
            else:
                teacher_cat, student_cat, logits_cat = model(
                    view1,
                    view2,
                )

            student1, student2 = student_cat.chunk(2)
            logits1, logits2 = logits_cat.chunk(2)

            inst_loss = nt_xent_loss(
                student1.float(), student2.float(), config.instance_temperature
            )
            cluster_loss, assign1, assign2 = swav_cluster_loss(
                logits_a=logits1.float(),
                logits_b=logits2.float(),
                temperature=config.cluster_temperature,
                sinkhorn_epsilon=config.sinkhorn_epsilon,
                sinkhorn_iters=config.sinkhorn_iters,
            )
            semantic_loss = student_cat.new_zeros(())
            semantic_retrieval_loss = student_cat.new_zeros(())
            prototype_anchor = student_cat.new_zeros(())
            if semantic_state is not None:
                if (
                    config.dataset == "ImageNet1K"
                    and config.semantic_mode == "tac_imagenet1k"
                    and (config.lambda_semantic > 0 or config.lambda_semantic_retrieval > 0)
                ):
                    semantic_loss, semantic_retrieval_loss = semantic_tac_alignment_losses(
                        logits=logits_cat,
                        student_features=student_cat,
                        teacher_features=teacher_cat,
                        semantic_state=semantic_state,
                        teacher_logit_scale=module.teacher_logit_scale(),
                        temperature=config.semantic_temperature,
                        target_topk=config.semantic_target_topk,
                        retrieval_topk=config.semantic_retrieval_topk,
                        class_blend=config.semantic_tac_class_blend,
                    )
                else:
                    if config.lambda_semantic > 0:
                        semantic_loss = semantic_cluster_alignment_loss(
                            logits=logits_cat,
                            teacher_features=teacher_cat,
                            text_features=semantic_state.class_text_features,
                            teacher_logit_scale=module.teacher_logit_scale(),
                            temperature=config.semantic_temperature,
                            target_topk=config.semantic_target_topk,
                        )
                    if config.lambda_semantic_retrieval > 0:
                        semantic_retrieval_loss = semantic_retrieval_alignment_loss(
                            student_features=student_cat,
                            teacher_features=teacher_cat,
                            text_features=semantic_state.retrieval_text_features,
                            teacher_logit_scale=module.teacher_logit_scale(),
                            retrieval_topk=config.semantic_retrieval_topk,
                        )
                if config.lambda_prototype_anchor > 0:
                    prototype_anchor = prototype_anchor_loss(
                        model=module,
                        target_prototypes=semantic_state.prototype_features.detach(),
                    )

            assignment_cat = torch.cat([assign1, assign2], dim=0)

            if step % config.subspace_every_n_steps == 0:
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
                dpp_loss, rho, _ = dpp_diversity_loss(
                    features=student_cat,
                    assignments=assignment_cat,
                    jitter=config.dpp_jitter,
                    class_aware=config.class_aware_dpp,
                    class_aware_min_mass=config.class_aware_dpp_min_mass,
                )
                epoch_progress = epoch + step / max(num_steps, 1)
                alpha_div = adaptive_diversity_weight(
                    rho=rho,
                    alpha_div_base=config.alpha_div_base,
                    alpha_div_mode=config.alpha_div_mode,
                    alpha_div_floor=config.alpha_div_floor,
                    alpha_div_rho_center=config.alpha_div_rho_center,
                    alpha_div_rho_scale=config.alpha_div_rho_scale,
                    alpha_div_max=config.alpha_div_max,
                    alpha_div_decay_start_epoch=config.alpha_div_decay_start_epoch,
                    alpha_div_decay_end_epoch=config.alpha_div_decay_end_epoch,
                    alpha_div_decay_target=config.alpha_div_decay_target,
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
                + config.lambda_prototype_anchor * prototype_anchor
                + config.lambda_subspace * subspace_loss
                + alpha_div * dpp_loss
            )

        if step_profile_enabled:
            maybe_profile_sync(device, step_profile_cuda_sync)
            backward_start = time.perf_counter()
            forward_ms = (backward_start - forward_start) * 1000.0

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        module.normalize_prototypes()
        queue.enqueue(teacher_cat, assignment_cat)

        if step_profile_enabled:
            maybe_profile_sync(device, step_profile_cuda_sync)
            step_end = time.perf_counter()
            backward_ms = (step_end - backward_start) * 1000.0
            step_total_ms = (step_end - transfer_start) * 1000.0
            previous_step_end = step_end
        else:
            previous_step_end = time.perf_counter()

        totals["loss"] += float(loss.item())
        totals["inst_loss"] += float(inst_loss.item())
        totals["cluster_loss"] += float(cluster_loss.item())
        totals["semantic_loss"] += float(semantic_loss.item())
        totals["semantic_retrieval_loss"] += float(semantic_retrieval_loss.item())
        totals["prototype_anchor_loss"] += float(prototype_anchor.item())
        totals["subspace_loss"] += float(subspace_loss.item())
        totals["dpp_loss"] += float(dpp_loss.item())
        totals["alpha_div"] += float(alpha_div.item())
        totals["rho"] += float(rho.item())

        if is_main_process(rank) and (
            (step + 1) % log_interval == 0 or step == 0 or (step + 1) == num_steps
        ):
            running_denom = step + 1
            logger.info(
                (
                    "epoch=%03d step=%04d/%04d "
                    "loss=%.4f inst=%.4f cluster=%.4f semantic=%.4f sem_ret=%.4f proto_reg=%.4f sub=%.4f dpp=%.4f alpha_div=%.4f rho=%.4f"
                ),
                epoch,
                step + 1,
                num_steps,
                totals["loss"] / running_denom,
                totals["inst_loss"] / running_denom,
                totals["cluster_loss"] / running_denom,
                totals["semantic_loss"] / running_denom,
                totals["semantic_retrieval_loss"] / running_denom,
                totals["prototype_anchor_loss"] / running_denom,
                totals["subspace_loss"] / running_denom,
                totals["dpp_loss"] / running_denom,
                totals["alpha_div"] / running_denom,
                totals["rho"] / running_denom,
            )
        if is_main_process(rank) and step_profile_enabled and should_log_step_profile(
            step,
            num_steps,
            config.step_profile_log_interval,
        ):
            logger.info(
                (
                    "epoch=%03d step=%04d/%04d "
                    "profile_ms data_wait=%.2f batch_to_device=%.2f forward=%.2f backward=%.2f step_total=%.2f"
                ),
                epoch,
                step + 1,
                num_steps,
                data_wait_ms,
                batch_to_device_ms,
                forward_ms,
                backward_ms,
                step_total_ms,
            )

    denom = max(num_steps, 1)
    metrics = {key: value / denom for key, value in totals.items()}
    return reduce_metrics_across_ranks(metrics, device, world_size)


def main():
    args = parse_args()
    rank, world_size, local_rank = distributed_env()
    device = resolve_parallel_device(args.device, world_size, local_rank)
    init_process_group_if_needed(device, world_size)

    try:
        config = load_experiment_config(args.config)
        set_seed(config.seed + rank)
        ensure_dir(config.output_dir)
        logger = setup_rank_logger(rank, config.output_dir)
        use_amp, amp_device_type, amp_dtype, use_grad_scaler = resolve_amp_settings(
            config.precision, device
        )
        if is_main_process(rank):
            logger.info("starting distributed training")
            logger.info(
                "rank=%d local_rank=%d world_size=%d device=%s output_dir=%s",
                rank,
                local_rank,
                world_size,
                device,
                config.output_dir,
            )
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
        if config.precision in {"fp16", "bf16"} and not use_amp and is_main_process(
            rank
        ):
            logger.warning(
                "precision=%s requested but AMP is unsupported on device=%s; falling back to fp32.",
                config.precision,
                device,
            )

        datasets = build_experiment_dataloaders(config)
        train_loader = datasets.train_loader
        train_sampler = None
        train_batch_info = {
            "requested_batch_size": int(train_loader.batch_size),
            "per_rank_batch_size": int(train_loader.batch_size),
            "effective_global_batch_size": int(train_loader.batch_size) * world_size,
        }
        if world_size > 1:
            train_loader, train_sampler, train_batch_info = build_distributed_train_loader(
                datasets.train_loader,
                rank,
                world_size,
                config.distributed_batch_size_mode,
            )
        train_prefetch_factor = getattr(train_loader, "prefetch_factor", None)
        if is_main_process(rank):
            logger.info("validation split: stratified=%s", datasets.val_is_stratified)
            logger.info(
                "train dataloader: workers=%d pin_memory=%s persistent_workers=%s prefetch_factor=%s world_size=%d distributed_sampler=%s",
                train_loader.num_workers,
                train_loader.pin_memory,
                train_loader.persistent_workers,
                train_prefetch_factor,
                world_size,
                train_sampler is not None,
            )
            if config.step_profile_enabled:
                logger.info(
                    "step profiling: enabled=True log_interval=%d cuda_sync=%s",
                    config.step_profile_log_interval,
                    config.step_profile_cuda_sync,
                )
            logger.info(
                "train batch: mode=%s requested=%d per_rank=%d effective_global=%d",
                config.distributed_batch_size_mode,
                train_batch_info["requested_batch_size"],
                train_batch_info["per_rank_batch_size"],
                train_batch_info["effective_global_batch_size"],
            )
            logger.info(
                "distributed loss semantics: local_per_rank batches with standard DDP gradient averaging"
            )
            if (
                world_size > 1
                and config.distributed_batch_size_mode == "global"
                and train_batch_info["effective_global_batch_size"]
                != train_batch_info["requested_batch_size"]
            ):
                logger.warning(
                    "requested global batch_size=%d is not divisible by world_size=%d; using effective_global_batch_size=%d.",
                    train_batch_info["requested_batch_size"],
                    world_size,
                    train_batch_info["effective_global_batch_size"],
                )

        if config.uses_imagenet1k_cached_teacher_views:
            if is_main_process(rank):
                logger.info(
                    "imagenet1k cached teacher views: enabled=True dir=%s num_views=%d dtype=%s sampled_views=2",
                    config.resolved_imagenet1k_cached_teacher_views_dir,
                    config.imagenet1k_cached_teacher_num_views,
                    config.imagenet1k_cached_teacher_dtype,
                )
            gpu_train_transform = None
        else:
            gpu_train_transform = build_gpu_train_transform(
                config.image_size, config.transform_profile
            ).to(device)
        base_model = HypersphereDPPModel(
            backbone_name=config.backbone_name,
            pretrained=config.pretrained,
            pretrained_checkpoint=config.pretrained_checkpoint,
            feature_dim=config.feature_dim,
            adapter_hidden_dim=config.adapter_hidden_dim,
            adapter_dropout=config.adapter_dropout,
            num_prototypes=config.class_num,
            device=device,
        ).to(device)
        semantic_state = None
        if config.semantic_prototype_enabled:
            semantic_state = build_semantic_prototype_state(
                backbone_model=base_model.backbone.model,
                backbone_name=config.backbone_name,
                dataset_name=config.dataset,
                dataset_dir=config.dataset_dir,
                prompt_style=config.semantic_prompt_style,
                aggregation=config.semantic_prototype_aggregation,
                imagenet_synonym_mode=config.semantic_imagenet_synonym_mode,
                semantic_primary_weight=config.semantic_primary_weight,
                semantic_template_topk=config.semantic_template_topk,
                semantic_mode=config.semantic_mode,
                device=device,
            )
            semantic_info = semantic_state.info
            # Initialize random prototypes with the semantic text features!
            with torch.no_grad():
                base_model.prototypes.prototypes.data.copy_(semantic_state.prototype_features)
                base_model.normalize_prototypes()

            if is_main_process(rank):
                logger.info(
                    "semantic prototypes: enabled=%s mode=%s lambda_semantic=%.4f lambda_semantic_retrieval=%.4f prompt_style=%s aggregation=%s synonym_mode=%s classes=%d templates_per_class=%d retrieval_bank=%d",
                    True,
                    semantic_info["semantic_mode"],
                    config.lambda_semantic,
                    config.lambda_semantic_retrieval,
                    semantic_info["prompt_style"],
                    semantic_info["aggregation"],
                    semantic_info["imagenet_synonym_mode"],
                    semantic_info["num_classes"],
                    semantic_info["num_templates"],
                    semantic_info["retrieval_bank_size"],
                )
        if is_main_process(rank):
            logger.info(
                "class-aware dpp: enabled=%s min_mass=%.4f",
                config.class_aware_dpp,
                config.class_aware_dpp_min_mass,
            )

        optimizer = build_optimizer(base_model, config)
        scheduler = build_lr_scheduler(optimizer, config)
        scaler = amp.GradScaler(
            device="cuda" if device.type == "cuda" else "cpu",
            enabled=use_grad_scaler,
        )
        queue = ClusterFeatureQueue(
            num_clusters=config.class_num,
            feature_dim=config.feature_dim,
            queue_size=config.subspace_queue_size,
            device=device,
        )

        start_epoch, global_step, best_metric = maybe_resume_training(
            model=base_model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            queue=queue,
            config=config,
            device=device,
        )
        if start_epoch > 0 and is_main_process(rank):
            logger.info(
                "resumed_from=%s epoch=%d global_step=%d best_metric=%.6f",
                config.output_dir / "last.pt",
                start_epoch,
                global_step,
                best_metric,
            )
        if is_main_process(rank):
                logger.info(
                    "optimizer: adapter_lr=%.8f prototype_lr=%.8f prototype_lr_multiplier=%.4f lambda_prototype_anchor=%.4f weight_decay=%.8f lr_scheduler=%s min_lr=%.8f",
                    current_learning_rate(optimizer),
                    float(optimizer.param_groups[1]["lr"]),
                    config.prototype_lr_multiplier,
                    config.lambda_prototype_anchor,
                    config.weight_decay,
                    config.lr_scheduler,
                    config.min_learning_rate,
                )

        model = base_model
        if world_size > 1:
            ddp_kwargs = {"device_ids": [local_rank]} if device.type == "cuda" else {}
            model = DistributedDataParallel(base_model, **ddp_kwargs)

        epochs_without_improvement = 0
        for epoch in range(start_epoch, config.epochs):
            set_prototype_trainability(
                model=base_model,
                freeze_epochs=config.prototype_freeze_epochs,
                epoch=epoch,
                logger=logger if is_main_process(rank) else None,
            )
            epoch_lr = current_learning_rate(optimizer)
            train_metrics = train_one_epoch_distributed(
                model=model,
                dataloader=train_loader,
                sampler=train_sampler,
                gpu_train_transform=gpu_train_transform,
                optimizer=optimizer,
                scaler=scaler,
                queue=queue,
                semantic_state=semantic_state,
                config=config,
                device=device,
                epoch=epoch,
                logger=logger,
                rank=rank,
                world_size=world_size,
            )
            global_step += len(train_loader)
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

            tracked_stop = False
            if is_main_process(rank):
                eval_model = model.module if isinstance(model, DistributedDataParallel) else model
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
                        queue_state=queue.state_dict(),
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
                    queue_state=queue.state_dict(),
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
                    tracked_stop = True

            best_metric, epochs_without_improvement, tracked_stop = broadcast_training_control(
                best_metric=best_metric,
                epochs_without_improvement=epochs_without_improvement,
                stop_training=tracked_stop,
                device=device,
                world_size=world_size,
            )
            if tracked_stop:
                break
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
