from __future__ import annotations

import argparse
from contextlib import nullcontext

import torch
from torch import amp
from torch import optim

from hypersphere_dpp.checkpoint import load_checkpoint, save_checkpoint
from hypersphere_dpp.config import load_experiment_config
from hypersphere_dpp.data import build_experiment_dataloaders
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
) -> dict[str, float]:
    model.train()
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
        "alpha_div": 0.0,
        "rho": 0.0,
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
        images = batch["image"].to(device, non_blocking=True)
        view1, view2 = gpu_train_transform(images)

        optimizer.zero_grad(set_to_none=True)
        autocast_kwargs = {"device_type": amp_device_type, "enabled": use_amp}
        if amp_dtype is not None:
            autocast_kwargs["dtype"] = amp_dtype
        amp_context = amp.autocast(**autocast_kwargs) if use_amp else nullcontext()
        with amp_context:
            teacher1 = model.encode_teacher(view1)
            teacher2 = model.encode_teacher(view2)
            teacher_cat = torch.cat([teacher1, teacher2], dim=0)
            student_cat = model.encode_student(teacher_features=teacher_cat)
            student1, student2 = student_cat.chunk(2)

            if config.clustering_mode == "spectral_free_k":
                spectral_output = model.spectral_cluster(
                    student_features=student_cat,
                    temperature=config.spectral_assignment_temperature,
                    min_clusters=config.min_spectral_clusters,
                    max_clusters=spectral_cluster_cap,
                    momentum=config.spectral_proto_momentum,
                    update_state=True,
                )
                logits_cat = spectral_output["logits"]
                assignment_cat = spectral_output["assignments"]
                estimated_k = float(spectral_output["estimated_k"])
                logits1, logits2 = logits_cat.chunk(2)
                assign1, assign2 = assignment_cat.chunk(2)
            else:
                logits_cat = model.prototype_logits(student_cat)
                logits1, logits2 = logits_cat.chunk(2)
                estimated_k = float(config.class_num)

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
            if semantic_text_features is not None and config.lambda_semantic > 0:
                semantic_loss = semantic_cluster_alignment_loss(
                    logits=logits_cat,
                    teacher_features=teacher_cat,
                    text_features=semantic_text_features,
                    teacher_logit_scale=model.teacher_logit_scale(),
                    temperature=config.semantic_temperature,
                )
            else:
                semantic_loss = student_cat.new_zeros(())

            assignment_cat = torch.cat([assign1, assign2], dim=0)

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
                + config.lambda_subspace * subspace_loss
                + alpha_div * dpp_loss
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        model.normalize_prototypes()
        if queue is not None:
            queue.enqueue(teacher_cat, assignment_cat)

        totals["loss"] += float(loss.item())
        totals["inst_loss"] += float(inst_loss.item())
        totals["cluster_loss"] += float(cluster_loss.item())
        totals["semantic_loss"] += float(semantic_loss.item())
        totals["subspace_loss"] += float(subspace_loss.item())
        totals["dpp_loss"] += float(dpp_loss.item())
        totals["alpha_div"] += float(alpha_div.item())
        totals["rho"] += float(rho.item())
        totals["estimated_k"] += estimated_k

        if (step + 1) % log_interval == 0 or step == 0 or (step + 1) == num_steps:
            running_denom = step + 1
            logger.info(
                (
                    "epoch=%03d step=%04d/%04d "
                    "loss=%.4f inst=%.4f cluster=%.4f semantic=%.4f sub=%.4f dpp=%.4f alpha_div=%.4f rho=%.4f k=%.2f"
                ),
                epoch,
                step + 1,
                num_steps,
                totals["loss"] / running_denom,
                totals["inst_loss"] / running_denom,
                totals["cluster_loss"] / running_denom,
                totals["semantic_loss"] / running_denom,
                totals["subspace_loss"] / running_denom,
                totals["dpp_loss"] / running_denom,
                totals["alpha_div"] / running_denom,
                totals["rho"] / running_denom,
                totals["estimated_k"] / running_denom,
            )

    denom = max(len(dataloader), 1)
    return {key: value / denom for key, value in totals.items()}


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
    config = load_experiment_config(args.config)
    device = resolve_device(args.device or config.device)
    set_seed(config.seed)
    ensure_dir(config.output_dir)
    logger = setup_training_logger(config.output_dir)
    use_amp, amp_device_type, amp_dtype, use_grad_scaler = resolve_amp_settings(
        config.precision, device
    )
    logger.info("starting training")
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
    train_prefetch_factor = getattr(datasets.train_loader, "prefetch_factor", None)
    logger.info("validation split: stratified=%s", datasets.val_is_stratified)
    logger.info(
        "train dataloader: workers=%d pin_memory=%s persistent_workers=%s prefetch_factor=%s",
        datasets.train_loader.num_workers,
        datasets.train_loader.pin_memory,
        datasets.train_loader.persistent_workers,
        train_prefetch_factor,
    )
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
    elif config.clustering_mode == "spectral_free_k" and config.semantic_prototype_enabled:
        logger.info(
            "semantic prototypes requested but disabled for clustering_mode=%s to keep free-K spectral assignments dimensionally consistent",
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
            "semantic prototypes: enabled=%s lambda_semantic=%.4f prompt_style=%s aggregation=%s classes=%d templates_per_class=%d",
            True,
            config.lambda_semantic,
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
        "clustering_mode=%s min_spectral_clusters=%d max_spectral_clusters=%d",
        config.clustering_mode,
        config.min_spectral_clusters,
        config.max_spectral_clusters,
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
        )
        global_step += len(datasets.train_loader)
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
            logger.info(
                "epoch=%03d lr=%.8f next_lr=%.8f scheduler=%s",
                epoch,
                epoch_lr,
                current_learning_rate(optimizer),
                config.lr_scheduler,
            )

        if (epoch + 1) % config.eval_interval != 0:
            continue

        eval_metrics = evaluate_splits(model, datasets.val_loader, device)
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
                model=model,
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
            model=model,
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
            break


if __name__ == "__main__":
    main()
