from __future__ import annotations

import argparse
import time
from contextlib import nullcontext

import torch
from torch import amp
from torch import optim

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
    swav_cluster_loss,
)
from hypersphere_dpp.metrics import (
    clustering_selection_score,
    evaluate_clustering,
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


def use_cached_teacher_feature_candidates(batch: dict) -> bool:
    return TEACHER_FEATURE_CANDIDATES_KEY in batch


def maybe_profile_sync(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def should_log_step_profile(step: int, num_steps: int, profile_log_interval: int) -> bool:
    return (
        step == 0
        or (step + 1) == num_steps
        or ((step + 1) % max(1, profile_log_interval) == 0)
    )


def train_one_epoch(
    model: HypersphereDPPModel,
    dataloader,
    gpu_train_transform,
    optimizer,
    scaler,
    queue: ClusterFeatureQueue,
    semantic_state,
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
            else:
                teacher1 = model.encode_teacher(view1)
                teacher2 = model.encode_teacher(view2)
            teacher_cat = torch.cat([teacher1, teacher2], dim=0)
            student_cat = model.encode_student(teacher_features=teacher_cat)
            logits_cat = model.prototype_logits(student_cat)

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
                        teacher_logit_scale=model.teacher_logit_scale(),
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
                            teacher_logit_scale=model.teacher_logit_scale(),
                            temperature=config.semantic_temperature,
                            target_topk=config.semantic_target_topk,
                        )
                    if config.lambda_semantic_retrieval > 0:
                        semantic_retrieval_loss = semantic_retrieval_alignment_loss(
                            student_features=student_cat,
                            teacher_features=teacher_cat,
                            text_features=semantic_state.retrieval_text_features,
                            teacher_logit_scale=model.teacher_logit_scale(),
                            retrieval_topk=config.semantic_retrieval_topk,
                        )
                if config.lambda_prototype_anchor > 0:
                    prototype_anchor = prototype_anchor_loss(
                        model=model,
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
        model.normalize_prototypes()
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

        if (step + 1) % log_interval == 0 or step == 0 or (step + 1) == num_steps:
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
        if step_profile_enabled and should_log_step_profile(
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


def build_optimizer(model, config):
    prototype_lr = config.learning_rate * config.prototype_lr_multiplier
    return optim.AdamW(
        [
            {
                "params": model.adapter_parameters(),
                "lr": config.learning_rate,
                "name": "adapter",
            },
            {
                "params": model.prototype_parameters(),
                "lr": prototype_lr,
                "name": "prototype",
            },
        ],
        weight_decay=config.weight_decay,
    )


def prototype_anchor_loss(model, target_prototypes: torch.Tensor) -> torch.Tensor:
    current = torch.nn.functional.normalize(model.prototypes.prototypes.float(), dim=-1)
    target = torch.nn.functional.normalize(target_prototypes.float(), dim=-1)
    cosine = (current * target).sum(dim=-1)
    return 1.0 - cosine.mean()


def set_prototype_trainability(model, freeze_epochs: int, epoch: int, logger=None) -> bool:
    trainable = epoch >= freeze_epochs
    model.set_prototypes_trainable(trainable)
    if logger is not None:
        logger.info(
            "epoch=%03d prototype_trainable=%s freeze_epochs=%d",
            epoch,
            trainable,
            freeze_epochs,
        )
    return trainable


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
    queue.load_state_dict(checkpoint.get("queue", {}))
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_metric = float(checkpoint.get("best_metric", float("-inf")))
    return start_epoch, global_step, best_metric


def eval_snapshot_checkpoint_path(config, epoch: int) -> Path:
    # Save eval-triggered snapshots in a dedicated folder so best/last semantics stay unchanged.
    return config.output_dir / "eval_checkpoints" / f"epoch_{epoch + 1:04d}.pt"


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
    if config.step_profile_enabled:
        logger.info(
            "step profiling: enabled=True log_interval=%d cuda_sync=%s",
            config.step_profile_log_interval,
            config.step_profile_cuda_sync,
        )
    if config.uses_imagenet1k_cached_teacher_views:
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
    model = HypersphereDPPModel(
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
            backbone_model=model.backbone.model,
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
            model.prototypes.prototypes.data.copy_(semantic_state.prototype_features)
            model.normalize_prototypes()

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
    logger.info(
        "class-aware dpp: enabled=%s min_mass=%.4f",
        config.class_aware_dpp,
        config.class_aware_dpp_min_mass,
    )
    optimizer = build_optimizer(model, config)
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
        "optimizer: adapter_lr=%.8f prototype_lr=%.8f prototype_lr_multiplier=%.4f lambda_prototype_anchor=%.4f weight_decay=%.8f lr_scheduler=%s min_lr=%.8f",
        current_learning_rate(optimizer),
        float(optimizer.param_groups[1]["lr"]),
        config.prototype_lr_multiplier,
        config.lambda_prototype_anchor,
        config.weight_decay,
        config.lr_scheduler,
        config.min_learning_rate,
    )

    epochs_without_improvement = 0
    for epoch in range(start_epoch, config.epochs):
        set_prototype_trainability(
            model=model,
            freeze_epochs=config.prototype_freeze_epochs,
            epoch=epoch,
            logger=logger,
        )
        epoch_lr = current_learning_rate(optimizer)
        train_metrics = train_one_epoch(
            model=model,
            dataloader=datasets.train_loader,
            gpu_train_transform=gpu_train_transform,
            optimizer=optimizer,
            scaler=scaler,
            queue=queue,
            semantic_state=semantic_state,
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

        eval_snapshot_path = eval_snapshot_checkpoint_path(config, epoch)
        save_checkpoint(
            checkpoint_path=eval_snapshot_path,
            model=model,
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
            "saved eval snapshot=%s trigger_epoch=%03d split=%s %s selection_score=%.6f",
            eval_snapshot_path,
            epoch,
            selected_split,
            format_clustering_metrics(selected_metrics),
            tracked_metric,
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
            model=model,
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
            break


if __name__ == "__main__":
    main()
