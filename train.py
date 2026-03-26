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
    swav_cluster_loss,
)
from hypersphere_dpp.metrics import (
    clustering_selection_score,
    evaluate_clustering,
    format_clustering_metrics,
)
from hypersphere_dpp.model import HypersphereDPPModel
from hypersphere_dpp.subspace import ClusterFeatureQueue, subspace_regularization_loss
from hypersphere_dpp.utils import ensure_dir, resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train the CUB ViT-L/14 hypersphere-DPP prototype.")
    parser.add_argument("--config", required=True, help="Path to the training config YAML.")
    parser.add_argument("--device", default=None, help="Override device string, e.g. cuda:0.")
    return parser.parse_args()


def resolve_amp_settings(precision: str, device: torch.device) -> tuple[bool, str, torch.dtype | None, bool]:
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


def train_one_epoch(
    model: HypersphereDPPModel,
    dataloader,
    optimizer,
    scaler,
    queue: ClusterFeatureQueue,
    config,
    device: torch.device,
    epoch: int,
    logger,
) -> dict[str, float]:
    model.train()
    use_amp, amp_device_type, amp_dtype, _ = resolve_amp_settings(config.precision, device)
    totals = {
        "loss": 0.0,
        "inst_loss": 0.0,
        "cluster_loss": 0.0,
        "subspace_loss": 0.0,
        "dpp_loss": 0.0,
        "alpha_div": 0.0,
        "rho": 0.0,
    }
    num_steps = len(dataloader)
    log_interval = max(1, num_steps // 5) if num_steps > 0 else 1

    for step, batch in enumerate(dataloader):
        view1 = batch["view1"].to(device, non_blocking=True)
        view2 = batch["view2"].to(device, non_blocking=True)

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
            logits_cat = model.prototype_logits(student_cat)

            student1, student2 = student_cat.chunk(2)
            logits1, logits2 = logits_cat.chunk(2)

            inst_loss = nt_xent_loss(student1.float(), student2.float(), config.instance_temperature)
            cluster_loss, assign1, assign2 = swav_cluster_loss(
                logits_a=logits1.float(),
                logits_b=logits2.float(),
                temperature=config.cluster_temperature,
                sinkhorn_epsilon=config.sinkhorn_epsilon,
                sinkhorn_iters=config.sinkhorn_iters,
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
                )
                epoch_progress = epoch + step / max(num_steps, 1)
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
                + config.lambda_subspace * subspace_loss
                + alpha_div * dpp_loss
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        model.normalize_prototypes()
        queue.enqueue(teacher_cat, assignment_cat)

        totals["loss"] += float(loss.item())
        totals["inst_loss"] += float(inst_loss.item())
        totals["cluster_loss"] += float(cluster_loss.item())
        totals["subspace_loss"] += float(subspace_loss.item())
        totals["dpp_loss"] += float(dpp_loss.item())
        totals["alpha_div"] += float(alpha_div.item())
        totals["rho"] += float(rho.item())

        if (step + 1) % log_interval == 0 or step == 0 or (step + 1) == num_steps:
            running_denom = step + 1
            logger.info(
                (
                    "epoch=%03d step=%04d/%04d "
                    "loss=%.4f inst=%.4f cluster=%.4f sub=%.4f dpp=%.4f alpha_div=%.4f rho=%.4f"
                ),
                epoch,
                step + 1,
                num_steps,
                totals["loss"] / running_denom,
                totals["inst_loss"] / running_denom,
                totals["cluster_loss"] / running_denom,
                totals["subspace_loss"] / running_denom,
                totals["dpp_loss"] / running_denom,
                totals["alpha_div"] / running_denom,
                totals["rho"] / running_denom,
            )

    denom = max(len(dataloader), 1)
    return {key: value / denom for key, value in totals.items()}


def evaluate_splits(model, datasets, device: torch.device) -> dict[str, dict[str, float]]:
    return {"val": evaluate_clustering(model, datasets.val_loader, device)}


def select_checkpoint_score(
    eval_metrics: dict[str, dict[str, float]],
    logger,
) -> tuple[str, dict[str, float], float]:
    split_name = "val"
    metrics = eval_metrics[split_name]
    score = clustering_selection_score(metrics)
    return split_name, metrics, score


def maybe_resume_training(model, optimizer, scaler, queue, config, device):
    last_checkpoint = config.output_dir / "last.pt"
    if not config.reload or not last_checkpoint.exists():
        return 0, 0, float("-inf")

    checkpoint = load_checkpoint(
        checkpoint_path=last_checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        map_location=device,
    )
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
    use_amp, amp_device_type, amp_dtype, use_grad_scaler = resolve_amp_settings(config.precision, device)
    logger.info("starting training")
    logger.info("device=%s output_dir=%s", device, config.output_dir)
    logger.info("config=%s", config.as_dict())
    logger.info("validation cadence: run val every %d epoch(s) based on config.eval_interval", config.eval_interval)
    logger.info(
        "precision=%s amp_enabled=%s amp_device_type=%s amp_dtype=%s grad_scaler=%s",
        config.precision,
        use_amp,
        amp_device_type,
        amp_dtype,
        use_grad_scaler,
    )
    if config.precision in {"fp16", "bf16"} and not use_amp:
        logger.warning("precision=%s requested but AMP is unsupported on device=%s; falling back to fp32.", config.precision, device)

    datasets = build_experiment_dataloaders(config)
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
    optimizer = optim.AdamW(
        model.trainable_parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
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

    epochs_without_improvement = 0
    for epoch in range(start_epoch, config.epochs):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=datasets.train_loader,
            optimizer=optimizer,
            scaler=scaler,
            queue=queue,
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
                "split": "train",
                **train_metrics,
            },
        )

        if (epoch + 1) % config.eval_interval != 0:
            continue

        eval_metrics = evaluate_splits(model, datasets, device)
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

        selected_split, selected_metrics, tracked_metric = select_checkpoint_score(eval_metrics, logger)
        is_best = tracked_metric > best_metric
        if is_best:
            best_metric = tracked_metric
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path=config.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
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
