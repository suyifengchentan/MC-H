from __future__ import annotations

import argparse

import torch
import torch.distributed as dist
from torch import amp, optim
from torch.nn.parallel import DistributedDataParallel

from hypersphere_dpp.checkpoint import save_checkpoint
from hypersphere_dpp.config import load_experiment_config
from hypersphere_dpp.data import build_experiment_dataloaders
from hypersphere_dpp.logging_utils import append_metrics_record
from hypersphere_dpp.model import HypersphereDPPModel
from hypersphere_dpp.runtime_accel import (
    configure_cuda_runtime,
    format_runtime_accel_state,
)
from hypersphere_dpp.semantic import build_semantic_prototype_state
from hypersphere_dpp.subspace import ClusterFeatureQueue
from hypersphere_dpp.transforms import build_gpu_train_transform
from hypersphere_dpp.utils import ensure_dir, set_seed
from train import (
    build_optimizer,
    build_lr_scheduler,
    current_learning_rate,
    evaluate_splits,
    maybe_resume_training,
    resolve_amp_settings,
    select_checkpoint_score,
    set_prototype_trainability,
)
from train_parallel import (
    broadcast_training_control,
    build_distributed_train_loader,
    distributed_env,
    init_process_group_if_needed,
    is_main_process,
    resolve_parallel_device,
    setup_rank_logger,
    train_one_epoch_distributed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Accelerated multi-GPU training entry with TF32 and cuDNN benchmark."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the training config YAML."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override for direct single-process runs.",
    )
    parser.add_argument(
        "--disable-tf32",
        action="store_true",
        help="Disable TF32 matmul/cudnn acceleration on CUDA.",
    )
    parser.add_argument(
        "--disable-cudnn-benchmark",
        action="store_true",
        help="Disable cudnn benchmark autotuning for fixed-size inputs.",
    )
    parser.add_argument(
        "--float32-matmul-precision",
        default="high",
        choices=("highest", "high", "medium"),
        help="torch.set_float32_matmul_precision setting applied on CUDA.",
    )
    return parser.parse_args()


def resolve_batch_sizes(
    requested_batch_size: int,
    world_size: int,
    batch_size_mode: str,
) -> dict[str, int]:
    if batch_size_mode == "global":
        per_rank_batch_size = requested_batch_size // world_size
        if per_rank_batch_size <= 0:
            raise ValueError(
                "batch_size is smaller than world_size under distributed_batch_size_mode='global'."
            )
    elif batch_size_mode == "per_rank":
        per_rank_batch_size = requested_batch_size
    else:
        raise ValueError(
            f"Unsupported distributed batch size mode: {batch_size_mode!r}"
        )
    return {
        "requested_batch_size": requested_batch_size,
        "per_rank_batch_size": per_rank_batch_size,
        "effective_global_batch_size": per_rank_batch_size * world_size,
    }


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

        runtime_state = configure_cuda_runtime(
            device,
            enable_tf32=not args.disable_tf32,
            cudnn_benchmark=not args.disable_cudnn_benchmark,
            float32_matmul_precision=args.float32_matmul_precision,
        )
        use_amp, amp_device_type, amp_dtype, use_grad_scaler = resolve_amp_settings(
            config.precision, device
        )

        if is_main_process(rank):
            logger.info("starting accelerated distributed training")
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
                "runtime acceleration: %s", format_runtime_accel_state(runtime_state)
            )
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

        datasets = build_experiment_dataloaders(config)
        if is_main_process(rank):
            logger.info("validation split: stratified=%s", datasets.val_is_stratified)

        batch_info = resolve_batch_sizes(
            requested_batch_size=int(datasets.train_loader.batch_size),
            world_size=world_size,
            batch_size_mode=config.distributed_batch_size_mode,
        )

        if device.type != "cuda":
            raise RuntimeError("The accelerated training entry requires CUDA.")

        train_loader, train_sampler, _ = build_distributed_train_loader(
            datasets.train_loader,
            rank=rank,
            world_size=world_size,
            batch_size_mode=config.distributed_batch_size_mode,
        )
        if is_main_process(rank):
            logger.info("train loader backend=native")

        if is_main_process(rank):
            logger.info(
                "train dataloader: workers=%d pin_memory=%s persistent_workers=%s prefetch_factor=%s world_size=%d distributed_sampler=%s",
                train_loader.num_workers,
                train_loader.pin_memory,
                train_loader.persistent_workers,
                train_loader.prefetch_factor,
                world_size,
                train_sampler is not None,
            )
            logger.info(
                "train batch: mode=%s requested=%d per_rank=%d effective_global=%d",
                config.distributed_batch_size_mode,
                batch_info["requested_batch_size"],
                batch_info["per_rank_batch_size"],
                batch_info["effective_global_batch_size"],
            )
            logger.info(
                "distributed loss semantics: local_per_rank batches with standard DDP gradient averaging"
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
        ddp_model = (
            DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                output_device=device.index if device.type == "cuda" else None,
            )
            if world_size > 1
            else model
        )

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
            with torch.no_grad():
                model.prototypes.prototypes.data.copy_(semantic_state.prototype_features)
                model.normalize_prototypes()
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
        if is_main_process(rank) and start_epoch > 0:
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

        epochs_without_improvement = 0
        for epoch in range(start_epoch, config.epochs):
            set_prototype_trainability(
                model=model,
                freeze_epochs=config.prototype_freeze_epochs,
                epoch=epoch,
                logger=logger if is_main_process(rank) else None,
            )
            epoch_lr = current_learning_rate(optimizer)
            train_metrics = train_one_epoch_distributed(
                model=ddp_model,
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
                eval_model = model
                eval_metrics = evaluate_splits(eval_model, datasets.val_loader, device)
                for split_name, metrics in eval_metrics.items():
                    selection_score = float(
                        metrics["acc"] + metrics["nmi"] + metrics["ari"]
                    )
                    logger.info(
                        "epoch=%03d split=%s acc=%.4f nmi=%.4f ari=%.4f selection_score=%.6f",
                        epoch,
                        split_name,
                        metrics["acc"],
                        metrics["nmi"],
                        metrics["ari"],
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

                selected_split, selected_metrics, tracked_metric = (
                    select_checkpoint_score(eval_metrics, logger)
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
                        "saved checkpoint=%s split=%s acc=%.4f nmi=%.4f ari=%.4f selection_score=%.6f",
                        config.output_dir / "best.pt",
                        selected_split,
                        selected_metrics["acc"],
                        selected_metrics["nmi"],
                        selected_metrics["ari"],
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

            best_metric, epochs_without_improvement, tracked_stop = (
                broadcast_training_control(
                    best_metric=best_metric,
                    epochs_without_improvement=epochs_without_improvement,
                    stop_training=tracked_stop,
                    device=device,
                    world_size=world_size,
                )
            )
            if tracked_stop:
                break
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
