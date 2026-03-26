from __future__ import annotations

import argparse

import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast

from hypersphere_dpp.checkpoint import load_checkpoint, save_checkpoint
from hypersphere_dpp.config import load_experiment_config
from hypersphere_dpp.data import build_cub_dataloaders
from hypersphere_dpp.losses import (
    adaptive_diversity_weight,
    dpp_diversity_loss,
    nt_xent_loss,
    swav_cluster_loss,
)
from hypersphere_dpp.metrics import evaluate_clustering
from hypersphere_dpp.model import HypersphereDPPModel
from hypersphere_dpp.subspace import ClusterFeatureQueue, subspace_regularization_loss
from hypersphere_dpp.utils import ensure_dir, resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train the CUB ViT-L/14 hypersphere-DPP prototype.")
    parser.add_argument("--config", required=True, help="Path to the training config YAML.")
    parser.add_argument("--device", default=None, help="Override device string, e.g. cuda:0.")
    return parser.parse_args()


def train_one_epoch(
    model: HypersphereDPPModel,
    dataloader,
    optimizer,
    scaler,
    queue: ClusterFeatureQueue,
    config,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    use_amp = config.precision == "fp16" and device.type == "cuda"
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

    for step, batch in enumerate(dataloader):
        view1 = batch["view1"].to(device, non_blocking=True)
        view2 = batch["view2"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            teacher1 = model.encode_teacher(view1)
            teacher2 = model.encode_teacher(view2)
            student1 = model.encode_student(teacher_features=teacher1)
            student2 = model.encode_student(teacher_features=teacher2)

            logits1 = model.prototype_logits(student1)
            logits2 = model.prototype_logits(student2)

            inst_loss = nt_xent_loss(student1.float(), student2.float(), config.instance_temperature)
            cluster_loss, assign1, assign2 = swav_cluster_loss(
                logits_a=logits1.float(),
                logits_b=logits2.float(),
                temperature=config.cluster_temperature,
                sinkhorn_epsilon=config.sinkhorn_epsilon,
                sinkhorn_iters=config.sinkhorn_iters,
            )

            student_cat = torch.cat([student1, student2], dim=0)
            teacher_cat = torch.cat([teacher1, teacher2], dim=0)
            assignment_cat = torch.cat([assign1, assign2], dim=0)

            subspace_loss, _ = subspace_regularization_loss(
                student_features=student_cat,
                teacher_features=teacher_cat,
                assignments=assignment_cat,
                queue=queue,
                rank=config.subspace_rank,
                min_mass=config.subspace_min_mass,
            )

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

    denom = max(len(dataloader), 1)
    return {key: value / denom for key, value in totals.items()}


def evaluate_splits(model, datasets, device: torch.device) -> dict[str, dict[str, float]]:
    results = {}
    if datasets.val_loader is not None:
        results["val"] = evaluate_clustering(model, datasets.val_loader, device)
    results["test"] = evaluate_clustering(model, datasets.test_loader, device)
    return results


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

    datasets = build_cub_dataloaders(config)
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
    scaler = GradScaler(enabled=config.precision == "fp16" and device.type == "cuda")
    queue = ClusterFeatureQueue(
        num_clusters=config.class_num,
        feature_dim=config.feature_dim,
        queue_size=config.subspace_queue_size,
    )

    start_epoch, global_step, best_metric = maybe_resume_training(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        queue=queue,
        config=config,
        device=device,
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
        )
        global_step += len(datasets.train_loader)
        print(f"[Epoch {epoch:03d}] train={train_metrics}")

        if (epoch + 1) % config.eval_interval != 0:
            continue

        eval_metrics = evaluate_splits(model, datasets, device)
        print(f"[Epoch {epoch:03d}] eval={eval_metrics}")

        tracked_metric = eval_metrics.get("val", eval_metrics["test"])["acc"]
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

        if epochs_without_improvement >= config.best_checkpoint_patience:
            print(f"Stopping early after {config.best_checkpoint_patience} eval rounds without improvement.")
            break


if __name__ == "__main__":
    main()
