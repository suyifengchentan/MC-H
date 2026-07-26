from __future__ import annotations

import argparse

import numpy as np
import torch

from hypersphere_dpp.checkpoint import load_checkpoint
from hypersphere_dpp.config import load_experiment_config
from hypersphere_dpp.data import build_experiment_dataloaders
from hypersphere_dpp.metrics import evaluate_clustering
from hypersphere_dpp.model import build_model_from_config
from hypersphere_dpp.utils import resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the CUB ViT-L/14 hypersphere-DPP prototype.")
    parser.add_argument("--config", required=True, help="Path to the evaluation config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path, e.g. best.pt.")
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    parser.add_argument(
        "--save-test-predictions",
        default=None,
        help="Optional .npy path for test predictions in official dataset order.",
    )
    parser.add_argument("--device", default=None, help="Override device string, e.g. cuda:0.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_experiment_config(args.config)
    device = resolve_device(args.device or config.device)
    set_seed(config.seed)

    datasets = build_experiment_dataloaders(config)
    model = build_model_from_config(config, device=device).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    if (
        config.clustering_mode == "spectral_free_k"
        and config.spectral_eval_refresh_enabled
    ):
        refresh_stats = model.refresh_spectral_prototypes(
            dataloader=datasets.train_eval_loader,
            device=device,
            assignment_temperature=config.spectral_eval_refresh_temperature,
            passes=config.spectral_eval_refresh_passes,
            min_cluster_mass=config.spectral_eval_refresh_min_mass,
        )
        print(
            "spectral_eval_refresh="
            f"{refresh_stats}"
        )
    model.eval()

    if args.save_test_predictions:
        predictions: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in datasets.test_loader:
                images = batch["image"].to(device, non_blocking=True)
                teacher = model.encode_teacher(images)
                student = model.encode_student(teacher_features=teacher)
                predictions.append(model.prototype_logits(student).argmax(dim=1).cpu().numpy())
        output_path = args.save_test_predictions
        np.save(output_path, np.concatenate(predictions).astype(np.int64, copy=False))
        print(f"saved_test_predictions={output_path}")

    if args.split in {"val", "both"}:
        if datasets.val_loader is None:
            raise ValueError("Validation split is disabled in the current config.")
        metrics = evaluate_clustering(model, datasets.val_loader, device)
        print(f"val={metrics}")
    if args.split in {"test", "both"}:
        metrics = evaluate_clustering(model, datasets.test_loader, device)
        print(f"test={metrics}")


if __name__ == "__main__":
    main()
