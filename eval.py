from __future__ import annotations

import argparse

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
    model.eval()

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
