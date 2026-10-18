from __future__ import annotations

import argparse

from src.checkpoint import load_checkpoint
from src.config import load_experiment_config
from src.data import build_experiment_dataloaders
from src.metrics import evaluate_clustering
from src.model import HypersphereDPPModel
from src.utils import resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a hypersphere-DPP checkpoint.")
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
    model = HypersphereDPPModel(
        backbone_name=config.backbone_name,
        pretrained=config.pretrained,
        pretrained_checkpoint=config.pretrained_checkpoint,
        feature_dim=config.feature_dim,
        adapter_hidden_dim=config.adapter_hidden_dim,
        adapter_dropout=config.adapter_dropout,
        num_prototypes=config.class_num,
        device=device,
        representation_dim=config.representation_dim,
    ).to(device)
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
