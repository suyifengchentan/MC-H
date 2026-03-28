import os
from pathlib import Path
from typing import Dict, List, Tuple

import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader


class OxfordFlowersDataset(datasets.ImageFolder):
    """Oxford Flowers train/valid split loader with numeric class ordering."""

    def __init__(
        self, root, train=True, split=None, transform=None, target_transform=None
    ):
        if split is None:
            split = "train" if train else "valid"

        base_root = Path(root)
        flower_root = (
            base_root / "flower_data"
            if (base_root / "flower_data").is_dir()
            else base_root
        )
        split_root = flower_root / split

        if split == "test":
            raise ValueError(
                "Oxford_Flowers test split under the current root is unlabeled; use split='valid' for evaluation."
            )

        if not split_root.is_dir():
            raise FileNotFoundError(
                f"Oxford_Flowers split directory not found: {split_root}"
            )

        super(OxfordFlowersDataset, self).__init__(
            root=str(split_root),
            loader=default_loader,
            transform=transform,
            target_transform=target_transform,
        )

    def __getitem__(self, index):
        sample, target = super(OxfordFlowersDataset, self).__getitem__(index)
        return sample, target, index

    @staticmethod
    def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        try:
            classes.sort(key=int)
        except ValueError:
            classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
