from __future__ import annotations

import os
from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class TinyImageNetDataset(Dataset):
    """
    Tiny-ImageNet 数据集包装器。

    - train: 直接复用 ImageFolder(root/train)
    - val: 解析 val_annotations.txt，因为标准 Tiny-ImageNet 的 val 目录不是按类别分文件夹存放
    """

    def __init__(self, root: str, split: str, transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        if split == "train":
            self._train_dataset = ImageFolder(os.path.join(root, "train"), transform=None)
            self.classes = self._train_dataset.classes
            self.class_to_idx = self._train_dataset.class_to_idx
            self.samples = self._train_dataset.samples
        elif split == "val":
            self._train_dataset = None
            self.class_to_idx = self._load_class_to_idx()
            self.classes = sorted(self.class_to_idx, key=self.class_to_idx.get)
            self.samples = self._load_val_samples()
        else:
            raise ValueError(f"TinyImageNetDataset only supports split='train' or 'val', got {split!r}.")

    def _load_class_to_idx(self) -> Dict[str, int]:
        wnids_path = os.path.join(self.root, "wnids.txt")
        if os.path.isfile(wnids_path):
            with open(wnids_path, "r", encoding="utf-8") as handle:
                classes = [line.strip() for line in handle if line.strip()]
            return {class_name: index for index, class_name in enumerate(classes)}

        train_root = os.path.join(self.root, "train")
        classes = sorted(entry.name for entry in os.scandir(train_root) if entry.is_dir())
        return {class_name: index for index, class_name in enumerate(classes)}

    def _load_val_samples(self) -> List[Tuple[str, int]]:
        val_root = os.path.join(self.root, "val")
        images_root = os.path.join(val_root, "images")
        annotation_path = os.path.join(val_root, "val_annotations.txt")
        if not os.path.isfile(annotation_path):
            raise FileNotFoundError(f"Expected Tiny-ImageNet val annotations at: {annotation_path}")

        samples: List[Tuple[str, int]] = []
        with open(annotation_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                image_name, class_name = parts[0], parts[1]
                if class_name not in self.class_to_idx:
                    continue
                image_path = os.path.join(images_root, image_name)
                samples.append((image_path, self.class_to_idx[class_name]))
        return samples

    def __len__(self) -> int:
        if self.split == "train":
            return len(self._train_dataset)
        return len(self.samples)

    def __getitem__(self, index: int):
        if self.split == "train":
            image, target = self._train_dataset[index]
            if self.transform:
                image = self.transform(image)
            return image, target, index

        image_path, target = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target, index

