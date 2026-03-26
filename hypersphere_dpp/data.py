from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from datasets.cub_dataset import CUB200Dataset

from .config import ExperimentConfig
from .transforms import build_eval_transform, build_train_transform


class PairViewDataset(Dataset):
    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label, sample_index = self.dataset[index]
        view1, view2 = self.transform(image)
        return {
            "view1": view1,
            "view2": view2,
            "label": label,
            "index": sample_index,
        }


class SingleViewDataset(Dataset):
    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label, sample_index = self.dataset[index]
        return {
            "image": self.transform(image),
            "label": label,
            "index": sample_index,
        }


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def split_train_indices(size: int, validation_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if size < 2:
        raise ValueError("Dataset must contain at least 2 samples to create train/val splits.")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(size, generator=generator).tolist()
    val_size = max(1, int(size * validation_ratio))
    if val_size >= size:
        raise ValueError("validation_ratio leaves no samples for training; reduce validation_ratio.")
    val_indices = sorted(indices[:val_size])
    train_indices = sorted(indices[val_size:])
    return train_indices, val_indices


def _build_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def build_cub_dataloaders(config: ExperimentConfig) -> DatasetBundle:
    train_base = CUB200Dataset(root=config.dataset_dir, train=True, transform=None)
    test_base = CUB200Dataset(root=config.dataset_dir, train=False, transform=None)

    train_indices, val_indices = split_train_indices(
        size=len(train_base),
        validation_ratio=config.validation_ratio,
        seed=config.validation_seed,
    )
    val_indices = val_indices[: config.batch_size]

    train_subset = Subset(train_base, train_indices)
    val_subset = Subset(train_base, val_indices)

    train_dataset = PairViewDataset(
        train_subset,
        build_train_transform(config.image_size, config.transform_profile),
    )
    val_dataset = SingleViewDataset(val_subset, build_eval_transform(config.image_size))
    test_dataset = SingleViewDataset(test_base, build_eval_transform(config.image_size))

    train_loader = _build_loader(
        train_dataset,
        batch_size=config.batch_size,
        workers=config.workers,
        shuffle=True,
        drop_last=True,
    )
    val_loader = _build_loader(
        val_dataset,
        batch_size=config.eval_batch_size,
        workers=config.workers,
        shuffle=False,
        drop_last=False,
    )
    test_loader = _build_loader(
        test_dataset,
        batch_size=config.eval_batch_size,
        workers=config.workers,
        shuffle=False,
        drop_last=False,
    )
    return DatasetBundle(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
