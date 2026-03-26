from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from datasets.cub_dataset import CUB200Dataset
from datasets.imagenet50_parquet_dataset import ImageNet50ParquetDataset
from datasets.indexed_image_folder_dataset import IndexedImageFolderDataset
from datasets.tiny_imagenet_dataset import TinyImageNetDataset

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


def _default_official_eval_split(dataset_name: str) -> str:
    if dataset_name in {"CUB_200_2011", "Image10"}:
        return "test"
    if dataset_name in {"Image50", "Image200"}:
        return "val"
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _build_base_datasets(config: ExperimentConfig) -> tuple[Dataset, Dataset]:
    official_eval_split = config.official_eval_split or _default_official_eval_split(config.dataset)

    if config.dataset == "CUB_200_2011":
        train_base = CUB200Dataset(root=config.dataset_dir, train=True, transform=None)
        eval_base = CUB200Dataset(root=config.dataset_dir, train=False, transform=None)
        return train_base, eval_base

    if config.dataset == "Image10":
        train_base = IndexedImageFolderDataset(root=config.dataset_dir, split=None, transform=None)
        eval_base = IndexedImageFolderDataset(root=config.dataset_dir, split=official_eval_split, transform=None)
        return train_base, eval_base

    if config.dataset == "Image50":
        if official_eval_split != "val":
            raise ValueError("Image50 only supports official_eval_split='val' in the current parquet dataset.")
        train_base = ImageNet50ParquetDataset(root=config.dataset_dir, split="train", transform=None)
        eval_base = ImageNet50ParquetDataset(root=config.dataset_dir, split="val", transform=None)
        return train_base, eval_base

    if config.dataset == "Image200":
        train_base = TinyImageNetDataset(root=config.dataset_dir, split="train", transform=None)
        eval_base = TinyImageNetDataset(root=config.dataset_dir, split=official_eval_split, transform=None)
        return train_base, eval_base

    raise ValueError(f"Unsupported dataset for dataloader construction: {config.dataset}")


def build_experiment_dataloaders(config: ExperimentConfig) -> DatasetBundle:
    train_base, test_base = _build_base_datasets(config)

    train_indices, val_indices = split_train_indices(
        size=len(train_base),
        validation_ratio=config.validation_ratio,
        seed=config.validation_seed,
    )

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
