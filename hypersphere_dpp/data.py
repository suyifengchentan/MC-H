from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from datasets.cub_dataset import CUB200Dataset
from datasets.imagenet_shard_dataset import ImageNetH5Dataset
from datasets.indexed_image_folder_dataset import IndexedImageFolderDataset

from .config import ExperimentConfig
from .labels import load_label_wnids
from .transforms import build_eval_transform, build_train_decode_transform


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


class TrainTensorDataset(Dataset):
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


def _clone_cpu_tensors_for_collate(sample):
    if torch.is_tensor(sample):
        return sample.clone().contiguous() if sample.device.type == "cpu" else sample
    if isinstance(sample, dict):
        return {key: _clone_cpu_tensors_for_collate(value) for key, value in sample.items()}
    if isinstance(sample, tuple):
        return tuple(_clone_cpu_tensors_for_collate(value) for value in sample)
    if isinstance(sample, list):
        return [_clone_cpu_tensors_for_collate(value) for value in sample]
    return sample


def safe_default_collate(batch):
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    elem = batch[0]

    if torch.is_tensor(elem):
        tensors = [_clone_cpu_tensors_for_collate(sample) for sample in batch]
        return torch.stack(tensors, dim=0)

    if isinstance(elem, Mapping):
        return {key: safe_default_collate([sample[key] for sample in batch]) for key in elem}

    if isinstance(elem, tuple) and hasattr(elem, "_fields"):
        return type(elem)(*(safe_default_collate(items) for items in zip(*batch)))

    if isinstance(elem, tuple):
        return tuple(safe_default_collate(list(items)) for items in zip(*batch))

    if isinstance(elem, Sequence) and not isinstance(elem, (str, bytes)):
        return [safe_default_collate(list(items)) for items in zip(*batch)]

    if isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)

    if isinstance(elem, int):
        return torch.tensor(batch)

    return batch


def _build_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    drop_last: bool,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
) -> DataLoader:
    if persistent_workers is None:
        persistent_workers = workers > 0

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if workers > 0 else False,
        "collate_fn": safe_default_collate,
    }
    if workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(
        **loader_kwargs,
    )


def _default_official_eval_split(dataset_name: str) -> str:
    if dataset_name in {"CUB_200_2011", "Image10"}:
        return "test"
    if dataset_name in {"Image50", "Image100", "Image200"}:
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
        label_wnids = load_label_wnids(config.dataset)
        train_base = ImageNetH5Dataset(root=config.dataset_dir, split="train", transform=None, class_names=label_wnids)
        eval_base = ImageNetH5Dataset(root=config.dataset_dir, split=official_eval_split, transform=None, class_names=label_wnids)
        return train_base, eval_base

    if config.dataset == "Image100":
        label_wnids = load_label_wnids(config.dataset)
        train_base = ImageNetH5Dataset(root=config.dataset_dir, split="train", transform=None, class_names=label_wnids)
        eval_base = ImageNetH5Dataset(root=config.dataset_dir, split=official_eval_split, transform=None, class_names=label_wnids)
        return train_base, eval_base

    if config.dataset == "Image200":
        label_wnids = load_label_wnids(config.dataset)
        train_base = ImageNetH5Dataset(root=config.dataset_dir, split="train", transform=None, class_names=label_wnids)
        eval_base = ImageNetH5Dataset(root=config.dataset_dir, split=official_eval_split, transform=None, class_names=label_wnids)
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

    train_dataset = TrainTensorDataset(
        train_subset,
        build_train_decode_transform(config.image_size, config.transform_profile),
    )
    val_dataset = SingleViewDataset(val_subset, build_eval_transform(config.image_size, config.transform_profile))
    test_dataset = SingleViewDataset(test_base, build_eval_transform(config.image_size, config.transform_profile))

    # HDF5 shard datasets benefit from a small pinned prefetch window. Keep the
    # queue shallow enough to avoid large host-memory spikes while still feeding the GPU.
    is_h5_dataset = config.dataset in {"Image50", "Image100", "Image200"}
    train_workers = min(config.workers, 4) if is_h5_dataset else config.workers
    eval_workers = min(config.workers, 4) if is_h5_dataset else config.workers
    pin_memory = is_h5_dataset
    persistent_workers = True if is_h5_dataset else None
    train_prefetch_factor = 2 if is_h5_dataset and train_workers > 0 else None
    eval_prefetch_factor = 1 if is_h5_dataset and eval_workers > 0 else (4 if eval_workers > 0 else None)

    train_loader = _build_loader(
        train_dataset,
        batch_size=config.batch_size,
        workers=train_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=train_prefetch_factor,
    )
    val_loader = _build_loader(
        val_dataset,
        batch_size=config.eval_batch_size,
        workers=eval_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=eval_prefetch_factor,
    )
    test_loader = _build_loader(
        test_dataset,
        batch_size=config.eval_batch_size,
        workers=eval_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=eval_prefetch_factor,
    )
    return DatasetBundle(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
