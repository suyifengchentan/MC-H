from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from datasets.cifar100_dataset import CIFAR100Dataset
from datasets.stl10_dataset import STL10Dataset
from torch.utils.data import DataLoader, Dataset, Subset

from datasets.aircraft_dataset import AircraftDataset
from datasets.cifar10_dataset import CIFAR10Dataset
from datasets.cub_dataset import CUB200Dataset
from datasets.imagenet50_parquet_dataset import ImageNetParquetDataset
from datasets.imagenet_shard_dataset import ImageNetH5Dataset
from datasets.indexed_image_folder_dataset import IndexedImageFolderDataset
from datasets.oxford_flowers_dataset import OxfordFlowersDataset
from datasets.oxford_pet_dataset import OxfordPetDataset
from datasets.stanford_dogs_dataset import StanfordDogsDataset

from .config import ExperimentConfig
from .labels import load_label_wnids
from .transforms import build_eval_transform, build_train_decode_transform


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
    val_is_stratified: bool = False


def split_train_indices(
    size: int,
    validation_ratio: float,
    seed: int,
    labels: Sequence[int] | None = None,
) -> tuple[list[int], list[int]]:
    if size < 2:
        raise ValueError(
            "Dataset must contain at least 2 samples to create train/val splits."
        )

    if labels is not None and len(labels) != size:
        raise ValueError("labels length must match dataset size for stratified split.")

    generator = torch.Generator().manual_seed(seed)

    if labels is None:
        indices = torch.randperm(size, generator=generator).tolist()
        val_size = max(1, int(size * validation_ratio))
        if val_size >= size:
            raise ValueError(
                "validation_ratio leaves no samples for training; reduce validation_ratio."
            )
        val_indices = sorted(indices[:val_size])
        train_indices = sorted(indices[val_size:])
        return train_indices, val_indices

    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        label_to_indices[int(label)].append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for label in sorted(label_to_indices):
        class_indices = label_to_indices[label]
        class_size = len(class_indices)
        if class_size < 2:
            raise ValueError(
                "Stratified validation split requires at least 2 samples per class."
            )

        class_perm = torch.randperm(class_size, generator=generator).tolist()
        shuffled_indices = [class_indices[i] for i in class_perm]
        class_val_size = max(1, int(round(class_size * validation_ratio)))
        class_val_size = min(class_val_size, class_size - 1)
        val_indices.extend(shuffled_indices[:class_val_size])
        train_indices.extend(shuffled_indices[class_val_size:])

    return sorted(train_indices), sorted(val_indices)


def _extract_dataset_labels(dataset: Dataset) -> list[int] | None:
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        return [int(label) for label in targets]

    if hasattr(dataset, "labels"):
        labels = getattr(dataset, "labels")
        return [int(label) for label in labels]

    if hasattr(dataset, "samples"):
        samples = getattr(dataset, "samples")
        if (
            isinstance(samples, list)
            and samples
            and isinstance(samples[0], tuple)
            and len(samples[0]) >= 2
        ):
            return [int(sample[1]) for sample in samples]

    if hasattr(dataset, "data"):
        data = getattr(dataset, "data")
        if isinstance(data, list) and data and isinstance(data[0], tuple) and len(data[0]) >= 2:
            return [int(sample[1]) for sample in data]

    return None


def _clone_cpu_tensors_for_collate(sample):
    if torch.is_tensor(sample):
        return sample.clone().contiguous() if sample.device.type == "cpu" else sample
    if isinstance(sample, dict):
        return {
            key: _clone_cpu_tensors_for_collate(value) for key, value in sample.items()
        }
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
        return {
            key: safe_default_collate([sample[key] for sample in batch]) for key in elem
        }

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
    if dataset_name in {
        "CIFAR10",
        "CIFAR100",
        "CUB_200_2011",
        "FGVC_Aircraft",
        "Image10",
        "ImageNet1K",
        "Oxford_Pet",
        "STL10",
        "Stanford_Dogs",
    }:
        return "test"
    if dataset_name in {"Image50", "Image100", "Image200"}:
        return "val"
    if dataset_name == "Oxford_Flowers":
        return "valid"
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _build_imagenet_h5_datasets(
    config: ExperimentConfig,
    official_eval_split: str,
) -> tuple[Dataset, Dataset]:
    label_wnids = load_label_wnids(config.dataset)
    train_base = ImageNetH5Dataset(
        root=config.dataset_dir,
        split="train",
        transform=None,
        class_names=label_wnids,
    )
    eval_base = ImageNetH5Dataset(
        root=config.dataset_dir,
        split=official_eval_split,
        transform=None,
        class_names=label_wnids,
    )
    return train_base, eval_base


def _build_imagenet_parquet_datasets(
    config: ExperimentConfig,
    official_eval_split: str,
) -> tuple[Dataset, Dataset]:
    train_base = ImageNetParquetDataset(
        root=config.dataset_dir,
        split="train",
        transform=None,
    )
    eval_base = ImageNetParquetDataset(
        root=config.dataset_dir,
        split=official_eval_split,
        transform=None,
    )
    return train_base, eval_base


def _build_base_datasets(config: ExperimentConfig) -> tuple[Dataset, Dataset]:
    official_eval_split = config.official_eval_split or _default_official_eval_split(
        config.dataset
    )

    if config.dataset == "CIFAR10":
        train_base = CIFAR10Dataset(
            root=config.dataset_dir, train=True, transform=None, download=False
        )
        eval_base = CIFAR10Dataset(
            root=config.dataset_dir, train=False, transform=None, download=False
        )
        return train_base, eval_base

    if config.dataset == "CIFAR100":
        train_base = CIFAR100Dataset(
            root=config.dataset_dir, train=True, transform=None, download=False
        )
        eval_base = CIFAR100Dataset(
            root=config.dataset_dir, train=False, transform=None, download=False
        )
        return train_base, eval_base

    if config.dataset == "CUB_200_2011":
        train_base = CUB200Dataset(root=config.dataset_dir, train=True, transform=None)
        eval_base = CUB200Dataset(root=config.dataset_dir, train=False, transform=None)
        return train_base, eval_base

    if config.dataset == "FGVC_Aircraft":
        train_base = AircraftDataset(
            root=config.dataset_dir, split="train", transform=None
        )
        eval_base = AircraftDataset(
            root=config.dataset_dir, split=official_eval_split, transform=None
        )
        return train_base, eval_base

    if config.dataset == "Image10":
        train_base = IndexedImageFolderDataset(
            root=config.dataset_dir, split=None, transform=None
        )
        eval_base = IndexedImageFolderDataset(
            root=config.dataset_dir, split=official_eval_split, transform=None
        )
        return train_base, eval_base

    if config.dataset == "ImageNet1K":
        return _build_imagenet_parquet_datasets(config, official_eval_split)

    if config.dataset in {"Image50", "Image100", "Image200"}:
        return _build_imagenet_h5_datasets(config, official_eval_split)

    if config.dataset == "Oxford_Flowers":
        train_base = OxfordFlowersDataset(
            root=config.dataset_dir, split="train", transform=None
        )
        eval_base = OxfordFlowersDataset(
            root=config.dataset_dir, split=official_eval_split, transform=None
        )
        return train_base, eval_base

    if config.dataset == "Oxford_Pet":
        train_base = OxfordPetDataset(
            root=config.dataset_dir, split="trainval", transform=None
        )
        eval_base = OxfordPetDataset(
            root=config.dataset_dir, split="test", transform=None
        )
        return train_base, eval_base

    if config.dataset == "STL10":
        train_base = STL10Dataset(
            root=config.dataset_dir, split="train", transform=None, download=False
        )
        eval_base = STL10Dataset(
            root=config.dataset_dir, split="test", transform=None, download=False
        )
        return train_base, eval_base

    if config.dataset == "Stanford_Dogs":
        train_base = StanfordDogsDataset(
            root=config.dataset_dir, train=True, transform=None
        )
        eval_base = StanfordDogsDataset(
            root=config.dataset_dir, train=False, transform=None
        )
        return train_base, eval_base

    raise ValueError(
        f"Unsupported dataset for dataloader construction: {config.dataset}"
    )


def build_experiment_dataloaders(config: ExperimentConfig) -> DatasetBundle:
    train_base, test_base = _build_base_datasets(config)
    train_labels = _extract_dataset_labels(train_base)
    val_is_stratified = train_labels is not None

    train_indices, val_indices = split_train_indices(
        size=len(train_base),
        validation_ratio=config.validation_ratio,
        seed=config.validation_seed,
        labels=train_labels,
    )

    train_subset = Subset(train_base, train_indices)
    val_subset = Subset(train_base, val_indices)

    train_dataset = TrainTensorDataset(
        train_subset,
        build_train_decode_transform(config.image_size, config.transform_profile),
    )
    val_dataset = SingleViewDataset(
        val_subset, build_eval_transform(config.image_size, config.transform_profile)
    )
    test_dataset = SingleViewDataset(
        test_base, build_eval_transform(config.image_size, config.transform_profile)
    )

    # Sharded ImageNet-style datasets benefit from a small pinned prefetch
    # window. Keep the queue shallow enough to avoid large host-memory spikes
    # while still feeding the GPU.
    is_sharded_imagenet_dataset = config.dataset in {
        "ImageNet1K",
        "Image50",
        "Image100",
        "Image200",
    }
    train_workers = (
        min(config.workers, 4) if is_sharded_imagenet_dataset else config.workers
    )
    eval_workers = (
        min(config.workers, 4) if is_sharded_imagenet_dataset else config.workers
    )
    pin_memory = is_sharded_imagenet_dataset
    persistent_workers = True if is_sharded_imagenet_dataset else None
    train_prefetch_factor = (
        2 if is_sharded_imagenet_dataset and train_workers > 0 else None
    )
    eval_prefetch_factor = (
        1
        if is_sharded_imagenet_dataset and eval_workers > 0
        else (4 if eval_workers > 0 else None)
    )

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
    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        val_is_stratified=val_is_stratified,
    )
