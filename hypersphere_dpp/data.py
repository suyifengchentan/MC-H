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
from datasets.image10_dataset import Image10Dataset
from datasets.imagenet_h5_dataset import ImageNetH5Dataset
from datasets.oxford_flowers_dataset import OxfordFlowersDataset
from datasets.oxford_pet_dataset import OxfordPetDataset
from datasets.stanford_dogs_dataset import StanfordDogsDataset
from datasets.tiny_imagenet_dataset import TinyImageNetDataset

from .config import ExperimentConfig
from .labels import load_label_wnids
from .teacher_cache import ImageNetTeacherFeatureCache
from .transforms import build_eval_transform, build_train_decode_transform

TEACHER_FEATURE_CANDIDATES_KEY = "teacher_feature_candidates"


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


class ImageNetTeacherFeatureTrainDataset(Dataset):
    def __init__(self, dataset: Subset, teacher_cache: ImageNetTeacherFeatureCache):
        if not isinstance(dataset, Subset):
            raise TypeError("ImageNetTeacherFeatureTrainDataset expects a torch.utils.data.Subset.")
        if not hasattr(dataset.dataset, "targets"):
            raise ValueError("The wrapped base dataset must expose targets for labels.")
        if len(teacher_cache) != len(dataset.dataset):
            raise ValueError(
                "Teacher feature cache size must match the full ImageNet training dataset size."
            )

        self.dataset = dataset
        self.base_dataset = dataset.dataset
        self.teacher_cache = teacher_cache

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        base_index = int(self.dataset.indices[index])
        label = int(self.base_dataset.targets[base_index])
        return {
            TEACHER_FEATURE_CANDIDATES_KEY: self.teacher_cache[base_index],
            "label": label,
            "index": base_index,
        }

    def __getitems__(self, indices: Sequence[int]):
        base_indices = [int(self.dataset.indices[int(index)]) for index in indices]
        labels = [int(self.base_dataset.targets[base_index]) for base_index in base_indices]
        teacher_batches = self.teacher_cache.get_many(base_indices)
        return [
            {
                TEACHER_FEATURE_CANDIDATES_KEY: teacher_batches[offset],
                "label": labels[offset],
                "index": base_indices[offset],
            }
            for offset in range(len(base_indices))
        ]


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
        try:
            # Fast path: avoid an extra deep copy for every sample when tensors
            # are already backed by stackable storage.
            return torch.stack(batch, dim=0)
        except RuntimeError as error:
            if "resizable" not in str(error).lower():
                raise
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


def sample_two_teacher_feature_views(
    teacher_feature_candidates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if teacher_feature_candidates.ndim != 3:
        raise ValueError(
            "teacher_feature_candidates must have shape [batch, num_views, feature_dim]."
        )
    num_candidates = teacher_feature_candidates.size(1)
    if num_candidates < 2:
        raise ValueError("teacher_feature_candidates must provide at least 2 candidate views.")

    random_scores = torch.rand(
        teacher_feature_candidates.size(0),
        num_candidates,
        device=teacher_feature_candidates.device,
    )
    sampled_indices = random_scores.topk(k=2, dim=1).indices
    gather_index = sampled_indices.unsqueeze(-1).expand(
        -1,
        -1,
        teacher_feature_candidates.size(-1),
    )
    sampled = torch.gather(teacher_feature_candidates, dim=1, index=gather_index)
    return sampled[:, 0], sampled[:, 1]


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
        "TinyImageNet200",
        "Oxford_Pet",
        "STL10",
        "Stanford_Dogs",
    }:
        return "test"
    if dataset_name == "ImageNet1K":
        return "val"
    if dataset_name == "TinyImageNet200":
        return "val"
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
    class_names = label_wnids or None
    train_base = ImageNetH5Dataset(
        root=config.dataset_dir,
        split="train",
        transform=None,
        class_names=class_names,
    )
    eval_base = ImageNetH5Dataset(
        root=config.dataset_dir,
        split=official_eval_split,
        transform=None,
        class_names=class_names,
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
        train_base = Image10Dataset(
            root=config.dataset_dir,
            split="train",
            transform=None,
        )
        eval_base = Image10Dataset(
            root=config.dataset_dir,
            split=official_eval_split,
            transform=None,
        )
        return train_base, eval_base

    if config.dataset == "ImageNet1K":
        return _build_imagenet_h5_datasets(config, official_eval_split)

    if config.dataset in {"Image50", "Image100", "Image200"}:
        return _build_imagenet_h5_datasets(config, official_eval_split)

    if config.dataset == "TinyImageNet200":
        train_base = TinyImageNetDataset(
            root=config.dataset_dir, split="train", transform=None
        )
        eval_base = TinyImageNetDataset(
            root=config.dataset_dir, split=official_eval_split, transform=None
        )
        return train_base, eval_base

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

    if config.uses_imagenet1k_cached_teacher_views:
        teacher_cache_dir = config.resolved_imagenet1k_cached_teacher_views_dir
        if teacher_cache_dir is None:
            raise ValueError("ImageNet1K cached teacher views are enabled but no cache dir was resolved.")
        teacher_cache = ImageNetTeacherFeatureCache(
            teacher_cache_dir,
            expected_num_views=config.imagenet1k_cached_teacher_num_views,
            expected_feature_dim=config.feature_dim,
        )
        train_dataset = ImageNetTeacherFeatureTrainDataset(
            train_subset,
            teacher_cache,
        )
    else:
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
    train_workers = config.workers
    eval_workers = config.workers
    pin_memory = is_sharded_imagenet_dataset
    if config.persistent_workers is not None:
        persistent_workers = config.persistent_workers
    else:
        persistent_workers = True if is_sharded_imagenet_dataset else None

    if config.train_prefetch_factor is not None:
        train_prefetch_factor = config.train_prefetch_factor if train_workers > 0 else None
    else:
        train_prefetch_factor = (
            2 if is_sharded_imagenet_dataset and train_workers > 0 else None
        )

    if config.eval_prefetch_factor is not None:
        eval_prefetch_factor = config.eval_prefetch_factor if eval_workers > 0 else None
    else:
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
