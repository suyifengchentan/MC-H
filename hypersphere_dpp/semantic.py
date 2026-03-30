from __future__ import annotations

import json
import os
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F

from datasets.aircraft_dataset import AircraftDataset
from datasets.cifar100_dataset import CIFAR100Dataset
from datasets.indexed_image_folder_dataset import IndexedImageFolderDataset
from datasets.stl10_dataset import STL10Dataset

from .labels import load_label_entries

_CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

_SEMANTIC_TEMPLATE_PRESETS = {
    "single": ["a photo of a {label}"],
    "clip_ensemble": [
        "a photo of a {label}",
        "a blurry photo of a {label}",
        "a close-up photo of a {label}",
        "a bright photo of a {label}",
        "a cropped photo of a {label}",
    ],
}

_SEMANTIC_DATASET_TEMPLATE_PRESETS = {
    "CUB_200_2011": "clip_ensemble",
    "FGVC_Aircraft": "clip_ensemble",
    "Image10": "clip_ensemble",
    "Image50": "clip_ensemble",
    "Image100": "clip_ensemble",
    "Image200": "clip_ensemble",
    "Oxford_Flowers": "clip_ensemble",
    "Oxford_Pet": "clip_ensemble",
    "Stanford_Dogs": "clip_ensemble",
}


def _resolve_prompt_style(dataset_name: str, prompt_style: str) -> tuple[list[str], str]:
    normalized = str(prompt_style).strip().lower().replace("-", "_")
    if normalized == "auto":
        normalized = _SEMANTIC_DATASET_TEMPLATE_PRESETS.get(dataset_name, "single")
    if normalized not in _SEMANTIC_TEMPLATE_PRESETS:
        raise ValueError(
            "Unsupported semantic_prompt_style "
            f"{prompt_style!r}; expected one of {sorted({'auto', *_SEMANTIC_TEMPLATE_PRESETS})}."
        )
    return list(_SEMANTIC_TEMPLATE_PRESETS[normalized]), normalized


def _load_cub_class_texts(dataset_dir: str) -> list[str]:
    classes_path = Path(dataset_dir) / "classes.txt"
    class_texts: list[str] = []
    with classes_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            _, class_name = raw_line.strip().split()
            class_texts.append(class_name.split(".", 1)[1].replace("_", " "))
    return class_texts


def _load_stanford_dogs_class_texts(dataset_dir: str) -> list[str]:
    root = Path(dataset_dir)
    candidate_dirs = [root / "Images", root / "images" / "Images"]
    image_dir = next((candidate for candidate in candidate_dirs if candidate.is_dir()), None)
    if image_dir is None:
        raise FileNotFoundError(
            f"Stanford_Dogs image directory not found under {dataset_dir}."
        )

    folder_names = sorted(
        entry.name
        for entry in image_dir.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )
    return [folder_name.split("-", 1)[1].replace("_", " ") for folder_name in folder_names]


def _load_oxford_flowers_class_texts(dataset_dir: str) -> list[str]:
    root = Path(dataset_dir)
    flower_root = root / "flower_data" if (root / "flower_data").is_dir() else root
    json_path = flower_root / "cat_to_name.json"
    train_dir = flower_root / "train"
    with json_path.open("r", encoding="utf-8") as handle:
        cat_to_name = json.load(handle)
    folder_names = sorted(
        entry.name for entry in train_dir.iterdir() if entry.is_dir() and not entry.name.startswith(".")
    )
    return [str(cat_to_name[folder_name]).strip() for folder_name in folder_names]


def _load_oxford_pet_class_texts(dataset_dir: str) -> list[str]:
    list_path = Path(dataset_dir) / "annotations" / "list.txt"
    id_to_name: dict[int, tuple[str, int]] = {}
    with list_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.startswith("#"):
                continue
            parts = raw_line.strip().split()
            if not parts:
                continue
            image_name, class_id, species_id = parts[0], int(parts[1]), int(parts[2])
            if class_id in id_to_name:
                continue
            breed = " ".join(image_name.split("_")[:-1])
            id_to_name[class_id] = (breed, species_id)
    return [
        f"{breed} {'cat' if species_id == 1 else 'dog'}"
        for class_id, (breed, species_id) in sorted(id_to_name.items())
    ]


def load_class_texts(dataset_name: str, dataset_dir: str) -> list[str]:
    label_entries = load_label_entries(dataset_name)
    if label_entries:
        return [entry.name for entry in label_entries]

    if dataset_name == "CIFAR10":
        return list(_CIFAR10_CLASS_NAMES)
    if dataset_name == "CIFAR100":
        return [
            name.replace("_", " ")
            for name in CIFAR100Dataset(root=dataset_dir, train=True, transform=None, download=False).classes
        ]
    if dataset_name == "STL10":
        return [
            name.replace("_", " ")
            for name in STL10Dataset(root=dataset_dir, split="train", transform=None, download=False).classes
        ]
    if dataset_name == "CUB_200_2011":
        return _load_cub_class_texts(dataset_dir)
    if dataset_name == "Stanford_Dogs":
        return _load_stanford_dogs_class_texts(dataset_dir)
    if dataset_name == "Oxford_Flowers":
        return _load_oxford_flowers_class_texts(dataset_dir)
    if dataset_name == "Oxford_Pet":
        return _load_oxford_pet_class_texts(dataset_dir)
    if dataset_name == "FGVC_Aircraft":
        return AircraftDataset(root=dataset_dir, split="train", transform=None).classes
    if dataset_name == "Image10":
        dataset = IndexedImageFolderDataset(root=dataset_dir, split="train", transform=None)
        return [class_name.replace("_", " ").replace("-", " ") for class_name in dataset.classes]

    raise ValueError(f"Semantic class texts are unsupported for dataset {dataset_name!r}.")


@torch.no_grad()
def build_semantic_text_features(
    backbone_model,
    backbone_name: str,
    dataset_name: str,
    dataset_dir: str,
    prompt_style: str,
    aggregation: str,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, int | str]]:
    class_texts = load_class_texts(dataset_name, dataset_dir)
    prompt_templates, resolved_prompt_style = _resolve_prompt_style(dataset_name, prompt_style)
    tokenizer = open_clip.get_tokenizer(backbone_name)

    prompt_groups = [
        [template.format(label=class_text).strip() for template in prompt_templates]
        for class_text in class_texts
    ]
    flat_prompts = [prompt for prompt_group in prompt_groups for prompt in prompt_group]
    tokenized = tokenizer(flat_prompts).to(device)
    encoded = backbone_model.encode_text(tokenized)
    encoded = F.normalize(encoded.float(), dim=-1)
    encoded = encoded.view(len(class_texts), len(prompt_templates), -1)

    normalized_aggregation = aggregation.strip().lower().replace("-", "_")
    if normalized_aggregation == "mean":
        pooled = encoded.mean(dim=1)
    elif normalized_aggregation == "max":
        pooled = encoded.max(dim=1).values
    elif normalized_aggregation == "first":
        pooled = encoded[:, 0, :]
    else:
        raise ValueError(f"Unsupported semantic aggregation: {aggregation!r}")

    return F.normalize(pooled, dim=-1), {
        "num_classes": len(class_texts),
        "num_templates": len(prompt_templates),
        "prompt_style": resolved_prompt_style,
        "aggregation": normalized_aggregation,
    }


def semantic_cluster_alignment_loss(
    logits: torch.Tensor,
    teacher_features: torch.Tensor,
    text_features: torch.Tensor,
    teacher_logit_scale: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if logits.size(0) == 0:
        return logits.new_zeros(())

    with torch.autocast(device_type=logits.device.type, enabled=False):
        teacher_features_f = F.normalize(teacher_features.float(), dim=-1)
        text_features_f = F.normalize(text_features.float(), dim=-1)
        teacher_logits = teacher_logit_scale.float() * (teacher_features_f @ text_features_f.t())
        target_probs = F.softmax(teacher_logits, dim=-1).detach()
        student_log_probs = F.log_softmax(logits.float() / temperature, dim=-1)
        return -(target_probs * student_log_probs).sum(dim=1).mean()
