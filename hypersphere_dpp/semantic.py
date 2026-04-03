from __future__ import annotations

import json
import os
import importlib.util
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F

from datasets.aircraft_dataset import AircraftDataset
from datasets.cifar100_dataset import CIFAR100Dataset
from datasets.indexed_image_folder_dataset import IndexedImageFolderDataset
from datasets.stl10_dataset import STL10Dataset
from datasets.tiny_imagenet_dataset import load_tiny_imagenet_class_texts

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
    # CLIP paper 80-template ensemble for ImageNet-scale fine-grained recognition.
    # Covers diverse contexts (rendered, embroidery, cartoon, origami, etc.) which
    # anchors fine-grained classes more precisely in the CLIP text embedding space.
    "imagenet1k_80": [
        "a bad photo of a {label}",
        "a photo of many {label}",
        "a sculpture of a {label}",
        "a photo of the hard to see {label}",
        "a low resolution photo of the {label}",
        "a rendering of a {label}",
        "graffiti of a {label}",
        "a bad photo of the {label}",
        "a cropped photo of the {label}",
        "a tattoo of a {label}",
        "the embroidered {label}",
        "a photo of a hard to see {label}",
        "a bright photo of a {label}",
        "a photo of a clean {label}",
        "a photo of a dirty {label}",
        "a dark photo of the {label}",
        "a drawing of a {label}",
        "a photo of my {label}",
        "the plastic {label}",
        "a photo of the cool {label}",
        "a close-up photo of a {label}",
        "a black and white photo of the {label}",
        "a painting of the {label}",
        "a painting of a {label}",
        "a pixelated photo of the {label}",
        "a sculpture of the {label}",
        "a bright photo of the {label}",
        "a cropped photo of a {label}",
        "a plastic {label}",
        "a photo of the dirty {label}",
        "a jpeg corrupted photo of a {label}",
        "a blurry photo of the {label}",
        "a photo of the {label}",
        "a good photo of the {label}",
        "a rendering of the {label}",
        "a {label} in a video game",
        "a photo of one {label}",
        "a doodle of a {label}",
        "a close-up photo of the {label}",
        "a photo of a {label}",
        "the origami {label}",
        "the {label} in a video game",
        "a sketch of a {label}",
        "a doodle of the {label}",
        "a origami {label}",
        "a low resolution photo of a {label}",
        "the toy {label}",
        "a rendition of the {label}",
        "a photo of the clean {label}",
        "a photo of a large {label}",
        "a rendition of a {label}",
        "a photo of a nice {label}",
        "a photo of a weird {label}",
        "a blurry photo of a {label}",
        "a cartoon {label}",
        "art of a {label}",
        "a sketch of the {label}",
        "a embroidered {label}",
        "a pixelated photo of a {label}",
        "itap of the {label}",
        "a jpeg corrupted photo of the {label}",
        "a good photo of a {label}",
        "a plushie {label}",
        "a photo of the nice {label}",
        "a photo of the small {label}",
        "a photo of the weird {label}",
        "the cartoon {label}",
        "art of the {label}",
        "a drawing of the {label}",
        "a photo of the large {label}",
        "a black and white photo of a {label}",
        "the plushie {label}",
        "a dark photo of a {label}",
        "itap of a {label}",
        "graffiti of the {label}",
        "a toy {label}",
        "itap of my {label}",
        "a photo of a cool {label}",
        "a photo of a small {label}",
        "a tattoo of the {label}",
    ],
}

_SEMANTIC_DATASET_TEMPLATE_PRESETS = {
    "CUB_200_2011": "clip_ensemble",
    "FGVC_Aircraft": "clip_ensemble",
    "Image10": "clip_ensemble",
    # Use the full 80-template ensemble for ImageNet1K: the 1000 fine-grained
    # classes demand richer textual context to produce well-separated prototypes.
    "ImageNet1K": "imagenet1k_80",
    "Image50": "clip_ensemble",
    "Image100": "clip_ensemble",
    "Image200": "imagenet1k_80",
    "TinyImageNet200": "imagenet1k_80",
    "Oxford_Flowers": "clip_ensemble",
    "Oxford_Pet": "clip_ensemble",
    "Stanford_Dogs": "clip_ensemble",
}


def _canonicalize_imagenet_class_name(name: str) -> str:
    normalized = str(name).strip().replace("_", " ")
    primary_label = normalized.split(",", 1)[0].strip()
    return primary_label or normalized


def _parse_imagenet_synonyms(name: str) -> list[str]:
    """Return all comma-separated WordNet synonyms for one ImageNet class.

    Example: ``"goldfish, Carassius auratus"`` → ``["goldfish", "Carassius auratus"]``
    """
    normalized = str(name).strip().replace("_", " ")
    synonyms = [part.strip() for part in normalized.split(",")]
    return [s for s in synonyms if s]


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


def _load_imagenet1k_class_texts(dataset_dir: str) -> list[str]:
    label_info_path = Path(dataset_dir) / "label_info.json"
    if label_info_path.is_file():
        with label_info_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)

        idx_to_class_name = raw.get("idx_to_class_name")
        if isinstance(idx_to_class_name, dict) and idx_to_class_name:
            sorted_pairs = sorted((int(index), name) for index, name in idx_to_class_name.items())
            return [
                _canonicalize_imagenet_class_name(name)
                for _, name in sorted_pairs
            ]

    classes_path = Path(dataset_dir) / "classes.py"
    spec = importlib.util.spec_from_file_location(
        "imagenet1k_classes_module",
        classes_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import ImageNet-1K classes from: {classes_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    imagenet_classes = getattr(module, "IMAGENET2012_CLASSES", None)
    if not isinstance(imagenet_classes, dict) or not imagenet_classes:
        raise ValueError(f"IMAGENET2012_CLASSES is missing or invalid in {classes_path}")
    return [_canonicalize_imagenet_class_name(name) for name in imagenet_classes.values()]


def _load_imagenet1k_class_synonyms(dataset_dir: str) -> list[list[str]]:
    """Return all WordNet synonyms for each ImageNet1K class, ordered by class index.

    Each element is a list of synonym strings for that class.  For classes that
    have a single name the list contains exactly one element.
    """
    label_info_path = Path(dataset_dir) / "label_info.json"
    if label_info_path.is_file():
        with label_info_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)

        idx_to_class_name = raw.get("idx_to_class_name")
        if isinstance(idx_to_class_name, dict) and idx_to_class_name:
            sorted_pairs = sorted((int(index), name) for index, name in idx_to_class_name.items())
            return [_parse_imagenet_synonyms(name) for _, name in sorted_pairs]

    classes_path = Path(dataset_dir) / "classes.py"
    spec = importlib.util.spec_from_file_location(
        "imagenet1k_classes_module",
        classes_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import ImageNet-1K classes from: {classes_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    imagenet_classes = getattr(module, "IMAGENET2012_CLASSES", None)
    if not isinstance(imagenet_classes, dict) or not imagenet_classes:
        raise ValueError(f"IMAGENET2012_CLASSES is missing or invalid in {classes_path}")
    return [_parse_imagenet_synonyms(name) for name in imagenet_classes.values()]


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
    if dataset_name == "ImageNet1K":
        return _load_imagenet1k_class_texts(dataset_dir)
    if dataset_name == "TinyImageNet200":
        return load_tiny_imagenet_class_texts(dataset_dir)

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
    prompt_templates, resolved_prompt_style = _resolve_prompt_style(dataset_name, prompt_style)
    tokenizer = open_clip.get_tokenizer(backbone_name)
    normalized_aggregation = aggregation.strip().lower().replace("-", "_")

    # ImageNet1K uses all available WordNet synonyms for rich semantic pooling.
    if dataset_name == "ImageNet1K":
        class_synonyms = _load_imagenet1k_class_synonyms(dataset_dir)
        num_classes = len(class_synonyms)

        pooled_class_features = []
        for synonyms in class_synonyms:
            # Generate prompts using all synonyms and all templates
            flat_prompts = [
                template.format(label=syn).strip()
                for syn in synonyms
                for template in prompt_templates
            ]
            tokenized = tokenizer(flat_prompts).to(device)
            # Batch encode
            encoded = backbone_model.encode_text(tokenized)
            encoded = F.normalize(encoded.float(), dim=-1)

            # Pool across all templates & synonyms for this single class
            if normalized_aggregation == "mean":
                pooled = encoded.mean(dim=0)
            elif normalized_aggregation == "max":
                pooled = encoded.max(dim=0).values
            elif normalized_aggregation == "first":
                pooled = encoded[0, :]
            else:
                raise ValueError(f"Unsupported semantic aggregation: {aggregation!r}")

            pooled_class_features.append(pooled)

        final_pooled = torch.stack(pooled_class_features, dim=0)
        num_templates_used = len(prompt_templates) * (sum(len(s) for s in class_synonyms) / max(1, num_classes))

    else:
        # Standard code path for other datasets using single class names
        class_texts = load_class_texts(dataset_name, dataset_dir)
        num_classes = len(class_texts)

        prompt_groups = [
            [template.format(label=class_text).strip() for template in prompt_templates]
            for class_text in class_texts
        ]
        flat_prompts = [prompt for prompt_group in prompt_groups for prompt in prompt_group]
        tokenized = tokenizer(flat_prompts).to(device)
        encoded = backbone_model.encode_text(tokenized)
        encoded = F.normalize(encoded.float(), dim=-1)
        encoded = encoded.view(len(class_texts), len(prompt_templates), -1)

        if normalized_aggregation == "mean":
            final_pooled = encoded.mean(dim=1)
        elif normalized_aggregation == "max":
            final_pooled = encoded.max(dim=1).values
        elif normalized_aggregation == "first":
            final_pooled = encoded[:, 0, :]
        else:
            raise ValueError(f"Unsupported semantic aggregation: {aggregation!r}")

        num_templates_used = len(prompt_templates)

    return F.normalize(final_pooled, dim=-1), {
        "num_classes": num_classes,
        "num_templates": num_templates_used,
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
