from __future__ import annotations

import json
import os
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F

from datasets.aircraft_dataset import AircraftDataset
from datasets.cifar100_dataset import CIFAR100Dataset
from datasets.stl10_dataset import STL10Dataset
from datasets.tiny_imagenet_dataset import load_tiny_imagenet_class_texts

from .labels import load_label_entries


@dataclass
class SemanticPrototypeState:
    prototype_features: torch.Tensor
    class_text_features: torch.Tensor
    retrieval_text_features: torch.Tensor
    retrieval_class_indices: torch.Tensor
    info: dict[str, int | str | float]


_DEFAULT_SEMANTIC_TEXT_BATCH_SIZE = 256


def _spherical_text_prototypes(
    text_features: torch.Tensor,
    num_prototypes: int,
    max_iter: int = 25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically compress class text features to an arbitrary K."""
    if text_features.ndim != 2:
        raise ValueError("text_features must be a 2D tensor.")
    num_texts = text_features.size(0)
    if not 1 < num_prototypes <= num_texts:
        raise ValueError(
            f"num_prototypes must be in [2, {num_texts}], got {num_prototypes}."
        )

    features = F.normalize(text_features.float(), dim=-1)
    if num_prototypes == num_texts:
        assignments = torch.arange(num_texts, device=features.device)
        return features, assignments

    # Farthest-first initialization followed by spherical Lloyd updates avoids
    # adding a random seed to semantic prototype construction.
    selected = [0]
    nearest_distance = 1.0 - features @ features[0]
    for _ in range(1, num_prototypes):
        next_index = int(nearest_distance.argmax().item())
        selected.append(next_index)
        distance = 1.0 - features @ features[next_index]
        nearest_distance = torch.minimum(nearest_distance, distance)

    centers = features[selected]
    assignments = torch.full(
        (num_texts,), -1, device=features.device, dtype=torch.long
    )
    for _ in range(max_iter):
        next_assignments = (features @ centers.t()).argmax(dim=1)
        if torch.equal(next_assignments, assignments):
            break
        assignments = next_assignments
        updated = []
        for cluster_index in range(num_prototypes):
            members = features[assignments == cluster_index]
            center = (
                centers[cluster_index]
                if members.numel() == 0
                else members.mean(dim=0)
            )
            updated.append(F.normalize(center, dim=0))
        centers = torch.stack(updated, dim=0)
    return centers, assignments

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

        idx_to_wnid = raw.get("idx_to_wnid")
        if isinstance(idx_to_wnid, dict) and idx_to_wnid:
            sorted_pairs = sorted((int(index), name) for index, name in idx_to_wnid.items())
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

        idx_to_wnid = raw.get("idx_to_wnid")
        if isinstance(idx_to_wnid, dict) and idx_to_wnid:
            sorted_pairs = sorted((int(index), name) for index, name in idx_to_wnid.items())
            return [[str(name).strip()] for _, name in sorted_pairs]

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


def _pool_prompt_embeddings(
    encoded: torch.Tensor,
    normalized_aggregation: str,
) -> torch.Tensor:
    if normalized_aggregation == "mean":
        return encoded.mean(dim=0)
    if normalized_aggregation == "max":
        return encoded.max(dim=0).values
    if normalized_aggregation == "first":
        return encoded[0, :]
    raise ValueError(f"Unsupported semantic aggregation: {normalized_aggregation!r}")


def _encode_text_prompts_batched(
    backbone_model,
    tokenizer,
    prompts: list[str],
    device: torch.device,
    batch_size: int = _DEFAULT_SEMANTIC_TEXT_BATCH_SIZE,
) -> torch.Tensor:
    if not prompts:
        raise ValueError("prompts must not be empty.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    encoded_chunks = []
    for start in range(0, len(prompts), batch_size):
        prompt_chunk = prompts[start : start + batch_size]
        tokenized = tokenizer(prompt_chunk).to(device)
        encoded_chunk = backbone_model.encode_text(tokenized)
        encoded_chunks.append(F.normalize(encoded_chunk.float(), dim=-1))
    return torch.cat(encoded_chunks, dim=0)


def _encode_label_text_feature(
    backbone_model,
    tokenizer,
    prompt_templates: list[str],
    label: str,
    normalized_aggregation: str,
    template_topk: int,
    device: torch.device,
    reference_feature: torch.Tensor | None = None,
) -> torch.Tensor:
    prompts = [template.format(label=label).strip() for template in prompt_templates]
    encoded = _encode_text_prompts_batched(
        backbone_model=backbone_model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
    )

    if template_topk > 0 and encoded.size(0) > template_topk:
        if reference_feature is None:
            template_reference = F.normalize(encoded.mean(dim=0, keepdim=True), dim=-1)
        else:
            template_reference = F.normalize(reference_feature.unsqueeze(0), dim=-1)
        template_scores = (encoded @ template_reference.t()).squeeze(1)
        keep_indices = template_scores.topk(k=template_topk).indices
        encoded = encoded[keep_indices]

    pooled = _pool_prompt_embeddings(encoded, normalized_aggregation)
    return F.normalize(pooled, dim=-1)


def _pool_imagenet1k_text_features(
    backbone_model,
    tokenizer,
    prompt_templates: list[str],
    synonyms: list[str],
    normalized_aggregation: str,
    synonym_mode: str,
    primary_weight: float,
    template_topk: int,
    device: torch.device,
) -> torch.Tensor:
    primary = synonyms[0]

    def encode_label_prompts(label: str) -> torch.Tensor:
        prompts = [template.format(label=label).strip() for template in prompt_templates]
        return _encode_text_prompts_batched(
            backbone_model=backbone_model,
            tokenizer=tokenizer,
            prompts=prompts,
            device=device,
        )

    primary_encoded = encode_label_prompts(primary)

    if template_topk > 0 and primary_encoded.size(0) > template_topk:
        template_reference = F.normalize(primary_encoded.mean(dim=0, keepdim=True), dim=-1)
        template_scores = (primary_encoded @ template_reference.t()).squeeze(1)
        keep_indices = template_scores.topk(k=template_topk).indices
        primary_encoded = primary_encoded[keep_indices]

    primary_pooled = _pool_prompt_embeddings(primary_encoded, normalized_aggregation)

    if synonym_mode == "mean_all" or len(synonyms) <= 1:
        if len(synonyms) == 1:
            pooled = primary_pooled
        else:
            pooled_features = [primary_pooled]
            for synonym in synonyms[1:]:
                synonym_encoded = encode_label_prompts(synonym)
                if template_topk > 0 and synonym_encoded.size(0) > template_topk:
                    template_reference = F.normalize(primary_pooled.unsqueeze(0), dim=-1)
                    template_scores = (synonym_encoded @ template_reference.t()).squeeze(1)
                    keep_indices = template_scores.topk(k=template_topk).indices
                    synonym_encoded = synonym_encoded[keep_indices]
                pooled_features.append(
                    _pool_prompt_embeddings(synonym_encoded, normalized_aggregation)
                )
            pooled = torch.stack(pooled_features, dim=0).mean(dim=0)
        return F.normalize(pooled, dim=-1)

    if synonym_mode == "primary_weighted":
        if len(synonyms) == 1:
            return F.normalize(primary_pooled, dim=-1)

        synonym_pooled_features = []
        for synonym in synonyms[1:]:
            synonym_encoded = encode_label_prompts(synonym)
            if template_topk > 0 and synonym_encoded.size(0) > template_topk:
                template_reference = F.normalize(primary_pooled.unsqueeze(0), dim=-1)
                template_scores = (synonym_encoded @ template_reference.t()).squeeze(1)
                keep_indices = template_scores.topk(k=template_topk).indices
                synonym_encoded = synonym_encoded[keep_indices]
            synonym_pooled_features.append(
                _pool_prompt_embeddings(synonym_encoded, normalized_aggregation)
            )

        synonym_pooled = torch.stack(synonym_pooled_features, dim=0).mean(dim=0)
        pooled = (
            primary_weight * F.normalize(primary_pooled, dim=-1)
            + (1.0 - primary_weight) * F.normalize(synonym_pooled, dim=-1)
        )
        return F.normalize(pooled, dim=-1)

    raise ValueError(f"Unsupported semantic_imagenet_synonym_mode: {synonym_mode!r}")


@torch.no_grad()
def build_semantic_prototype_state(
    backbone_model,
    backbone_name: str,
    dataset_name: str,
    dataset_dir: str,
    cifar100_label_mode: str,
    prompt_style: str,
    aggregation: str,
    imagenet_synonym_mode: str,
    semantic_primary_weight: float,
    semantic_template_topk: int,
    semantic_mode: str,
    device: torch.device,
    num_prototypes: int | None = None,
) -> SemanticPrototypeState:
    class_text_features, info = build_semantic_text_features(
        backbone_model=backbone_model,
        backbone_name=backbone_name,
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        cifar100_label_mode=cifar100_label_mode,
        prompt_style=prompt_style,
        aggregation=aggregation,
        imagenet_synonym_mode=imagenet_synonym_mode,
        semantic_primary_weight=semantic_primary_weight,
        semantic_template_topk=semantic_template_topk,
        device=device,
    )

    source_text_features = class_text_features
    source_class_count = class_text_features.size(0)
    target_prototype_count = num_prototypes or source_class_count

    if not (dataset_name == "ImageNet1K" and semantic_mode == "tac_imagenet1k"):
        class_text_features, retrieval_class_indices = _spherical_text_prototypes(
            source_text_features,
            target_prototype_count,
        )
        return SemanticPrototypeState(
            prototype_features=class_text_features,
            class_text_features=class_text_features,
            retrieval_text_features=source_text_features,
            retrieval_class_indices=retrieval_class_indices,
            info={
                **info,
                "num_source_classes": source_class_count,
                "num_classes": target_prototype_count,
                "semantic_mode": semantic_mode,
                "retrieval_bank_size": source_class_count,
            },
        )

    if target_prototype_count != source_class_count:
        raise ValueError(
            "tac_imagenet1k requires num_prototypes to match the number of classes."
        )

    prompt_templates, resolved_prompt_style = _resolve_prompt_style(dataset_name, prompt_style)
    tokenizer = open_clip.get_tokenizer(backbone_name)
    normalized_aggregation = aggregation.strip().lower().replace("-", "_")
    class_synonyms = _load_imagenet1k_class_synonyms(dataset_dir)

    class_features = []
    retrieval_features = []
    retrieval_class_indices = []
    for class_index, synonyms in enumerate(class_synonyms):
        primary_label = synonyms[0]
        primary_feature = _encode_label_text_feature(
            backbone_model=backbone_model,
            tokenizer=tokenizer,
            prompt_templates=prompt_templates,
            label=primary_label,
            normalized_aggregation=normalized_aggregation,
            template_topk=semantic_template_topk,
            device=device,
        )

        alias_features = [primary_feature]
        for synonym in synonyms[1:]:
            alias_features.append(
                _encode_label_text_feature(
                    backbone_model=backbone_model,
                    tokenizer=tokenizer,
                    prompt_templates=prompt_templates,
                    label=synonym,
                    normalized_aggregation=normalized_aggregation,
                    template_topk=semantic_template_topk,
                    device=device,
                    reference_feature=primary_feature,
                )
            )

        if len(alias_features) == 1 or imagenet_synonym_mode == "primary_weighted":
            if len(alias_features) == 1:
                class_feature = primary_feature
            else:
                synonym_feature = torch.stack(alias_features[1:], dim=0).mean(dim=0)
                class_feature = F.normalize(
                    semantic_primary_weight * primary_feature
                    + (1.0 - semantic_primary_weight) * synonym_feature,
                    dim=-1,
                )
        else:
            class_feature = F.normalize(torch.stack(alias_features, dim=0).mean(dim=0), dim=-1)

        class_features.append(class_feature)
        retrieval_features.extend(alias_features)
        retrieval_class_indices.extend([class_index] * len(alias_features))

    class_text_features = F.normalize(torch.stack(class_features, dim=0), dim=-1)
    retrieval_text_features = F.normalize(torch.stack(retrieval_features, dim=0), dim=-1)
    retrieval_class_indices_tensor = torch.tensor(
        retrieval_class_indices, device=device, dtype=torch.long
    )
    num_templates_used = (
        min(len(prompt_templates), semantic_template_topk)
        if semantic_template_topk > 0
        else len(prompt_templates)
    )
    return SemanticPrototypeState(
        prototype_features=class_text_features,
        class_text_features=class_text_features,
        retrieval_text_features=retrieval_text_features,
        retrieval_class_indices=retrieval_class_indices_tensor,
        info={
            "num_classes": len(class_synonyms),
            "num_templates": num_templates_used,
            "prompt_style": resolved_prompt_style,
            "aggregation": normalized_aggregation,
            "imagenet_synonym_mode": imagenet_synonym_mode,
            "semantic_mode": semantic_mode,
            "retrieval_bank_size": retrieval_text_features.size(0),
        },
    )


def load_class_texts(
    dataset_name: str,
    dataset_dir: str,
    cifar100_label_mode: str = "fine",
) -> list[str]:
    label_entries = load_label_entries(dataset_name)
    if label_entries:
        return [entry.name for entry in label_entries]

    if dataset_name == "CIFAR10":
        return list(_CIFAR10_CLASS_NAMES)
    if dataset_name == "CIFAR100":
        return [
            name.replace("_", " ")
            for name in CIFAR100Dataset(
                root=dataset_dir,
                train=True,
                transform=None,
                download=False,
                label_mode=cifar100_label_mode,
            ).classes
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
        return _load_imagenet1k_class_texts(dataset_dir)
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
    cifar100_label_mode: str,
    prompt_style: str,
    aggregation: str,
    imagenet_synonym_mode: str,
    semantic_primary_weight: float,
    semantic_template_topk: int,
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
            pooled = _pool_imagenet1k_text_features(
                backbone_model=backbone_model,
                tokenizer=tokenizer,
                prompt_templates=prompt_templates,
                synonyms=synonyms,
                normalized_aggregation=normalized_aggregation,
                synonym_mode=imagenet_synonym_mode,
                primary_weight=semantic_primary_weight,
                template_topk=semantic_template_topk,
                device=device,
            )
            pooled_class_features.append(pooled)

        final_pooled = torch.stack(pooled_class_features, dim=0)
        templates_per_synonym = (
            min(len(prompt_templates), semantic_template_topk)
            if semantic_template_topk > 0
            else len(prompt_templates)
        )
        num_templates_used = templates_per_synonym

    else:
        # Standard code path for other datasets using single class names
        class_texts = load_class_texts(
            dataset_name,
            dataset_dir,
            cifar100_label_mode=cifar100_label_mode,
        )
        num_classes = len(class_texts)

        prompt_groups = [
            [template.format(label=class_text).strip() for template in prompt_templates]
            for class_text in class_texts
        ]
        flat_prompts = [prompt for prompt_group in prompt_groups for prompt in prompt_group]
        encoded = _encode_text_prompts_batched(
            backbone_model=backbone_model,
            tokenizer=tokenizer,
            prompts=flat_prompts,
            device=device,
        )
        encoded = encoded.view(len(class_texts), len(prompt_templates), -1)

        if 0 < semantic_template_topk < encoded.size(1):
            template_reference = F.normalize(encoded.mean(dim=1), dim=-1)
            template_scores = (encoded * template_reference.unsqueeze(1)).sum(dim=-1)
            keep_indices = template_scores.topk(k=semantic_template_topk, dim=1).indices
            gather_index = keep_indices.unsqueeze(-1).expand(-1, -1, encoded.size(-1))
            encoded = torch.gather(encoded, dim=1, index=gather_index)

        if normalized_aggregation == "mean":
            final_pooled = encoded.mean(dim=1)
        elif normalized_aggregation == "max":
            final_pooled = encoded.max(dim=1).values
        elif normalized_aggregation == "first":
            final_pooled = encoded[:, 0, :]
        else:
            raise ValueError(f"Unsupported semantic aggregation: {aggregation!r}")

        num_templates_used = (
            min(len(prompt_templates), semantic_template_topk)
            if semantic_template_topk > 0
            else len(prompt_templates)
        )

    return F.normalize(final_pooled, dim=-1), {
        "num_classes": num_classes,
        "num_templates": num_templates_used,
        "prompt_style": resolved_prompt_style,
        "aggregation": normalized_aggregation,
        "imagenet_synonym_mode": imagenet_synonym_mode if dataset_name == "ImageNet1K" else "n/a",
    }


def semantic_cluster_alignment_loss(
    logits: torch.Tensor,
    teacher_features: torch.Tensor,
    text_features: torch.Tensor,
    teacher_logit_scale: torch.Tensor,
    temperature: float,
    target_topk: int = 0,
) -> torch.Tensor:
    if logits.size(0) == 0:
        return logits.new_zeros(())

    with torch.autocast(device_type=logits.device.type, enabled=False):
        teacher_features_f = F.normalize(teacher_features.float(), dim=-1)
        text_features_f = F.normalize(text_features.float(), dim=-1)
        teacher_logits = teacher_logit_scale.float() * (teacher_features_f @ text_features_f.t())
        if 0 < target_topk < teacher_logits.size(1):
            topk_values, topk_indices = teacher_logits.topk(k=target_topk, dim=-1)
            sparse_logits = torch.full_like(teacher_logits, float("-inf"))
            sparse_logits.scatter_(1, topk_indices, topk_values)
            target_probs = F.softmax(sparse_logits, dim=-1).detach()
        else:
            target_probs = F.softmax(teacher_logits, dim=-1).detach()
        student_log_probs = F.log_softmax(logits.float() / temperature, dim=-1)
        return -(target_probs * student_log_probs).sum(dim=1).mean()


def semantic_retrieval_alignment_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    text_features: torch.Tensor,
    teacher_logit_scale: torch.Tensor,
    retrieval_topk: int = 0,
) -> torch.Tensor:
    if student_features.size(0) == 0:
        return student_features.new_zeros(())

    with torch.autocast(device_type=student_features.device.type, enabled=False):
        student_features_f = F.normalize(student_features.float(), dim=-1)
        teacher_features_f = F.normalize(teacher_features.float(), dim=-1)
        text_features_f = F.normalize(text_features.float(), dim=-1)
        retrieval_logits = teacher_logit_scale.float() * (
            teacher_features_f @ text_features_f.t()
        )

        if 0 < retrieval_topk < retrieval_logits.size(1):
            topk_values, topk_indices = retrieval_logits.topk(k=retrieval_topk, dim=-1)
            sparse_logits = torch.full_like(retrieval_logits, float("-inf"))
            sparse_logits.scatter_(1, topk_indices, topk_values)
            retrieval_weights = F.softmax(sparse_logits, dim=-1).detach()
        else:
            retrieval_weights = F.softmax(retrieval_logits, dim=-1).detach()

        retrieved_text_features = retrieval_weights @ text_features_f
        retrieved_text_features = F.normalize(retrieved_text_features, dim=-1)
        cosine_similarity = (student_features_f * retrieved_text_features).sum(dim=-1)
        return 1.0 - cosine_similarity.mean()


def semantic_tac_alignment_losses(
    logits: torch.Tensor,
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    semantic_state: SemanticPrototypeState,
    teacher_logit_scale: torch.Tensor,
    temperature: float,
    target_topk: int = 0,
    retrieval_topk: int = 0,
    class_blend: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.size(0) == 0:
        zero = logits.new_zeros(())
        return zero, zero

    with torch.autocast(device_type=logits.device.type, enabled=False):
        teacher_features_f = F.normalize(teacher_features.float(), dim=-1)
        student_features_f = F.normalize(student_features.float(), dim=-1)
        class_text_features_f = F.normalize(semantic_state.class_text_features.float(), dim=-1)
        retrieval_text_features_f = F.normalize(
            semantic_state.retrieval_text_features.float(), dim=-1
        )

        retrieval_logits = teacher_logit_scale.float() * (
            teacher_features_f @ retrieval_text_features_f.t()
        )
        if 0 < retrieval_topk < retrieval_logits.size(1):
            topk_values, topk_indices = retrieval_logits.topk(k=retrieval_topk, dim=-1)
            sparse_logits = torch.full_like(retrieval_logits, float("-inf"))
            sparse_logits.scatter_(1, topk_indices, topk_values)
            retrieval_weights = F.softmax(sparse_logits, dim=-1).detach()
        else:
            retrieval_weights = F.softmax(retrieval_logits, dim=-1).detach()

        retrieved_text_features = retrieval_weights @ retrieval_text_features_f
        retrieved_text_features = F.normalize(retrieved_text_features, dim=-1)
        retrieval_cosine = (student_features_f * retrieved_text_features).sum(dim=-1)
        retrieval_loss = 1.0 - retrieval_cosine.mean()

        retrieval_class_probs = logits.new_zeros(
            logits.size(0), class_text_features_f.size(0), dtype=torch.float32
        )
        retrieval_class_indices = semantic_state.retrieval_class_indices.unsqueeze(0).expand(
            retrieval_weights.size(0), -1
        )
        retrieval_class_probs.scatter_add_(1, retrieval_class_indices, retrieval_weights)
        retrieval_class_probs = retrieval_class_probs / retrieval_class_probs.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)

        teacher_class_logits = teacher_logit_scale.float() * (
            teacher_features_f @ class_text_features_f.t()
        )
        if 0 < target_topk < teacher_class_logits.size(1):
            topk_values, topk_indices = teacher_class_logits.topk(k=target_topk, dim=-1)
            sparse_logits = torch.full_like(teacher_class_logits, float("-inf"))
            sparse_logits.scatter_(1, topk_indices, topk_values)
            direct_class_probs = F.softmax(sparse_logits, dim=-1).detach()
        else:
            direct_class_probs = F.softmax(teacher_class_logits, dim=-1).detach()

        target_probs = F.normalize(
            class_blend * direct_class_probs + (1.0 - class_blend) * retrieval_class_probs,
            p=1,
            dim=-1,
        )
        student_log_probs = F.log_softmax(logits.float() / temperature, dim=-1)
        class_loss = -(target_probs * student_log_probs).sum(dim=1).mean()
        return class_loss, retrieval_loss
