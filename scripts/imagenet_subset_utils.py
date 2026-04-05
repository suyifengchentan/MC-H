from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypersphere_dpp.labels import LabelEntry, default_label_config_path, load_label_entries


DEFAULT_OUTPUT_ROOTS = {
    "ImageNet1K": Path("/home/geyingjie/hypersphere-data/ImageNet1KWebDataset"),
    "Image50": Path("/home/geyingjie/hypersphere-data/Image50WebDataset"),
    "Image100": Path("/home/geyingjie/hypersphere-data/Image100WebDataset"),
    "Image200": Path("/home/geyingjie/hypersphere-data/Image200WebDataset"),
}
DEFAULT_DATASETS = ["ImageNet1K", "Image50", "Image100", "Image200"]


@dataclass(frozen=True)
class SubsetSpec:
    dataset_name: str
    output_root: Path
    entries: list[LabelEntry]
    source_label_to_subset_label: dict[int, int]
    subset_label_to_wnid: dict[int, str]
    subset_label_to_name: dict[int, str]


def load_imagenet_1k_classes(imagenet_root: Path) -> list[tuple[str, str]]:
    classes_path = imagenet_root / "classes.py"
    if not classes_path.is_file():
        raise FileNotFoundError(f"Expected classes.py at: {classes_path}")

    spec = importlib.util.spec_from_file_location("imagenet_classes_module", classes_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import ImageNet classes from: {classes_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    imagenet_classes = getattr(module, "IMAGENET2012_CLASSES", None)
    if imagenet_classes is None:
        raise AttributeError(f"IMAGENET2012_CLASSES not found in {classes_path}")

    return list(imagenet_classes.items())


def build_subset_specs(dataset_names: list[str], imagenet_items: list[tuple[str, str]]) -> list[SubsetSpec]:
    wnid_to_source_label = {wnid: index for index, (wnid, _) in enumerate(imagenet_items)}
    subset_specs: list[SubsetSpec] = []

    for dataset_name in dataset_names:
        if dataset_name == "ImageNet1K":
            label_entries = [LabelEntry(wnid=wnid, name=name) for wnid, name in imagenet_items]
        else:
            label_entries = load_label_entries(dataset_name)
            if not label_entries:
                raise FileNotFoundError(
                    f"No label entries found for {dataset_name} at {default_label_config_path(dataset_name)}"
                )

        source_label_to_subset_label: dict[int, int] = {}
        subset_label_to_wnid: dict[int, str] = {}
        subset_label_to_name: dict[int, str] = {}

        for subset_label, entry in enumerate(label_entries):
            if entry.wnid not in wnid_to_source_label:
                raise KeyError(f"{dataset_name} label {entry.wnid!r} does not exist in ImageNet-1k classes.")
            source_label = wnid_to_source_label[entry.wnid]
            source_label_to_subset_label[source_label] = subset_label
            subset_label_to_wnid[subset_label] = entry.wnid
            subset_label_to_name[subset_label] = entry.name

        subset_specs.append(
            SubsetSpec(
                dataset_name=dataset_name,
                output_root=DEFAULT_OUTPUT_ROOTS[dataset_name],
                entries=label_entries,
                source_label_to_subset_label=source_label_to_subset_label,
                subset_label_to_wnid=subset_label_to_wnid,
                subset_label_to_name=subset_label_to_name,
            )
        )

    return subset_specs


def discover_source_splits(source_data_root: Path) -> list[str]:
    split_prefixes = [
        split_prefix
        for split_prefix in ("train", "validation", "test")
        if any(source_data_root.glob(f"{split_prefix}-*.parquet"))
    ]
    if not split_prefixes:
        raise FileNotFoundError(f"No parquet shards found under {source_data_root}")
    return split_prefixes


def build_label_info(spec: SubsetSpec) -> dict[str, dict[str, str | int]]:
    return {
        "idx_to_wnid": {str(index): spec.subset_label_to_wnid[index] for index in range(len(spec.entries))},
        "idx_to_class_name": {str(index): spec.subset_label_to_name[index] for index in range(len(spec.entries))},
        "class_to_idx": {spec.subset_label_to_wnid[index]: index for index in range(len(spec.entries))},
    }


def write_label_info(output_root: Path, spec: SubsetSpec) -> None:
    label_info = build_label_info(spec)
    with (output_root / "label_info.json").open("w", encoding="utf-8") as handle:
        json.dump(label_info, handle, ensure_ascii=False, indent=2)
