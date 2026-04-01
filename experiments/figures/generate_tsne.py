#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypersphere_dpp.checkpoint import load_checkpoint
from hypersphere_dpp.config import load_experiment_config
from hypersphere_dpp.data import SingleViewDataset, build_experiment_dataloaders, safe_default_collate
from hypersphere_dpp.labels import load_label_name_map
from hypersphere_dpp.model import HypersphereDPPModel
from hypersphere_dpp.utils import resolve_device, set_seed


def apply_neurips_style() -> None:
    """Apply a publication-friendly style with Computer Modern typography."""
    neurips_rc = {
        # --- Font -----------------------------------------------------------
        "font.family": "serif",
        "font.serif": [
            "Computer Modern Roman",
            "CMU Serif",
            "cmr10",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",
        "mathtext.rm": "cmr10",
        "axes.formatter.use_mathtext": True,
        "font.size": 11,
        "font.weight": "semibold",
        # --- Axes -----------------------------------------------------------
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "semibold",
        "axes.linewidth": 0.0,
        "axes.edgecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "axes.grid": False,
        # --- Ticks ----------------------------------------------------------
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        # --- Legend ----------------------------------------------------------
        "legend.fontsize": 7.6,
        "legend.frameon": False,
        "legend.borderpad": 0.3,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.0,
        # --- Figure ---------------------------------------------------------
        "figure.facecolor": "#ffffff",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # --- Scatter --------------------------------------------------------
        "scatter.edgecolors": "face",
        # --- PDF / PS -------------------------------------------------------
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    matplotlib.rcParams.update(neurips_rc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dataset t-SNE plots before and after Hypersphere-DPP clustering."
    )
    parser.add_argument(
        "--config",
        default="config/test/coarse_grained/cifar100_vitb16_hypersphere_dpp_config.yaml",
        help="Path to the evaluation config YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. If omitted, the script resolves it from eval_results.csv.",
    )
    parser.add_argument(
        "--results-csv",
        default="eval_results.csv",
        help="CSV used to auto-resolve the CIFAR100 checkpoint directory.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default=None,
        help="Dataset split used for visualization. Defaults to the config official eval split.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=100,
        help="Balanced number of images per class for t-SNE.",
    )
    parser.add_argument(
        "--top-k-classes",
        type=int,
        default=10,
        help="Select the top-k largest classes by split frequency.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Inference batch size used while extracting features.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader workers used for plotting.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=35.0,
        help="t-SNE perplexity.",
    )
    parser.add_argument(
        "--learning-rate",
        default="auto",
        help="t-SNE learning rate passed to scikit-learn.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device string, e.g. cuda:0.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--raw-image-size",
        type=int,
        default=40,
        help="Resize size used to represent original images before t-SNE.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output figure path.",
    )
    return parser.parse_args()


def _resolve_checkpoint(
    config_path: Path,
    checkpoint_arg: str | None,
    results_csv: Path,
) -> Path:
    if checkpoint_arg is not None:
        checkpoint_path = Path(checkpoint_arg)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    config = load_experiment_config(config_path)
    if not results_csv.exists():
        raise FileNotFoundError(
            "Checkpoint was not provided and eval_results.csv was not found. "
            "Please pass --checkpoint explicitly."
        )

    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        if row.get("dataset") != config.dataset:
            continue
        checkpoint_dir = row.get("checkpoint_dir")
        if not checkpoint_dir:
            continue
        candidate = Path(checkpoint_dir) / "best.pt"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve a best checkpoint for dataset={config.dataset!r} from {results_csv}."
    )


def _balanced_indices(
    labels: list[int] | np.ndarray,
    selected_class_ids: list[int],
    samples_per_class: int,
    seed: int,
) -> list[int]:
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive.")

    label_array = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected: list[int] = []

    for class_id in selected_class_ids:
        class_indices = np.flatnonzero(label_array == class_id)
        if class_indices.size == 0:
            continue
        rng.shuffle(class_indices)
        chosen = class_indices[:samples_per_class]
        selected.extend(chosen.tolist())

    if not selected:
        raise ValueError("No samples were selected for visualization.")

    selected.sort()
    return selected


def _select_top_classes(
    labels: list[int] | np.ndarray,
    class_names: list[str] | None,
    top_k_classes: int,
) -> tuple[list[int], list[str]]:
    label_array = np.asarray(labels, dtype=np.int64)
    unique_ids, counts = np.unique(label_array, return_counts=True)
    ranked_pairs = sorted(
        zip(unique_ids.tolist(), counts.tolist()),
        key=lambda item: (-item[1], item[0]),
    )

    selected_ids = [class_id for class_id, _ in ranked_pairs[:top_k_classes]]
    if class_names is None:
        selected_names = [f"class_{class_id}" for class_id in selected_ids]
    else:
        selected_names = [class_names[class_id] for class_id in selected_ids]
    return selected_ids, selected_names


def _extract_labels_and_class_names(
    dataset,
    dataset_name: str,
) -> tuple[list[int], list[str] | None]:
    if isinstance(dataset, Subset):
        parent_labels, parent_class_names = _extract_labels_and_class_names(
            dataset.dataset,
            dataset_name=dataset_name,
        )
        subset_labels = [int(parent_labels[index]) for index in dataset.indices]
        return subset_labels, parent_class_names

    targets = getattr(dataset, "targets", None)
    if targets is not None:
        class_names = getattr(dataset, "classes", None)
        return [int(label) for label in targets], list(class_names) if class_names is not None else None

    labels = getattr(dataset, "labels", None)
    if labels is not None:
        class_names = getattr(dataset, "classes", None)
        return [int(label) for label in labels], list(class_names) if class_names is not None else None

    data = getattr(dataset, "data", None)
    if isinstance(data, list) and data and isinstance(data[0], tuple) and len(data[0]) >= 2:
        parsed_labels = [int(sample[1]) for sample in data]
        class_names = getattr(dataset, "classes", None)
        if class_names is not None:
            return parsed_labels, list(class_names)

        root = Path(getattr(dataset, "root", ""))
        classes_txt = root / "classes.txt"
        if classes_txt.is_file():
            parsed_names: list[str] = []
            for raw_line in classes_txt.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                _, class_name = line.split(maxsplit=1)
                parsed_names.append(class_name.replace("_", " "))
            return parsed_labels, parsed_names

        return parsed_labels, None

    if hasattr(dataset, "shards") and hasattr(dataset, "_get_h5_file"):
        shard_labels: list[int] = []
        for shard in dataset.shards:
            handle = dataset._get_h5_file(shard.path)
            shard_labels.extend(np.asarray(handle["label"][:], dtype=np.int64).tolist())

        class_names = getattr(dataset, "classes", None)
        if class_names is not None:
            class_names = list(class_names)
            label_name_map = load_label_name_map(dataset_name)
            if label_name_map:
                class_names = [label_name_map.get(name, name) for name in class_names]
        return shard_labels, class_names

    raise ValueError("The selected dataset does not expose class labels for balanced sampling.")


def _build_visualization_loader(
    config_path: Path,
    split: str,
    samples_per_class: int,
    top_k_classes: int,
    batch_size: int,
    workers: int,
    seed: int,
    device: torch.device,
) -> tuple[DataLoader, Subset, list[int], list[str]]:
    config = load_experiment_config(config_path)
    config.eval_batch_size = batch_size
    config.workers = workers

    bundles = build_experiment_dataloaders(config)
    source_dataset = bundles.val_loader.dataset if split == "val" else bundles.test_loader.dataset
    if not isinstance(source_dataset, SingleViewDataset):
        raise TypeError("Expected a SingleViewDataset for visualization.")

    base_dataset = source_dataset.dataset
    labels, class_names = _extract_labels_and_class_names(
        base_dataset,
        dataset_name=config.dataset,
    )

    selected_class_ids, selected_class_names = _select_top_classes(
        labels=labels,
        class_names=class_names,
        top_k_classes=top_k_classes,
    )

    selected_indices = _balanced_indices(
        labels=labels,
        selected_class_ids=selected_class_ids,
        samples_per_class=samples_per_class,
        seed=seed,
    )
    subset = Subset(base_dataset, selected_indices)
    visualization_dataset = SingleViewDataset(subset, source_dataset.transform)
    loader = DataLoader(
        visualization_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=safe_default_collate,
    )
    return loader, subset, selected_class_ids, selected_class_names


def _extract_features(
    model: HypersphereDPPModel,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    student_features: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            batch_labels = batch["label"].cpu().numpy()

            teacher = model.encode_teacher(images)
            student = model.encode_student(teacher_features=teacher)

            student_features.append(student.cpu().numpy())
            labels.append(batch_labels)

    return (
        np.concatenate(student_features, axis=0),
        np.concatenate(labels, axis=0),
    )


def _extract_raw_image_features(
    raw_subset: Subset,
    raw_image_size: int,
) -> np.ndarray:
    raw_transform = transforms.Compose(
        [
            transforms.Resize((raw_image_size, raw_image_size)),
            transforms.ToTensor(),
        ]
    )

    features: list[np.ndarray] = []
    for index in range(len(raw_subset)):
        image, _, _ = raw_subset[index]
        image_tensor = raw_transform(image)
        features.append(image_tensor.flatten().numpy())
    return np.stack(features, axis=0)


def _run_tsne(
    features: np.ndarray,
    perplexity: float,
    learning_rate: str | float,
    seed: int,
) -> np.ndarray:
    if features.shape[0] < 3:
        raise ValueError("At least 3 samples are required for t-SNE.")

    pca_dim = min(50, features.shape[0], features.shape[1])
    reduced = PCA(n_components=pca_dim, random_state=seed).fit_transform(features)
    effective_perplexity = min(float(perplexity), max(2.0, (features.shape[0] - 1) / 3.0))
    tsne_learning_rate = (
        learning_rate
        if learning_rate == "auto"
        else float(learning_rate)
    )
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate=tsne_learning_rate,
        init="pca",
        random_state=seed,
        max_iter=1500,
    )
    return tsne.fit_transform(reduced)


# ---------------------------------------------------------------------------
# NeurIPS-friendly colour palette – 10 distinct, print- & CB-safe colours
# Based on Tol's muted qualitative scheme with minor adjustments.
# ---------------------------------------------------------------------------

def _build_label_colormap(class_num: int) -> ListedColormap:
    neurips_colors = [
        "#332288",  # indigo
        "#88CCEE",  # cyan
        "#44AA99",  # teal
        "#117733",  # green
        "#999933",  # olive
        "#DDCC77",  # sand
        "#CC6677",  # rose
        "#882255",  # wine
        "#AA4499",  # purple
        "#DDDDDD",  # light grey
    ]
    return ListedColormap(neurips_colors[:class_num])


# ---------------------------------------------------------------------------
# Panel drawing – clean NeurIPS two-panel layout
# ---------------------------------------------------------------------------

def _scatter_panel(
    ax,
    embedding: np.ndarray,
    labels: np.ndarray,
    cmap,
    title: str,
    subtitle: str,
) -> None:
    ax.set_facecolor("#ffffff")
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap=cmap,
        s=18,
        alpha=0.88,
        linewidths=0.18,
        edgecolors="white",
        rasterized=True,
    )
    ax.set_title(title, fontsize=17, pad=10, fontweight="bold", y=1.055)
    ax.text(
        0.5,
        1.005,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.2,
        fontweight="semibold",
        color="#444444",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)
    ax.margins(0.03)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    args = parse_args()

    # ---- Apply NeurIPS style instead of IEEE ----
    apply_neurips_style()

    set_seed(args.seed)

    config_path = Path(args.config)
    config = load_experiment_config(config_path)
    checkpoint_path = _resolve_checkpoint(
        config_path=config_path,
        checkpoint_arg=args.checkpoint,
        results_csv=Path(args.results_csv),
    )
    default_output = (
        f"experiments/generated/figures/{config.dataset.lower()}_tsne_before_after.png"
    )
    output_path = Path(args.output or default_output)
    output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device or config.device)
    split = args.split or config.official_eval_split or "test"
    loader, raw_subset, selected_class_ids, selected_class_names = _build_visualization_loader(
        config_path=config_path,
        split=split,
        samples_per_class=args.samples_per_class,
        top_k_classes=args.top_k_classes,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        device=device,
    )

    model = HypersphereDPPModel(
        backbone_name=config.backbone_name,
        pretrained=config.pretrained,
        pretrained_checkpoint=config.pretrained_checkpoint,
        feature_dim=config.feature_dim,
        adapter_hidden_dim=config.adapter_hidden_dim,
        adapter_dropout=config.adapter_dropout,
        num_prototypes=config.class_num,
        device=device,
    ).to(device)
    load_checkpoint(checkpoint_path, model=model, map_location=device)

    raw_image_features = _extract_raw_image_features(
        raw_subset=raw_subset,
        raw_image_size=args.raw_image_size,
    )
    student_features, labels = _extract_features(model, loader, device)
    raw_tsne = _run_tsne(
        raw_image_features,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    student_tsne = _run_tsne(
        student_features,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    label_map = {class_id: index for index, class_id in enumerate(selected_class_ids)}
    display_labels = np.asarray([label_map[int(label)] for label in labels], dtype=np.int64)

    cmap = _build_label_colormap(len(selected_class_ids))
    fig, axes = plt.subplots(
        1, 2,
        figsize=(9.2, 5.8),
        constrained_layout=False,
    )
    fig.patch.set_facecolor("#ffffff")

    _scatter_panel(
        axes[0],
        raw_tsne,
        display_labels,
        cmap=cmap,
        title="Original Distribution",
        subtitle=f"Raw pixel features ({args.raw_image_size}x{args.raw_image_size})",
    )
    _scatter_panel(
        axes[1],
        student_tsne,
        display_labels,
        cmap=cmap,
        title="After Clustering",
        subtitle="Hypersphere-DPP student features",
    )

    # ---- Legend handles -----------------------------------------------------
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=cmap.colors[index],
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=5,
            label=class_name,
        )
        for index, class_name in enumerate(selected_class_names)
    ]

    meta_caption = (
        f"Top-{len(selected_class_ids)} classes on the {split} split, "
        f"{args.samples_per_class} images per class, {labels.shape[0]} samples total."
    )
    fig.text(
        0.5,
        0.085,
        meta_caption,
        ha="center",
        va="top",
        fontsize=8.2,
        fontweight="semibold",
        color="#666666",
    )

    ncol = min(5, len(selected_class_names))
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=ncol,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0.0,
        prop={"size": 7.6, "weight": "semibold"},
    )

    fig.subplots_adjust(
        left=0.04,
        right=0.98,
        top=0.83,
        bottom=0.22,
        wspace=0.10,
    )

    dataset_title = config.dataset.replace("_", " ")
    left_caption = f"{dataset_title}\noriginal image manifold"
    right_caption = "Hypersphere-DPP\ncompact and separated embedding"
    left_center_x = (axes[0].get_position().x0 + axes[0].get_position().x1) / 2.0
    right_center_x = (axes[1].get_position().x0 + axes[1].get_position().x1) / 2.0
    caption_y = 0.145
    for center_x, caption in (
        (left_center_x, left_caption),
        (right_center_x, right_caption),
    ):
        fig.text(
            center_x,
            caption_y,
            caption,
            ha="center",
            va="top",
            fontsize=10.6,
            fontweight="bold",
            color="#222222",
            linespacing=1.05,
        )

    fig.savefig(output_path, dpi=args.dpi)
    print(f"Saved figure to {output_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {split}")
    print(f"Samples: {labels.shape[0]}")
    print(f"Selected classes: {selected_class_names}")
    print(f"Device: {device}")


if __name__ == "__main__":
    main()
