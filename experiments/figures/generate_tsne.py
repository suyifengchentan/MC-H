#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import pickle
import re
import sys
import time
import zipfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap, to_rgba_array
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypersphere_dpp.checkpoint import load_checkpoint
from hypersphere_dpp.config import load_experiment_config
from hypersphere_dpp.data import (
    SingleViewDataset,
    build_experiment_dataloaders,
    safe_default_collate,
)
from hypersphere_dpp.labels import load_label_name_map
from hypersphere_dpp.metrics import clustering_accuracy
from hypersphere_dpp.model import HypersphereDPPModel
from hypersphere_dpp.utils import resolve_device, set_seed
from datasets.tiny_imagenet_dataset import load_tiny_imagenet_class_texts

DEFAULT_CONFIG = (
    "config/test/coarse_grained/cifar100_vitb16_hypersphere_dpp_config.yaml"
)
DEFAULT_OURS_ACC_TEX = "all_benchmarks_results_body.tex"
DATASET_CONFIG_MAP = {
    "cifar10": "config/test/coarse_grained/cifar10_vitb16_hypersphere_dpp_config.yaml",
    "cifar100": "config/test/coarse_grained/cifar100_vitb16_hypersphere_dpp_config.yaml",
    "cub": "config/test/fine_grained/cub_vitb16_hypersphere_dpp_config.yaml",
    "cub_200_2011": "config/test/fine_grained/cub_vitb16_hypersphere_dpp_config.yaml",
    "aircraft": "config/test/fine_grained/fgvc_aircraft_vitb16_hypersphere_dpp_config.yaml",
    "fgvc_aircraft": "config/test/fine_grained/fgvc_aircraft_vitb16_hypersphere_dpp_config.yaml",
    "image10": "config/test/coarse_grained/image10_vitb16_hypersphere_dpp_config.yaml",
    "imagenet1k": "config/test/coarse_grained/imagenet1k_vitb16_hypersphere_dpp_config.yaml",
    "imagenet-1k": "config/test/coarse_grained/imagenet1k_vitb16_hypersphere_dpp_config.yaml",
    "image1k": "config/test/coarse_grained/imagenet1k_vitb16_hypersphere_dpp_config.yaml",
    "image50": "config/test/medium_grained/image50_vitb16_hypersphere_dpp_config.yaml",
    "image100": "config/test/medium_grained/image100_vitb16_hypersphere_dpp_config.yaml",
    "image200": "config/test/medium_grained/image200_vitb16_hypersphere_dpp_config.yaml",
    "tinyimagenet200": "config/test/medium_grained/tiny_imagenet200_vitb16_hypersphere_dpp_config.yaml",
    "tiny-imagenet-200": "config/test/medium_grained/tiny_imagenet200_vitb16_hypersphere_dpp_config.yaml",
    "tiny_imagenet200": "config/test/medium_grained/tiny_imagenet200_vitb16_hypersphere_dpp_config.yaml",
    "tiny200": "config/test/medium_grained/tiny_imagenet200_vitb16_hypersphere_dpp_config.yaml",
    "flowers": "config/test/fine_grained/oxford_flowers_vitb16_hypersphere_dpp_config.yaml",
    "oxford_flowers": "config/test/fine_grained/oxford_flowers_vitb16_hypersphere_dpp_config.yaml",
    "pet": "config/test/fine_grained/oxford_pet_vitb16_hypersphere_dpp_config.yaml",
    "oxford_pet": "config/test/fine_grained/oxford_pet_vitb16_hypersphere_dpp_config.yaml",
    "stl10": "config/test/coarse_grained/stl10_vitb16_hypersphere_dpp_config.yaml",
    "dogs": "config/test/fine_grained/stanford_dogs_vitb16_hypersphere_dpp_config.yaml",
    "stanford_dogs": "config/test/fine_grained/stanford_dogs_vitb16_hypersphere_dpp_config.yaml",
}


@dataclass
class TSNEResult:
    dataset_name: str
    checkpoint_path: Path
    split: str
    sample_count: int
    teacher_tsne: np.ndarray
    student_tsne: np.ndarray
    true_labels: np.ndarray
    teacher_baseline_method: str
    teacher_baseline_labels: np.ndarray
    student_prototype_labels: np.ndarray
    teacher_baseline_aligned_labels: np.ndarray
    student_prototype_aligned_labels: np.ndarray
    teacher_baseline_acc: float
    teacher_baseline_nmi: float
    teacher_baseline_ari: float
    student_prototype_acc: float
    student_prototype_nmi: float
    student_prototype_ari: float


def apply_neurips_style() -> None:
    """Apply a publication-friendly style with Arial sans-serif typography."""
    neurips_rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "font.weight": "semibold",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "axes.linewidth": 0.0,
        "axes.edgecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "axes.grid": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.facecolor": "#ffffff",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "scatter.edgecolors": "face",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    matplotlib.rcParams.update(neurips_rc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate before/after t-SNE plots for one or more datasets."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--results-csv", default="eval_results.csv")
    parser.add_argument("--split", choices=["val", "test"], default=None)
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--top-k-classes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--perplexity", type=float, default=35.0)
    parser.add_argument("--learning-rate", default="auto")
    parser.add_argument(
        "--tsne-backend", choices=["auto", "cpu", "gpu"], default="auto"
    )
    parser.add_argument("--tsne-jobs", type=int, default=-1)
    parser.add_argument(
        "--pca-threads",
        type=int,
        default=1,
        help=(
            "Limit BLAS/OpenMP threads used during PCA. "
            "Use 1 to avoid OpenBLAS/OpenMP stalls on some systems; "
            "set <= 0 to keep library defaults."
        ),
    )
    parser.add_argument(
        "--tsne-verbose",
        action="store_true",
        help="Enable verbose progress logs from the selected t-SNE backend.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--teacher-baseline",
        choices=["kmeans"],
        default="kmeans",
        help="Teacher-side baseline for the top row. Uses Raw CLIP features clustered by KMeans.",
    )
    parser.add_argument("--raw-image-size", type=int, default=40)
    parser.add_argument(
        "--ours-acc-tex",
        default=DEFAULT_OURS_ACC_TEX,
        help="Path to the LaTeX benchmark table used to source displayed Ours ACC values.",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _log_progress(message: str) -> None:
    print(message, flush=True)


def _resolve_training_config_path(config_path: Path) -> Path:
    resolved = config_path.resolve()
    parts = resolved.parts
    try:
        config_index = parts.index("config")
    except ValueError:
        return resolved

    if config_index + 1 >= len(parts):
        return resolved

    mode = parts[config_index + 1]
    if mode == "train":
        return resolved
    if mode != "test":
        return resolved

    train_parts = list(parts)
    train_parts[config_index + 1] = "train"
    candidate = Path(*train_parts)
    if candidate.is_file():
        return candidate
    return resolved


def _resolve_model_path_prefix(config_path: Path) -> Path:
    train_config_path = _resolve_training_config_path(config_path)
    config = load_experiment_config(train_config_path)
    model_path = Path(config.model_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    return model_path


def _resolve_checkpoint(config_path: Path, checkpoint_arg: str | None) -> Path:
    if checkpoint_arg is not None:
        checkpoint_path = Path(checkpoint_arg)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model_keys = _peek_checkpoint_model_keys(checkpoint_path)
        if "prototypes" not in model_keys:
            raise ValueError(
                "The provided checkpoint is incompatible with t-SNE visualization: "
                f"{checkpoint_path} has model keys {model_keys}, but a prototype-based "
                "checkpoint with model keys ['adapter', 'prototypes'] is required."
            )
        return checkpoint_path

    model_path_prefix = _resolve_model_path_prefix(config_path)
    candidate_dirs: list[Path] = []

    if model_path_prefix.is_dir():
        candidate_dirs.append(model_path_prefix)

    parent = model_path_prefix.parent
    if parent.is_dir():
        matched_dirs = sorted(
            path for path in parent.glob(f"{model_path_prefix.name}*") if path.is_dir()
        )
        for path in matched_dirs:
            if path not in candidate_dirs:
                candidate_dirs.append(path)

    for directory in candidate_dirs:
        best_checkpoint = directory / "best.pt"
        if best_checkpoint.is_file() and _checkpoint_supports_prototype_labels(
            best_checkpoint
        ):
            return best_checkpoint
        last_checkpoint = directory / "last.pt"
        if last_checkpoint.is_file() and _checkpoint_supports_prototype_labels(
            last_checkpoint
        ):
            return last_checkpoint

    incompatible_candidates: list[str] = []
    for directory in candidate_dirs:
        for checkpoint_name in ("best.pt", "last.pt"):
            checkpoint_path = directory / checkpoint_name
            if checkpoint_path.is_file():
                model_keys = _peek_checkpoint_model_keys(checkpoint_path)
                incompatible_candidates.append(
                    f"{checkpoint_path} (model keys: {model_keys})"
                )

    raise FileNotFoundError(
        "Could not resolve a prototype-compatible checkpoint from model_path prefix "
        f"{model_path_prefix} derived from config {config_path}. "
        f"Checked candidates: {incompatible_candidates or 'none'}"
    )


def _resolve_config_path_list(args: argparse.Namespace) -> list[Path]:
    if args.datasets:
        config_paths: list[Path] = []
        for dataset_name in args.datasets:
            dataset_key = dataset_name.strip().lower()
            config_rel = DATASET_CONFIG_MAP.get(dataset_key)
            if config_rel is None:
                supported = ", ".join(sorted(DATASET_CONFIG_MAP))
                raise ValueError(
                    f"Unsupported dataset {dataset_name!r}. Supported aliases: {supported}."
                )
            config_paths.append(Path(config_rel))
        return config_paths

    if args.configs:
        return [Path(config_path) for config_path in args.configs]

    return [Path(args.config)]


def _peek_checkpoint_model_keys(checkpoint_path: Path) -> list[str]:
    class _DummyOrderedDict(dict):
        pass

    class _PeekUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "collections" and name == "OrderedDict":
                return _DummyOrderedDict
            if module in ("torch._utils", "torch"):
                def _stub(*args, **kwargs):
                    return ("TORCH_OBJ", module, name, args)

                return _stub
            return super().find_class(module, name)

        def persistent_load(self, pid):
            return ("PERSISTENT", pid)

    if not zipfile.is_zipfile(checkpoint_path):
        return []

    with zipfile.ZipFile(checkpoint_path) as archive:
        data_members = [name for name in archive.namelist() if name.endswith("data.pkl")]
        if not data_members:
            return []
        payload = archive.read(data_members[0])

    try:
        checkpoint = _PeekUnpickler(io.BytesIO(payload)).load()
    except Exception:
        return []

    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        return []
    return list(model_state.keys())


def _checkpoint_supports_prototype_labels(checkpoint_path: Path) -> bool:
    return "prototypes" in _peek_checkpoint_model_keys(checkpoint_path)


def _balanced_indices(
    labels: list[int] | np.ndarray,
    selected_class_ids: list[int],
    samples_per_class: int,
    seed: int,
) -> list[int]:
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
    labels: list[int] | np.ndarray, class_names: list[str] | None, top_k_classes: int
) -> tuple[list[int], list[str]]:
    label_array = np.asarray(labels, dtype=np.int64)
    unique_ids, counts = np.unique(label_array, return_counts=True)
    ranked_pairs = sorted(
        zip(unique_ids.tolist(), counts.tolist()), key=lambda item: (-item[1], item[0])
    )

    selected_ids = [class_id for class_id, _ in ranked_pairs[:top_k_classes]]
    if class_names is None:
        selected_names = [f"class_{class_id}" for class_id in selected_ids]
    else:
        selected_names = [class_names[class_id] for class_id in selected_ids]
    return selected_ids, selected_names


def _extract_labels_and_class_names(
    dataset, dataset_name: str
) -> tuple[list[int], list[str] | None]:
    if isinstance(dataset, Subset):
        parent_labels, parent_class_names = _extract_labels_and_class_names(
            dataset.dataset, dataset_name=dataset_name
        )
        subset_labels = [int(parent_labels[index]) for index in dataset.indices]
        return subset_labels, parent_class_names

    targets = getattr(dataset, "targets", None)
    if targets is not None:
        class_names = getattr(dataset, "classes", None)
        return [int(label) for label in targets], list(
            class_names
        ) if class_names is not None else None

    labels = getattr(dataset, "labels", None)
    if labels is not None:
        class_names = getattr(dataset, "classes", None)
        return [int(label) for label in labels], list(
            class_names
        ) if class_names is not None else None

    tuple_samples = getattr(dataset, "data", None)
    if tuple_samples is None:
        tuple_samples = getattr(dataset, "samples", None)
    if (
        isinstance(tuple_samples, list)
        and tuple_samples
        and isinstance(tuple_samples[0], tuple)
        and len(tuple_samples[0]) >= 2
    ):
        parsed_labels = [int(sample[1]) for sample in tuple_samples]
        class_names = getattr(dataset, "classes", None)
        if class_names is not None:
            resolved_class_names = list(class_names)
            label_name_map = load_label_name_map(dataset_name)
            if label_name_map:
                resolved_class_names = [
                    label_name_map.get(name, name) for name in resolved_class_names
                ]
            elif dataset_name == "TinyImageNet200":
                try:
                    tiny_imagenet_class_names = load_tiny_imagenet_class_texts(
                        str(getattr(dataset, "root", ""))
                    )
                except Exception:
                    tiny_imagenet_class_names = []
                if len(tiny_imagenet_class_names) == len(resolved_class_names):
                    resolved_class_names = tiny_imagenet_class_names
            return parsed_labels, resolved_class_names

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

    raise ValueError(
        "The selected dataset does not expose class labels for balanced sampling."
    )


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
    source_dataset = (
        bundles.val_loader.dataset if split == "val" else bundles.test_loader.dataset
    )
    if not isinstance(source_dataset, SingleViewDataset):
        raise TypeError("Expected a SingleViewDataset for visualization.")

    base_dataset = source_dataset.dataset
    labels, class_names = _extract_labels_and_class_names(
        base_dataset, dataset_name=config.dataset
    )

    selected_class_ids, selected_class_names = _select_top_classes(
        labels=labels, class_names=class_names, top_k_classes=top_k_classes
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
    model: HypersphereDPPModel, dataloader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    teacher_features: list[np.ndarray] = []
    student_features: list[np.ndarray] = []
    student_prototype_labels: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            batch_labels = batch["label"].cpu().numpy()
            teacher = model.encode_teacher(images)
            student = model.encode_student(teacher_features=teacher)
            student_logits = model.prototype_logits(student)
            teacher_features.append(teacher.cpu().numpy())
            student_features.append(student.cpu().numpy())
            student_prototype_labels.append(student_logits.argmax(dim=1).cpu().numpy())
            labels.append(batch_labels)

    return (
        np.concatenate(teacher_features, axis=0),
        np.concatenate(student_features, axis=0),
        np.concatenate(student_prototype_labels, axis=0),
        np.concatenate(labels, axis=0),
    )


def _predict_kmeans_labels(
    features: np.ndarray,
    n_clusters: int,
    seed: int,
    stage_label: str,
) -> np.ndarray:
    _log_progress(
        f"[{stage_label}] kmeans assignment start: samples={features.shape[0]}, "
        f"clusters={n_clusters}"
    )
    assign_start = time.perf_counter()
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit_predict(
        features
    )
    labels = labels.astype(np.int64, copy=False)
    _log_progress(
        f"[{stage_label}] kmeans assignment done in "
        f"{time.perf_counter() - assign_start:.2f}s"
    )
    return labels


def _resolve_tsne_backend(requested_backend: str) -> str:
    if requested_backend == "cpu":
        return "cpu"

    try:
        import cuml.manifold  # noqa: F401
    except ImportError as exc:
        if requested_backend == "gpu":
            raise RuntimeError(
                "GPU t-SNE backend was requested, but cuML is not installed in this environment."
            ) from exc
        return "cpu"

    return "gpu"


def _run_tsne_cpu(
    reduced: np.ndarray,
    perplexity: float,
    learning_rate: str | float,
    seed: int,
    tsne_jobs: int,
    verbose: bool,
) -> np.ndarray:
    effective_perplexity = min(
        float(perplexity), max(2.0, (reduced.shape[0] - 1) / 3.0)
    )
    tsne_learning_rate = (
        learning_rate if learning_rate == "auto" else float(learning_rate)
    )
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate=tsne_learning_rate,
        init="pca",
        random_state=seed,
        max_iter=1500,
        method="barnes_hut",
        angle=0.5,
        n_jobs=tsne_jobs,
        verbose=1 if verbose else 0,
    )
    return tsne.fit_transform(reduced)


def _run_tsne_gpu(
    reduced: np.ndarray,
    perplexity: float,
    learning_rate: str | float,
    seed: int,
    verbose: bool,
) -> np.ndarray:
    from cuml.manifold import TSNE as CuMLTSNE

    effective_perplexity = min(
        float(perplexity), max(2.0, (reduced.shape[0] - 1) / 3.0)
    )
    tsne_learning_rate = 200.0 if learning_rate == "auto" else float(learning_rate)
    n_neighbors = max(90, int(np.ceil(effective_perplexity * 3.0)))
    tsne = CuMLTSNE(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate=tsne_learning_rate,
        init="pca",
        random_state=seed,
        max_iter=1500,
        method="fft",
        n_neighbors=n_neighbors,
        verbose=verbose,
        output_type="numpy",
    )
    embedding = tsne.fit_transform(reduced.astype(np.float32, copy=False))
    return np.asarray(embedding)


def _run_tsne(
    features: np.ndarray,
    perplexity: float,
    learning_rate: str | float,
    seed: int,
    backend: str,
    tsne_jobs: int,
    pca_threads: int,
    verbose: bool,
    stage_label: str,
) -> np.ndarray:
    _log_progress(
        f"[{stage_label}] PCA start: samples={features.shape[0]}, dim={features.shape[1]}"
    )
    pca_start = time.perf_counter()
    pca_dim = min(50, features.shape[0], features.shape[1])
    pca_solver = (
        "randomized" if pca_dim < min(features.shape[0], features.shape[1]) else "auto"
    )
    threadpool_context = nullcontext()
    if pca_threads > 0:
        try:
            from threadpoolctl import threadpool_limits

            threadpool_context = threadpool_limits(limits=pca_threads)
        except ImportError:
            _log_progress(
                f"[{stage_label}] threadpoolctl is unavailable; using default PCA threading"
            )

    with threadpool_context:
        reduced = PCA(
            n_components=pca_dim, random_state=seed, svd_solver=pca_solver
        ).fit_transform(features)
    reduced = reduced.astype(np.float32, copy=False)
    _log_progress(
        f"[{stage_label}] PCA done in {time.perf_counter() - pca_start:.2f}s -> "
        f"reduced_dim={reduced.shape[1]}, pca_threads="
        f"{pca_threads if pca_threads > 0 else 'default'}"
    )

    effective_perplexity = min(
        float(perplexity), max(2.0, (reduced.shape[0] - 1) / 3.0)
    )
    tsne_learning_rate = (
        200.0
        if backend == "gpu" and learning_rate == "auto"
        else learning_rate
    )
    _log_progress(
        f"[{stage_label}] t-SNE start: backend={backend}, samples={reduced.shape[0]}, "
        f"perplexity={effective_perplexity:.2f}, learning_rate={tsne_learning_rate}"
    )
    tsne_start = time.perf_counter()

    if backend == "gpu":
        embedding = _run_tsne_gpu(reduced, perplexity, learning_rate, seed, verbose)
    else:
        embedding = _run_tsne_cpu(
            reduced, perplexity, learning_rate, seed, tsne_jobs, verbose
        )
    _log_progress(
        f"[{stage_label}] t-SNE done in {time.perf_counter() - tsne_start:.2f}s"
    )
    return embedding


def _remap_labels(labels: np.ndarray) -> np.ndarray:
    unique_labels = np.unique(labels)
    label_map = {int(label): index for index, label in enumerate(unique_labels.tolist())}
    return np.asarray([label_map[int(label)] for label in labels], dtype=np.int64)


def _align_predictions_to_truth(
    true_labels: np.ndarray, predicted_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    true_unique = np.unique(true_labels)
    pred_unique = np.unique(predicted_labels)
    contingency = np.zeros((pred_unique.size, true_unique.size), dtype=np.int64)

    pred_index = {int(label): index for index, label in enumerate(pred_unique.tolist())}
    true_index = {int(label): index for index, label in enumerate(true_unique.tolist())}

    for true_label, pred_label in zip(true_labels, predicted_labels):
        contingency[pred_index[int(pred_label)], true_index[int(true_label)]] += 1

    if contingency.size == 0:
        return np.empty_like(true_labels), np.zeros_like(true_labels, dtype=bool)

    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
    assignment = {
        int(pred_unique[row]): int(true_unique[col])
        for row, col in zip(row_ind.tolist(), col_ind.tolist())
    }

    unmatched_label = int(true_unique.size)
    aligned_labels = np.asarray(
        [assignment.get(int(pred_label), unmatched_label) for pred_label in predicted_labels],
        dtype=np.int64,
    )
    mismatch_mask = aligned_labels != true_labels
    return aligned_labels, mismatch_mask


def _compute_clustering_metrics(
    true_labels: np.ndarray, predicted_labels: np.ndarray
) -> tuple[float, float, float]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    acc = clustering_accuracy(true_labels, predicted_labels)
    nmi = float(normalized_mutual_info_score(true_labels, predicted_labels))
    ari = float(adjusted_rand_score(true_labels, predicted_labels))
    return acc, nmi, ari


def _build_label_colormap(class_num: int) -> ListedColormap:
    neurips_colors = [
        "#332288",
        "#88CCEE",
        "#44AA99",
        "#117733",
        "#999933",
        "#DDCC77",
        "#CC6677",
        "#882255",
        "#AA4499",
        "#DDDDDD",
    ]
    if class_num <= len(neurips_colors):
        return ListedColormap(neurips_colors[:class_num])

    tab20 = plt.get_cmap("tab20", class_num)
    return ListedColormap(tab20(np.arange(class_num)))


def _build_consistent_comparison_colormaps(
    class_num: int,
) -> tuple[ListedColormap, ListedColormap]:
    base_cmap = _build_label_colormap(class_num)
    base_colors = to_rgba_array(base_cmap.colors)
    unmatched_color = np.array([[0.72, 0.72, 0.72, 1.0]], dtype=np.float32)
    cluster_colors = np.concatenate([base_colors, unmatched_color], axis=0)
    return ListedColormap(base_colors), ListedColormap(cluster_colors)


def _scatter_panel(ax, embedding: np.ndarray, labels: np.ndarray, cmap) -> None:
    """仅保留纯粹的绘图逻辑，标题与坐标轴由外部网格统一设置。"""
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
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)
    ax.margins(0.03)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _scatter_cluster_panel(
    ax,
    embedding: np.ndarray,
    aligned_labels: np.ndarray,
    cmap,
) -> None:
    ax.set_facecolor("#ffffff")
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=aligned_labels,
        cmap=cmap,
        s=18,
        alpha=0.78,
        linewidths=0.12,
        edgecolors="white",
        rasterized=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)
    ax.margins(0.03)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _build_output_paths(output_arg: str | None, results: list[TSNEResult]) -> tuple[Path, Path]:
    if len(results) == 1:
        default_base = (
            f"experiments/generated/figures/{results[0].dataset_name.lower()}_tsne"
        )
    else:
        default_base = "experiments/generated/figures/multi_dataset_tsne"

    base_path = Path(output_arg or default_base)
    if base_path.suffix:
        base_path = base_path.with_suffix("")

    representation_path = base_path.parent / f"{base_path.name}_representation.png"
    clustering_path = base_path.parent / f"{base_path.name}_clustering.png"
    return representation_path, clustering_path


def _normalize_dataset_key(dataset_name: str) -> str:
    dataset_key = dataset_name.strip().lower().replace("_", "-")
    alias_map = {
        "cifar10": "cifar10",
        "cifar-10": "cifar10",
        "cifar100": "cifar100",
        "cifar-100": "cifar100",
        "stl10": "stl10",
        "stl-10": "stl10",
        "image10": "imagenet10",
        "imagenet10": "imagenet10",
        "imagenet-10": "imagenet10",
        "image50": "imagenet50",
        "imagenet50": "imagenet50",
        "imagenet-50": "imagenet50",
        "image100": "imagenet100",
        "imagenet100": "imagenet100",
        "imagenet-100": "imagenet100",
        "image200": "imagenet200",
        "imagenet200": "imagenet200",
        "imagenet-200": "imagenet200",
        "image1k": "imagenet1k",
        "imagenet1k": "imagenet1k",
        "imagenet-1k": "imagenet1k",
        "stanford-dogs": "stanforddogs",
        "stanford dogs": "stanforddogs",
        "stanford_dogs": "stanforddogs",
        "dogs": "stanforddogs",
        "oxford-flowers": "oxfordflowers",
        "oxford flowers": "oxfordflowers",
        "oxford_flowers": "oxfordflowers",
        "flowers": "oxfordflowers",
        "oxford-iiit-pet": "oxfordiiitpet",
        "oxford iiit pet": "oxfordiiitpet",
        "oxford-iiit pet": "oxfordiiitpet",
        "oxford_pet": "oxfordiiitpet",
        "pet": "oxfordiiitpet",
        "cub-200-2011": "cub2002011",
        "cub_200_2011": "cub2002011",
        "cub": "cub2002011",
    }
    return alias_map.get(dataset_key, re.sub(r"[^a-z0-9]+", "", dataset_key))


def _strip_latex_markup(text: str) -> str:
    cleaned = text.strip()
    while True:
        updated = re.sub(
            r"\\(?:textbf|underline|multirow)\{[^{}]*\}\{[^{}]*\}(?:\{([^{}]*)\})?",
            lambda match: match.group(1) or "",
            cleaned,
        )
        updated = re.sub(
            r"\\(?:textbf|underline)\{([^{}]*)\}",
            r"\1",
            updated,
        )
        if updated == cleaned:
            break
        cleaned = updated
    cleaned = re.sub(r"\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^{}]*\})*", " ", cleaned)
    cleaned = cleaned.replace("{", " ").replace("}", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned


def _extract_acc_value(acc_cell: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)", _strip_latex_markup(acc_cell))
    if match is None:
        return None
    return float(match.group(1)) / 100.0


def _load_ours_acc_lookup(tex_path: Path) -> dict[str, float]:
    if not tex_path.is_absolute():
        tex_path = PROJECT_ROOT / tex_path
    if not tex_path.is_file():
        return {}

    lookup: dict[str, float] = {}
    current_dataset: str | None = None

    for raw_line in tex_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        dataset_match = re.search(
            r"^&\s*\\multirow\{[^{}]*\}\{[^{}]*\}\{([^{}]+)\}",
            line,
        )
        if dataset_match is not None:
            current_dataset = _strip_latex_markup(dataset_match.group(1))

        if "&" not in line or not line.endswith(r"\\"):
            continue
        if "Method" in line and "ACC" in line:
            continue

        columns = [column.strip() for column in line[:-2].split("&")]
        if len(columns) < 5:
            continue

        if len(columns) >= 7:
            dataset_cell = _strip_latex_markup(columns[1])
            if dataset_cell:
                current_dataset = dataset_cell
            method_cell = _strip_latex_markup(columns[2])
            acc_cell = columns[4]
        else:
            method_cell = _strip_latex_markup(columns[1])
            acc_cell = columns[3]

        if current_dataset is None or method_cell.lower() != "ours":
            continue

        acc_value = _extract_acc_value(acc_cell)
        if acc_value is None:
            continue
        lookup[_normalize_dataset_key(current_dataset)] = acc_value

    return lookup


def _display_student_acc(
    result: TSNEResult,
    ours_acc_lookup: dict[str, float] | None = None,
) -> float:
    if ours_acc_lookup:
        mapped_acc = ours_acc_lookup.get(_normalize_dataset_key(result.dataset_name))
        if mapped_acc is not None:
            return mapped_acc
    return result.student_prototype_acc


def _format_dataset_title(dataset_name: str) -> str:
    dataset_key = dataset_name.strip().lower()
    title_map = {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "stl10": "STL-10",
        "image10": "ImageNet-10",
        "image50": "ImageNet-50",
        "image100": "ImageNet-100",
        "image200": "ImageNet-200",
        "imagenet1k": "ImageNet-1K",
        "imagenet-1k": "ImageNet-1K",
        "image1k": "ImageNet-1K",
        "stanford_dogs": "Stanford Dogs",
        "dogs": "Stanford Dogs",
        "oxford_flowers": "Oxford Flowers",
        "flowers": "Oxford Flowers",
    }
    return title_map.get(dataset_key, dataset_name.replace("_", " "))


def _render_representation_figure(results: list[TSNEResult], output_path: Path, dpi: int) -> None:
    n_cols = len(results)
    fig_width = max(5.0, 3.5 * n_cols)
    fig_height = 6.8
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(fig_width, fig_height),
        constrained_layout=False,
        squeeze=False,
    )
    fig.patch.set_facecolor("#ffffff")

    for col_index, result in enumerate(results):
        cmap = _build_label_colormap(int(result.true_labels.max()) + 1)
        ax_teacher = axes[0, col_index]
        ax_student = axes[1, col_index]

        _scatter_panel(ax_teacher, result.teacher_tsne, result.true_labels, cmap=cmap)
        _scatter_panel(ax_student, result.student_tsne, result.true_labels, cmap=cmap)

        dataset_title = _format_dataset_title(result.dataset_name)
        ax_teacher.set_title(dataset_title, fontsize=15, pad=15, fontweight="bold")
        if col_index == 0:
            ax_teacher.set_ylabel("Raw CLIP", fontsize=13, fontweight="normal", labelpad=20)
            ax_student.set_ylabel(
                "Ours",
                fontsize=13,
                fontweight="normal",
                labelpad=20,
            )

    fig.subplots_adjust(
        left=0.08 if n_cols > 1 else 0.18,
        right=0.98,
        top=0.92,
        bottom=0.06,
        hspace=0.10,
        wspace=0.05,
    )

    if n_cols > 1:
        separator_y0 = min(
            axes[1, col_index].get_position().y0 for col_index in range(n_cols)
        )
        separator_y1 = max(
            axes[0, col_index].get_position().y1 for col_index in range(n_cols)
        )
        for col_index in range(n_cols - 1):
            left_box = axes[0, col_index].get_position()
            right_box = axes[0, col_index + 1].get_position()
            separator_x = (left_box.x1 + right_box.x0) / 2.0
            fig.add_artist(
                Line2D(
                    [separator_x, separator_x],
                    [separator_y0, separator_y1],
                    transform=fig.transFigure,
                    color="#8f8f8f",
                    linewidth=1.0,
                    linestyle=(0, (4, 4)),
                    alpha=0.9,
                )
            )

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _render_clustering_figure(
    results: list[TSNEResult],
    output_path: Path,
    dpi: int,
    ours_acc_lookup: dict[str, float] | None = None,
) -> None:
    n_cols = len(results)
    fig_width = max(5.0, 3.5 * n_cols)
    fig_height = 6.3
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(fig_width, fig_height),
        constrained_layout=False,
        squeeze=False,
    )
    fig.patch.set_facecolor("#ffffff")

    for col_index, result in enumerate(results):
        gt_cmap, cluster_cmap = _build_consistent_comparison_colormaps(
            int(result.true_labels.max()) + 1
        )

        ax_gt = axes[0, col_index]
        ax_ours = axes[1, col_index]

        _scatter_panel(ax_gt, result.student_tsne, result.true_labels, cmap=gt_cmap)
        _scatter_cluster_panel(
            ax_ours,
            result.student_tsne,
            result.student_prototype_aligned_labels,
            cmap=cluster_cmap,
        )

        dataset_title = _format_dataset_title(result.dataset_name)

        ax_gt.set_title(dataset_title, fontsize=15, pad=15, fontweight="bold")
        ax_ours.text(
            0.02,
            0.98,
            f"ACC {_display_student_acc(result, ours_acc_lookup=ours_acc_lookup):.1%}",
            transform=ax_ours.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            fontweight="normal",
            color="#222222",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.80,
                "boxstyle": "round,pad=0.14",
            },
        )

        if col_index == 0:
            ax_gt.set_ylabel(
                "GT",
                fontsize=13,
                fontweight="normal",
                labelpad=20,
            )
            ax_ours.set_ylabel(
                "Ours",
                fontsize=13,
                fontweight="normal",
                labelpad=20,
            )

    fig.subplots_adjust(
        left=0.08 if n_cols > 1 else 0.18,
        right=0.98,
        top=0.93,
        bottom=0.06,
        hspace=0.08,
        wspace=0.05,
    )

    if n_cols > 1:
        separator_y0 = min(
            axes[1, col_index].get_position().y0 for col_index in range(n_cols)
        )
        separator_y1 = max(
            axes[0, col_index].get_position().y1 for col_index in range(n_cols)
        )
        for col_index in range(n_cols - 1):
            left_box = axes[0, col_index].get_position()
            right_box = axes[0, col_index + 1].get_position()
            separator_x = (left_box.x1 + right_box.x0) / 2.0
            fig.add_artist(
                Line2D(
                    [separator_x, separator_x],
                    [separator_y0, separator_y1],
                    transform=fig.transFigure,
                    color="#8f8f8f",
                    linewidth=1.0,
                    linestyle=(0, (4, 4)),
                    alpha=0.9,
                )
            )

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _generate_tsne_result(
    config_path: Path, args: argparse.Namespace, checkpoint_arg: str | None = None
) -> TSNEResult:
    config = load_experiment_config(config_path)
    checkpoint_path = _resolve_checkpoint(config_path, checkpoint_arg)
    tsne_backend = _resolve_tsne_backend(args.tsne_backend)

    device = resolve_device(args.device or config.device)
    split = args.split or config.official_eval_split or "test"
    loader, _, selected_class_ids, _ = _build_visualization_loader(
        config_path,
        split,
        args.samples_per_class,
        args.top_k_classes,
        args.batch_size,
        args.workers,
        args.seed,
        device,
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

    (
        teacher_features,
        student_features,
        student_prototype_labels,
        labels,
    ) = _extract_features(
        model, loader, device
    )
    teacher_baseline_method = args.teacher_baseline
    _log_progress(
        f"[{config.dataset}] t-SNE backend={tsne_backend}, tsne_jobs={args.tsne_jobs}, "
        f"samples={student_features.shape[0]}, teacher_baseline={teacher_baseline_method}"
    )
    teacher_tsne = _run_tsne(
        teacher_features,
        args.perplexity,
        args.learning_rate,
        args.seed,
        backend=tsne_backend,
        tsne_jobs=args.tsne_jobs,
        pca_threads=args.pca_threads,
        verbose=args.tsne_verbose,
        stage_label=f"{config.dataset}/teacher",
    )
    student_tsne = _run_tsne(
        student_features,
        args.perplexity,
        args.learning_rate,
        args.seed,
        backend=tsne_backend,
        tsne_jobs=args.tsne_jobs,
        pca_threads=args.pca_threads,
        verbose=args.tsne_verbose,
        stage_label=f"{config.dataset}/student",
    )
    label_map = {class_id: index for index, class_id in enumerate(selected_class_ids)}
    true_labels = np.asarray([label_map[int(label)] for label in labels], dtype=np.int64)
    teacher_baseline_labels = _predict_kmeans_labels(
        teacher_features,
        n_clusters=len(selected_class_ids),
        seed=args.seed,
        stage_label=f"{config.dataset}/teacher-kmeans",
    )
    student_prototype_labels = _remap_labels(student_prototype_labels)
    teacher_baseline_aligned_labels, _ = _align_predictions_to_truth(
        true_labels, teacher_baseline_labels
    )
    student_prototype_aligned_labels, _ = _align_predictions_to_truth(
        true_labels, student_prototype_labels
    )
    teacher_baseline_acc, teacher_baseline_nmi, teacher_baseline_ari = (
        _compute_clustering_metrics(true_labels, teacher_baseline_labels)
    )
    student_prototype_acc, student_prototype_nmi, student_prototype_ari = (
        _compute_clustering_metrics(true_labels, student_prototype_labels)
    )

    return TSNEResult(
        dataset_name=config.dataset,
        checkpoint_path=checkpoint_path,
        split=split,
        sample_count=int(student_features.shape[0]),
        teacher_tsne=teacher_tsne,
        student_tsne=student_tsne,
        true_labels=true_labels,
        teacher_baseline_method=teacher_baseline_method,
        teacher_baseline_labels=teacher_baseline_labels,
        student_prototype_labels=student_prototype_labels,
        teacher_baseline_aligned_labels=teacher_baseline_aligned_labels,
        student_prototype_aligned_labels=student_prototype_aligned_labels,
        teacher_baseline_acc=teacher_baseline_acc,
        teacher_baseline_nmi=teacher_baseline_nmi,
        teacher_baseline_ari=teacher_baseline_ari,
        student_prototype_acc=student_prototype_acc,
        student_prototype_nmi=student_prototype_nmi,
        student_prototype_ari=student_prototype_ari,
    )


def main() -> None:
    args = parse_args()
    apply_neurips_style()
    set_seed(args.seed)
    ours_acc_lookup = _load_ours_acc_lookup(Path(args.ours_acc_tex))

    config_paths = _resolve_config_path_list(args)
    if args.checkpoint is not None and len(config_paths) > 1:
        raise ValueError("--checkpoint only supports single-config mode.")

    checkpoint_arg = args.checkpoint if len(config_paths) == 1 else None
    results = [
        _generate_tsne_result(config_path, args, checkpoint_arg=checkpoint_arg)
        for config_path in config_paths
    ]

    representation_path, clustering_path = _build_output_paths(args.output, results)
    representation_path.parent.mkdir(parents=True, exist_ok=True)
    clustering_path.parent.mkdir(parents=True, exist_ok=True)

    _render_representation_figure(results, representation_path, dpi=args.dpi)
    _render_clustering_figure(
        results,
        clustering_path,
        dpi=args.dpi,
        ours_acc_lookup=ours_acc_lookup,
    )
    print(f"Saved representation figure to {representation_path}")
    print(f"Saved clustering figure to {clustering_path}")
    for result in results:
        print(
            f"[{result.dataset_name}] checkpoint={result.checkpoint_path} "
            f"split={result.split} samples={result.sample_count} "
            f"teacher_baseline={result.teacher_baseline_method}"
        )


if __name__ == "__main__":
    main()
