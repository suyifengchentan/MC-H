from __future__ import annotations

import io
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

try:
    import zarr
except ImportError:  # pragma: no cover - optional dependency in some environments
    zarr = None

from datasets.imagenet_h5_dataset import ImageNetH5Dataset
from hypersphere_dpp.config import ExperimentConfig
from hypersphere_dpp.data import TEACHER_FEATURE_CANDIDATES_KEY, build_experiment_dataloaders
from hypersphere_dpp.teacher_cache import ImageNetTeacherFeatureCache

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (8, 8), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_teacher_cache_zarr(path: Path, teacher_features: np.ndarray) -> None:
    if zarr is None:
        raise unittest.SkipTest("zarr is not installed")
    path.parent.mkdir(parents=True, exist_ok=True)
    array = zarr.open(
        path,
        mode="w",
        shape=teacher_features.shape,
        chunks=teacher_features.shape,
        dtype=teacher_features.dtype,
    )
    array[:] = teacher_features


def _load_module(script_name: str, module_name: str):
    module_path = PROJECT_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_precompute_module():
    return _load_module(
        "precompute_imagenet1k_teacher_cache.py",
        "precompute_imagenet1k_teacher_cache_test",
    )


def _write_classes(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "classes.py").write_text(
        "IMAGENET2012_CLASSES = {\n"
        "    'n00000001': 'red bird',\n"
        "    'n00000002': 'green bird',\n"
        "}\n",
        encoding="utf-8",
    )


def _write_label_info(root: Path) -> None:
    with (root / "label_info.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "idx_to_wnid": {"0": "n00000001", "1": "n00000002"},
                "idx_to_class_name": {"0": "red bird", "1": "green bird"},
            },
            handle,
        )


def _write_h5_shard(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_dtype = h5py.vlen_dtype(np.dtype("uint8"))
    text_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        image_ds = handle.create_dataset("image_bytes", shape=(len(rows),), dtype=image_dtype)
        label_ds = handle.create_dataset("label", data=np.asarray([row["label"] for row in rows], dtype=np.int32))
        path_ds = handle.create_dataset("path", shape=(len(rows),), dtype=text_dtype)
        wnid_ds = handle.create_dataset("wnid", shape=(len(rows),), dtype=text_dtype)
        for index, row in enumerate(rows):
            image_ds[index] = np.frombuffer(row["image_bytes"], dtype=np.uint8)
            path_ds[index] = row["path"]
            wnid_ds[index] = row["wnid"]
        label_ds[:] = np.asarray([row["label"] for row in rows], dtype=np.int32)


def _write_h5_root(
    root: Path,
    *,
    train_rows: list[dict],
    validation_rows: list[dict],
) -> None:
    _write_classes(root)
    _write_label_info(root)
    data_root = root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    _write_h5_shard(data_root / "train-00000-of-00001.h5", train_rows)
    _write_h5_shard(data_root / "validation-00000-of-00001.h5", validation_rows)


class ImageNetH5DatasetTest(unittest.TestCase):
    def test_dataset_reads_h5_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "imagenet_h5"
            _write_h5_root(
                root,
                train_rows=[
                    {
                        "image_bytes": _jpeg_bytes((255, 0, 0)),
                        "label": 0,
                        "path": "train-red.jpg",
                        "wnid": "n00000001",
                    },
                    {
                        "image_bytes": _jpeg_bytes((0, 255, 0)),
                        "label": 1,
                        "path": "train-green.jpg",
                        "wnid": "n00000002",
                    },
                ],
                validation_rows=[
                    {
                        "image_bytes": _jpeg_bytes((0, 0, 255)),
                        "label": 1,
                        "path": "validation-blue.jpg",
                        "wnid": "n00000002",
                    }
                ],
            )

            train_dataset = ImageNetH5Dataset(root=str(root), split="train")
            val_dataset = ImageNetH5Dataset(root=str(root), split="val")

            image, label, sample_index = train_dataset[1]

            self.assertEqual(len(train_dataset), 2)
            self.assertEqual(len(val_dataset), 1)
            self.assertEqual(train_dataset.class_to_idx["n00000002"], 1)
            self.assertEqual(image.size, (8, 8))
            self.assertEqual(label, 1)
            self.assertEqual(sample_index, 1)

    def test_dataset_can_fall_back_to_explicit_class_names(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "imagenet_h5"
            _write_h5_root(
                root,
                train_rows=[
                    {
                        "image_bytes": _jpeg_bytes((255, 0, 0)),
                        "label": 0,
                        "path": "train-red.jpg",
                        "wnid": "n00000001",
                    }
                ],
                validation_rows=[
                    {
                        "image_bytes": _jpeg_bytes((0, 0, 255)),
                        "label": 1,
                        "path": "validation-blue.jpg",
                        "wnid": "n00000002",
                    }
                ],
            )
            (root / "label_info.json").unlink()

            dataset = ImageNetH5Dataset(
                root=str(root),
                split="train",
                class_names=["n00000001", "n00000002"],
            )

            self.assertEqual(dataset.classes, ["n00000001", "n00000002"])
            _, label, sample_index = dataset[0]
            self.assertEqual(label, 0)
            self.assertEqual(sample_index, 0)

    def test_dataset_exposes_labels_for_stratified_splits(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "imagenet_h5"

            train_rows = []
            for label in (0, 1):
                for sample_idx in range(10):
                    train_rows.append(
                        {
                            "image_bytes": _jpeg_bytes((label * 127, sample_idx, 0)),
                            "label": label,
                            "path": f"train-{label}-{sample_idx}.jpg",
                            "wnid": f"n0000000{label + 1}",
                        }
                    )
            _write_h5_root(
                root,
                train_rows=train_rows,
                validation_rows=[
                    {
                        "image_bytes": _jpeg_bytes((0, 0, 255)),
                        "label": 0,
                        "path": "val-0.jpg",
                        "wnid": "n00000001",
                    }
                ],
            )

            dataset = ImageNetH5Dataset(root=str(root), split="train")
            self.assertEqual(dataset.labels.count(0), 10)
            self.assertEqual(dataset.labels.count(1), 10)
            self.assertEqual(dataset.targets, dataset.labels)

            config = ExperimentConfig(
                dataset="ImageNet1K",
                dataset_dir=str(root),
                class_num=2,
                image_size=8,
                feature_dim=512,
                backbone_name="ViT-B-16",
                pretrained="laion2b_s34b_b88k",
                pretrained_checkpoint="/root/clip-pretrained/open_clip_pytorch_model.bin",
                batch_size=4,
                eval_batch_size=4,
                workers=0,
                validation_ratio=0.2,
                official_eval_split="val",
            )
            bundle = build_experiment_dataloaders(config)
            self.assertTrue(bundle.val_is_stratified)

    def test_dataloaders_can_use_imagenet_teacher_feature_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "imagenet_h5"

            train_rows = []
            for label in (0, 1):
                for sample_idx in range(10):
                    train_rows.append(
                        {
                            "image_bytes": _jpeg_bytes((label * 100, sample_idx, 0)),
                            "label": label,
                            "path": f"train-{label}-{sample_idx}.jpg",
                            "wnid": f"n0000000{label + 1}",
                        }
                    )
            _write_h5_root(
                root,
                train_rows=train_rows,
                validation_rows=[
                    {
                        "image_bytes": _jpeg_bytes((0, 0, 255)),
                        "label": 0,
                        "path": "val-0.jpg",
                        "wnid": "n00000001",
                    }
                ],
            )

            cache_root = Path(tmp_dir) / "teacher_cache_vitb16_imagenet1k_two_view_views6"
            cache_root.mkdir(parents=True, exist_ok=True)
            with (cache_root / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "num_views": 6,
                        "feature_dim": 512,
                        "dtype": "float16",
                        "storage_format": "zarr",
                        "shards": [
                            {
                                "name": "train-00000-of-00001.zarr",
                                "num_rows": 20,
                                "start_index": 0,
                            }
                        ],
                    },
                    handle,
                )
            teacher_features = np.random.randn(20, 6, 512).astype(np.float16)
            _write_teacher_cache_zarr(
                cache_root / "data" / "train-00000-of-00001.zarr",
                teacher_features,
            )

            config = ExperimentConfig(
                dataset="ImageNet1K",
                dataset_dir=str(root),
                class_num=2,
                image_size=8,
                feature_dim=512,
                backbone_name="ViT-B-16",
                pretrained="laion2b_s34b_b88k",
                pretrained_checkpoint="/root/clip-pretrained/open_clip_pytorch_model.bin",
                batch_size=4,
                eval_batch_size=4,
                workers=0,
                validation_ratio=0.2,
                official_eval_split="val",
                imagenet1k_cached_teacher_views_enabled=True,
                imagenet1k_cached_teacher_num_views=6,
                imagenet1k_cached_teacher_views_dir=str(cache_root),
            )

            bundle = build_experiment_dataloaders(config)
            batch = next(iter(bundle.train_loader))

            self.assertIn(TEACHER_FEATURE_CANDIDATES_KEY, batch)
            self.assertNotIn("image", batch)
            self.assertEqual(batch[TEACHER_FEATURE_CANDIDATES_KEY].shape[1:], (6, 512))

    @unittest.skipIf(zarr is None, "zarr is not installed")
    def test_precompute_shard_progress_reads_rows_written(self):
        module = _load_precompute_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            shard_path = Path(tmp_dir) / "train-00000-of-00001.zarr"
            shard_array = zarr.open(
                shard_path,
                mode="w",
                shape=(10, 6, 512),
                chunks=(2, 6, 512),
                dtype="float16",
            )
            shard_array.attrs["rows_written"] = 4

            teacher_features_ds, rows_written = module.prepare_cache_shard(
                shard_path,
                expected_num_rows=10,
                expected_num_views=6,
                expected_feature_dim=512,
                dtype_name="float16",
                batch_size=2,
                overwrite=False,
            )
            teacher_features_shape = tuple(int(dim) for dim in teacher_features_ds.shape)

            self.assertEqual(teacher_features_shape, (10, 6, 512))
            self.assertEqual(rows_written, 4)

    @unittest.skipIf(zarr is None, "zarr is not installed")
    def test_teacher_cache_get_many_preserves_order_for_zarr(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir)
            (cache_root / "data").mkdir(parents=True, exist_ok=True)
            teacher_features = np.arange(12 * 6 * 4, dtype=np.float16).reshape(12, 6, 4)
            _write_teacher_cache_zarr(
                cache_root / "data" / "train-00000-of-00001.zarr",
                teacher_features,
            )
            with (cache_root / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "num_views": 6,
                        "feature_dim": 4,
                        "dtype": "float16",
                        "storage_format": "zarr",
                        "shards": [
                            {
                                "name": "train-00000-of-00001.zarr",
                                "num_rows": 12,
                                "start_index": 0,
                            }
                        ],
                    },
                    handle,
                )

            cache = ImageNetTeacherFeatureCache(cache_root)
            batch = cache.get_many([9, 0, 3])

            expected = torch.from_numpy(teacher_features[[9, 0, 3]])
            self.assertTrue(torch.equal(batch, expected))


if __name__ == "__main__":
    unittest.main()
