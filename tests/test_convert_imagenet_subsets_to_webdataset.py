from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from datasets.imagenet_shard_dataset import ImageNetWebDataset


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "convert_imagenet_subsets_to_webdataset.py"
SPEC = importlib.util.spec_from_file_location("convert_imagenet_subsets_to_webdataset", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Unable to load converter script from {SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (8, 8), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_parquet(path: Path, rows: list[dict]) -> None:
    image_type = pa.struct(
        [
            ("bytes", pa.binary()),
            ("path", pa.string()),
        ]
    )
    table = pa.table(
        {
            "image": pa.array([row["image"] for row in rows], type=image_type),
            "label": pa.array([row["label"] for row in rows], type=pa.int64()),
        }
    )
    pq.write_table(table, path, row_group_size=1)


class ConvertImageNetSubsetsToWebdatasetTest(unittest.TestCase):
    def test_convert_one_shard_filters_subset_labels_and_writes_webdataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            shard_path = root / "train-00000-of-00001.parquet"
            tar_path = root / "train-00000-of-00001.tar"
            index_path = root / "train-00000-of-00001.index.npz"

            _write_parquet(
                shard_path,
                [
                    {"image": {"bytes": _jpeg_bytes((255, 0, 0)), "path": "red.jpg"}, "label": 7},
                    {"image": {"bytes": _jpeg_bytes((0, 255, 0)), "path": "green.png"}, "label": 3},
                    {"image": {"bytes": _jpeg_bytes((0, 0, 255)), "path": "blue.unknown"}, "label": 7},
                ],
            )

            num_samples = MODULE.convert_one_shard(
                shard_path=shard_path,
                tar_path=tar_path,
                index_path=index_path,
                label_map={7: 0, 3: 1},
                split="train",
                start_index=0,
            )

            self.assertEqual(num_samples, 3)
            with np.load(index_path, allow_pickle=False) as index_data:
                self.assertEqual(index_data["labels"].tolist(), [0, 1, 0])
                self.assertEqual(index_data["offsets"].shape[0], 3)
                self.assertEqual(index_data["sizes"].shape[0], 3)

            with tarfile.open(tar_path, "r") as archive:
                member_names = archive.getnames()
                self.assertEqual(
                    member_names,
                    [
                        "train-00000000.jpg",
                        "train-00000000.cls",
                        "train-00000001.png",
                        "train-00000001.cls",
                        "train-00000002.jpg",
                        "train-00000002.cls",
                    ],
                )
                cls_payload = archive.extractfile("train-00000001.cls")
                self.assertIsNotNone(cls_payload)
                self.assertEqual(cls_payload.read().decode("utf-8"), "1\n")

    def test_convert_subset_produces_readable_webdataset_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            imagenet_root = Path(tmp_dir) / "imagenet"
            data_root = imagenet_root / "data"
            data_root.mkdir(parents=True, exist_ok=True)
            (imagenet_root / "classes.py").write_text(
                "IMAGENET2012_CLASSES = {\n"
                "    'n00000001': 'red bird',\n"
                "    'n00000002': 'green bird',\n"
                "}\n",
                encoding="utf-8",
            )

            _write_parquet(
                data_root / "train-00000-of-00001.parquet",
                [
                    {"image": {"bytes": _jpeg_bytes((255, 0, 0)), "path": "train-red.jpg"}, "label": 0},
                    {"image": {"bytes": _jpeg_bytes((0, 255, 0)), "path": "train-green.jpg"}, "label": 1},
                ],
            )
            _write_parquet(
                data_root / "validation-00000-of-00001.parquet",
                [
                    {"image": {"bytes": _jpeg_bytes((0, 0, 255)), "path": "val-blue.jpg"}, "label": 1},
                ],
            )

            output_root = Path(tmp_dir) / "webdataset"
            [spec] = MODULE.resolve_subset_specs(
                imagenet_root=imagenet_root,
                dataset_names=["ImageNet1K"],
                output_root=output_root,
            )
            split_summaries = MODULE.convert_subset(
                imagenet_root=imagenet_root,
                spec=spec,
                max_shards_per_split=None,
                overwrite=True,
            )
            MODULE.write_metadata_files(
                spec=spec,
                imagenet_root=imagenet_root,
                split_summaries=split_summaries,
            )

            train_dataset = ImageNetWebDataset(root=str(output_root), split="train")
            val_dataset = ImageNetWebDataset(root=str(output_root), split="val")

            image, label, sample_index = train_dataset[1]

            self.assertEqual(len(train_dataset), 2)
            self.assertEqual(len(val_dataset), 1)
            self.assertEqual(train_dataset.class_to_idx["n00000002"], 1)
            self.assertEqual(image.size, (8, 8))
            self.assertEqual(label, 1)
            self.assertEqual(sample_index, 1)

            with (output_root / "metadata.json").open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["format"], "webdataset")
            self.assertEqual(metadata["class_num"], 2)
            self.assertEqual(metadata["splits"]["train"]["num_samples"], 2)


if __name__ == "__main__":
    unittest.main()
