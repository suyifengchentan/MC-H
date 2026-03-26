from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from datasets.imagenet50_parquet_dataset import ImageNet50ParquetDataset


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
            "wnid": pa.array([row["wnid"] for row in rows], type=pa.string()),
            "class_name": pa.array([row["class_name"] for row in rows], type=pa.string()),
        }
    )
    pq.write_table(table, path, row_group_size=1)


class ImageNet50ParquetDatasetTest(unittest.TestCase):
    def test_dataset_reads_parquet_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_root = root / "data"
            data_root.mkdir(parents=True)

            with (root / "label_info.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "idx_to_wnid": {"0": "n00000001", "1": "n00000002"},
                    },
                    handle,
                )

            _write_parquet(
                data_root / "train-00000-of-00001.parquet",
                [
                    {
                        "image": {"bytes": _jpeg_bytes((255, 0, 0)), "path": "red.jpg"},
                        "label": 0,
                        "wnid": "n00000001",
                        "class_name": "red",
                    },
                    {
                        "image": {"bytes": _jpeg_bytes((0, 255, 0)), "path": "green.jpg"},
                        "label": 1,
                        "wnid": "n00000002",
                        "class_name": "green",
                    },
                ],
            )
            _write_parquet(
                data_root / "validation-00000-of-00001.parquet",
                [
                    {
                        "image": {"bytes": _jpeg_bytes((0, 0, 255)), "path": "blue.jpg"},
                        "label": 1,
                        "wnid": "n00000002",
                        "class_name": "blue",
                    }
                ],
            )

            train_dataset = ImageNet50ParquetDataset(root=str(root), split="train")
            val_dataset = ImageNet50ParquetDataset(root=str(root), split="val")

            image, label, sample_index = train_dataset[1]

            self.assertEqual(len(train_dataset), 2)
            self.assertEqual(len(val_dataset), 1)
            self.assertEqual(train_dataset.class_to_idx["n00000002"], 1)
            self.assertEqual(image.size, (8, 8))
            self.assertEqual(label, 1)
            self.assertEqual(sample_index, 1)


if __name__ == "__main__":
    unittest.main()
