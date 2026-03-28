from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from datasets.imagenet_shard_dataset import ImageNetH5Dataset


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (8, 8), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_h5(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vlen_dtype = h5py.vlen_dtype(np.dtype("uint8"))
    utf8_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as handle:
        image_dataset = handle.create_dataset("image_bytes", shape=(len(rows),), dtype=vlen_dtype, chunks=(1,))
        for index, row in enumerate(rows):
            image_dataset[index] = np.frombuffer(row["image_bytes"], dtype=np.uint8)
        handle.create_dataset("label", data=np.asarray([row["label"] for row in rows], dtype=np.int32), chunks=(1,))
        handle.create_dataset("path", data=np.asarray([row["path"] for row in rows], dtype=object), dtype=utf8_dtype)
        handle.create_dataset("wnid", data=np.asarray([row["wnid"] for row in rows], dtype=object), dtype=utf8_dtype)


class ImageNetH5DatasetTest(unittest.TestCase):
    def test_dataset_reads_h5_rows(self):
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

            _write_h5(
                data_root / "train-00000-of-00001.h5",
                [
                    {
                        "image_bytes": _jpeg_bytes((255, 0, 0)),
                        "label": 0,
                        "path": "red.jpg",
                        "wnid": "n00000001",
                    },
                    {
                        "image_bytes": _jpeg_bytes((0, 255, 0)),
                        "label": 1,
                        "path": "green.jpg",
                        "wnid": "n00000002",
                    },
                ],
            )
            _write_h5(
                data_root / "validation-00000-of-00001.h5",
                [
                    {
                        "image_bytes": _jpeg_bytes((0, 0, 255)),
                        "label": 1,
                        "path": "blue.jpg",
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
            root = Path(tmp_dir)
            _write_h5(
                root / "data" / "train-00000-of-00001.h5",
                [
                    {
                        "image_bytes": _jpeg_bytes((255, 255, 0)),
                        "label": 0,
                        "path": "yellow.jpg",
                        "wnid": "n00000010",
                    }
                ],
            )

            dataset = ImageNetH5Dataset(
                root=str(root),
                split="train",
                class_names=["n00000010"],
            )

            self.assertEqual(dataset.classes, ["n00000010"])
            _, label, sample_index = dataset[0]
            self.assertEqual(label, 0)
            self.assertEqual(sample_index, 0)


if __name__ == "__main__":
    unittest.main()
