from __future__ import annotations

import io
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "convert_imagenet_subsets_to_hdf5.py"
SPEC = importlib.util.spec_from_file_location("convert_imagenet_subsets_to_hdf5", SCRIPT_PATH)
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


class ConvertImageNetSubsetsToHdf5Test(unittest.TestCase):
    def test_build_subset_specs_supports_imagenet1k_identity_mapping(self):
        imagenet_items = [
            ("n00000001", "class one"),
            ("n00000002", "class two"),
        ]

        [spec] = MODULE.build_subset_specs(["ImageNet1K"], imagenet_items)

        self.assertEqual(spec.dataset_name, "ImageNet1K")
        self.assertEqual(spec.output_root, Path("/home/geyingjie/hypersphere-data/ImageNet1K"))
        self.assertEqual([entry.wnid for entry in spec.entries], ["n00000001", "n00000002"])
        self.assertEqual(spec.source_label_to_subset_label, {0: 0, 1: 1})
        self.assertEqual(spec.subset_label_to_wnid, {0: "n00000001", 1: "n00000002"})
        self.assertEqual(spec.subset_label_to_name, {0: "class one", 1: "class two"})

    def test_discover_source_splits_detects_test_shards(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)
            (data_root / "train-00000-of-00001.parquet").touch()
            (data_root / "test-00000-of-00001.parquet").touch()

            self.assertEqual(MODULE.discover_source_splits(data_root), ["train", "test"])

    def test_convert_one_shard_filters_subset_labels_and_remaps_them(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            shard_path = root / "train-00000-of-00001.parquet"
            output_path = root / "train-00000-of-00001.h5"

            image_type = pa.struct(
                [
                    ("bytes", pa.binary()),
                    ("path", pa.string()),
                ]
            )
            table = pa.table(
                {
                    "image": pa.array(
                        [
                            {"bytes": _jpeg_bytes((255, 0, 0)), "path": "red.jpg"},
                            {"bytes": _jpeg_bytes((0, 255, 0)), "path": "green.jpg"},
                            {"bytes": _jpeg_bytes((0, 0, 255)), "path": "blue.jpg"},
                        ],
                        type=image_type,
                    ),
                    "label": pa.array([7, 3, 7], type=pa.int64()),
                }
            )
            pq.write_table(table, shard_path)

            num_samples = MODULE.convert_one_shard(
                shard_path=shard_path,
                output_path=output_path,
                label_map={7: 0, 3: 1},
                subset_label_to_wnid={0: "n00000007", 1: "n00000003"},
                chunk_size=2,
            )

            self.assertEqual(num_samples, 3)
            with h5py.File(output_path, "r") as handle:
                self.assertEqual(handle["label"][:].tolist(), [0, 1, 0])
                self.assertEqual([item.decode("utf-8") for item in handle["path"][:]], ["red.jpg", "green.jpg", "blue.jpg"])
                self.assertEqual(
                    [item.decode("utf-8") for item in handle["wnid"][:]],
                    ["n00000007", "n00000003", "n00000007"],
                )


if __name__ == "__main__":
    unittest.main()
