from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class _CacheShard:
    path: Path
    start_index: int
    num_rows: int


class ImageNetTeacherFeatureCache:
    """
    In-memory reader for precomputed ImageNet teacher feature candidates.

    Expected layout:
    - cache_root/metadata.json
    - cache_root/data/train-*.mmap
    - cache_root/data/train-*.mmap.progress.json

    On initialization, the full memmap cache is loaded into host RAM so
    training workers only read from memory afterward.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        expected_num_views: int | None = None,
        expected_feature_dim: int | None = None,
    ):
        self.root = Path(root)
        self.data_root = self.root / "data"
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"Expected teacher cache shard directory at: {self.data_root}")

        self.metadata = self._load_metadata()
        storage_format = str(self.metadata.get("storage_format", "memmap")).lower()
        if storage_format not in {"memmap", "mmap"}:
            raise ValueError(
                "Teacher cache metadata must declare storage_format='memmap'. "
                f"Got {storage_format!r}."
            )

        self.num_views = int(self.metadata["num_views"])
        self.feature_dim = int(self.metadata["feature_dim"])
        self.dtype = str(self.metadata["dtype"])
        self.numpy_dtype = np.dtype(self.dtype)
        self.torch_dtype = torch.from_numpy(np.empty((), dtype=self.numpy_dtype)).dtype

        if expected_num_views is not None and self.num_views != expected_num_views:
            raise ValueError(
                f"Teacher cache num_views mismatch: expected {expected_num_views}, got {self.num_views}."
            )
        if expected_feature_dim is not None and self.feature_dim != expected_feature_dim:
            raise ValueError(
                "Teacher cache feature_dim mismatch: "
                f"expected {expected_feature_dim}, got {self.feature_dim}."
            )

        self.shards = self._build_shards()
        self._size = sum(shard.num_rows for shard in self.shards)
        self.features = self._load_all_features()

    def _load_metadata(self) -> dict:
        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Expected teacher cache metadata at: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid teacher cache metadata format: {metadata_path}")
        for key in ("num_views", "feature_dim", "dtype", "shards"):
            if key not in metadata:
                raise KeyError(f"Teacher cache metadata is missing required key {key!r}: {metadata_path}")
        return metadata

    def _build_shards(self) -> list[_CacheShard]:
        metadata_shards = self.metadata["shards"]
        if not isinstance(metadata_shards, list) or not metadata_shards:
            raise ValueError("Teacher cache metadata 'shards' must be a non-empty list.")

        shards: list[_CacheShard] = []
        expected_start_index = 0
        for item in metadata_shards:
            if not isinstance(item, dict):
                raise ValueError("Teacher cache metadata 'shards' entries must be objects.")

            name = item.get("name")
            start_index = item.get("start_index")
            num_rows = item.get("num_rows")
            if not isinstance(name, str):
                raise KeyError("Teacher cache metadata shard entry is missing string field 'name'.")
            if start_index is None or num_rows is None:
                raise KeyError(
                    "Teacher cache metadata shard entry must provide 'start_index' and 'num_rows'."
                )

            start_index = int(start_index)
            num_rows = int(num_rows)
            if start_index != expected_start_index:
                raise ValueError(
                    "Teacher cache metadata shards must form a contiguous index range. "
                    f"Expected start_index {expected_start_index}, got {start_index}."
                )

            shard_path = self.data_root / name
            self._validate_shard_file(
                shard_path,
                expected_num_rows=num_rows,
                expected_num_views=self.num_views,
                expected_feature_dim=self.feature_dim,
            )
            self._validate_progress_file(
                shard_path,
                expected_num_rows=num_rows,
                expected_num_views=self.num_views,
                expected_feature_dim=self.feature_dim,
            )

            shards.append(
                _CacheShard(
                    path=shard_path,
                    start_index=start_index,
                    num_rows=num_rows,
                )
            )
            expected_start_index += num_rows

        return shards

    def _validate_shard_file(
        self,
        shard_path: Path,
        *,
        expected_num_rows: int,
        expected_num_views: int,
        expected_feature_dim: int,
    ) -> None:
        if not shard_path.is_file():
            raise FileNotFoundError(f"Teacher cache shard listed in metadata does not exist: {shard_path}")

        expected_nbytes = (
            math.prod((expected_num_rows, expected_num_views, expected_feature_dim))
            * self.numpy_dtype.itemsize
        )
        actual_nbytes = shard_path.stat().st_size
        if actual_nbytes != expected_nbytes:
            raise ValueError(
                f"Teacher cache shard {shard_path} has size {actual_nbytes} bytes, "
                f"expected {expected_nbytes} bytes."
            )

    def _validate_progress_file(
        self,
        shard_path: Path,
        *,
        expected_num_rows: int,
        expected_num_views: int,
        expected_feature_dim: int,
    ) -> None:
        progress_path = self._progress_path(shard_path)
        if not progress_path.is_file():
            raise FileNotFoundError(f"Teacher cache shard progress file is missing: {progress_path}")

        with progress_path.open("r", encoding="utf-8") as handle:
            progress = json.load(handle)

        if not isinstance(progress, dict):
            raise ValueError(f"Invalid teacher cache shard progress format: {progress_path}")

        expected_shape = [expected_num_rows, expected_num_views, expected_feature_dim]
        if progress.get("shape") != expected_shape:
            raise ValueError(
                f"Teacher cache shard progress {progress_path} has shape {progress.get('shape')}, "
                f"expected {expected_shape}."
            )
        if str(progress.get("dtype")) != self.dtype:
            raise ValueError(
                f"Teacher cache shard progress {progress_path} has dtype {progress.get('dtype')!r}, "
                f"expected {self.dtype!r}."
            )

        rows_written = int(progress.get("rows_written", -1))
        if rows_written != expected_num_rows:
            raise RuntimeError(
                f"Teacher cache shard {shard_path} is incomplete: {rows_written}/{expected_num_rows} rows written."
            )

    @staticmethod
    def _progress_path(shard_path: Path) -> Path:
        return shard_path.with_suffix(shard_path.suffix + ".progress.json")

    def _load_all_features(self) -> torch.Tensor:
        merged = np.empty(
            (self._size, self.num_views, self.feature_dim),
            dtype=self.numpy_dtype,
        )
        for shard in self.shards:
            shard_memmap = np.memmap(
                shard.path,
                dtype=self.numpy_dtype,
                mode="r",
                shape=(shard.num_rows, self.num_views, self.feature_dim),
            )
            start = shard.start_index
            stop = start + shard.num_rows
            merged[start:stop] = shard_memmap

        return torch.from_numpy(np.ascontiguousarray(merged))

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for teacher cache of size {len(self)}")
        return self.features[index]

    def get_many(self, indices: Sequence[int]) -> torch.Tensor:
        if not indices:
            return torch.empty(0, self.num_views, self.feature_dim, dtype=self.torch_dtype)

        normalized_indices: list[int] = []
        for index in indices:
            resolved = int(index)
            if resolved < 0:
                resolved += len(self)
            if resolved < 0 or resolved >= len(self):
                raise IndexError(f"Index {resolved} out of range for teacher cache of size {len(self)}")
            normalized_indices.append(resolved)

        index_tensor = torch.tensor(normalized_indices, dtype=torch.long)
        return self.features.index_select(0, index_tensor)
