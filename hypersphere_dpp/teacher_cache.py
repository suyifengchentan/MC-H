from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import zarr


@dataclass(frozen=True)
class _CacheShard:
    path: Path
    start_index: int
    num_rows: int


class ImageNetTeacherFeatureCache:
    """
    Sharded zarr reader for precomputed ImageNet teacher feature candidates.

    Expected layout:
    - cache_root/metadata.json
    - cache_root/data/train-*.zarr

    Each zarr shard must store a rank-3 array with shape:
    - [num_rows, num_views, feature_dim]
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
        storage_format = str(self.metadata.get("storage_format", "zarr")).lower()
        if storage_format != "zarr":
            raise ValueError(
                "Teacher cache metadata must declare storage_format='zarr'. "
                f"Got {storage_format!r}."
            )

        self.num_views = int(self.metadata["num_views"])
        self.feature_dim = int(self.metadata["feature_dim"])
        self.dtype = str(self.metadata["dtype"])
        self.numpy_dtype = np.dtype(self.dtype)
        self.torch_dtype = torch.from_numpy(np.empty((), dtype=self.numpy_dtype)).dtype
        self.shards = self._build_shards()
        self._shard_start_indices = [shard.start_index for shard in self.shards]
        self._size = sum(shard.num_rows for shard in self.shards)
        self._zarr_arrays: dict[Path, zarr.Array] = {}

        if expected_num_views is not None and self.num_views != expected_num_views:
            raise ValueError(
                f"Teacher cache num_views mismatch: expected {expected_num_views}, got {self.num_views}."
            )
        if expected_feature_dim is not None and self.feature_dim != expected_feature_dim:
            raise ValueError(
                "Teacher cache feature_dim mismatch: "
                f"expected {expected_feature_dim}, got {self.feature_dim}."
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zarr_arrays"] = {}
        return state

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

            shard_path = self.data_root / name
            if not shard_path.exists():
                raise FileNotFoundError(f"Teacher cache shard listed in metadata does not exist: {shard_path}")

            array = zarr.open(shard_path, mode="r")
            shape = tuple(int(dim) for dim in array.shape)
            expected_shape = (int(num_rows), self.num_views, self.feature_dim)
            if shape != expected_shape:
                raise ValueError(
                    f"Teacher cache shard {shard_path} has shape {shape}, expected {expected_shape}."
                )

            shards.append(
                _CacheShard(
                    path=shard_path,
                    start_index=int(start_index),
                    num_rows=int(num_rows),
                )
            )
        return shards

    def _get_zarr_array(self, shard: _CacheShard) -> zarr.Array:
        handle = self._zarr_arrays.get(shard.path)
        if handle is None:
            handle = zarr.open(shard.path, mode="r")
            self._zarr_arrays[shard.path] = handle
        return handle

    def _lookup_shard(self, index: int) -> tuple[_CacheShard, int]:
        shard_index = bisect.bisect_right(self._shard_start_indices, index) - 1
        shard = self.shards[shard_index]
        local_index = index - shard.start_index
        return shard, local_index

    def _read_slice(self, shard: _CacheShard, start: int, stop: int) -> np.ndarray:
        return np.asarray(self._get_zarr_array(shard)[start:stop], dtype=self.numpy_dtype)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for teacher cache of size {len(self)}")

        shard, local_index = self._lookup_shard(index)
        features = self._read_slice(shard, local_index, local_index + 1)[0].copy()
        return torch.from_numpy(features)

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

        ordered_positions = sorted(
            range(len(normalized_indices)),
            key=lambda position: normalized_indices[position],
        )
        sorted_indices = [normalized_indices[position] for position in ordered_positions]
        sorted_result = np.empty(
            (len(sorted_indices), self.num_views, self.feature_dim),
            dtype=self.numpy_dtype,
        )

        grouped: dict[Path, list[tuple[int, int, _CacheShard]]] = {}
        for sorted_position, global_index in enumerate(sorted_indices):
            shard, local_index = self._lookup_shard(global_index)
            grouped.setdefault(shard.path, []).append((sorted_position, local_index, shard))

        for items in grouped.values():
            local_sorted_items = sorted(items, key=lambda item: item[1])
            run_start = 0
            while run_start < len(local_sorted_items):
                run_end = run_start + 1
                while (
                    run_end < len(local_sorted_items)
                    and local_sorted_items[run_end][1] == local_sorted_items[run_end - 1][1] + 1
                ):
                    run_end += 1

                run_items = local_sorted_items[run_start:run_end]
                shard = run_items[0][2]
                run_slice = self._read_slice(shard, run_items[0][1], run_items[-1][1] + 1)
                for relative_offset, (sorted_position, _local_index, _shard) in enumerate(run_items):
                    sorted_result[sorted_position] = run_slice[relative_offset]
                run_start = run_end

        result = np.empty_like(sorted_result)
        for sorted_position, original_position in enumerate(ordered_positions):
            result[original_position] = sorted_result[sorted_position]
        return torch.from_numpy(result.copy())
