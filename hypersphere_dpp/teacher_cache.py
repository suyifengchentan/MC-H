from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import h5py
try:
    import zarr
except ImportError:  # pragma: no cover - optional dependency in some environments
    zarr = None

from .hdf5_utils import open_readonly_h5


@dataclass(frozen=True)
class _CacheShard:
    path: Path
    start_index: int
    num_rows: int


class ImageNetTeacherFeatureCache:
    """
    Sharded HDF5 reader for precomputed ImageNet-1K teacher feature candidates.

    Expected layout:
    - cache_root/metadata.json
    - cache_root/data/train-*.h5

    Each HDF5 shard must contain:
    - teacher_features: [num_rows, num_views, feature_dim]
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
        self.num_views = int(self.metadata["num_views"])
        self.feature_dim = int(self.metadata["feature_dim"])
        self.dtype = str(self.metadata["dtype"])
        self.storage_format = str(self.metadata.get("storage_format", "hdf5")).lower()
        self.numpy_dtype = np.dtype(self.dtype)
        self.shards = self._build_shards()
        self._shard_start_indices = [shard.start_index for shard in self.shards]
        self._size = sum(shard.num_rows for shard in self.shards)
        self._h5_files: dict[Path, h5py.File] = {}
        self._memmaps: dict[Path, np.memmap] = {}
        self._zarr_arrays: dict[Path, object] = {}

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
        state["_h5_files"] = {}
        state["_memmaps"] = {}
        state["_zarr_arrays"] = {}
        return state

    def _open_readonly_h5(self, path: Path) -> h5py.File:
        return open_readonly_h5(path, swmr=True)

    def _load_metadata(self) -> dict:
        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Expected teacher cache metadata at: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid teacher cache metadata format: {metadata_path}")
        for key in ("num_views", "feature_dim", "dtype"):
            if key not in metadata:
                raise KeyError(f"Teacher cache metadata is missing required key {key!r}: {metadata_path}")
        return metadata

    def _build_shards(self) -> list[_CacheShard]:
        metadata_shards = self.metadata.get("shards")
        if metadata_shards:
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
                shards.append(
                    _CacheShard(
                        path=shard_path,
                        start_index=int(start_index),
                        num_rows=int(num_rows),
                    )
                )
            return shards

        if self.storage_format != "hdf5":
            raise ValueError(
                "Teacher cache metadata must include explicit shard entries for non-HDF5 storage formats."
            )

        shard_paths = sorted(self.data_root.glob("train-*.h5"))
        if not shard_paths:
            raise FileNotFoundError(
                f"Couldn't find teacher cache shards matching train-*.h5 under {self.data_root}"
            )

        shards: list[_CacheShard] = []
        start_index = 0
        for shard_path in shard_paths:
            with self._open_readonly_h5(shard_path) as handle:
                if "teacher_features" not in handle:
                    raise KeyError(
                        f"Teacher cache shard is missing required dataset 'teacher_features': {shard_path}"
                    )
                features = handle["teacher_features"]
                if features.ndim != 3:
                    raise ValueError(
                        f"Teacher cache shard must store a rank-3 tensor, got shape {features.shape} in {shard_path}"
                    )
                num_rows = int(features.shape[0])

            shards.append(
                _CacheShard(
                    path=shard_path,
                    start_index=start_index,
                    num_rows=num_rows,
                )
            )
            start_index += num_rows

        return shards

    def _get_h5_file(self, path: Path) -> h5py.File:
        handle = self._h5_files.get(path)
        if handle is None:
            handle = self._open_readonly_h5(path)
            self._h5_files[path] = handle
        return handle

    def _get_memmap(self, shard: _CacheShard) -> np.memmap:
        handle = self._memmaps.get(shard.path)
        if handle is None:
            handle = np.memmap(
                shard.path,
                mode="r",
                dtype=self.numpy_dtype,
                shape=(shard.num_rows, self.num_views, self.feature_dim),
            )
            self._memmaps[shard.path] = handle
        return handle

    def _get_zarr_array(self, shard: _CacheShard) -> zarr.Array:
        if zarr is None:
            raise ImportError(
                "Teacher cache storage_format='zarr' requires the 'zarr' package to be installed."
            )
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
        if self.storage_format == "hdf5":
            handle = self._get_h5_file(shard.path)
            return np.asarray(handle["teacher_features"][start:stop], dtype=self.numpy_dtype)
        if self.storage_format == "memmap":
            return np.asarray(self._get_memmap(shard)[start:stop], dtype=self.numpy_dtype)
        if self.storage_format == "zarr":
            return np.asarray(self._get_zarr_array(shard)[start:stop], dtype=self.numpy_dtype)
        raise ValueError(f"Unsupported teacher cache storage_format: {self.storage_format!r}")

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
            return torch.empty(0, self.num_views, self.feature_dim, dtype=torch.from_numpy(np.empty((0,), dtype=self.numpy_dtype)).dtype)

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

                first_sorted_position, first_local_index, shard = local_sorted_items[run_start]
                block = self._read_slice(
                    shard,
                    first_local_index,
                    local_sorted_items[run_end - 1][1] + 1,
                )
                for block_offset, (sorted_position, local_index, _shard) in enumerate(
                    local_sorted_items[run_start:run_end]
                ):
                    expected_local_index = first_local_index + block_offset
                    if local_index != expected_local_index:
                        raise RuntimeError("Teacher cache contiguous block assembly lost index alignment.")
                    sorted_result[sorted_position] = block[block_offset]

                run_start = run_end

        result = np.empty_like(sorted_result)
        for sorted_position, original_position in enumerate(ordered_positions):
            result[original_position] = sorted_result[sorted_position]

        return torch.from_numpy(result)
