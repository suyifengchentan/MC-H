from __future__ import annotations

import bisect
import json
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch


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
        self.shards = self._build_shards()
        self._shard_start_indices = [shard.start_index for shard in self.shards]
        self._size = sum(shard.num_rows for shard in self.shards)
        self._h5_files: dict[Path, h5py.File] = {}

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
        return state

    def _open_readonly_h5(self, path: Path) -> h5py.File:
        last_error: BlockingIOError | None = None
        for attempt in range(20):
            try:
                return h5py.File(path, "r", locking=False, swmr=True)
            except TypeError:
                # Older h5py builds may not expose locking/swmr kwargs.
                return h5py.File(path, "r")
            except BlockingIOError as error:
                last_error = error
                if attempt == 19:
                    break
                time.sleep(0.5)

        raise BlockingIOError(
            f"Unable to open teacher cache shard after 20 attempts due to file locking: {path}"
        ) from last_error

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

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for teacher cache of size {len(self)}")

        shard_index = bisect.bisect_right(self._shard_start_indices, index) - 1
        shard = self.shards[shard_index]
        local_index = index - shard.start_index
        handle = self._get_h5_file(shard.path)
        features = np.asarray(handle["teacher_features"][local_index])
        return torch.from_numpy(features.copy())
