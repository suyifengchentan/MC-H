from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


class TeacherFeatureCache:
    """Read the augmented frozen-teacher feature cache produced on ``main``."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_num_views: int,
        expected_feature_dim: int,
    ):
        self.root = Path(root)
        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Expected teacher cache metadata at: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        self.num_views = int(metadata["num_views"])
        self.feature_dim = int(metadata["feature_dim"])
        self.dtype = np.dtype(str(metadata["dtype"]))
        if self.num_views != expected_num_views:
            raise ValueError(
                f"Teacher cache num_views mismatch: expected {expected_num_views}, got {self.num_views}."
            )
        if self.feature_dim != expected_feature_dim:
            raise ValueError(
                "Teacher cache feature_dim mismatch: "
                f"expected {expected_feature_dim}, got {self.feature_dim}."
            )

        shards = metadata.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError("Teacher cache metadata must contain non-empty shards.")
        features: list[np.ndarray] = []
        expected_start = 0
        for shard in shards:
            start = int(shard["start_index"])
            rows = int(shard["num_rows"])
            if start != expected_start:
                raise ValueError("Teacher cache shards must form a contiguous index range.")
            path = self.root / "data" / str(shard["name"])
            if not path.is_file():
                raise FileNotFoundError(f"Teacher cache shard is missing: {path}")
            expected_bytes = rows * self.num_views * self.feature_dim * self.dtype.itemsize
            if path.stat().st_size != expected_bytes:
                raise ValueError(f"Teacher cache shard has unexpected size: {path}")
            features.append(
                np.memmap(
                    path,
                    mode="r",
                    dtype=self.dtype,
                    shape=(rows, self.num_views, self.feature_dim),
                )
            )
            expected_start += rows

        self.features = np.concatenate(features, axis=0)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(np.array(self.features[int(index)], copy=True))

    def get_many(self, indices: Sequence[int]) -> torch.Tensor:
        index_array = np.asarray(indices, dtype=np.int64)
        return torch.from_numpy(np.array(self.features[index_array], copy=True))
