"""Shared clustering baselines and evaluation utilities for rebuttal experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_1d_integer_array(values: np.ndarray | list[int]) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1-D label array, got shape {array.shape}.")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"Expected integer labels, got dtype {array.dtype}.")
    return array.astype(np.int64, copy=False)


def rectangular_hungarian_accuracy(
    true_labels: np.ndarray | list[int],
    predicted_labels: np.ndarray | list[int],
) -> float:
    """One-to-one clustering accuracy that remains well-defined when K differs."""
    from scipy.optimize import linear_sum_assignment

    truth = _as_1d_integer_array(true_labels)
    predictions = _as_1d_integer_array(predicted_labels)
    if truth.shape != predictions.shape:
        raise ValueError("true_labels and predicted_labels must have the same length.")
    if truth.size == 0:
        return 0.0

    true_ids, true_inverse = np.unique(truth, return_inverse=True)
    predicted_ids, predicted_inverse = np.unique(predictions, return_inverse=True)
    contingency = np.zeros((predicted_ids.size, true_ids.size), dtype=np.int64)
    np.add.at(contingency, (predicted_inverse, true_inverse), 1)
    row_indices, column_indices = linear_sum_assignment(contingency.max() - contingency)
    return float(contingency[row_indices, column_indices].sum() / truth.size)


def clustering_metrics(
    true_labels: np.ndarray | list[int],
    predicted_labels: np.ndarray | list[int],
) -> dict[str, float]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    truth = _as_1d_integer_array(true_labels)
    predictions = _as_1d_integer_array(predicted_labels)
    if truth.shape != predictions.shape:
        raise ValueError("true_labels and predicted_labels must have the same length.")
    return {
        "acc_1to1": rectangular_hungarian_accuracy(truth, predictions),
        "nmi": float(normalized_mutual_info_score(truth, predictions)),
        "ari": float(adjusted_rand_score(truth, predictions)),
    }


def normalize_rows(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D feature matrix, got shape {values.shape}.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Spherical k-means requires non-zero feature vectors.")
    return values / norms


@dataclass(frozen=True)
class SphericalKMeans:
    centers: np.ndarray
    objective: float
    iterations: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        normalized = normalize_rows(features)
        return (normalized @ self.centers.T).argmax(axis=1).astype(np.int64, copy=False)


def _cosine_kmeans_plus_plus(
    features: np.ndarray,
    n_clusters: int,
    generator: np.random.Generator,
) -> np.ndarray:
    centers = np.empty((n_clusters, features.shape[1]), dtype=np.float64)
    first_index = int(generator.integers(features.shape[0]))
    centers[0] = features[first_index]
    closest_distance = 1.0 - features @ centers[0]

    for cluster_index in range(1, n_clusters):
        weights = np.maximum(closest_distance, 0.0)
        total_weight = float(weights.sum())
        if total_weight <= 1e-12:
            selected_index = int(generator.integers(features.shape[0]))
        else:
            selected_index = int(generator.choice(features.shape[0], p=weights / total_weight))
        centers[cluster_index] = features[selected_index]
        closest_distance = np.minimum(
            closest_distance,
            1.0 - features @ centers[cluster_index],
        )
    return normalize_rows(centers)


def fit_spherical_kmeans(
    features: np.ndarray,
    n_clusters: int,
    *,
    seed: int,
    n_init: int = 20,
    max_iter: int = 300,
    tolerance: float = 1e-5,
) -> SphericalKMeans:
    """Fit cosine/spherical k-means with k-means++ initialization and restarts."""
    values = normalize_rows(features)
    if not 1 < n_clusters <= values.shape[0]:
        raise ValueError(
            f"n_clusters must be in [2, {values.shape[0]}], got {n_clusters}."
        )
    if n_init <= 0:
        raise ValueError("n_init must be positive.")

    master_generator = np.random.default_rng(seed)
    best_model: SphericalKMeans | None = None
    for _ in range(n_init):
        generator = np.random.default_rng(int(master_generator.integers(2**32 - 1)))
        centers = _cosine_kmeans_plus_plus(values, n_clusters, generator)
        previous_assignments: np.ndarray | None = None

        for iteration in range(1, max_iter + 1):
            similarities = values @ centers.T
            assignments = similarities.argmax(axis=1)
            updated_centers = np.empty_like(centers)
            for cluster_id in range(n_clusters):
                members = values[assignments == cluster_id]
                if members.size == 0:
                    nearest_similarity = similarities.max(axis=1)
                    replacement = int(nearest_similarity.argmin())
                    updated_centers[cluster_id] = values[replacement]
                else:
                    updated_centers[cluster_id] = members.mean(axis=0)
            updated_centers = normalize_rows(updated_centers)
            center_shift = float(np.max(np.linalg.norm(updated_centers - centers, axis=1)))
            centers = updated_centers
            if previous_assignments is not None and np.array_equal(assignments, previous_assignments):
                break
            previous_assignments = assignments
            if center_shift <= tolerance:
                break

        final_similarities = values @ centers.T
        objective = float(final_similarities.max(axis=1).sum())
        candidate = SphericalKMeans(
            centers=centers,
            objective=objective,
            iterations=iteration,
        )
        if best_model is None or candidate.objective > best_model.objective:
            best_model = candidate

    assert best_model is not None
    return best_model

