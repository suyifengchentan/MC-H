from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SubspaceStats:
    active_clusters: int
    mean_teacher_mass: float
    mean_student_mass: float


class ClusterFeatureQueue:
    def __init__(self, num_clusters: int, feature_dim: int, queue_size: int):
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.queue_size = queue_size
        self.features = [torch.empty(0, feature_dim) for _ in range(num_clusters)]
        self.weights = [torch.empty(0) for _ in range(num_clusters)]

    @torch.no_grad()
    def enqueue(self, teacher_features: torch.Tensor, assignments: torch.Tensor) -> None:
        confidences, cluster_ids = assignments.detach().max(dim=1)
        features_cpu = teacher_features.detach().float().cpu()
        confidences_cpu = confidences.detach().float().cpu()
        cluster_ids_cpu = cluster_ids.detach().cpu()

        for cluster_id in cluster_ids_cpu.unique().tolist():
            cluster_mask = cluster_ids_cpu == cluster_id
            if not torch.any(cluster_mask):
                continue

            new_features = features_cpu[cluster_mask]
            new_weights = confidences_cpu[cluster_mask]

            merged_features = torch.cat([self.features[cluster_id], new_features], dim=0)
            merged_weights = torch.cat([self.weights[cluster_id], new_weights], dim=0)

            if merged_features.size(0) > self.queue_size:
                merged_features = merged_features[-self.queue_size :]
                merged_weights = merged_weights[-self.queue_size :]

            self.features[cluster_id] = merged_features
            self.weights[cluster_id] = merged_weights

    def get(self, cluster_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[cluster_id].to(device), self.weights[cluster_id].to(device)

    def state_dict(self) -> dict:
        return {
            "num_clusters": self.num_clusters,
            "feature_dim": self.feature_dim,
            "queue_size": self.queue_size,
            "features": self.features,
            "weights": self.weights,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.features = state_dict.get("features", self.features)
        self.weights = state_dict.get("weights", self.weights)


def weighted_covariance(features: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    total_weight = weights.sum().clamp_min(eps)
    normalized_weights = weights / total_weight
    mean = torch.sum(normalized_weights.unsqueeze(1) * features, dim=0, keepdim=True)
    centered = features - mean
    covariance = centered.t() @ (normalized_weights.unsqueeze(1) * centered)
    return covariance


def top_principal_subspace(
    features: torch.Tensor,
    weights: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    total_weight = weights.sum().clamp_min(1e-6)
    normalized_weights = weights / total_weight
    mean = torch.sum(normalized_weights.unsqueeze(1) * features, dim=0, keepdim=True)
    centered = (features - mean) * normalized_weights.sqrt().unsqueeze(1)
    _, _, v = torch.linalg.svd(centered, full_matrices=False)
    return v[:rank].t()


def subspace_regularization_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    assignments: torch.Tensor,
    queue: ClusterFeatureQueue,
    rank: int,
    min_mass: float,
) -> tuple[torch.Tensor, SubspaceStats]:
    num_clusters = assignments.size(1)
    losses = []
    teacher_masses = []
    student_masses = []

    detached_assignments = assignments.detach()
    for cluster_id in range(num_clusters):
        student_weights = detached_assignments[:, cluster_id].float()
        student_mass = float(student_weights.sum().item())
        if student_mass < min_mass:
            continue

        queued_features, queued_weights = queue.get(cluster_id, teacher_features.device)
        batch_teacher_weights = detached_assignments[:, cluster_id].float()
        teacher_sources = [teacher_features.detach().float()]
        teacher_weight_sources = [batch_teacher_weights]

        if queued_features.numel() > 0:
            teacher_sources.append(queued_features)
            teacher_weight_sources.append(queued_weights)

        merged_teacher_features = torch.cat(teacher_sources, dim=0)
        merged_teacher_weights = torch.cat(teacher_weight_sources, dim=0)
        teacher_mass = float(merged_teacher_weights.sum().item())
        if teacher_mass < min_mass or merged_teacher_features.size(0) < rank:
            continue

        anchor_basis = top_principal_subspace(
            features=merged_teacher_features,
            weights=merged_teacher_weights,
            rank=rank,
        )
        sigma_student = weighted_covariance(
            features=student_features.float(),
            weights=student_weights,
        )
        projection = anchor_basis @ anchor_basis.t()
        projected_sigma = projection @ sigma_student @ projection
        residual = sigma_student - projected_sigma
        losses.append(torch.linalg.norm(residual, ord="fro"))
        teacher_masses.append(teacher_mass)
        student_masses.append(student_mass)

    if not losses:
        return student_features.new_zeros(()), SubspaceStats(0, 0.0, 0.0)

    stacked_losses = torch.stack(losses)
    stats = SubspaceStats(
        active_clusters=len(losses),
        mean_teacher_mass=sum(teacher_masses) / len(teacher_masses),
        mean_student_mass=sum(student_masses) / len(student_masses),
    )
    return stacked_losses.mean(), stats

