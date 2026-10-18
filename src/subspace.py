from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SubspaceStats:
    active_clusters: int
    mean_teacher_mass: float
    mean_student_mass: float


class ClusterFeatureQueue:
    def __init__(
        self,
        num_clusters: int,
        feature_dim: int,
        queue_size: int,
        device: torch.device | str = "cpu",
    ):
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.queue_size = queue_size
        self.device = torch.device(device)
        self.features = [torch.empty(0, feature_dim, device=self.device) for _ in range(num_clusters)]
        self.weights = [torch.empty(0, device=self.device) for _ in range(num_clusters)]

    @torch.no_grad()
    def enqueue(self, teacher_features: torch.Tensor, assignments: torch.Tensor) -> None:
        confidences, cluster_ids = assignments.detach().max(dim=1)
        features = teacher_features.detach().float().to(self.device)
        weights = confidences.detach().float().to(self.device)
        cluster_ids_local = cluster_ids.detach().to(self.device)

        for cluster_id in cluster_ids_local.unique().tolist():
            cluster_mask = cluster_ids_local == cluster_id
            if not torch.any(cluster_mask):
                continue

            new_features = features[cluster_mask]
            new_weights = weights[cluster_mask]

            merged_features = torch.cat([self.features[cluster_id], new_features], dim=0)
            merged_weights = torch.cat([self.weights[cluster_id], new_weights], dim=0)

            if merged_features.size(0) > self.queue_size:
                merged_features = merged_features[-self.queue_size :]
                merged_weights = merged_weights[-self.queue_size :]

            self.features[cluster_id] = merged_features
            self.weights[cluster_id] = merged_weights

    def get(self, cluster_id: int, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if device is None or torch.device(device) == self.device:
            return self.features[cluster_id], self.weights[cluster_id]
        return self.features[cluster_id].to(device), self.weights[cluster_id].to(device)

    def state_dict(self) -> dict:
        return {
            "num_clusters": self.num_clusters,
            "feature_dim": self.feature_dim,
            "queue_size": self.queue_size,
            "device": str(self.device),
            "features": [feature.detach().cpu() for feature in self.features],
            "weights": [weight.detach().cpu() for weight in self.weights],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        features = state_dict.get("features")
        weights = state_dict.get("weights")
        if features is not None:
            self.features = [feature.to(self.device) for feature in features]
        if weights is not None:
            self.weights = [weight.to(self.device) for weight in weights]


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
    q_offset: int = 4,
) -> torch.Tensor:
    total_weight = weights.sum().clamp_min(1e-6)
    normalized_weights = weights / total_weight
    mean = torch.sum(normalized_weights.unsqueeze(1) * features, dim=0, keepdim=True)
    centered = (features - mean) * normalized_weights.sqrt().unsqueeze(1)
    max_rank = min(centered.shape)
    q = min(max_rank, rank + q_offset)
    _, _, v = torch.pca_lowrank(centered, q=q, center=False)
    return v[:, :rank]


def subspace_regularization_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    assignments: torch.Tensor,
    queue: ClusterFeatureQueue,
    rank: int,
    min_mass: float,
    pca_q_offset: int = 4,
    topk_clusters: int | None = None,
) -> tuple[torch.Tensor, SubspaceStats]:
    losses = []
    teacher_masses = []
    student_masses = []

    with torch.autocast(device_type=student_features.device.type, enabled=False):
        detached_assignments = assignments.detach().float()
        teacher_features_f = teacher_features.detach().float()
        student_features_f = student_features.float()
        cluster_masses = detached_assignments.sum(dim=0)
        active_cluster_ids = torch.nonzero(cluster_masses >= min_mass, as_tuple=False).flatten()
        if topk_clusters is not None and active_cluster_ids.numel() > topk_clusters:
            active_masses = cluster_masses[active_cluster_ids]
            _, topk_indices = torch.topk(active_masses, k=topk_clusters, largest=True, sorted=False)
            active_cluster_ids = active_cluster_ids[topk_indices]
        active_cluster_ids = active_cluster_ids.tolist()

        for cluster_id in active_cluster_ids:
            student_weights = detached_assignments[:, cluster_id]
            student_mass = float(student_weights.sum().item())

            queued_features, queued_weights = queue.get(cluster_id, teacher_features.device)
            batch_teacher_weights = detached_assignments[:, cluster_id]
            teacher_sources = [teacher_features_f]
            teacher_weight_sources = [batch_teacher_weights]

            if queued_features.numel() > 0:
                teacher_sources.append(queued_features.float())
                teacher_weight_sources.append(queued_weights.float())

            merged_teacher_features = torch.cat(teacher_sources, dim=0)
            merged_teacher_weights = torch.cat(teacher_weight_sources, dim=0)
            teacher_mass = float(merged_teacher_weights.sum().item())
            if teacher_mass < min_mass or merged_teacher_features.size(0) < rank:
                continue

            anchor_basis = top_principal_subspace(
                features=merged_teacher_features,
                weights=merged_teacher_weights,
                rank=rank,
                q_offset=pca_q_offset,
            )
            sigma_student = weighted_covariance(
                features=student_features_f,
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
