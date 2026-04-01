from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import open_clip


class FrozenCLIPBackbone(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: str,
        pretrained_checkpoint: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        pretrained_source = pretrained_checkpoint or pretrained
        model, _, _ = open_clip.create_model_and_transforms(
            backbone_name,
            pretrained=pretrained_source,
            device=device,
        )
        self.model = model
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image(images)
        return F.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, tokenized_text: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_text(tokenized_text)
        return F.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def logit_scale(self) -> torch.Tensor:
        return self.model.logit_scale.exp().float()


class FeatureAdapter(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, feature_dim)

    def forward(self, teacher_features: torch.Tensor) -> torch.Tensor:
        residual = teacher_features
        x = self.norm(teacher_features)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.normalize(residual + x, dim=-1)


class PrototypeHead(nn.Module):
    def __init__(self, num_prototypes: int, feature_dim: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        self.normalize_()

    @torch.no_grad()
    def normalize_(self) -> None:
        self.prototypes.data = F.normalize(self.prototypes.data, dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized_prototypes = F.normalize(self.prototypes, dim=-1)
        return features @ normalized_prototypes.t()


class SpectralQualityHead(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.proj = nn.Linear(feature_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)


class HypersphereDPPModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: str,
        pretrained_checkpoint: Optional[str],
        feature_dim: int,
        adapter_hidden_dim: int,
        adapter_dropout: float,
        num_prototypes: int,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.backbone = FrozenCLIPBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            pretrained_checkpoint=pretrained_checkpoint,
            device=device,
        )
        self.adapter = FeatureAdapter(
            feature_dim=feature_dim,
            hidden_dim=adapter_hidden_dim,
            dropout=adapter_dropout,
        )
        self.prototypes = PrototypeHead(num_prototypes=num_prototypes, feature_dim=feature_dim)
        self.clustering_mode = "fixed_k"

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def encode_teacher(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def encode_student(
        self,
        images: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if teacher_features is None:
            if images is None:
                raise ValueError("Either images or teacher_features must be provided.")
            teacher_features = self.encode_teacher(images)
        return self.adapter(teacher_features)

    def prototype_logits(self, student_features: torch.Tensor) -> torch.Tensor:
        return self.prototypes(student_features)

    @torch.no_grad()
    def teacher_logit_scale(self) -> torch.Tensor:
        return self.backbone.logit_scale()

    @torch.no_grad()
    def normalize_prototypes(self) -> None:
        self.prototypes.normalize_()

    def trainable_parameters(self):
        return list(self.adapter.parameters()) + list(self.prototypes.parameters())

    def trainable_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "adapter": self.adapter.state_dict(),
            "prototypes": self.prototypes.state_dict(),
        }

    def load_trainable_state_dict(self, state_dict: dict[str, dict[str, torch.Tensor]]) -> None:
        self.adapter.load_state_dict(state_dict["adapter"])
        self.prototypes.load_state_dict(state_dict["prototypes"])


class HypersphereFreeKModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: str,
        pretrained_checkpoint: Optional[str],
        feature_dim: int,
        adapter_hidden_dim: int,
        adapter_dropout: float,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.backbone = FrozenCLIPBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            pretrained_checkpoint=pretrained_checkpoint,
            device=device,
        )
        self.adapter = FeatureAdapter(
            feature_dim=feature_dim,
            hidden_dim=adapter_hidden_dim,
            dropout=adapter_dropout,
        )
        self.quality_head = SpectralQualityHead(feature_dim=feature_dim)
        self.feature_dim = int(feature_dim)
        self.clustering_mode = "spectral_free_k"
        self.register_buffer("spectral_prototypes", torch.empty(0, feature_dim))
        self.register_buffer("spectral_eigenvalues", torch.empty(0))
        self.register_buffer(
            "estimated_num_clusters", torch.tensor(0, dtype=torch.long)
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def encode_teacher(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def encode_student(
        self,
        images: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if teacher_features is None:
            if images is None:
                raise ValueError("Either images or teacher_features must be provided.")
            teacher_features = self.encode_teacher(images)
        return self.adapter(teacher_features)

    @torch.no_grad()
    def teacher_logit_scale(self) -> torch.Tensor:
        return self.backbone.logit_scale()

    def _estimate_cluster_count(
        self,
        eigenvalues: torch.Tensor,
        min_clusters: int,
        max_clusters: int,
        num_samples: int,
    ) -> int:
        expected = float((eigenvalues / (1.0 + eigenvalues)).sum().item())
        estimated = int(round(expected))
        lower = min(max(1, min_clusters), max(1, num_samples))
        upper = max(1, num_samples)
        if max_clusters > 0:
            upper = min(upper, max_clusters)
        positive = int((eigenvalues > 1e-6).sum().item())
        if positive > 0:
            upper = min(upper, positive)
        return max(lower, min(max(1, estimated), upper))

    def _prototypes_from_spectrum(
        self,
        phi: torch.Tensor,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        estimated_k: int,
    ) -> torch.Tensor:
        top_vectors = eigenvectors[:, :estimated_k]
        top_values = eigenvalues[:estimated_k].clamp_min(1e-6)
        prototypes = phi.t() @ top_vectors
        prototypes = prototypes / top_values.sqrt().unsqueeze(0)
        return F.normalize(prototypes.t(), dim=-1)

    def _stabilize_prototypes(
        self,
        prototypes: torch.Tensor,
        momentum: float,
    ) -> torch.Tensor:
        previous = self.spectral_prototypes
        if previous.numel() == 0 or prototypes.numel() == 0:
            return prototypes

        similarity = prototypes @ previous.t()
        row_ind, col_ind = linear_sum_assignment(
            (-similarity.detach().cpu().numpy())
        )
        matched_pairs = sorted(zip(row_ind.tolist(), col_ind.tolist()), key=lambda item: item[1])

        aligned: list[torch.Tensor] = []
        used_rows: set[int] = set()
        for row_index, col_index in matched_pairs:
            if row_index >= prototypes.size(0) or col_index >= previous.size(0):
                continue
            current = prototypes[row_index]
            reference = previous[col_index]
            if torch.dot(current, reference) < 0:
                current = -current
            stabilized = F.normalize(
                momentum * reference + (1.0 - momentum) * current,
                dim=0,
            )
            aligned.append(stabilized)
            used_rows.add(row_index)

        for row_index in range(prototypes.size(0)):
            if row_index in used_rows:
                continue
            aligned.append(prototypes[row_index])

        return torch.stack(aligned, dim=0) if aligned else prototypes

    def spectral_cluster(
        self,
        student_features: torch.Tensor,
        temperature: float,
        min_clusters: int,
        max_clusters: int,
        momentum: float,
        update_state: bool,
    ) -> dict[str, torch.Tensor | int]:
        with torch.autocast(device_type=student_features.device.type, enabled=False):
            normalized = F.normalize(student_features.float(), dim=-1)
            raw_log_quality = self.quality_head(normalized)
            centered_log_quality = raw_log_quality - raw_log_quality.mean()
            relative_quality = centered_log_quality.exp().clamp_min(1e-6)
            phi = normalized * relative_quality.unsqueeze(1)
            kernel = 0.5 * (phi @ phi.t() + (phi @ phi.t()).t())
            eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
            order = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[order].clamp_min(0.0)
            eigenvectors = eigenvectors[:, order]
            estimated_k = self._estimate_cluster_count(
                eigenvalues=eigenvalues,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                num_samples=student_features.size(0),
            )
            prototypes = self._prototypes_from_spectrum(
                phi=phi,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                estimated_k=estimated_k,
            )
        if update_state:
            prototypes = self._stabilize_prototypes(prototypes, momentum=momentum)
            self.spectral_prototypes = prototypes.detach()
            self.spectral_eigenvalues = eigenvalues[:estimated_k].detach()
            self.estimated_num_clusters = torch.tensor(
                int(prototypes.size(0)),
                device=self.estimated_num_clusters.device,
                dtype=torch.long,
            )
        logits = normalized @ prototypes.t()
        logits = logits / temperature
        assignments = F.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "assignments": assignments,
            "prototypes": prototypes,
            "relative_quality": relative_quality,
            "eigenvalues": eigenvalues[:estimated_k],
            "estimated_k": int(prototypes.size(0)),
            "phi": phi,
        }

    def prototype_logits(self, student_features: torch.Tensor) -> torch.Tensor:
        if self.spectral_prototypes.numel() == 0:
            raise RuntimeError(
                "No spectral prototypes are available yet. Train or load a checkpoint first."
            )
        normalized = F.normalize(student_features, dim=-1)
        prototypes = F.normalize(self.spectral_prototypes, dim=-1)
        return normalized @ prototypes.t()

    @torch.no_grad()
    def normalize_prototypes(self) -> None:
        if self.spectral_prototypes.numel() > 0:
            self.spectral_prototypes = F.normalize(self.spectral_prototypes, dim=-1)

    def trainable_parameters(self):
        return list(self.adapter.parameters()) + list(self.quality_head.parameters())

    def trainable_state_dict(self) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        return {
            "adapter": self.adapter.state_dict(),
            "quality_head": self.quality_head.state_dict(),
            "spectral_prototypes": self.spectral_prototypes.detach().cpu(),
            "spectral_eigenvalues": self.spectral_eigenvalues.detach().cpu(),
            "estimated_num_clusters": self.estimated_num_clusters.detach().cpu(),
        }

    def load_trainable_state_dict(self, state_dict: dict[str, dict[str, torch.Tensor] | torch.Tensor]) -> None:
        self.adapter.load_state_dict(state_dict["adapter"])
        self.quality_head.load_state_dict(state_dict["quality_head"])
        spectral_prototypes = state_dict.get("spectral_prototypes")
        spectral_eigenvalues = state_dict.get("spectral_eigenvalues")
        estimated_num_clusters = state_dict.get("estimated_num_clusters")
        if spectral_prototypes is not None:
            self.spectral_prototypes = spectral_prototypes.to(self.spectral_prototypes.device)
        if spectral_eigenvalues is not None:
            self.spectral_eigenvalues = spectral_eigenvalues.to(self.spectral_eigenvalues.device)
        if estimated_num_clusters is not None:
            self.estimated_num_clusters = estimated_num_clusters.to(self.estimated_num_clusters.device)


def build_model_from_config(config, device: Optional[torch.device] = None):
    common_kwargs = {
        "backbone_name": config.backbone_name,
        "pretrained": config.pretrained,
        "pretrained_checkpoint": config.pretrained_checkpoint,
        "feature_dim": config.feature_dim,
        "adapter_hidden_dim": config.adapter_hidden_dim,
        "adapter_dropout": config.adapter_dropout,
        "device": device,
    }
    if config.clustering_mode == "spectral_free_k":
        return HypersphereFreeKModel(**common_kwargs)
    return HypersphereDPPModel(
        **common_kwargs,
        num_prototypes=config.class_num,
    )
