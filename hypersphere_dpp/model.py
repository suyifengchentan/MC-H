from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

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
