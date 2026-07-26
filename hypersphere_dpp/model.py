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

    @torch.no_grad()
    def encode_text(self, tokenized_text: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_text(tokenized_text)
        return F.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def logit_scale(self) -> torch.Tensor:
        return self.model.logit_scale.exp().float()


class FeatureAdapter(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        dropout: float,
        input_dim: int | None = None,
    ):
        super().__init__()
        input_dim = feature_dim if input_dim is None else input_dim
        self.block_hidden_dim = max(1, hidden_dim // 2)
        self.input_projection: nn.Module
        if input_dim == feature_dim:
            self.input_projection = nn.Identity()
        else:
            self.input_projection = nn.Linear(input_dim, feature_dim, bias=False)
        self.main_scale = nn.Parameter(torch.tensor(1.0))
        self.norm = nn.LayerNorm(feature_dim)
        self.fc1 = nn.Linear(feature_dim, self.block_hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.block_hidden_dim, feature_dim)
        self.refine_scale = nn.Parameter(torch.tensor(1.0))
        self.refine_norm = nn.LayerNorm(feature_dim)
        self.refine_fc1 = nn.Linear(feature_dim, self.block_hidden_dim)
        self.refine_fc2 = nn.Linear(self.block_hidden_dim, feature_dim)

    def _apply_residual_block(
        self,
        teacher_features: torch.Tensor,
        norm: nn.LayerNorm,
        fc1: nn.Linear,
        fc2: nn.Linear,
        scale: nn.Parameter,
    ) -> torch.Tensor:
        residual = teacher_features
        x = norm(teacher_features)
        x = fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = fc2(x)
        return residual + scale * x

    def project_teacher_features(self, teacher_features: torch.Tensor) -> torch.Tensor:
        # Cached frozen CLIP features are stored in FP16, while this projection
        # is initialized in FP32 and can also be called outside autocast.
        return F.normalize(self.input_projection(teacher_features.float()), dim=-1)

    def forward(self, teacher_features: torch.Tensor) -> torch.Tensor:
        teacher_features = self.project_teacher_features(teacher_features)
        x = self._apply_residual_block(
            teacher_features,
            self.norm,
            self.fc1,
            self.fc2,
            self.main_scale,
        )
        x = self._apply_residual_block(
            x,
            self.refine_norm,
            self.refine_fc1,
            self.refine_fc2,
            self.refine_scale,
        )
        return F.normalize(x, dim=-1)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return self.state_dict()

    def load_trainable_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        try:
            self.load_state_dict(state_dict)
        except RuntimeError as exc:
            adapter_keys = sorted(state_dict.keys())
            raise RuntimeError(
                "FeatureAdapter checkpoint is incompatible with the current "
                "two-block residual adapter. "
                f"Expected two smaller residual blocks with bottleneck width "
                f"{self.block_hidden_dim}, but received adapter keys {adapter_keys}. "
                "Use a checkpoint trained with the same adapter architecture."
            ) from exc


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

    def set_trainable(self, trainable: bool) -> None:
        self.prototypes.requires_grad_(trainable)


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
        representation_dim: int | None = None,
    ):
        super().__init__()
        self.backbone = FrozenCLIPBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            pretrained_checkpoint=pretrained_checkpoint,
            device=device,
        )
        representation_dim = feature_dim if representation_dim is None else representation_dim
        self.adapter = FeatureAdapter(
            feature_dim=representation_dim,
            hidden_dim=adapter_hidden_dim,
            dropout=adapter_dropout,
            input_dim=feature_dim,
        )
        self.prototypes = PrototypeHead(
            num_prototypes=num_prototypes, feature_dim=representation_dim
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def encode_teacher(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def forward(
        self,
        view1: torch.Tensor | None = None,
        view2: torch.Tensor | None = None,
        teacher1: torch.Tensor | None = None,
        teacher2: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if teacher1 is None or teacher2 is None:
            if view1 is None or view2 is None:
                raise ValueError("Provide either two image views or two teacher feature views.")
            teacher1 = self.encode_teacher(view1)
            teacher2 = self.encode_teacher(view2)
        teacher_cat = torch.cat([teacher1, teacher2], dim=0)
        student_cat = self.encode_student(teacher_features=teacher_cat)
        logits_cat = self.prototype_logits(student_cat)
        return teacher_cat, student_cat, logits_cat

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

    def project_teacher_features(self, teacher_features: torch.Tensor) -> torch.Tensor:
        return self.adapter.project_teacher_features(teacher_features)

    def prototype_logits(self, student_features: torch.Tensor) -> torch.Tensor:
        return self.prototypes(student_features)

    @torch.no_grad()
    def teacher_logit_scale(self) -> torch.Tensor:
        return self.backbone.logit_scale()

    @torch.no_grad()
    def normalize_prototypes(self) -> None:
        self.prototypes.normalize_()

    def set_prototypes_trainable(self, trainable: bool) -> None:
        self.prototypes.set_trainable(trainable)

    def adapter_parameters(self):
        return list(self.adapter.parameters())

    def prototype_parameters(self):
        return list(self.prototypes.parameters())

    def trainable_parameters(self):
        return self.adapter_parameters() + self.prototype_parameters()

    def trainable_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "adapter": self.adapter.trainable_state_dict(),
            "prototypes": self.prototypes.state_dict(),
        }

    def load_trainable_state_dict(self, state_dict: dict[str, dict[str, torch.Tensor]]) -> None:
        self.adapter.load_trainable_state_dict(state_dict["adapter"])
        self.prototypes.load_state_dict(state_dict["prototypes"])
