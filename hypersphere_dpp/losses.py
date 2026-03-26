from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(features_a: torch.Tensor, features_b: torch.Tensor, temperature: float) -> torch.Tensor:
    batch_size = features_a.size(0)
    if batch_size <= 1:
        return features_a.new_zeros(())

    features = torch.cat([features_a, features_b], dim=0)
    logits = features @ features.t()
    logits = logits / temperature

    diagonal = torch.eye(2 * batch_size, device=features.device, dtype=torch.bool)
    logits = logits.masked_fill(diagonal, float("-inf"))

    targets = torch.arange(batch_size, device=features.device)
    targets = torch.cat([targets + batch_size, targets], dim=0)
    return F.cross_entropy(logits, targets)


@torch.no_grad()
def sinkhorn_assignments(logits: torch.Tensor, epsilon: float, iterations: int) -> torch.Tensor:
    scores = logits.float()
    scores = scores - scores.max(dim=1, keepdim=True).values
    q = torch.exp(scores / epsilon).t()
    b = q.shape[1]
    k = q.shape[0]

    q = q / q.sum().clamp_min(1e-12)
    for _ in range(iterations):
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        q = q / k
        q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        q = q / b
    q = q * b
    return q.t()


def swav_cluster_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    temperature: float,
    sinkhorn_epsilon: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assignments_a = sinkhorn_assignments(logits_a.detach(), sinkhorn_epsilon, sinkhorn_iters)
    assignments_b = sinkhorn_assignments(logits_b.detach(), sinkhorn_epsilon, sinkhorn_iters)

    log_probs_a = F.log_softmax(logits_a / temperature, dim=-1)
    log_probs_b = F.log_softmax(logits_b / temperature, dim=-1)
    loss = -0.5 * (
        (assignments_a * log_probs_b).sum(dim=1).mean()
        + (assignments_b * log_probs_a).sum(dim=1).mean()
    )
    return loss, assignments_a, assignments_b


def mean_pairwise_cosine(features: torch.Tensor) -> torch.Tensor:
    features = F.normalize(features.float(), dim=-1)
    similarity = features @ features.t()
    batch_size = similarity.size(0)
    if batch_size <= 1:
        return similarity.new_zeros(())
    mask = ~torch.eye(batch_size, device=similarity.device, dtype=torch.bool)
    return similarity[mask].mean()


def dpp_diversity_loss(
    features: torch.Tensor,
    assignments: torch.Tensor,
    jitter: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = F.normalize(features.float(), dim=-1)
    quality = assignments.detach().max(dim=1).values.clamp_min(1e-6)
    centered_log_quality = quality.log() - quality.log().mean()
    relative_quality = centered_log_quality.exp()

    phi = normalized * relative_quality.unsqueeze(1)
    kernel = phi @ phi.t()
    identity = torch.eye(kernel.size(0), device=kernel.device, dtype=kernel.dtype)
    matrix = identity + kernel + jitter * identity

    sign, logabsdet = torch.linalg.slogdet(matrix)
    if not torch.all(sign > 0):
        raise RuntimeError("DPP matrix lost positive definiteness even after jitter.")

    rho = mean_pairwise_cosine(normalized).detach()
    return -logabsdet, rho, relative_quality


def adaptive_diversity_weight(
    rho: torch.Tensor,
    alpha_div_base: float,
    epoch_progress: float,
    warmup_epochs: int,
) -> torch.Tensor:
    if warmup_epochs <= 0:
        curriculum = 1.0
    else:
        curriculum = min(max(epoch_progress / warmup_epochs, 0.0), 1.0)
    rho_value = rho.detach().clamp(min=0.0)
    return rho_value.new_tensor(curriculum) * torch.clamp(alpha_div_base * rho_value, max=1.0)

