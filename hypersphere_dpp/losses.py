from __future__ import annotations

import torch
import torch.nn.functional as F


def _spd_logdet(matrix: torch.Tensor) -> torch.Tensor:
    chol, info = torch.linalg.cholesky_ex(matrix)
    if torch.any(info > 0):
        sign, logabsdet = torch.linalg.slogdet(matrix)
        if not torch.all(sign > 0):
            raise RuntimeError("DPP matrix lost positive definiteness even after jitter.")
        return logabsdet
    return 2.0 * torch.log(torch.diagonal(chol)).sum()


def _dpp_logdet_from_phi(phi: torch.Tensor, jitter: float) -> torch.Tensor:
    num_samples, feature_dim = phi.shape
    diagonal_shift = phi.new_tensor(1.0 + jitter)

    # Compute the determinant in the smaller space when possible via Sylvester's identity.
    if feature_dim < num_samples:
        gram = phi.t() @ phi
        matrix = diagonal_shift * torch.eye(feature_dim, device=phi.device, dtype=phi.dtype) + gram
        return _spd_logdet(matrix) + (num_samples - feature_dim) * torch.log(diagonal_shift)

    kernel = phi @ phi.t()
    matrix = diagonal_shift * torch.eye(num_samples, device=phi.device, dtype=phi.dtype) + kernel
    return _spd_logdet(matrix)


def nt_xent_loss(features_a: torch.Tensor, features_b: torch.Tensor, temperature: float) -> torch.Tensor:
    batch_size = features_a.size(0)
    if batch_size <= 1:
        return features_a.new_zeros(())

    features = torch.cat([features_a, features_b], dim=0)
    logits = features @ features.t()
    logits = logits / temperature
    logits.fill_diagonal_(float("-inf"))

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
    off_diagonal_sum = similarity.sum() - batch_size
    return off_diagonal_sum / (batch_size * (batch_size - 1))


def dpp_diversity_loss(
    features: torch.Tensor,
    assignments: torch.Tensor,
    jitter: float = 1e-5,
    class_aware: bool = False,
    class_aware_min_mass: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type=features.device.type, enabled=False):
        normalized = F.normalize(features.float(), dim=-1)
        detached_assignments = assignments.detach().float()

        if class_aware:
            cluster_mass = detached_assignments.sum(dim=0)
            active_cluster_ids = torch.nonzero(
                cluster_mass >= class_aware_min_mass, as_tuple=False
            ).flatten()
            if active_cluster_ids.numel() <= 1:
                zero = normalized.new_zeros(())
                return zero, zero.detach(), normalized.new_ones((0,))

            selected_assignments = detached_assignments[:, active_cluster_ids]
            mass = selected_assignments.sum(dim=0).clamp_min(1e-6)
            cluster_features = selected_assignments.t() @ normalized
            cluster_features = cluster_features / mass.unsqueeze(1)
            normalized = F.normalize(cluster_features, dim=-1)
            quality = mass
        else:
            quality = detached_assignments.max(dim=1).values.clamp_min(1e-6)

        centered_log_quality = quality.log() - quality.log().mean()
        relative_quality = centered_log_quality.exp()

        phi = normalized * relative_quality.unsqueeze(1)
        logabsdet = _dpp_logdet_from_phi(phi, jitter)

    rho = mean_pairwise_cosine(normalized).detach()
    return -logabsdet, rho, relative_quality


def adaptive_diversity_weight(
    rho: torch.Tensor,
    alpha_div_base: float,
    alpha_div_mode: str,
    alpha_div_floor: float,
    alpha_div_rho_center: float,
    alpha_div_rho_scale: float,
    alpha_div_max: float,
    epoch_progress: float,
    warmup_epochs: int,
) -> torch.Tensor:
    if warmup_epochs <= 0:
        curriculum = 1.0
    else:
        curriculum = min(max(epoch_progress / warmup_epochs, 0.0), 1.0)
    curriculum_tensor = rho.detach().new_tensor(curriculum)

    if alpha_div_mode == "clamp_positive":
        rho_value = rho.detach().clamp(min=0.0)
        return curriculum_tensor * torch.clamp(
            alpha_div_base * rho_value,
            max=alpha_div_max,
        )

    if alpha_div_mode == "sigmoid_floor":
        rho_value = rho.detach()
        adaptive_gate = torch.sigmoid(
            (rho_value - alpha_div_rho_center) / alpha_div_rho_scale
        )
        floor = rho_value.new_tensor(alpha_div_floor)
        weight = alpha_div_base * (
            floor + (1.0 - floor) * adaptive_gate
        )
        return curriculum_tensor * torch.clamp(weight, max=alpha_div_max)

    raise ValueError(f"Unsupported alpha_div_mode: {alpha_div_mode}")
