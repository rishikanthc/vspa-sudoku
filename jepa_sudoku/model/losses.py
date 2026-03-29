from __future__ import annotations

import torch
from einops import einsum, reduce
from jaxtyping import Float
from torch import Tensor


def cosine_loss(
    a: Float[Tensor, "b s d"],
    b: Float[Tensor, "b s d"],
    eps: float = 1e-8,
) -> Float[Tensor, ""]:
    a_norm = torch.linalg.norm(a, dim=-1, keepdim=True).clamp_min(eps)
    b_norm = torch.linalg.norm(b, dim=-1, keepdim=True).clamp_min(eps)

    a = a / a_norm
    b = b / b_norm

    cos_sim = einsum(a, b, "b s d, b s d -> b s")
    losses = 1.0 - cos_sim
    return reduce(losses, "b s ->", "mean")


def masked_cosine_loss(
    pred_vectors: Float[Tensor, "b s d"],
    target_vectors: Float[Tensor, "b s d"],
    include_mask: Tensor,
    eps: float = 1e-8,
) -> Float[Tensor, ""]:
    if include_mask.ndim != 2:
        raise ValueError(f"Expected include_mask with shape (B, S), got {tuple(include_mask.shape)}.")

    pred_norm = pred_vectors / torch.linalg.norm(
        pred_vectors, dim=-1, keepdim=True
    ).clamp_min(eps)
    target_norm = target_vectors / torch.linalg.norm(
        target_vectors, dim=-1, keepdim=True
    ).clamp_min(eps)

    cos_sim = einsum(pred_norm, target_norm, "b s d, b s d -> b s")
    losses = 1.0 - cos_sim
    include_mask = include_mask.to(device=losses.device, dtype=losses.dtype)
    denom = include_mask.sum().clamp_min(1.0)
    return (losses * include_mask).sum() / denom
