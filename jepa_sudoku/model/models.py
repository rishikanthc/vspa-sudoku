import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor

from .ssp import TwoAxisSSP, TwoAxisSSPConfig


@dataclass
class TransformerConfig:
    context_size: int = 81
    n_heads: int = 4
    head_dim: int = 128
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1

    @property
    def d_model(self) -> int:
        return self.n_heads * self.head_dim


class SudokuRepresentation(nn.Module):
    def __init__(self, d_model: int, seed: int):
        super().__init__()
        self.d_model = d_model
        self.coordinate_encoder = TwoAxisSSP(TwoAxisSSPConfig(dim=d_model, seed=seed))
        prototypes = self._build_digit_codebook(d_model=d_model, seed=seed + 1)
        self.register_buffer("_digit_prototypes", prototypes)

    @staticmethod
    def _normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
        return x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(eps)

    @staticmethod
    def _build_digit_codebook(d_model: int, seed: int) -> Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        prototypes = torch.randn(9, d_model, generator=generator)
        if d_model >= 9:
            q, _ = torch.linalg.qr(prototypes.T, mode="reduced")
            prototypes = q.T
        return F.normalize(prototypes, dim=-1)

    def digit_prototypes(self) -> Float[Tensor, "9 d"]:
        return self._digit_prototypes

    def encode_coordinates(
        self, xy_or_xyz: Float[Tensor, "... c"]
    ) -> Float[Tensor, "... d"]:
        if xy_or_xyz.shape[-1] not in {2, 3}:
            raise ValueError(
                f"Expected coordinates with last dim 2 or 3, got {tuple(xy_or_xyz.shape)}"
            )
        encoded = self.coordinate_encoder.encode(xy_or_xyz[..., :2])
        if encoded.shape[-1] != self.d_model:
            raise ValueError(
                f"Coordinate encoding dim ({encoded.shape[-1]}) must match d_model ({self.d_model})."
            )
        return encoded

    def bind(
        self,
        coord_vecs: Float[Tensor, "... d"],
        digit_vecs: Float[Tensor, "... d"],
    ) -> Float[Tensor, "... d"]:
        return self._normalize(coord_vecs * digit_vecs)

    def _lookup_digit_vectors(self, digits: Tensor) -> Tensor:
        digit_indices = digits.to(dtype=torch.long) - 1
        if digit_indices.numel() > 0:
            min_digit = int(digits.min().item())
            max_digit = int(digits.max().item())
            if min_digit < 1 or max_digit > 9:
                raise ValueError(
                    f"Digit values must lie in [1, 9], got range [{min_digit}, {max_digit}]."
                )
        return self._digit_prototypes[digit_indices]

    def encode_board(
        self, board_xyz: Float[Tensor, "b s 3"]
    ) -> Float[Tensor, "b s d"]:
        coord_vecs = self.encode_coordinates(board_xyz)
        digits = board_xyz[..., 2]
        encoded = coord_vecs.clone()
        digit_mask = digits.ne(0)
        if digit_mask.any():
            digit_vecs = self._lookup_digit_vectors(digits[digit_mask])
            encoded[digit_mask] = self.bind(coord_vecs[digit_mask], digit_vecs)
        return encoded

    def encode_targets(
        self, solution_xyz: Float[Tensor, "b s 3"]
    ) -> Float[Tensor, "b s d"]:
        return self.encode_board(solution_xyz)

    def candidate_vectors(
        self, coordinates_xyz: Float[Tensor, "b s 3"]
    ) -> Float[Tensor, "b s v d"]:
        coord_vecs = self.encode_coordinates(coordinates_xyz)
        digit_vecs = self.digit_prototypes().to(coord_vecs.device)
        bound = coord_vecs.unsqueeze(-2) * digit_vecs.unsqueeze(0).unsqueeze(0)
        return self._normalize(bound)

    def logits_from_predictions(
        self,
        pred_vectors: Float[Tensor, "b s d"],
        coordinates_xyz: Float[Tensor, "b s 3"],
        logit_scale: float = 1.0,
    ) -> Float[Tensor, "b s 9"]:
        if logit_scale <= 0.0:
            raise ValueError(f"logit_scale must be positive, got {logit_scale}.")
        normalized_pred = self._normalize(pred_vectors)
        candidates = self.candidate_vectors(coordinates_xyz)
        return logit_scale * torch.einsum("bsd,bsvd->bsv", normalized_pred, candidates)


class SA(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.drop1 = nn.Dropout(config.dropout)
        self.drop2 = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, x: Float[Tensor, "b s d"]) -> Float[Tensor, "b s d"]:
        q = rearrange(self.q_proj(x), "b s (h d) -> b h s d", h=self.n_heads)
        k = rearrange(self.k_proj(x), "b s (h d) -> b h s d", h=self.n_heads)
        v = rearrange(self.v_proj(x), "b s (h d) -> b h s d", h=self.n_heads)

        attn_scores = einsum(q, k, "b h s_q d, b h s_k d -> b h s_q s_k") / math.sqrt(
            self.head_dim
        )
        attn_probs = self.drop1(F.softmax(attn_scores, dim=-1))
        out = einsum(v, attn_probs, "b h s_k d, b h s_q s_k -> b h s_q d")
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.drop2(self.out_proj(out))


class FFN(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Float[Tensor, "b s d"]) -> Float[Tensor, "b s d"]:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attention = SA(config)
        self.ln_sa = nn.LayerNorm(config.d_model)
        self.ln_ffn = nn.LayerNorm(config.d_model)
        self.ffn = FFN(config)

    def forward(self, x: Float[Tensor, "b s d"]) -> Float[Tensor, "b s d"]:
        out = x + self.self_attention(self.ln_sa(x))
        return out + self.ffn(self.ln_ffn(out))


class Encoder(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        representation: SudokuRepresentation | None = None,
        embedding: nn.Module | None = None,
    ):
        super().__init__()
        self.config = config
        if representation is None and embedding is None:
            raise ValueError("Pass a shared representation instance.")
        if representation is not None and embedding is not None:
            raise ValueError("Pass either representation or embedding, not both.")

        token_source = representation if representation is not None else embedding
        assert token_source is not None
        if hasattr(token_source, "d_model") and token_source.d_model != config.d_model:
            raise ValueError(
                f"Representation dim ({token_source.d_model}) must match model d_model ({config.d_model})."
            )
        if hasattr(token_source, "config") and token_source.config.dim != config.d_model:
            raise ValueError(
                f"Embedding dim ({token_source.config.dim}) must match model d_model ({config.d_model})."
            )

        self.representation = token_source
        self.embedding = token_source
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.drop = nn.Dropout(config.dropout)
        self.out_ln = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: Float[Tensor, "b s 3"]) -> Float[Tensor, "b s d"]:
        if hasattr(self.representation, "encode_board"):
            tokens = self.representation.encode_board(x)
        else:
            tokens = self.representation(x)
        out = self.drop(tokens * math.sqrt(self.config.d_model))
        for block in self.blocks:
            out = block(out)
        return self.head(self.out_ln(out))
