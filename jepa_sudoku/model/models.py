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


def _expand_attention_mask(mask: Tensor | None, target_length: int) -> Tensor | None:
    if mask is None:
        return None
    if mask.ndim != 2:
        raise ValueError(f"Expected attention mask with shape (B, S), got {tuple(mask.shape)}.")
    return mask[:, None, None, :].expand(-1, 1, target_length, -1)


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
        digit_vecs = self._lookup_digit_vectors(digits)
        return self.bind(coord_vecs, digit_vecs)

    def encode_targets(
        self, solution_xyz: Float[Tensor, "b s 3"]
    ) -> Float[Tensor, "b s d"]:
        return self.encode_board(solution_xyz)

    def candidate_vectors(
        self, coordinates: Float[Tensor, "b s c"]
    ) -> Float[Tensor, "b s v d"]:
        coord_vecs = self.encode_coordinates(coordinates)
        digit_vecs = self.digit_prototypes().to(coord_vecs.device)
        bound = coord_vecs.unsqueeze(-2) * digit_vecs.unsqueeze(0).unsqueeze(0)
        return self._normalize(bound)

    def logits_from_predictions(
        self,
        pred_vectors: Float[Tensor, "b s d"],
        coordinates: Float[Tensor, "b s c"],
        logit_scale: float = 1.0,
    ) -> Float[Tensor, "b s 9"]:
        if logit_scale <= 0.0:
            raise ValueError(f"logit_scale must be positive, got {logit_scale}.")
        normalized_pred = self._normalize(pred_vectors)
        candidates = self.candidate_vectors(coordinates)
        return logit_scale * torch.einsum("bsd,bsvd->bsv", normalized_pred, candidates)


class SelfAttention(nn.Module):
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

    def forward(
        self,
        x: Float[Tensor, "b s d"],
        key_padding_mask: Tensor | None = None,
    ) -> Float[Tensor, "b s d"]:
        q = rearrange(self.q_proj(x), "b s (h d) -> b h s d", h=self.n_heads)
        k = rearrange(self.k_proj(x), "b s (h d) -> b h s d", h=self.n_heads)
        v = rearrange(self.v_proj(x), "b s (h d) -> b h s d", h=self.n_heads)

        attn_scores = einsum(q, k, "b h s_q d, b h s_k d -> b h s_q s_k") / math.sqrt(
            self.head_dim
        )
        expanded_mask = _expand_attention_mask(key_padding_mask, target_length=x.shape[1])
        if expanded_mask is not None:
            attn_scores = attn_scores.masked_fill(~expanded_mask, -1.0e9)
        attn_probs = F.softmax(attn_scores, dim=-1)
        if expanded_mask is not None:
            attn_probs = attn_probs * expanded_mask.to(dtype=attn_probs.dtype)
        attn_probs = self.drop1(attn_probs)
        out = einsum(v, attn_probs, "b h s_k d, b h s_q s_k -> b h s_q d")
        out = rearrange(out, "b h s d -> b s (h d)")
        out = self.drop2(self.out_proj(out))
        if key_padding_mask is not None:
            out = out * key_padding_mask.unsqueeze(-1).to(dtype=out.dtype)
        return out


class CrossAttention(nn.Module):
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

    def forward(
        self,
        queries: Float[Tensor, "b q d"],
        context: Float[Tensor, "b s d"],
        *,
        query_mask: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> Float[Tensor, "b q d"]:
        if context.shape[1] == 0:
            return torch.zeros_like(queries)

        q = rearrange(self.q_proj(queries), "b s (h d) -> b h s d", h=self.n_heads)
        k = rearrange(self.k_proj(context), "b s (h d) -> b h s d", h=self.n_heads)
        v = rearrange(self.v_proj(context), "b s (h d) -> b h s d", h=self.n_heads)

        attn_scores = einsum(q, k, "b h s_q d, b h s_k d -> b h s_q s_k") / math.sqrt(
            self.head_dim
        )
        expanded_mask = _expand_attention_mask(context_mask, target_length=queries.shape[1])
        if expanded_mask is not None:
            attn_scores = attn_scores.masked_fill(~expanded_mask, -1.0e9)
        attn_probs = F.softmax(attn_scores, dim=-1)
        if expanded_mask is not None:
            attn_probs = attn_probs * expanded_mask.to(dtype=attn_probs.dtype)
        attn_probs = self.drop1(attn_probs)
        out = einsum(v, attn_probs, "b h s_k d, b h s_q s_k -> b h s_q d")
        out = rearrange(out, "b h s d -> b s (h d)")
        out = self.drop2(self.out_proj(out))
        if query_mask is not None:
            out = out * query_mask.unsqueeze(-1).to(dtype=out.dtype)
        return out


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


class EncoderBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attention = SelfAttention(config)
        self.ln_sa = nn.LayerNorm(config.d_model)
        self.ln_ffn = nn.LayerNorm(config.d_model)
        self.ffn = FFN(config)

    def forward(
        self,
        x: Float[Tensor, "b s d"],
        attention_mask: Tensor | None = None,
    ) -> Float[Tensor, "b s d"]:
        out = x + self.self_attention(self.ln_sa(x), key_padding_mask=attention_mask)
        out = out + self.ffn(self.ln_ffn(out))
        if attention_mask is not None:
            out = out * attention_mask.unsqueeze(-1).to(dtype=out.dtype)
        return out


class PredictorBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attention = SelfAttention(config)
        self.cross_attention = CrossAttention(config)
        self.ln_sa = nn.LayerNorm(config.d_model)
        self.ln_ca = nn.LayerNorm(config.d_model)
        self.ln_ffn = nn.LayerNorm(config.d_model)
        self.ffn = FFN(config)

    def forward(
        self,
        queries: Float[Tensor, "b q d"],
        context: Float[Tensor, "b s d"],
        *,
        query_mask: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> Float[Tensor, "b q d"]:
        out = queries + self.self_attention(self.ln_sa(queries), key_padding_mask=query_mask)
        out = out + self.cross_attention(
            self.ln_ca(out),
            context,
            query_mask=query_mask,
            context_mask=context_mask,
        )
        out = out + self.ffn(self.ln_ffn(out))
        if query_mask is not None:
            out = out * query_mask.unsqueeze(-1).to(dtype=out.dtype)
        return out


class Encoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layers)])
        self.drop = nn.Dropout(config.dropout)
        self.out_ln = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        tokens: Float[Tensor, "b s d"],
        attention_mask: Tensor | None = None,
    ) -> Float[Tensor, "b s d"]:
        out = self.drop(tokens * math.sqrt(self.config.d_model))
        for block in self.blocks:
            out = block(out, attention_mask=attention_mask)
        out = self.head(self.out_ln(out))
        if attention_mask is not None:
            out = out * attention_mask.unsqueeze(-1).to(dtype=out.dtype)
        return out


class Predictor(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([PredictorBlock(config) for _ in range(config.n_layers)])
        self.query_drop = nn.Dropout(config.dropout)
        self.context_drop = nn.Dropout(config.dropout)
        self.out_ln = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        encoded_context: Float[Tensor, "b s d"],
        query_tokens: Float[Tensor, "b q d"],
        *,
        context_mask: Tensor | None = None,
        query_mask: Tensor | None = None,
    ) -> Float[Tensor, "b q d"]:
        context = self.context_drop(encoded_context)
        queries = self.query_drop(query_tokens * math.sqrt(self.config.d_model))
        for block in self.blocks:
            queries = block(
                queries,
                context,
                query_mask=query_mask,
                context_mask=context_mask,
            )
        out = self.head(self.out_ln(queries))
        if query_mask is not None:
            out = out * query_mask.unsqueeze(-1).to(dtype=out.dtype)
        return out
