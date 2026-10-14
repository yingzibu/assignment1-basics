from __future__ import annotations

import torch
from torch import nn

from .attention import scaled_dot_product_attention
from .linear import Linear
from .rope import RotaryPositionalEmbedding


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError(
                "d_model must be positive and divisible by num_heads > 0"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        if rope is not None and rope.d_k != self.head_dim:
            raise ValueError("RoPE d_k must equal d_model // num_heads")
        self.rope = rope

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(
            *x.shape[:-1], self.num_heads, self.head_dim
        ).transpose(-3, -2)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        Q = self._split_heads(self.q_proj(x))
        K = self._split_heads(self.k_proj(x))
        V = self._split_heads(self.v_proj(x))

        positions = torch.arange(x.shape[-2], device=x.device)

        if self.rope is not None:
            if token_positions is None:
                token_positions = positions
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        causal_mask = positions[:, None] >= positions[None, :]
        context = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        merged = context.transpose(-3, -2).reshape(x.shape)
        return self.output_proj(merged)