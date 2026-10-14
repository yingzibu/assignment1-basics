from __future__ import annotations

import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError("d_k must be a positive even integer")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (
            theta ** (
                torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
                / d_k
            )
        )
        positions = torch.arange(
            max_seq_len, device=device, dtype=torch.float32
        )
        angles = positions[:, None] * inv_freq[None, :]

        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[-1] != self.d_k:
            raise ValueError("The last dimension of x must equal d_k")

        in_dtype = x.dtype
        if in_dtype in (torch.float16, torch.bfloat16):
            x = x.to(torch.float32)

        positions = token_positions.to(
            device=self.cos_cached.device, dtype=torch.long
        )
        cos = self.cos_cached[positions].to(dtype=x.dtype)
        sin = self.sin_cached[positions].to(dtype=x.dtype)

        # Broadcast over extra attention-head dimensions.
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        result = torch.stack(
            (
                x_even * cos - x_odd * sin,
                x_even * sin + x_odd * cos,
            ),
            dim=-1,
        )
        return result.flatten(-2).to(in_dtype)