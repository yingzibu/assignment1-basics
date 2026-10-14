from __future__ import annotations

import math

import torch
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # W 的形状为 (d_out, d_in)
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = math.sqrt(
            2.0 / (self.in_features + self.out_features)
        )

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (..., d_in) @ (d_in, d_out) -> (..., d_out)
        return x @ self.weight.transpose(-2, -1)