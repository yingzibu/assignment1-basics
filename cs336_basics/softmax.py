from __future__ import annotations

import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    in_dtype = x.dtype
    if in_dtype in (torch.float16, torch.bfloat16):
        x = x.to(torch.float32)

    shifted = x - x.amax(dim=dim, keepdim=True)
    exp_values = torch.exp(shifted)
    probabilities = exp_values / exp_values.sum(dim=dim, keepdim=True)
    return probabilities.to(in_dtype)