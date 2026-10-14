from __future__ import annotations

import math

import torch

from .softmax import softmax


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    in_dtype = Q.dtype
    if in_dtype in (torch.float16, torch.bfloat16):
        Q, K, V = Q.float(), K.float(), V.float()

    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])

    if mask is not None:
        mask = mask.to(device=scores.device)
        scores = scores.masked_fill(~mask, float("-inf"))

        # Avoid an all-negative-infinity row before softmax.
        has_valid_key = mask.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~has_valid_key, 0.0)

    attention_weights = softmax(scores, dim=-1)

    if mask is not None:
        attention_weights = attention_weights.masked_fill(~mask, 0.0)

    return (attention_weights @ V).to(in_dtype)