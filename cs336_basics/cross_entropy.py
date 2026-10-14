from __future__ import annotations

import torch


def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim < 1:
        raise ValueError("logits must have a vocabulary dimension")
    if targets.shape != logits.shape[:-1]:
        raise ValueError("targets must have shape logits.shape[:-1]")

    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.to(torch.float32)
    targets = targets.to(device=logits.device, dtype=torch.long)

    shifted = logits - logits.amax(dim=-1, keepdim=True)
    log_normalizer = shifted.exp().sum(dim=-1).log()
    target_logits = shifted.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    return (log_normalizer - target_logits).mean()
