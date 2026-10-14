"""Global L2 gradient clipping."""

from collections.abc import Iterable

import torch


@torch.no_grad()
def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    """Clip existing gradients in place using their combined L2 norm."""
    if max_l2_norm < 0:
        raise ValueError("max_l2_norm must be nonnegative")

    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads:
        return

    # Accumulate low-precision gradient norms in float32.
    norms = [
        torch.linalg.vector_norm(
            grad.float() if grad.dtype in (torch.float16, torch.bfloat16) else grad
        )
        for grad in grads
    ]
    total_norm = torch.linalg.vector_norm(torch.stack(norms))

    if total_norm >= max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for grad in grads:
            grad.mul_(scale)

