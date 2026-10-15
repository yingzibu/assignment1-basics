"""Sample language-modeling batches from an integer NumPy array."""

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample input sequences and next-token targets with replacement."""
    if dataset.ndim != 1:
        raise ValueError("dataset must be a one-dimensional array")
    if not np.issubdtype(dataset.dtype, np.integer):
        raise TypeError("dataset must contain integer token IDs")
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size and context_length must be positive")

    num_starts = len(dataset) - context_length
    if num_starts <= 0:
        raise ValueError("dataset must contain at least context_length + 1 tokens")

    # randint excludes its upper bound.
    starts = np.random.randint(0, num_starts, size=batch_size)
    indices = starts[:, None] + np.arange(context_length)[None, :]

    # Read and convert only the sampled windows.
    inputs_np = np.asarray(dataset[indices], dtype=np.int64)
    targets_np = np.asarray(dataset[indices + 1], dtype=np.int64)

    inputs = torch.from_numpy(inputs_np).to(device=device)
    targets = torch.from_numpy(targets_np).to(device=device)
    return inputs, targets

