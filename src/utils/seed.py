"""Reproducibility seeding utility for ResInVAR-RL and related models.

Per ResInVAR-RL spec rule 8: every entrypoint must call seed_all(seed)
before any tensor creation, random sampling, or stochastic layer init.
"""
from __future__ import annotations


def seed_all(seed: int) -> None:
    """Seed numpy, torch (CPU and CUDA), and Python's random module.

    Also forces cuDNN to deterministic mode (no benchmark) so per-cell
    runs are bit-identical across re-launches with the same seed.

    Args:
        seed: nonnegative integer seed. Must be an int; raises otherwise.
    """
    import random

    import numpy as np
    import torch

    if not isinstance(seed, int):
        raise TypeError(
            f"[ERR] seed_all expects int seed, got {type(seed).__name__}"
        )
    if seed < 0:
        raise ValueError(f"[ERR] seed_all expects nonnegative seed, got {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
