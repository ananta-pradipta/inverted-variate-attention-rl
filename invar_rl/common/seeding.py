"""Seed management for Python, NumPy, and PyTorch.

A single entry point so that every training run, evaluation, and stochastic
data operation in the project is reproducible under the fixed seed set.
"""

from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except ImportError:  # torch is optional for the pure-data Phase 0 path.
    torch = None  # type: ignore[assignment]


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all relevant random number generators.

    Args:
        seed: The integer seed. Project convention uses the set
            {42, 43, 44, 45, 46}.
        deterministic: If True, request deterministic algorithms where
            feasible. Some operations have no deterministic implementation;
            this flag enables the cuDNN deterministic path and disables the
            benchmark autotuner without forcing a hard error on unsupported
            kernels.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed)!r}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def make_rng(seed: int) -> np.random.Generator:
    """Return an isolated NumPy generator for a given seed.

    Using a dedicated generator avoids coupling stochastic data operations to
    the global NumPy state, which makes per-component reproducibility explicit.

    Args:
        seed: The integer seed.

    Returns:
        A seeded ``numpy.random.Generator``.
    """
    return np.random.default_rng(seed)
