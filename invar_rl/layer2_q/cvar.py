"""CVaR helper for the InVAR-RL-Q actor objective.

The actor's objective is a blend of mean and CVaR over the predicted
return-distribution quantiles. CVaR at level alpha is the mean of the
lower-alpha fraction of the quantiles (i.e. the expected return on the
worst-case alpha tail).
"""

from __future__ import annotations

import torch


def cvar_from_quantiles(q: torch.Tensor, alpha: float) -> torch.Tensor:
    """Lower-tail CVaR over the last dimension of a quantile tensor.

    Args:
        q: Quantile tensor of shape ``[..., Nq]`` where the last dim
            holds Nq predicted quantile values for one sample. The
            quantiles do NOT need to be pre-sorted; this helper sorts
            ascending along the last dim and averages the lower-alpha
            fraction.
        alpha: Tail level in ``(0, 1]``. At least one quantile is always
            included; ``alpha * Nq`` is rounded down with a floor of 1.

    Returns:
        Tensor of shape ``q.shape[:-1]`` with the per-sample CVaR.
    """
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError(
            f"alpha must be in (0, 1]; got {alpha}"
        )
    if q.dim() == 0:
        raise ValueError("q must have at least one dim (the quantile axis)")
    n_q = int(q.shape[-1])
    k = max(1, int(alpha * n_q))
    q_sorted, _ = torch.sort(q, dim=-1)
    return q_sorted[..., :k].mean(dim=-1)
