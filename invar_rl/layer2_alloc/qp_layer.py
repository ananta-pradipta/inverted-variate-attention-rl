"""Differentiable mean-variance quadratic program (the primary allocation).

Solves, for the day's score vector s and estimated covariance Sigma:

    minimize over w   (negative s dot w) plus (gamma over two) w' Sigma w
    subject to        sum of w equals 0          (dollar neutral)
                      L1 norm of w at most 1     (unit gross leverage)
                      negative b at most w_i at most b   (per-name bound)

The program is convex with a unique solution, so cvxpylayers gives the
gradient with respect to s and Sigma through the optimality conditions
automatically. The covariance enters through its lower Cholesky factor so
the parametrised problem is DPP-compliant and differentiable in Sigma.

The active-stock count varies day to day; one compiled CvxpyLayer is built
and cached per distinct cross-section size.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

from invar_rl.common.config import Layer2Config
from invar_rl.layer2_alloc.summary import portfolio_summary

_JITTER = 1e-6


class MeanVarianceQP:
    """Differentiable dollar-neutral mean-variance allocator."""

    def __init__(self, cfg: Layer2Config) -> None:
        """Initialise the allocator.

        Args:
            cfg: Layer 2 configuration providing gamma (risk aversion),
                b (per-name bound), and the gross-leverage cap.
        """
        self._gamma = float(cfg.risk_aversion)
        self._bound = float(cfg.per_name_bound)
        self._gross = float(cfg.gross_leverage)
        self._layers: Dict[int, CvxpyLayer] = {}

    def _build(self, n: int) -> CvxpyLayer:
        """Build and cache the compiled problem for ``n`` stocks."""
        w = cp.Variable(n)
        s = cp.Parameter(n)
        root = cp.Parameter((n, n))  # upper factor, Sigma = root' root

        objective = cp.Minimize(
            -s @ w + 0.5 * self._gamma * cp.sum_squares(root @ w)
        )
        constraints = [
            cp.sum(w) == 0,
            cp.norm(w, 1) <= self._gross,
            w <= self._bound,
            w >= -self._bound,
        ]
        problem = cp.Problem(objective, constraints)
        if not problem.is_dpp():
            raise RuntimeError("QP is not DPP-compliant")
        layer = CvxpyLayer(problem, parameters=[s, root], variables=[w])
        self._layers[n] = layer
        return layer

    def __call__(
        self, scores: torch.Tensor, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Solve the QP for one day.

        Args:
            scores: Expected-return proxy s, shape (n,).
            sigma: Estimated covariance Sigma, shape (n, n).

        Returns:
            A pair ``(weights, summary)``. ``weights`` has shape (n,);
            ``summary`` holds ``predicted_vol``, ``effective_positions``,
            ``gross_exposure``, and ``net_exposure`` (all scalar tensors).
        """
        if scores.dim() != 1:
            raise ValueError(
                f"scores must be 1-D, got shape {tuple(scores.shape)}"
            )
        n = scores.shape[0]
        if sigma.shape != (n, n):
            raise ValueError(
                f"sigma must be ({n}, {n}), got {tuple(sigma.shape)}"
            )
        layer = self._layers.get(n) or self._build(n)

        eye = torch.eye(n, dtype=sigma.dtype, device=sigma.device)
        chol = torch.linalg.cholesky(sigma + _JITTER * eye)
        root = chol.transpose(-1, -2)

        # SCS is a first-order conic solver; tighten its accuracy so the
        # box and equality constraints hold to a tight tolerance and the
        # differentiated solution is numerically stable.
        (weights,) = layer(
            scores,
            root,
            solver_args={
                "eps_abs": 1e-9,
                "eps_rel": 1e-9,
                "max_iters": 100000,
            },
        )
        return weights, portfolio_summary(weights, sigma)
