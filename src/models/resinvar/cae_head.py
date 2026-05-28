"""CAE-Head: Cross-sectional Auto-Encoder factor head for ResInVAR-RL.

Per-day low-rank factor model on the next-day cross-section of returns:

    r_t  ~  B_t f_t  +  eps_t,    B_t in R^{N_t x K}

B_t is produced by a per-stock MLP on x_t; f_t is solved in closed form
by ridge regression of r_t on B_t; eps_t is the identifiability target
that Stage 2 finetune uses in place of the raw return.

Only stocks flagged active contribute to centering, factor regression,
residual, and reconstruction loss; inactive positions are zeroed and
excluded from gradient flow.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn


DEFAULT_K_LATENT = 3
DEFAULT_HIDDEN_WIDTH = 64
DEFAULT_DROPOUT = 0.1
DEFAULT_RIDGE_LAMBDA = 1e-3
DEFAULT_LAMBDA_ORTH = 0.01


def _factor_regression(
    B_active: torch.Tensor,
    r_active: torch.Tensor,
    ridge_lambda: float,
) -> torch.Tensor:
    """Closed-form ridge factor regression on active stocks.

    Solves (B^T B + lambda I) f = B^T r via torch.linalg.solve.

    Args:
        B_active: loading matrix on active stocks, shape [N_active, K].
        r_active: return vector on active stocks, shape [N_active].
        ridge_lambda: ridge penalty on the K x K Gram matrix.

    Returns:
        f: factor vector of shape [K].

    Raises:
        ValueError: if B_active or r_active are empty or shape-mismatched.
    """
    if B_active.dim() != 2:
        raise ValueError(
            f"[ERR] _factor_regression expects B_active of rank 2, got "
            f"shape {tuple(B_active.shape)}"
        )
    if r_active.dim() != 1:
        raise ValueError(
            f"[ERR] _factor_regression expects r_active of rank 1, got "
            f"shape {tuple(r_active.shape)}"
        )
    if B_active.shape[0] != r_active.shape[0]:
        raise ValueError(
            f"[ERR] _factor_regression shape mismatch: B_active rows "
            f"{B_active.shape[0]} vs r_active len {r_active.shape[0]}"
        )
    if B_active.shape[0] == 0:
        raise ValueError("[ERR] _factor_regression got zero active stocks")

    k = B_active.shape[1]
    gram = B_active.transpose(0, 1) @ B_active
    eye = torch.eye(k, device=B_active.device, dtype=B_active.dtype)
    rhs = B_active.transpose(0, 1) @ r_active
    f = torch.linalg.solve(gram + ridge_lambda * eye, rhs)
    return f


class CAEHead(nn.Module):
    """Cross-sectional Auto-Encoder factor head.

    Produces per-stock loadings B_t = MLP(x_t), cross-sectionally centers
    them on active stocks, regresses r_t on B_t in closed form to obtain
    f_t, and emits r_hat_t = B_t f_t and eps_t = r_t - r_hat_t.
    """

    def __init__(
        self,
        feature_dim: int,
        k_latent: int = DEFAULT_K_LATENT,
        hidden_width: int = DEFAULT_HIDDEN_WIDTH,
        dropout: float = DEFAULT_DROPOUT,
        ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
        orthogonality_penalty_weight: float = DEFAULT_LAMBDA_ORTH,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError(
                f"[ERR] CAEHead feature_dim must be positive, got {feature_dim}"
            )
        if k_latent <= 0:
            raise ValueError(
                f"[ERR] CAEHead k_latent must be positive, got {k_latent}"
            )
        self.feature_dim = feature_dim
        self.k_latent = k_latent
        self.hidden_width = hidden_width
        self.dropout = dropout
        self.ridge_lambda = ridge_lambda
        self.orthogonality_penalty_weight = orthogonality_penalty_weight

        self.loading_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_width),
            nn.LayerNorm(hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, k_latent),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        r_t: torch.Tensor,
        active_mask_t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward one day.

        Args:
            x_t: panel features for day t, shape [N_t, F].
            r_t: next-day return per stock, shape [N_t].
            active_mask_t: boolean or 0/1 mask of active stocks, shape [N_t].

        Returns:
            dict with keys B [N_t, K], f [K], r_hat [N_t], eps [N_t].
            Inactive rows in B, r_hat, eps are zeroed.

        Raises:
            ValueError on shape mismatch or empty active set.
        """
        if x_t.dim() != 2 or x_t.shape[1] != self.feature_dim:
            raise ValueError(
                f"[ERR] CAEHead.forward expects x_t of shape [N, "
                f"{self.feature_dim}], got {tuple(x_t.shape)}"
            )
        if r_t.shape != (x_t.shape[0],):
            raise ValueError(
                f"[ERR] CAEHead.forward r_t shape {tuple(r_t.shape)} "
                f"incompatible with N_t={x_t.shape[0]}"
            )
        if active_mask_t.shape != (x_t.shape[0],):
            raise ValueError(
                f"[ERR] CAEHead.forward active_mask_t shape "
                f"{tuple(active_mask_t.shape)} incompatible with N_t="
                f"{x_t.shape[0]}"
            )

        mask_bool = active_mask_t.to(dtype=torch.bool)
        n_active = int(mask_bool.sum().item())
        if n_active == 0:
            raise ValueError("[ERR] CAEHead.forward got zero active stocks")

        mask_float = mask_bool.to(dtype=x_t.dtype).unsqueeze(-1)
        b_raw = self.loading_mlp(x_t)
        b_masked = b_raw * mask_float

        col_sum = b_masked.sum(dim=0, keepdim=True)
        col_mean = col_sum / float(n_active)
        b_centered = (b_raw - col_mean) * mask_float

        b_active = b_centered[mask_bool]
        r_active = r_t[mask_bool]
        f = _factor_regression(b_active, r_active, self.ridge_lambda)

        r_hat = (b_centered @ f) * mask_bool.to(dtype=x_t.dtype)
        eps = (r_t - r_hat) * mask_bool.to(dtype=x_t.dtype)
        return {"B": b_centered, "f": f, "r_hat": r_hat, "eps": eps}

    def reconstruction_loss(
        self,
        eps_t: torch.Tensor,
        active_mask_t: torch.Tensor,
    ) -> torch.Tensor:
        """Mean squared residual over active stocks.

        Args:
            eps_t: residual vector, shape [N_t].
            active_mask_t: 0/1 mask, shape [N_t].

        Returns:
            Scalar tensor: mean of eps_t**2 over active positions.

        Raises:
            ValueError on shape mismatch or empty active set.
        """
        if eps_t.dim() != 1:
            raise ValueError(
                f"[ERR] reconstruction_loss expects eps_t rank 1, got "
                f"shape {tuple(eps_t.shape)}"
            )
        if active_mask_t.shape != eps_t.shape:
            raise ValueError(
                f"[ERR] reconstruction_loss mask shape "
                f"{tuple(active_mask_t.shape)} != eps shape "
                f"{tuple(eps_t.shape)}"
            )
        mask_float = active_mask_t.to(dtype=eps_t.dtype)
        n_active = mask_float.sum()
        if float(n_active.item()) == 0.0:
            raise ValueError("[ERR] reconstruction_loss got zero active stocks")
        sq = (eps_t * mask_float) ** 2
        return sq.sum() / n_active

    def orthogonality_penalty(
        self,
        B_t: torch.Tensor,
        active_mask_t: torch.Tensor,
    ) -> torch.Tensor:
        """Frobenius penalty on (B_active^T B_active)/N_active vs identity.

        Args:
            B_t: loading matrix, shape [N_t, K]. Typically already centered.
            active_mask_t: 0/1 mask, shape [N_t].

        Returns:
            Scalar tensor: lambda_orth * || B^T B / N - I ||_F^2.

        Raises:
            ValueError on shape mismatch or empty active set.
        """
        if B_t.dim() != 2 or B_t.shape[1] != self.k_latent:
            raise ValueError(
                f"[ERR] orthogonality_penalty expects B_t shape [N, "
                f"{self.k_latent}], got {tuple(B_t.shape)}"
            )
        if active_mask_t.shape != (B_t.shape[0],):
            raise ValueError(
                f"[ERR] orthogonality_penalty mask shape "
                f"{tuple(active_mask_t.shape)} incompatible with N="
                f"{B_t.shape[0]}"
            )
        mask_bool = active_mask_t.to(dtype=torch.bool)
        n_active = int(mask_bool.sum().item())
        if n_active == 0:
            raise ValueError("[ERR] orthogonality_penalty got zero active stocks")
        b_active = B_t[mask_bool]
        b_centered = b_active - b_active.mean(dim=0, keepdim=True)
        gram = (b_centered.transpose(0, 1) @ b_centered) / float(n_active)
        eye = torch.eye(self.k_latent, device=B_t.device, dtype=B_t.dtype)
        diff = gram - eye
        return self.orthogonality_penalty_weight * (diff ** 2).sum()
