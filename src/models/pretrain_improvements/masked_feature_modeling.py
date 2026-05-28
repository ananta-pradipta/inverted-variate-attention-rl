"""C2 (2026-05-27): Masked Feature Modeling pretrain head.

Provides a BERT/MAE-style pretext for the Stage-1 per-ticker temporal
encoder: per stock per day a random subset of feature dimensions in the
lookback window is masked (zeroed); the encoder consumes the masked
window and a small linear decoder reconstructs the original feature
values at the masked positions. Loss is MSE on masked positions only.
The inductive bias is DENSE FEATURE DEPENDENCIES (which features
predict which other features across the lookback) rather than
day-level regime coherence (canonical / B1) or per-day sector coherence
(C3).

Design
------
Different from canonical / B1 / C3 in three orthogonal ways:

  * Granularity: pretext target is per-stock per-day per-feature scalar
    reconstruction, not a cross-sectional contrastive cohort.
  * Loss type: MSE on masked positions, not InfoNCE.
  * Augmentation: per-stock random feature mask on the INPUT window
    (positions zeroed) so the encoder must use neighbouring features
    and earlier time steps to predict the held-out values.

The encoder itself is the SAME ``PerTickerTemporalEncoder`` the rest
of the pretrain trainer uses; only the head differs. After Stage 1 the
masked-feature head is discarded and ONLY the encoder ``state_dict``
flows into Stage 2 (the canonical strict-load contract is unchanged).

Forbidden inputs: future returns, the target panel y, val days, test
days. Mask positions are sampled on the (N_active, T, F) standardised
LOOKBACK ONLY (already restricted to TRAIN-day windows in
``run_stage1_pretrain``). The reconstruction target is the ORIGINAL
(pre-mask) standardised feature values at the masked positions.

Canonical-preserve invariant: callers MUST only invoke this module
when the caller-side flag ``pretrain_method == "masked_feature"``. With
the default ``"infonce_kmeans"`` (or ``"infonce_hmm"`` / ``"infonce_
sector"``) the pretrain loop does not touch this file and the existing
InfoNCE path runs byte-identically.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


# Default BERT mask ratio. Held outside the module so the trainer and
# tests reference the same source.
DEFAULT_MASK_RATIO: float = 0.15


class MaskedFeatureHead(nn.Module):
    """Linear decoder ``d_model -> feature_dim`` for the C2 pretext.

    Consumes the per-ticker encoder output (``(N_active, d_model)``,
    last-step pooled) and predicts the ORIGINAL standardised feature
    values for the LAST time step of the lookback. We reconstruct
    the last time step only because the encoder's last-step pooling
    discards the earlier time steps; predicting the masked positions at
    step T-1 from the d-dim summary of the masked window is the cleanest
    pretext that matches the encoder's information bottleneck.

    Args:
        d_model: encoder output dimensionality.
        feature_dim: number of input feature channels per time step
            (F in the ``(N, T, F)`` window). The decoder's output width.
    """

    def __init__(self, d_model: int, feature_dim: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.feature_dim = int(feature_dim)
        # SimCLR-style 2-layer head sizes felt over-parameterised for a
        # tiny F = 26 (lattice_native); a single Linear keeps the
        # parameter count tiny and makes the inductive bias purely
        # "is the d-dim encoding rich enough to predict the masked
        # feature scalars". Mirrors the B2 aux-head sizing
        # (nn.Linear(d_model, 1)).
        self.decoder = nn.Linear(self.d_model, self.feature_dim)

    def forward(self, embeddings: Tensor) -> Tensor:
        """Reconstruct ``(N_active, feature_dim)`` from per-stock embeddings.

        Args:
            embeddings: ``(N_active, d_model)`` encoder output for the
                day's active stocks (last-step pooled).

        Returns:
            ``(N_active, feature_dim)`` reconstructed feature row at the
            LAST time step of each stock's lookback.
        """
        return self.decoder(embeddings)


def random_feature_mask(
    features: Tensor,
    mask_ratio: float = DEFAULT_MASK_RATIO,
    generator: torch.Generator | None = None,
) -> Tuple[Tensor, Tensor]:
    """Per-stock random feature mask on the LAST-step feature row.

    For each row (stock) of ``features`` independently sample a Bernoulli
    mask over the F feature channels with success probability
    ``mask_ratio``; zero out the masked positions in the returned
    ``masked_features`` and return a matching ``(N, F)`` 1.0/0.0
    indicator tensor. The encoder consumes the masked LAST-step row in
    place of the original LAST-step row of its lookback window; the
    earlier T-1 time steps of the lookback are unmodified (so the
    encoder must use the time history to fill in the masked positions).

    Args:
        features: ``(N, F)`` original standardised feature row at the
            last time step.
        mask_ratio: per-position Bernoulli probability of being masked.
            Default :data:`DEFAULT_MASK_RATIO` (0.15, BERT default).
        generator: optional :class:`torch.Generator` for reproducible
            mask sampling in tests. ``None`` (default) uses the global
            RNG, matching the training-loop convention.

    Returns:
        Tuple ``(masked_features, mask_indicator)``:
            masked_features: ``(N, F)`` features with masked positions
                set to 0.0. Same device, same dtype.
            mask_indicator: ``(N, F)`` float mask, 1.0 where masked, 0.0
                otherwise. Same device, same dtype as ``features``.
    """
    if mask_ratio < 0.0 or mask_ratio > 1.0:
        raise ValueError(
            f"mask_ratio must be in [0, 1]; got {mask_ratio!r}"
        )
    # Sample Bernoulli(mask_ratio) per (n, f). torch.rand_like gives
    # float in [0, 1); positions strictly less than mask_ratio are
    # marked as masked.
    if generator is None:
        rand = torch.rand(features.shape, device=features.device)
    else:
        rand = torch.rand(
            features.shape,
            device=features.device,
            generator=generator,
        )
    mask = (rand < float(mask_ratio)).to(features.dtype)
    masked_features = features * (1.0 - mask)
    return masked_features, mask


def masked_feature_loss(
    reconstructed: Tensor,
    original: Tensor,
    mask: Tensor,
) -> Tensor:
    """Mean-squared-error on masked positions only.

    Args:
        reconstructed: ``(N, F)`` decoder output.
        original: ``(N, F)`` original (pre-mask) feature row.
        mask: ``(N, F)`` 1.0 where masked, 0.0 otherwise.

    Returns:
        Scalar MSE averaged over masked positions only. Returns 0.0 on
        the input device / dtype if no position is masked (would be a
        no-op step; the trainer skips the backward in that case).
    """
    if reconstructed.shape != original.shape:
        raise ValueError(
            "reconstructed.shape != original.shape: "
            f"{tuple(reconstructed.shape)} vs {tuple(original.shape)}"
        )
    if mask.shape != original.shape:
        raise ValueError(
            "mask.shape != original.shape: "
            f"{tuple(mask.shape)} vs {tuple(original.shape)}"
        )
    sq = (reconstructed - original) ** 2
    num = (sq * mask).sum()
    den = mask.sum().clamp_min(1.0e-8)
    return num / den


__all__ = [
    "DEFAULT_MASK_RATIO",
    "MaskedFeatureHead",
    "masked_feature_loss",
    "random_feature_mask",
]
