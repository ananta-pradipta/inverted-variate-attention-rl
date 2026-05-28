"""External-baseline bundle: wrap precomputed Layer-1 scores from
existing universal-panel baseline runs (MASTER, FactorVAE, iTransformer,
StockMixer, DySTAGE, MERA, SWA-InVAR) as a Layer-1 forward provider so
the InVAR-RL Layer 2 + Layer 3 pipeline can be run on top of them
WITHOUT retraining the ranker.

This is the cleanest "whole-stack" baseline construction for the paper:
hold Layer 2 (cvxpy MV-QP) and Layer 3 (SAC / PPO) constant, swap only
Layer 1. Any difference in the final portfolio Sharpe is attributable
to the Layer 1 ranker.

Data source: ``results/baselines_universal_two_regime_val/{baseline}/foldF_seedS_predictions.npz``
with keys ``y_hat`` (T, N) per-day predicted score, ``tradable_mask``
(T, N), ``tickers`` (N,), ``dates`` (T,).

Usage::

    bundle = load_external_baseline(
        baseline_name="master",
        fold=1, seed=42,
        bridge=bridge,
    )
    # ... use the bundle exactly like a TrainedInVARBundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from invar_rl.data.lattice_bridge import LatticePanelBatch


@dataclass
class ExternalBaselineBundle:
    """Per-day Layer-1 scores from a precomputed baseline npz.

    The interface matches :class:`TrainedInVARBundle.forward_day` but
    no model is held; the forward is just an array lookup. This keeps
    Layer 2 + Layer 3 oblivious to whether the scores came from
    canonical InVAR or from an external baseline.
    """

    bridge: LatticePanelBatch
    y_hat: np.ndarray
    tradable_mask: np.ndarray
    macro_input: np.ndarray
    baseline_name: str
    fold: int
    seed: int

    def forward_day(self, day_idx: int) -> Dict[str, torch.Tensor]:
        """Pull Layer-1 scores for one day from the precomputed npz.

        Mirrors :meth:`TrainedInVARBundle.forward_day` exactly so
        downstream callers (precompute_tape_canonical) work unchanged.
        """
        active = np.nonzero(self.tradable_mask[day_idx])[0]
        if active.size == 0:
            raise RuntimeError(
                f"no active tickers at day_idx={day_idx} "
                f"for baseline {self.baseline_name}"
            )
        scores_np = self.y_hat[day_idx, active].astype(np.float64)
        # Some baselines emit NaN for padded slots; coerce to 0.
        scores_np = np.where(np.isfinite(scores_np), scores_np, 0.0)
        scores = torch.from_numpy(scores_np.astype(np.float32))
        score_dispersion = torch.tensor(
            float(scores.std(unbiased=False).item()), dtype=torch.float32
        )
        macro_input = torch.from_numpy(
            self.macro_input[day_idx].astype(np.float32)
        )
        return {
            "scores": scores.detach(),
            "active_indices": torch.from_numpy(
                active.astype(np.int64)
            ),
            "macro_input": macro_input.detach(),
            "score_dispersion": score_dispersion.detach(),
            "y": torch.from_numpy(
                self.bridge.y[day_idx, active].numpy().astype(np.float32)
                if isinstance(self.bridge.y, torch.Tensor)
                else self.bridge.y[day_idx, active].astype(np.float32)
            ),
        }


def load_external_baseline(
    baseline_name: str,
    fold: int,
    seed: int,
    bridge: LatticePanelBatch,
    npz_root: str = (
        "results/baselines_universal_two_regime_val"
    ),
) -> ExternalBaselineBundle:
    """Build an ExternalBaselineBundle from a baseline's predictions npz.

    Validates that the npz's panel shape, tickers, and date alignment
    match the bridge before returning.

    Args:
        baseline_name: e.g., "master", "factorvae", "itransformer".
        fold: 1..5.
        seed: 42..46.
        bridge: :class:`LatticePanelBatch` for the same fold + panel.
        npz_root: directory containing one sub-dir per baseline.

    Returns:
        :class:`ExternalBaselineBundle`.

    Raises:
        FileNotFoundError: if the npz is missing.
        ValueError: if shape/ticker alignment with the bridge fails.
    """
    npz_path = (
        Path(npz_root) / baseline_name
        / f"fold{fold}_seed{seed}_predictions.npz"
    )
    if not npz_path.exists():
        raise FileNotFoundError(
            f"baseline predictions not found: {npz_path}"
        )
    blob = np.load(npz_path, allow_pickle=False)
    y_hat = blob["y_hat"]
    tradable_mask = blob["tradable_mask"]
    tickers = [str(t) for t in blob["tickers"].tolist()]

    if y_hat.shape != (len(bridge.dates), len(bridge.tickers)):
        raise ValueError(
            f"baseline {baseline_name} y_hat shape {y_hat.shape} "
            f"!= bridge ({len(bridge.dates)}, {len(bridge.tickers)})"
        )
    if tickers != list(bridge.tickers):
        # Soft-fail: report mismatch but proceed if cardinality matches
        # (some baselines reorder); reordering is too lossy to allow
        # silently, so raise.
        mismatched = sum(1 for a, b in zip(tickers, bridge.tickers)
                         if a != b)
        if mismatched > 0:
            raise ValueError(
                f"baseline {baseline_name} ticker order does not "
                f"match bridge ({mismatched} mismatches at fold={fold} "
                f"seed={seed}). Cannot safely align scores."
            )

    return ExternalBaselineBundle(
        bridge=bridge,
        y_hat=y_hat.astype(np.float32),
        tradable_mask=tradable_mask.astype(bool),
        macro_input=bridge.macro_arr.astype(np.float32),
        baseline_name=baseline_name,
        fold=int(fold),
        seed=int(seed),
    )


__all__ = [
    "ExternalBaselineBundle",
    "load_external_baseline",
]
