"""Robust-InVAR-RL Phase 2: per-day score-spread + L1 uncertainty tape.

The canonical :class:`invar_rl.layer3_control.precompute.EpisodeTape`
already carries the realised wrapper PnL (``base_return``), the
score-dispersion summary, and the daily IC. The Kelly-style prior
needs two further per-day statistics:

- ``score_spread_topk_t``: top-K score mean minus bottom-K score mean,
  with the same ``K`` that the wrapper uses. The calibrator is fit on
  ``(score_spread_topk_t -> profitable_indicator_t)`` over the val
  segment.
- ``score_uncertainty_t``: cross-sectional std of active scores on day
  ``t`` (a monotone proxy for L1 uncertainty).

This module exposes :func:`compute_phase2_aux` that runs a single
``no-grad`` pass over the requested day indices and returns these two
arrays aligned with the tape's ``days`` axis. The pass is independent
from :func:`precompute_tape_canonical` to keep the canonical precompute
byte-identical for the Phase 1 / canonical baselines that do not need
the aux fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from invar_rl.data.lattice_bridge import LatticePanelBatch
from invar_rl.layer1_ranker.canonical_runner import TrainedInVARBundle


@dataclass
class Phase2AuxTape:
    """Per-day per-tape auxiliary statistics for the Kelly prior."""

    days: np.ndarray             # (T,) global trading-day indices
    score_spread_topk: np.ndarray  # (T,) top-K mean minus bottom-K mean
    score_uncertainty: np.ndarray  # (T,) cross-sectional score std
    daily_profitable: np.ndarray   # (T,) 0/1 indicator (base_return > 0)


def compute_phase2_aux(
    bundle: TrainedInVARBundle,
    bridge: LatticePanelBatch,
    tape_days: Sequence[int],
    base_return: np.ndarray,
    K: int,
) -> Phase2AuxTape:
    """Compute per-day top-K spread + score uncertainty for the tape days.

    Args:
        bundle: The frozen InVAR bundle that produced the tape.
        bridge: The lattice bridge for the same fold/seed.
        tape_days: The ``EpisodeTape.days`` array values (global day
            indices) the aux statistics should align with.
        base_return: The tape's ``base_return`` array; used to derive
            the binary profitable indicator for the calibrator.
        K: Wrapper per-side K (e.g. 50 for SP500).

    Returns:
        :class:`Phase2AuxTape` with arrays of length ``len(tape_days)``.
    """
    if int(K) < 1:
        raise ValueError(f"[ERR] K must be >= 1; got {K}")
    days_list = [int(d) for d in tape_days]
    T = len(days_list)
    if T == 0:
        return Phase2AuxTape(
            days=np.asarray([], dtype=np.int64),
            score_spread_topk=np.asarray([], dtype=np.float64),
            score_uncertainty=np.asarray([], dtype=np.float64),
            daily_profitable=np.asarray([], dtype=np.float64),
        )
    if base_return.shape[0] != T:
        raise ValueError(
            "[ERR] base_return length must match tape_days; "
            f"got {base_return.shape[0]} vs {T}"
        )
    spread = np.zeros(T, dtype=np.float64)
    unc = np.zeros(T, dtype=np.float64)
    with torch.no_grad():
        for i, day in enumerate(days_list):
            out = bundle.forward_day(day)
            scores = out["scores"].detach().cpu().numpy().astype(np.float64)
            if scores.size < 2:
                continue
            K_eff = int(max(1, min(int(K), scores.size // 2)))
            order = np.argsort(scores)
            bot = float(scores[order[:K_eff]].mean())
            top = float(scores[order[-K_eff:]].mean())
            spread[i] = top - bot
            unc[i] = float(scores.std(ddof=1)) if scores.size >= 2 else 0.0
    profitable = (np.asarray(base_return, dtype=np.float64) > 0.0).astype(
        np.float64
    )
    return Phase2AuxTape(
        days=np.asarray(days_list, dtype=np.int64),
        score_spread_topk=spread,
        score_uncertainty=unc,
        daily_profitable=profitable,
    )


__all__ = ["Phase2AuxTape", "compute_phase2_aux"]
