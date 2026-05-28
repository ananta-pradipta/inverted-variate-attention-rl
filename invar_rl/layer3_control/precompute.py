"""Precompute frozen Layer 1 and Layer 2 outputs for the Layer 3 environment.

Layers 1 and 2 are frozen in Phase 4. Their per-day outputs are computed once
under ``torch.no_grad`` and stored as plain NumPy arrays. The environment
serves only these detached values, so there is no live gradient connection
from the environment to Layer 1 or Layer 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch

from invar_rl.common.config import Layer2Config
from invar_rl.layer1_ranker.invar import INVAR
from invar_rl.layer2_alloc.qp_layer import MeanVarianceQP
from invar_rl.training.stage2_decision import (
    _day_inputs,
    _portfolio_return,
    build_one_day_return_matrix,
    covariance_for_day,
)


@dataclass
class EpisodeTape:
    """Detached per-day record consumed by the environment.

    Every array is indexed by step position within ``days`` (not by global
    trading-day index). All values are realised or known as of that day; no
    future information is stored.
    """

    days: np.ndarray            # (T,) global trading-day indices
    score_dispersion: np.ndarray  # (T,)
    macro_encoding: np.ndarray  # (T, d)
    pred_vol: np.ndarray        # (T,)
    eff_positions: np.ndarray   # (T,)
    base_return: np.ndarray     # (T,) realised return of the base book
    base_gross: np.ndarray      # (T,) gross exposure of the base book
    daily_ic: np.ndarray        # (T,) Spearman IC of Layer 1 scores vs fwd ret

    def __len__(self) -> int:
        return int(self.days.shape[0])

    @property
    def macro_dim(self) -> int:
        return int(self.macro_encoding.shape[1])


def precompute_tape(
    model: INVAR,
    qp: MeanVarianceQP,
    panel,
    day_indices: Sequence[int],
    layer2: Layer2Config,
    train_start: int,
    stride: int = 1,
) -> EpisodeTape:
    """Run frozen Layer 1 and Layer 2 over ``day_indices`` and store outputs.

    Args:
        model: A frozen, trained Layer 1 (no gradients are taken).
        qp: The Layer 2 mean-variance allocator.
        panel: A data-contract implementation.
        day_indices: Global trading-day indices for this episode window.
        layer2: Layer 2 configuration (covariance settings).
        train_start: Earliest day allowed in the covariance window, so the
            real-time, training-fold-only convention is preserved.
        stride: Subsample ``day_indices`` by this step. The per-day QP solve
            dominates wall time; ``stride > 1`` trades tape resolution for a
            roughly proportional speedup. ``stride = 1`` (default) keeps
            every day and is behaviour-preserving.

    Returns:
        An :class:`EpisodeTape`.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    day_indices = list(day_indices)[::stride]
    ret1_full = build_one_day_return_matrix(panel)
    warmup = panel.lookback - 1

    days: List[int] = []
    disp: List[float] = []
    enc: List[np.ndarray] = []
    pvol: List[float] = []
    effn: List[float] = []
    bret: List[float] = []
    bgross: List[float] = []
    ic: List[float] = []

    model.eval()
    with torch.no_grad():
        for day in day_indices:
            if day < warmup:
                continue
            feats, macro, g_idx, fwd, finite = _day_inputs(panel, day)
            if finite.sum() < 5:
                continue
            sigma = torch.from_numpy(
                covariance_for_day(
                    ret1_full, g_idx, day, layer2.cov_lookback,
                    train_start, layer2,
                )
            ).float()
            out = model(feats, macro)
            weights, summary = qp(out.scores, sigma)

            days.append(int(day))
            disp.append(float(out.summary["score_dispersion"]))
            enc.append(out.macro_regime_encoding.detach().cpu().numpy())
            pvol.append(float(summary["predicted_vol"]))
            effn.append(float(summary["effective_positions"]))
            bret.append(float(_portfolio_return(weights, fwd, finite)))
            bgross.append(float(summary["gross_exposure"]))

            s = out.scores.detach().cpu().numpy()[finite]
            r = fwd[finite]
            if s.size >= 2 and np.std(s) > 0 and np.std(r) > 0:
                rs = np.argsort(np.argsort(s)).astype(np.float64)
                rr = np.argsort(np.argsort(r)).astype(np.float64)
                ic.append(float(np.corrcoef(rs, rr)[0, 1]))
            else:
                ic.append(0.0)

    return EpisodeTape(
        days=np.asarray(days, dtype=np.int64),
        score_dispersion=np.asarray(disp, dtype=np.float64),
        macro_encoding=np.asarray(enc, dtype=np.float64),
        pred_vol=np.asarray(pvol, dtype=np.float64),
        eff_positions=np.asarray(effn, dtype=np.float64),
        base_return=np.asarray(bret, dtype=np.float64),
        base_gross=np.asarray(bgross, dtype=np.float64),
        daily_ic=np.asarray(ic, dtype=np.float64),
    )
