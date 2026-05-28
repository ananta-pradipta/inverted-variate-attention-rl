"""Precompute layer-3 tapes from canonical InVAR + direct cvxpy QP.

Mirrors :class:`invar_rl.layer3_control.precompute.EpisodeTape` and
:func:`precompute_tape` but sources the layer-1 forward from the
canonical InVAR adapter (``invar_rl.layer1_ranker.canonical_runner``)
and solves the layer-2 QP directly via cvxpy (no differentiable
CvxpyLayer; layer 3 trains on detached observations anyway). The
output schema is the same EpisodeTape dataclass so downstream
:class:`invar_rl.layer3_control.env.ExposureEnv` works unchanged.

Used by ``invar_rl.training.stage3_rl_canonical`` to drive the RL
controller on top of canonical InVAR.
"""

from __future__ import annotations

from typing import List, Sequence

import cvxpy as cp
import numpy as np
import torch

from invar_rl.common.config import Layer2Config
from invar_rl.data.lattice_bridge import LatticePanelBatch
from invar_rl.layer1_ranker.canonical_runner import TrainedInVARBundle
from invar_rl.layer2_alloc.covariance import estimate_covariance
from invar_rl.layer3_control.precompute import EpisodeTape


_MAX_QP_NAMES = 200


def _solve_mvqp(
    s: np.ndarray,
    sigma: np.ndarray,
    gamma: float,
    bound: float,
    gross: float,
    long_only: bool = False,
) -> tuple[np.ndarray | None, dict]:
    n = int(s.shape[0])
    w = cp.Variable(n)
    objective = cp.Minimize(
        -s @ w + 0.5 * gamma * cp.quad_form(w, cp.psd_wrap(sigma))
    )
    if long_only:
        constraints = [
            cp.sum(w) == 1.0,
            w >= 0,
            w <= bound,
        ]
    else:
        constraints = [
            cp.sum(w) == 0,
            cp.norm(w, 1) <= gross,
            w <= bound,
            w >= -bound,
        ]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.SCS, verbose=False)
    except Exception:
        return None, {}
    if problem.status not in ("optimal", "optimal_inaccurate"):
        return None, {}
    w_np = np.asarray(w.value, dtype=np.float64)
    pred_vol = float(np.sqrt(max(0.0, w_np @ sigma @ w_np)))
    gross_exp = float(np.sum(np.abs(w_np)))
    eff_pos = float(np.sum(w_np ** 2) ** 2 /
                    max(np.sum(w_np ** 4), 1e-12))
    return w_np, {
        "predicted_vol": pred_vol,
        "gross_exposure": gross_exp,
        "effective_positions": eff_pos,
    }


def _covariance_for_day(
    bridge: LatticePanelBatch,
    day_idx: int,
    active_global: np.ndarray,
    cov_lookback: int,
    estimator: str,
    factor_rank: int,
) -> np.ndarray:
    train_start = int(bridge.train_idx[0])
    lo = max(train_start, day_idx - cov_lookback)
    hi = day_idx
    window = bridge.log_returns_1d[lo:hi, :][:, active_global]
    window = np.where(np.isfinite(window), window, 0.0)
    if window.shape[0] < 2:
        n = active_global.shape[0]
        return np.eye(n, dtype=np.float64)
    return estimate_covariance(window, estimator, factor_rank)


def _equal_weight_topk(
    scores: np.ndarray, k: int = 50, gross: float = 1.0,
    long_only: bool = False,
) -> tuple[np.ndarray, dict]:
    """Equal-weight long top-k / short bottom-k portfolio.

    Returns weights summing to ~0 with L1 = gross. Used as the layer-2
    ablation that strips out the differentiable QP.

    If ``long_only`` is True, returns the equal-weight long-only top-k
    portfolio (weights sum to 1, no shorts). ``k`` is per side for L/S
    and total for L/O; tune via :func:`precompute_tape_canonical` arg
    ``equal_topk_k`` (default 50 mirrors SP500 ablation; use 20 for
    NDX-100 which has fewer active names).
    """
    n = int(scores.size)
    w = np.zeros(n, dtype=np.float64)
    if long_only:
        if n < k:
            k = max(1, n)
        order = np.argsort(-scores)
        long_idx = order[:k]
        w[long_idx] = 1.0 / k
    else:
        if n < 2 * k:
            k = max(1, n // 2)
        order = np.argsort(scores)
        short_idx = order[:k]
        long_idx = order[-k:]
        per_name = gross / (2.0 * k)
        w[long_idx] = per_name
        w[short_idx] = -per_name
    return w, {
        "predicted_vol": 0.0,
        "gross_exposure": float(np.sum(np.abs(w))),
        "effective_positions": float(
            np.sum(w ** 2) ** 2 / max(np.sum(w ** 4), 1e-12)
        ),
    }


def precompute_tape_canonical(
    bundle: TrainedInVARBundle,
    bridge: LatticePanelBatch,
    day_indices: Sequence[int],
    layer2: Layer2Config,
    stride: int = 1,
    score_mode: str = "canonical",
    weighting_mode: str = "qp",
    ablation_seed: int = 0,
    long_only: bool = False,
    equal_topk_k: int = 50,
) -> EpisodeTape:
    """Build an EpisodeTape for layer 3 using canonical InVAR + direct QP.

    The output schema matches the legacy
    :func:`invar_rl.layer3_control.precompute.precompute_tape` so the
    :class:`ExposureEnv` consumes it unchanged. All layer-1 + layer-2
    outputs are computed under ``torch.no_grad`` and detached to CPU.

    Args:
        bundle: Frozen canonical InVAR loaded by
            :func:`load_trained_invar`.
        bridge: The :class:`LatticePanelBatch` the bundle was paired
            against.
        day_indices: Trading-day indices for this episode window.
        layer2: Layer 2 config (covariance + QP knobs).
        stride: Subsample factor (>=1).
        score_mode: ``"canonical"`` (default) uses InVAR forward;
            ``"random"`` replaces the per-day scores with i.i.d.
            N(0, 1) noise seeded by ``ablation_seed`` and the day idx.
            Ablation 1 (layer-1 ablation).
        weighting_mode: ``"qp"`` (default) uses the cvxpy
            mean-variance QP; ``"equal_topk"`` replaces the QP with an
            equal-weight long-top-k / short-bottom-k portfolio
            (k=50, gross=1.0). Ablation 2 (layer-2 ablation).
        ablation_seed: Seed for the random-score stream when
            ``score_mode == "random"``. Ignored otherwise.
        long_only: If True, the QP / equal-weight portfolio enforces a
            long-only fully-invested book (sum_i w_i = 1, w_i >= 0).
        equal_topk_k: Per-side ``k`` for ``weighting_mode="equal_topk"``
            (default 50 mirrors SP500 Phase 6 ablation; pass 20 for
            NDX-100 since the active universe is roughly 1/5 the size).

    Returns:
        :class:`EpisodeTape`.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if score_mode not in ("canonical", "random"):
        raise ValueError(f"unknown score_mode: {score_mode!r}")
    if weighting_mode not in ("qp", "equal_topk"):
        raise ValueError(
            f"unknown weighting_mode: {weighting_mode!r}"
        )
    rng = np.random.RandomState(int(ablation_seed))
    day_indices = list(day_indices)[::stride]

    days: List[int] = []
    disp: List[float] = []
    enc: List[np.ndarray] = []
    pvol: List[float] = []
    effn: List[float] = []
    bret: List[float] = []
    bgross: List[float] = []
    ic: List[float] = []

    for day in day_indices:
        if day < bridge.temporal_window - 1:
            continue
        if day + 1 >= bridge.log_returns_1d.shape[0]:
            continue
        active_global = np.nonzero(bridge.tradable[day])[0]
        if active_global.size < 5:
            continue
        if score_mode == "canonical":
            out = bundle.forward_day(day)
            all_scores = out["scores"].detach().cpu().numpy().astype(
                np.float64
            )
            macro_enc_day = out["macro_input"].cpu().numpy().astype(
                np.float64
            )
            score_dispersion_day = float(
                out["score_dispersion"].cpu().item()
            )
            # Match stage 2 eval: bridge default active mask equals
            # tradable, so all_scores aligns with active_global.
            if all_scores.size != active_global.size:
                active_global = out["active_indices"].cpu().numpy().astype(
                    np.int64
                )
        else:  # score_mode == "random"
            all_scores = rng.standard_normal(active_global.size).astype(
                np.float64
            )
            macro_enc_day = np.zeros(
                bundle.bridge.macro_dim, dtype=np.float64
            )
            score_dispersion_day = float(all_scores.std())
        if active_global.size > _MAX_QP_NAMES:
            keep = np.argsort(-np.abs(all_scores))[:_MAX_QP_NAMES]
            keep.sort()
            s_np = all_scores[keep]
            active_qp = active_global[keep]
        else:
            s_np = all_scores
            active_qp = active_global
        if weighting_mode == "qp":
            sigma_np = _covariance_for_day(
                bridge,
                day,
                active_qp,
                layer2.cov_lookback,
                layer2.estimator,
                layer2.factor_rank,
            )
            w_np, summary = _solve_mvqp(
                s_np,
                sigma_np,
                gamma=float(layer2.risk_aversion),
                bound=float(layer2.per_name_bound),
                gross=float(layer2.gross_leverage),
                long_only=long_only,
            )
            if w_np is None:
                continue
        else:  # weighting_mode == "equal_topk"
            w_np, summary = _equal_weight_topk(
                s_np, k=int(equal_topk_k),
                gross=float(layer2.gross_leverage),
                long_only=long_only,
            )
        next_ret = bridge.log_returns_1d[day + 1, active_qp]
        next_ret = np.where(np.isfinite(next_ret), next_ret, 0.0)
        port_ret = float((w_np * next_ret).sum())

        days.append(int(day))
        disp.append(score_dispersion_day)
        enc.append(macro_enc_day)
        pvol.append(float(summary["predicted_vol"]))
        effn.append(float(summary["effective_positions"]))
        bret.append(port_ret)
        bgross.append(float(summary["gross_exposure"]))

        r = next_ret
        if s_np.size >= 2 and np.std(s_np) > 0 and np.std(r) > 0:
            rs = np.argsort(np.argsort(s_np)).astype(np.float64)
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


__all__ = ["precompute_tape_canonical"]
