"""Biotech NBI Phase 4 Layer 2 mean-variance QP driver.

Direct mirror of :mod:`invar_rl.training.nasdaq100_layer2_qp` for the
biotech NBI ENRICHED universe. For one ``(fold, seed)`` cell, this driver loads
(or regenerates) the Layer 1 scores parquet, estimates a daily
Ledoit-Wolf shrinkage covariance from a 120-day trailing window, and
solves a convex MV-QP via cvxpy + SCS for two protocols (long-short
dollar-neutral and long-only), writing one weight parquet per
protocol.

Policy P1 reuse: gamma=5.0, per-name cap=0.05, cov_lookback=120,
top-K cap = min(200, n_active) (NBI active count ~270 per day, so
the cap actually bites here unlike NDX-100).

CLI::

    python -m invar_rl.training.biotech_nbi_enriched_layer2_qp \
        --fold F --seed S [--protocol ls|lo|both]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Defer cvxpy import so --help and the scores-regen path work without it.
try:
    import cvxpy as cp
except Exception:  # pragma: no cover
    cp = None

from invar_rl.layer2_alloc.covariance import (
    ledoit_wolf_constant_correlation,
)

PROTOCOLS = ("ls", "lo")
GAMMA = 5.0
PER_NAME_CAP = 0.05
COV_LOOKBACK = 120
TOPK_CAP = 200
EPS_ACTIVE = 1e-6


def _ensure_scores(
    fold: int, seed: int, layer1_root: Path, panel_end: str,
) -> Path:
    """Load (or regenerate) the Layer 1 scores parquet.

    Mirrors the NDX-100 regen path: if the parquet is missing but the
    full-state ckpt exists, regenerate via the biotech NBI ENRICHED Phase 3
    helper so scores are byte-identical to what Phase 3 would write.
    """
    scores_path = layer1_root / "scores" / f"fold{fold}_seed{seed}.parquet"
    if scores_path.exists():
        if len(pd.read_parquet(scores_path, columns=["date"])) > 0:
            print(
                f"[biotech_nbi_enriched_layer2] scores parquet present at "
                f"{scores_path}; skipping regen",
                flush=True,
            )
            return scores_path
    full_path = layer1_root / "_ckpt" / f"fold{fold}_seed{seed}_full.pt"
    if not full_path.exists():
        raise FileNotFoundError(
            f"cannot regenerate scores: no full ckpt at {full_path}; "
            f"re-run Phase 3 for this cell first"
        )
    print(
        f"[biotech_nbi_enriched_layer2] regenerating scores for fold={fold} "
        f"seed={seed} from {full_path}",
        flush=True,
    )
    from invar_rl.training.biotech_nbi_enriched_layer1_eval import (
        _persist_scores_and_macro,
    )
    _persist_scores_and_macro(
        fold=fold, seed=seed,
        output_dir_root=layer1_root,
        scores_dir=layer1_root / "scores",
        macro_dir=layer1_root / "macro_enc",
        panel_end=panel_end,
    )
    if not scores_path.exists():
        raise RuntimeError(
            f"scores regen completed but {scores_path} still missing"
        )
    return scores_path


def _load_returns(
    fold: int, seed: int, panel_end: str,
) -> pd.DataFrame:
    """Build the (T x N) one-day log-return panel via lattice_bridge."""
    from invar_rl.data.lattice_bridge import build_lattice_bridge
    from src.baselines.train_invar_stx_v2 import InvarSTXV2Config
    import torch

    cfg = InvarSTXV2Config(fold=fold, seed=seed)
    cfg.panel_kind = "biotech_nbi_enriched"
    cfg.two_regime_val = True
    cfg.panel_end = panel_end
    bridge = build_lattice_bridge(cfg, device=torch.device("cpu"))
    dates = pd.DatetimeIndex(
        [pd.Timestamp(d).normalize() for d in bridge.dates],
        name="date",
    )
    return pd.DataFrame(
        np.asarray(bridge.log_returns_1d, dtype=np.float64),
        index=dates, columns=list(bridge.tickers),
    )


def _solve_qp(
    s: np.ndarray, sigma: np.ndarray, protocol: str,
) -> Tuple[Optional[np.ndarray], str]:
    """Solve one MV-QP; try SCS, then OSQP, then ECOS."""
    if cp is None:
        raise RuntimeError("cvxpy is required for the Layer 2 QP")
    n = int(s.shape[0])
    w = cp.Variable(n)
    sigma_p = cp.psd_wrap(sigma)
    obj = cp.Maximize(s @ w - 0.5 * GAMMA * cp.quad_form(w, sigma_p))
    if protocol == "ls":
        constraints = [
            cp.sum(w) == 0, cp.norm(w, 1) <= 1.0,
            w <= PER_NAME_CAP, w >= -PER_NAME_CAP,
        ]
    elif protocol == "lo":
        constraints = [
            cp.sum(w) == 1.0, w >= 0.0, w <= PER_NAME_CAP,
        ]
    else:
        raise ValueError(f"unknown protocol {protocol!r}")
    problem = cp.Problem(obj, constraints)
    for name, kw in (
        ("SCS", dict(solver=cp.SCS, verbose=False, eps=1e-7,
                     max_iters=20000)),
        ("OSQP", dict(solver=cp.OSQP, verbose=False, eps_abs=1e-8,
                      eps_rel=1e-8, max_iter=50000)),
        ("ECOS", dict(solver=cp.ECOS, verbose=False, abstol=1e-8,
                      reltol=1e-8, max_iters=200)),
    ):
        try:
            problem.solve(**kw)
        except Exception:  # pragma: no cover
            continue
        if problem.status in ("optimal", "optimal_inaccurate"):
            wv = np.asarray(w.value, dtype=np.float64)
            if wv is not None and np.all(np.isfinite(wv)):
                return wv, f"{name}:{problem.status}"
    return None, "infeasible_or_unbounded"


def _project(w: np.ndarray, protocol: str) -> np.ndarray:
    """Project near-feasible weights onto the strict feasible set."""
    w = np.where(np.abs(w) < EPS_ACTIVE, 0.0, w)
    w = np.clip(w, -PER_NAME_CAP, PER_NAME_CAP)
    if protocol == "lo":
        w = np.maximum(w, 0.0)
        total = float(w.sum())
        if total > 0.0:
            w = w / total
        w = np.clip(w, 0.0, PER_NAME_CAP)
        total = float(w.sum())
        if total > 0.0:
            w = w / total
        w = np.clip(w, 0.0, PER_NAME_CAP)
    else:  # ls
        w = w - float(w.mean())
        l1 = float(np.abs(w).sum())
        if l1 > 1.0:
            w = w / l1
        w = np.clip(w, -PER_NAME_CAP, PER_NAME_CAP)
        w = np.where(np.abs(w) < EPS_ACTIVE, 0.0, w)
    return w


def _run_protocol(
    fold: int, seed: int,
    scores_df: pd.DataFrame, returns_df: pd.DataFrame,
    out_path: Path, protocol: str,
) -> Dict[str, object]:
    """Iterate test days; solve QP; persist non-zero weights."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    test_dates = sorted(scores_df["date"].unique())
    print(
        f"[biotech_nbi_enriched_layer2] fold={fold} seed={seed} protocol={protocol}: "
        f"{len(test_dates):,} test days",
        flush=True,
    )
    col_to_idx = {t: i for i, t in enumerate(returns_df.columns)}
    qp_total = qp_fail = 0
    active_pos: List[int] = []
    n_long: List[int] = []
    n_short: List[int] = []
    gross: List[float] = []
    net: List[float] = []
    rows: List[dict] = []
    t0 = time.time()
    last_log = t0
    for di, day in enumerate(test_dates):
        day_ts = pd.Timestamp(day).normalize()
        loc = returns_df.index.searchsorted(day_ts)
        qp_total += 1
        if loc >= len(returns_df.index) or loc - COV_LOOKBACK < 0:
            lo_i = max(0, loc - COV_LOOKBACK)
        else:
            lo_i = loc - COV_LOOKBACK
        if loc - lo_i < 5:
            qp_fail += 1
            continue
        day_scores = scores_df.loc[
            scores_df["date"] == day_ts, ["ticker", "score"]
        ].drop_duplicates(subset=["ticker"], keep="last")
        if day_scores.empty:
            qp_fail += 1
            continue
        day_scores = day_scores.set_index("ticker")["score"]
        tickers_day = [t for t in day_scores.index if t in col_to_idx]
        if len(tickers_day) < 5:
            qp_fail += 1
            continue
        day_scores = day_scores.loc[tickers_day]
        col_idx = np.array([col_to_idx[t] for t in tickers_day],
                            dtype=np.int64)
        window = returns_df.iloc[lo_i:loc, :].values[:, col_idx]
        window = np.where(np.isfinite(window), window, 0.0)
        if window.shape[0] < 5:
            qp_fail += 1
            continue
        try:
            sigma = ledoit_wolf_constant_correlation(window)
        except Exception as exc:
            print(f"[biotech_nbi_enriched_layer2] cov fail day={day_ts} err={exc}",
                  flush=True)
            qp_fail += 1
            continue
        s = day_scores.values.astype(np.float64)
        n_active = int(s.shape[0])
        k = min(TOPK_CAP, n_active)
        if k < n_active:
            order = np.argsort(-np.abs(s))
            keep = np.sort(order[:k])
            s = s[keep]
            sigma = sigma[np.ix_(keep, keep)]
            tickers_day = [tickers_day[i] for i in keep]
        w, _status = _solve_qp(s, sigma, protocol)
        if w is None:
            qp_fail += 1
            continue
        w = _project(w, protocol)
        mask = np.abs(w) >= EPS_ACTIVE
        active_pos.append(int(mask.sum()))
        n_long.append(int(((w > 0) & mask).sum()))
        n_short.append(int(((w < 0) & mask).sum()))
        gross.append(float(np.abs(w).sum()))
        net.append(float(w.sum()))
        for j in np.nonzero(mask)[0]:
            rows.append({"date": day_ts,
                         "ticker": tickers_day[int(j)],
                         "weight": float(w[int(j)])})
        now = time.time()
        if now - last_log > 30.0 or di == len(test_dates) - 1:
            print(
                f"[biotech_nbi_enriched_layer2] {protocol} fold={fold} seed={seed} "
                f"day {di + 1}/{len(test_dates)} "
                f"elapsed={now - t0:6.1f}s "
                f"fails={qp_fail}/{qp_total}",
                flush=True,
            )
            last_log = now
    if rows:
        wdf = pd.DataFrame(rows).sort_values(["date", "ticker"])
        wdf.reset_index(drop=True, inplace=True)
        wdf.to_parquet(out_path, index=False)
        print(
            f"[biotech_nbi_enriched_layer2] wrote {out_path}: {len(wdf):,} non-zero "
            f"weights across {wdf['date'].nunique()} days",
            flush=True,
        )
    else:
        print(f"[biotech_nbi_enriched_layer2] WARN no non-zero weights for {out_path}",
              flush=True)
    return {
        "fold": fold, "seed": seed, "protocol": protocol,
        "n_test_days": int(len(test_dates)),
        "qp_total": int(qp_total),
        "qp_fail": int(qp_fail),
        "qp_fail_rate": float(qp_fail / max(1, qp_total)),
        "avg_active": float(np.mean(active_pos)) if active_pos else 0.0,
        "avg_active_L": float(np.mean(n_long)) if n_long else 0.0,
        "avg_active_S": float(np.mean(n_short)) if n_short else 0.0,
        "avg_gross": float(np.mean(gross)) if gross else 0.0,
        "avg_net": float(np.mean(net)) if net else 0.0,
        "out_path": str(out_path),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True,
                   choices=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--protocol", type=str, default="both",
                   choices=["ls", "lo", "both"])
    p.add_argument("--output-dir-root", type=str,
                   default="outputs/biotech_nbi_enriched")
    p.add_argument("--panel-end", type=str, default="2025-12-31")
    args = p.parse_args()

    layer1_root = Path(args.output_dir_root) / "layer1"
    layer2_root = Path(args.output_dir_root) / "layer2"
    summary_dir = layer2_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"fold{args.fold}_seed{args.seed}.json"

    scores_path = _ensure_scores(
        fold=args.fold, seed=args.seed,
        layer1_root=layer1_root, panel_end=args.panel_end,
    )
    scores_df = pd.read_parquet(scores_path)
    scores_df["date"] = pd.to_datetime(scores_df["date"]).dt.normalize()
    returns_df = _load_returns(
        fold=args.fold, seed=args.seed, panel_end=args.panel_end,
    )

    protocols = PROTOCOLS if args.protocol == "both" else (args.protocol,)
    per_protocol: Dict[str, Dict[str, object]] = {}
    for protocol in protocols:
        out_path = (
            layer2_root / "weights" / protocol
            / f"fold{args.fold}_seed{args.seed}.parquet"
        )
        if out_path.exists():
            print(f"[biotech_nbi_enriched_layer2] {out_path} exists; reading summary",
                  flush=True)
            wdf = pd.read_parquet(out_path)
            wdf["date"] = pd.to_datetime(wdf["date"]).dt.normalize()
            gp = wdf.groupby("date")["weight"]
            per_protocol[protocol] = {
                "fold": args.fold, "seed": args.seed, "protocol": protocol,
                "n_test_days": int(wdf["date"].nunique()),
                "qp_total": int(wdf["date"].nunique()),
                "qp_fail": 0, "qp_fail_rate": 0.0,
                "avg_active": float(gp.size().mean()),
                "avg_active_L": float(gp.apply(
                    lambda x: int((x > EPS_ACTIVE).sum())).mean()),
                "avg_active_S": float(gp.apply(
                    lambda x: int((x < -EPS_ACTIVE).sum())).mean()),
                "avg_gross": float(gp.apply(
                    lambda x: float(np.abs(x).sum())).mean()),
                "avg_net": float(gp.sum().mean()),
                "out_path": str(out_path),
            }
            continue
        per_protocol[protocol] = _run_protocol(
            fold=args.fold, seed=args.seed,
            scores_df=scores_df, returns_df=returns_df,
            out_path=out_path, protocol=protocol,
        )

    payload = {
        "universe": "biotech_nbi_enriched",
        "fold": args.fold, "seed": args.seed,
        "panel_end": args.panel_end, "gamma": GAMMA,
        "per_name_cap": PER_NAME_CAP, "cov_lookback": COV_LOOKBACK,
        "topk_cap": TOPK_CAP, "per_protocol": per_protocol,
    }
    with open(summary_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"[biotech_nbi_enriched_layer2] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
