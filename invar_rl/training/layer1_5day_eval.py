"""Layer-1 ranker portfolio eval on the RAG-STAR-paper convention.

5-day non-overlapping rebalance, top-25 long / bottom-25 short
equal-weight, 5 bps round-trip transaction cost on each rebalance.
This is the convention used by the RAG-STAR universal paper Table 4
(``drafts/universal_paper_aaai/sections/07_results.tex``) so the
output number lives in the same comparison table as the lifted
MASTER / FactorVAE / StockMixer / DySTAGE / MERA / iTransformer
portfolio numbers.

Two sources of Layer-1 scores supported:
- ``--source canonical``: load the canonical InVAR full-state ckpt and
  forward through every test day. (Headline.)
- ``--source baseline {master,factorvae,...}``: pull y_hat from
  ``results/baselines_universal_two_regime_val/{baseline}/foldF_seedS_predictions.npz``
  so we can re-derive Table A entries for baselines under our
  identical pipeline as a cross-check on the lifted numbers.

Output: ``invar_rl/results/layer1_5day/{source}/foldF_seedS.json``,
schema matching stage3 with a ``methods.top25_5day`` block plus
``ir_5day_annual`` (the published-convention IR) and the standard
daily-Sharpe equivalent for cross-reference.

Annualisation:
- ir_5day_annual = mean(5-day log returns) / std(5-day log returns)
  * sqrt(252 / 5)
- daily Sharpe equivalent (for cross-reference vs Table B):
  mean(5-day log return) / std(5-day log return) * sqrt(252)

Usage::

    python -m invar_rl.training.layer1_5day_eval \
        --source canonical --fold 1 --seed 42 \
        --layer1-ckpt invar_rl/results/stage1/_ckpt/fold1_seed42_full.pt

    python -m invar_rl.training.layer1_5day_eval \
        --source baseline --baseline master --fold 1 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from src.invar import InVARConfig

from invar_rl.data.lattice_bridge import build_lattice_bridge


def _build_canonical_y_hat(
    bridge,
    ckpt_path: Path,
    day_indices: Sequence[int],
    device: torch.device,
) -> np.ndarray:
    """Forward canonical InVAR on every requested day, return (T_panel, N)
    score matrix (zeros outside the day_indices set)."""
    from invar_rl.layer1_ranker.canonical_runner import load_trained_invar
    bundle = load_trained_invar(
        ckpt_path=ckpt_path, bridge=bridge, device=device,
    )
    T, N = bridge.log_returns_1d.shape
    y = np.zeros((T, N), dtype=np.float32)
    for d in day_indices:
        try:
            out = bundle.forward_day(int(d))
        except (ValueError, RuntimeError):
            continue
        active = out["active_indices"].cpu().numpy().astype(np.int64)
        scores = out["scores"].detach().cpu().numpy().astype(np.float32)
        y[d, active] = scores
    return y


def _load_baseline_y_hat(
    baseline: str, fold: int, seed: int, npz_root: Path,
) -> np.ndarray:
    p = (
        Path(npz_root) / baseline
        / f"fold{fold}_seed{seed}_predictions.npz"
    )
    if not p.exists():
        raise FileNotFoundError(f"baseline npz not found: {p}")
    return np.load(p, allow_pickle=False)["y_hat"].astype(np.float32)


def _five_day_topk_ls(
    y_hat: np.ndarray,
    tradable: np.ndarray,
    log_returns_1d: np.ndarray,
    day_indices: Sequence[int],
    k: int = 25,
    cost_bps_roundtrip: float = 5.0,
    hold_days: int = 5,
) -> dict:
    """5-day non-overlapping top-k L/S equal-weight portfolio with cost."""
    days = list(day_indices)
    rebalance_days = list(range(0, len(days), hold_days))
    five_day_returns = []
    turnovers = []
    prev_w = None
    held_global = None
    cost_per_round = cost_bps_roundtrip / 10000.0
    for ri in rebalance_days:
        d = days[ri]
        if d + hold_days >= log_returns_1d.shape[0]:
            break
        active = np.nonzero(tradable[d])[0]
        if active.size < 2 * k:
            continue
        scores = y_hat[d, active].astype(np.float64)
        valid = np.isfinite(scores)
        if valid.sum() < 2 * k:
            continue
        scored = np.where(valid, scores, -np.inf)
        order = np.argsort(scored)
        short_local = order[:k]
        long_local = order[-k:]
        per_name = 1.0 / (2.0 * k)
        w = np.zeros(active.size, dtype=np.float64)
        w[long_local] = per_name
        w[short_local] = -per_name
        held_global_new = active.copy()
        if prev_w is None:
            turnover = 1.0
        else:
            full_prev = np.zeros(log_returns_1d.shape[1])
            full_prev[held_global] = prev_w
            full_curr = np.zeros(log_returns_1d.shape[1])
            full_curr[held_global_new] = w
            turnover = float(np.abs(full_curr - full_prev).sum())
        cost = cost_per_round * turnover
        r_window = log_returns_1d[d + 1: d + 1 + hold_days, held_global_new]
        r_window = np.where(np.isfinite(r_window), r_window, 0.0)
        # 5-day log return per stock = sum of 5 daily log returns
        r_5d = r_window.sum(axis=0)
        port_ret_5d = float((w * r_5d).sum()) - cost
        five_day_returns.append(port_ret_5d)
        turnovers.append(turnover)
        prev_w = w
        held_global = held_global_new
    arr = np.asarray(five_day_returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "ir_5day_annual": 0.0,
            "mean_return_5d": 0.0,
            "volatility_5d": 0.0,
            "sharpe_daily_equivalent": 0.0,
            "final_equity": 1.0,
            "n_5d_periods": 0,
            "mean_turnover": 0.0,
        }
    mean = float(arr.mean())
    vol = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ir_5day = (mean / vol) * np.sqrt(252.0 / 5.0) if vol > 0 else 0.0
    sharpe_daily_eq = (mean / vol) * np.sqrt(252.0) if vol > 0 else 0.0
    return {
        "ir_5day_annual": float(ir_5day),
        "mean_return_5d": float(mean),
        "volatility_5d": float(vol),
        "sharpe_daily_equivalent": float(sharpe_daily_eq),
        "final_equity": float(np.exp(arr.sum())),
        "n_5d_periods": int(arr.size),
        "mean_turnover": (
            float(np.mean(turnovers)) if turnovers else 0.0
        ),
    }


def run_one_cell(
    source: str,
    fold: int,
    seed: int,
    ckpt_path: Optional[Path],
    baseline: Optional[str],
    npz_root: Path,
    output_dir: Path,
    panel_kind: str,
    panel_end: str,
    two_regime_val: bool,
    k: int,
    cost_bps_roundtrip: float,
) -> dict:
    cfg = InVARConfig(fold=fold, seed=seed)
    cfg.panel_kind = panel_kind
    cfg.two_regime_val = two_regime_val
    cfg.panel_end = panel_end
    bridge = build_lattice_bridge(cfg)

    if source == "canonical":
        if ckpt_path is None or not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"--source canonical needs --layer1-ckpt; got {ckpt_path}"
            )
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        y_hat = _build_canonical_y_hat(
            bridge=bridge,
            ckpt_path=Path(ckpt_path),
            day_indices=list(bridge.test_idx),
            device=device,
        )
        tag = "canonical_invar"
    elif source == "baseline":
        if not baseline:
            raise ValueError("--source baseline needs --baseline")
        y_hat = _load_baseline_y_hat(
            baseline=baseline, fold=fold, seed=seed,
            npz_root=Path(npz_root),
        )
        tag = baseline
    else:
        raise ValueError(f"unknown source: {source}")

    method_key = f"top{k}_5d_nonoverlap_{int(cost_bps_roundtrip)}bps"
    res = _five_day_topk_ls(
        y_hat=y_hat,
        tradable=bridge.tradable,
        log_returns_1d=bridge.log_returns_1d,
        day_indices=list(bridge.test_idx),
        k=k,
        cost_bps_roundtrip=cost_bps_roundtrip,
        hold_days=5,
    )

    payload = {
        "source": source,
        "tag": tag,
        "fold": fold,
        "seed": seed,
        "model": (
            "Layer-1 ranker, 5-day non-overlap top-K L/S "
            f"(source={tag}, k={k}, cost={cost_bps_roundtrip}bps)"
        ),
        "n_test_days": int(len(bridge.test_idx)),
        "methods": {method_key: res},
        "config": {
            "panel_kind": panel_kind,
            "two_regime_val": two_regime_val,
            "panel_end": panel_end,
            "k_long": k,
            "k_short": k,
            "cost_bps_roundtrip": cost_bps_roundtrip,
            "hold_days": 5,
            "annualisation": "sqrt(252/5)",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"fold{fold}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(
        f"[layer1_5day_eval source={source} tag={tag}] "
        f"fold={fold} seed={seed} ir_5day_annual={res['ir_5day_annual']:+.3f} "
        f"sharpe_daily_eq={res['sharpe_daily_equivalent']:+.3f} "
        f"n_periods={res['n_5d_periods']} -> {out_path}"
    )
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Layer-1 5-day non-overlap top-K L/S portfolio eval."
    )
    p.add_argument(
        "--source", type=str, required=True,
        choices=["canonical", "baseline"],
    )
    p.add_argument("--fold", type=int, required=True, choices=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--layer1-ckpt", type=str, default=None)
    p.add_argument("--baseline", type=str, default=None)
    p.add_argument(
        "--npz-root", type=str,
        default="results/baselines_universal_two_regime_val",
    )
    p.add_argument(
        "--output-dir-root", type=str,
        default="invar_rl/results/layer1_5day",
    )
    p.add_argument(
        "--panel_kind", type=str, default="lattice_native",
        choices=["biotech", "lattice_native"],
    )
    p.add_argument("--panel_end", type=str, default="2025-12-31")
    p.add_argument("--two_regime_val", action="store_true", default=True)
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--cost-bps-roundtrip", type=float, default=5.0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.source == "canonical":
        out_dir = Path(args.output_dir_root) / "canonical"
    else:
        out_dir = Path(args.output_dir_root) / args.baseline
    run_one_cell(
        source=args.source,
        fold=args.fold,
        seed=args.seed,
        ckpt_path=Path(args.layer1_ckpt) if args.layer1_ckpt else None,
        baseline=args.baseline,
        npz_root=Path(args.npz_root),
        output_dir=out_dir,
        panel_kind=args.panel_kind,
        panel_end=args.panel_end,
        two_regime_val=args.two_regime_val,
        k=args.k,
        cost_bps_roundtrip=args.cost_bps_roundtrip,
    )


if __name__ == "__main__":
    main()
