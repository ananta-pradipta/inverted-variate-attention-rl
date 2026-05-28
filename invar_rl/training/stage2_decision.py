"""Stage 2: decision-focused training.

Composes Layer 1 (the InVAR ranker) and the Layer 2 mean-variance QP into a
single differentiable pipeline: features go in, scores come out of Layer 1,
weights come out of Layer 2. The training loss is the negative realised
return of the Layer 2 portfolio on the day's realised stock returns, with an
optional portfolio-variance penalty; its gradient flows through the QP into
Layer 1.

Two training variants are selected by configuration. Variant A fine-tunes
only the Layer 1 score head and freezes the rest of Layer 1. Variant B
fine-tunes all of Layer 1. A pure-ranking control (Stage 1 only, no
decision-focused training) is evaluated through the same Layer 2 allocation
at inference, so the first research question can be addressed in preliminary
form. Runs across the seed set and the walk-forward folds, all on Wulver.

Covariance is estimated from a trailing window of realised one-day returns
strictly before the decision day, assembled only through the data contract,
so no future information enters the estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from invar_rl.common.config import (
    BaseConfig,
    Layer1Config,
    Layer2Config,
    Stage2Config,
    load_base_config,
    load_folds_config,
    load_layer1_config,
    load_layer2_config,
    load_stage2_config,
)
from invar_rl.common.logging_utils import get_logger
from invar_rl.common.seeding import set_global_seed
from invar_rl.common.splits import FoldSplit
from invar_rl.data.contract import PanelDataContract
from invar_rl.data.panel_factory import build_panel, build_splits
from invar_rl.layer1_ranker.invar import INVAR
from invar_rl.layer2_alloc.covariance import estimate_covariance
from invar_rl.layer2_alloc.qp_layer import MeanVarianceQP

LOGGER = get_logger(__name__)


def _ticker_col_map(panel: PanelDataContract) -> dict:
    """Panel-agnostic ticker -> global column index, cached on the panel.

    Works for any ticker universe (real symbols or synthetic names); does
    not rely on a naming convention.
    """
    m = getattr(panel, "_tli_tmap", None)
    if m is None:
        m = {t: i for i, t in enumerate(panel.all_tickers())}
        panel._tli_tmap = m
    return m


def build_one_day_return_matrix(
    panel: PanelDataContract,
) -> np.ndarray:
    """Assemble a (n_days, n_tickers) realised one-day return matrix.

    Built only through the data contract (``realized_returns`` per day), so
    later phases stay decoupled from any raw data format. Entry [t, i] is the
    realised return over (t, t+1] for global ticker i, NaN where inactive or
    past the panel end.
    """
    n_days = len(panel.trading_days())
    t2c = _ticker_col_map(panel)
    mat = np.full((n_days, len(t2c)), np.nan, dtype=np.float64)
    for tau in range(n_days):
        tickers, rets = panel.realized_returns(tau, horizon=1)
        for name, val in zip(tickers, rets):
            mat[tau, t2c[name]] = val
    return mat


def covariance_for_day(
    ret1_full: np.ndarray,
    active_global_idx: np.ndarray,
    day_index: int,
    lookback: int,
    train_start: int,
    layer2: Layer2Config,
) -> np.ndarray:
    """Estimate Sigma from past one-day returns of the active tickers.

    The window is [max(train_start, day_index - lookback), day_index - 1],
    strictly before the decision day and clamped to the fold's training
    start, so no future or pre-fold information enters.
    """
    lo = max(train_start, day_index - lookback)
    hi = day_index  # exclusive: rows lo .. day_index-1
    window = ret1_full[lo:hi, :][:, active_global_idx]
    window = np.where(np.isfinite(window), window, 0.0)
    if window.shape[0] < 2:
        n = active_global_idx.shape[0]
        return np.eye(n, dtype=np.float64)
    return estimate_covariance(
        window, layer2.estimator, layer2.factor_rank
    )


def _set_variant_trainable(model: INVAR, variant: str) -> None:
    """Freeze parameters according to the decision-focused variant."""
    if variant == "A":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("score_head")
    else:  # "B": all of Layer 1 is trainable
        for p in model.parameters():
            p.requires_grad = True


def _load_stage1_into(
    model: INVAR, ckpt_dir: Path, fold: str, seed: int
) -> bool:
    """Warm-start Layer 1 from a Stage 1 checkpoint if one exists."""
    path = ckpt_dir / f"layer1_{fold}_seed{seed}.pt"
    if not path.is_file():
        return False
    blob = torch.load(path, map_location="cpu")
    model.load_state_dict(blob["state_dict"])
    return True


def _day_inputs(
    panel: PanelDataContract, day: int
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """Features, macro, active global indices, horizon returns, finite mask."""
    tickers, feats = panel.feature_window(day)
    macro = panel.macro_vector(day)
    t2c = _ticker_col_map(panel)
    g_idx = np.array([t2c[t] for t in tickers])
    _, fwd = panel.realized_returns(day, horizon=panel.label_horizon)
    finite = np.isfinite(fwd)
    return (
        torch.from_numpy(feats).float(),
        torch.from_numpy(macro).float(),
        g_idx,
        fwd,
        finite,
    )


def _portfolio_return(
    weights: torch.Tensor, fwd: np.ndarray, finite: np.ndarray
) -> torch.Tensor:
    """Realised portfolio return, ignoring names with no label that day."""
    r = torch.from_numpy(np.where(finite, fwd, 0.0)).to(weights.dtype)
    return weights @ r


def _evaluate(
    model: INVAR,
    qp: MeanVarianceQP,
    panel: PanelDataContract,
    ret1_full: np.ndarray,
    days: Sequence[int],
    lookback: int,
    train_start: int,
    layer2: Layer2Config,
) -> Dict[str, float]:
    """Mean and volatility of the realised daily portfolio return."""
    model.eval()
    warmup = panel.lookback - 1
    rets: List[float] = []
    with torch.no_grad():
        for day in days:
            if day < warmup:
                continue
            feats, macro, g_idx, fwd, finite = _day_inputs(panel, day)
            if finite.sum() < 5:
                continue
            sigma_np = covariance_for_day(
                ret1_full, g_idx, day, lookback, train_start, layer2
            )
            sigma = torch.from_numpy(sigma_np).float()
            scores = model(feats, macro).scores
            weights, _ = qp(scores, sigma)
            rets.append(float(_portfolio_return(weights, fwd, finite)))
    if not rets:
        return {"mean_return": 0.0, "volatility": 0.0, "n_days": 0}
    arr = np.asarray(rets)
    return {
        "mean_return": float(arr.mean()),
        "volatility": float(arr.std()),
        "n_days": len(rets),
    }


def train_cell(
    base: BaseConfig,
    layer1: Layer1Config,
    layer2: Layer2Config,
    stage2: Stage2Config,
    fold: FoldSplit,
    seed: int,
    variant: str,
) -> Dict[str, float]:
    """Decision-focused training for one (fold, seed, variant)."""
    set_global_seed(seed)
    panel = build_panel(
        base, seed=seed, train_end_index=int(fold.train_idx[-1])
    )
    ret1_full = build_one_day_return_matrix(panel)
    lookback = layer2.cov_lookback
    train_start = int(fold.train_idx[0])

    model = INVAR(
        layer1.model,
        n_features=panel.n_features,
        lookback=panel.lookback,
        macro_dim=panel.macro_dim,
    )
    ckpt_dir = Path(base.paths.checkpoint_dir)
    warm = (
        _load_stage1_into(model, ckpt_dir, fold.name, seed)
        if stage2.init_from_stage1
        else False
    )
    _set_variant_trainable(model, variant)
    qp = MeanVarianceQP(layer2)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=stage2.learning_rate,
        weight_decay=stage2.weight_decay,
    )

    warmup = panel.lookback - 1
    train_days = [
        d
        for d in list(fold.train_idx)[:: stage2.train_day_stride]
        if d >= warmup
    ]
    for epoch in range(stage2.epochs):
        model.train()
        running = 0.0
        n_steps = 0
        for day in train_days:
            feats, macro, g_idx, fwd, finite = _day_inputs(panel, day)
            if finite.sum() < 5:
                continue
            sigma_np = covariance_for_day(
                ret1_full, g_idx, day, lookback, train_start, layer2
            )
            sigma = torch.from_numpy(sigma_np).float()
            scores = model(feats, macro).scores
            weights, summary = qp(scores, sigma)
            port_ret = _portfolio_return(weights, fwd, finite)
            variance = weights @ (sigma @ weights)
            loss = -port_ret + stage2.variance_penalty * variance
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, stage2.grad_clip)
            optimizer.step()
            running += float(loss.detach())
            n_steps += 1
        LOGGER.info(
            "variant %s fold %s seed %d epoch %d: mean_loss %.6f "
            "(warm_start=%s)",
            variant,
            fold.name,
            seed,
            epoch,
            running / max(1, n_steps),
            warm,
        )

    test_perf = _evaluate(
        model, qp, panel, ret1_full, list(fold.test_idx),
        lookback, train_start, layer2,
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = (
        ckpt_dir / f"stage2_{variant}_{fold.name}_seed{seed}.pt"
    )
    torch.save(
        {
            "state_dict": {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            },
            "variant": variant,
            "fold": fold.name,
            "seed": seed,
            "test_perf": test_perf,
        },
        ckpt_path,
    )
    LOGGER.info(
        "variant %s fold %s seed %d: test mean_return %.6f vol %.6f",
        variant,
        fold.name,
        seed,
        test_perf["mean_return"],
        test_perf["volatility"],
    )
    return test_perf


def control_cell(
    base: BaseConfig,
    layer1: Layer1Config,
    layer2: Layer2Config,
    stage2: Stage2Config,
    fold: FoldSplit,
    seed: int,
) -> Dict[str, float]:
    """Pure-ranking control: Stage 1 Layer 1 evaluated through Layer 2."""
    set_global_seed(seed)
    panel = build_panel(
        base, seed=seed, train_end_index=int(fold.train_idx[-1])
    )
    ret1_full = build_one_day_return_matrix(panel)
    model = INVAR(
        layer1.model,
        n_features=panel.n_features,
        lookback=panel.lookback,
        macro_dim=panel.macro_dim,
    )
    _load_stage1_into(
        model, Path(base.paths.checkpoint_dir), fold.name, seed
    )
    qp = MeanVarianceQP(layer2)
    return _evaluate(
        model, qp, panel, ret1_full, list(fold.test_idx),
        layer2.cov_lookback, int(fold.train_idx[0]), layer2,
    )


def run(
    base_path: str,
    layer1_path: str,
    layer2_path: str,
    stage2_path: str,
    folds_path: str,
    seeds: Optional[List[int]],
    fold_names: Optional[List[str]],
) -> Dict[str, dict]:
    """Run the control and both decision-focused variants; build the table."""
    base = load_base_config(base_path)
    layer1 = load_layer1_config(layer1_path)
    layer2 = load_layer2_config(layer2_path)
    stage2 = load_stage2_config(stage2_path)
    splits = build_splits(base, folds_path)
    if fold_names:
        splits = [s for s in splits if s.name in set(fold_names)]
    use_seeds = seeds or base.seeds

    raw: List[dict] = []
    for seed in use_seeds:
        for fold in splits:
            if stage2.run_control:
                c = control_cell(base, layer1, layer2, stage2, fold, seed)
                raw.append(
                    {"method": "control", "fold": fold.name,
                     "seed": seed, **c}
                )
            for variant in ("A", "B"):
                p = train_cell(
                    base, layer1, layer2, stage2, fold, seed, variant
                )
                raw.append(
                    {"method": f"variant_{variant}", "fold": fold.name,
                     "seed": seed, **p}
                )

    table: Dict[str, dict] = {}
    for fold in splits:
        table[fold.name] = {}
        for method in ("control", "variant_A", "variant_B"):
            cells = [
                r for r in raw
                if r["fold"] == fold.name and r["method"] == method
            ]
            if not cells:
                continue
            mr = np.mean([c["mean_return"] for c in cells])
            vol = np.mean([c["volatility"] for c in cells])
            table[fold.name][method] = {
                "mean_return": float(mr),
                "volatility": float(vol),
                "n_seeds": len(cells),
            }

    out_dir = Path(base.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stage2_comparison.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({"per_fold": table, "raw": raw}, fh, indent=2)

    LOGGER.info("Stage 2 comparison (per fold, seed-mean):")
    for fname, methods in table.items():
        for method, m in methods.items():
            LOGGER.info(
                "  %s %s: mean_return %.6f volatility %.6f (n_seeds %d)",
                fname,
                method,
                m["mean_return"],
                m["volatility"],
                m["n_seeds"],
            )
    LOGGER.info("Stage 2 comparison written to %s", out_path)
    return {"per_fold": table}


def _parse_args() -> argparse.Namespace:
    cfg = Path(__file__).resolve().parents[2] / "configs"
    p = argparse.ArgumentParser(description="Stage 2: decision-focused.")
    p.add_argument("--base", default=str(cfg / "base.yaml"))
    p.add_argument("--layer1", default=str(cfg / "layer1.yaml"))
    p.add_argument("--layer2", default=str(cfg / "layer2.yaml"))
    p.add_argument("--stage2", default=str(cfg / "stage2.yaml"))
    p.add_argument("--folds", default=str(cfg / "folds.yaml"))
    p.add_argument("--seeds", default=None)
    p.add_argument("--fold-names", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    fold_names = (
        [s for s in args.fold_names.split(",")]
        if args.fold_names
        else None
    )
    run(
        args.base,
        args.layer1,
        args.layer2,
        args.stage2,
        args.folds,
        seeds,
        fold_names,
    )
    LOGGER.info("Stage 2 decision-focused training complete")


if __name__ == "__main__":
    main()
