"""NASDAQ-100 InVAR Layer 1 + top-25 L/S wrapper eval (Phase 5.5 row).

Companion to :mod:`invar_rl.training.nasdaq100_baseline_eval`. For each
``(fold, seed)`` cell on NDX-100, this script:

  1. Builds the NASDAQ-100 ``lattice_bridge`` for the fold (canonical
     panel routing, tradable mask, 1-day log-return panel).
  2. Loads the canonical InVAR Layer-1 full state_dict from
     ``outputs/nasdaq100/layer1/_ckpt/fold{F}_seed{S}_full.pt``
     (written by ``invar_rl/training/nasdaq100_layer1_eval.py`` when
     env ``INVAR_SAVE_FULL_STATE=1``).
  3. Forwards the trained InvarSTXModel over every test day, building
     the full (T, N) score matrix ``y_hat`` with NaN at non-active cells.
  4. Applies the SAME top-25 long / bottom-25 short wrapper used for
     MASTER / FactorVAE / etc. (gross = 2.0, net = 0.0, daily rebalance,
     equal weight, dollar-neutral).
  5. Writes
     ``outputs/nasdaq100/baselines/invar_l1/fold{F}_seed{S}.json``
     with the same JSON schema as the other Phase 5.5 baselines, so the
     existing rollup picks it up uncritically.

The purpose is to isolate two contributions in the NDX-100 audit:

  * InVAR Layer 1 RANKER QUALITY (this row, top-25 L/S wrapper, no QP,
    no SAC). Directly comparable to MASTER / FactorVAE / DySTAGE /
    StockMixer / SWA-InVAR rows.
  * InVAR-RL full stack VALUE-ADD (Layer 3 SAC row in
    ``reports/nasdaq100/phase_5_layer3_sharpe.md``, mean-variance QP +
    learned exposure). Difference vs the L1+wrapper row is the marginal
    contribution of the Layer 2 QP and Layer 3 SAC controller on NDX-100.

Usage (per cell)::

    PYTHONPATH=$PWD python3 -m invar_rl.training.nasdaq100_invar_l1_wrapper_eval \\
        --fold 1 --seed 42

Usage (sweep all 5 seeds within one fold via single bridge build)::

    PYTHONPATH=$PWD python3 -m invar_rl.training.nasdaq100_invar_l1_wrapper_eval \\
        --fold 1 --sweep-fold

Usage (sweep all 25 cells, single process)::

    PYTHONPATH=$PWD python3 -m invar_rl.training.nasdaq100_invar_l1_wrapper_eval \\
        --sweep-all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from invar_rl.data.lattice_bridge import build_lattice_bridge
from invar_rl.training.nasdaq100_baseline_eval import (
    _topk_long_short_portfolio,
    _TOP_K_LS,
)


_BASELINE_NAME: str = "invar_l1"
_TOP_K_LS_NATIVE: int = _TOP_K_LS  # 25, symmetric with other Phase 5.5 rows.


def _build_bridge(fold: int, panel_end: str, two_regime_val: bool,
                  device: torch.device):
    """Build the NDX-100 lattice bridge for one fold (seed-agnostic)."""
    from src.baselines.train_invar_stx_v2 import InvarSTXV2Config
    cfg = InvarSTXV2Config(fold=fold, seed=42)
    cfg.panel_kind = "nasdaq100"
    cfg.two_regime_val = two_regime_val
    cfg.panel_end = panel_end
    cfg.enable_retrieval_bank = False
    return build_lattice_bridge(cfg, device=device)


def _load_trained_invar(
    fold: int,
    seed: int,
    output_dir_root: Path,
    bridge,
    panel_end: str,
    two_regime_val: bool,
    device: torch.device,
):
    """Load the canonical InVAR full state_dict and populate day memory."""
    from src.baselines.train_invar_stx_v2 import (
        InvarSTXModel, InvarSTXV2Config,
    )

    cfg = InvarSTXV2Config(fold=fold, seed=seed)
    cfg.panel_kind = "nasdaq100"
    cfg.two_regime_val = two_regime_val
    cfg.panel_end = panel_end
    cfg.output_dir = str(output_dir_root)
    cfg.enable_retrieval_bank = False
    # day_value_dim is inferred from the bridge (panel shape only).
    cfg.day_value_dim = int(bridge.day_values.shape[1])

    full_path = (
        Path(cfg.output_dir) / "_ckpt"
        / f"fold{fold}_seed{seed}_full.pt"
    )
    if not full_path.exists():
        raise FileNotFoundError(
            f"InVAR Layer-1 full ckpt not found: {full_path}. "
            f"Run nasdaq100_layer1_eval.py with INVAR_SAVE_FULL_STATE=1 "
            f"to produce it."
        )
    ckpt = torch.load(full_path, map_location=device)

    model = InvarSTXModel(
        cfg,
        n_features=int(ckpt["n_features"]),
        day_key_dim=int(ckpt["day_key_dim"]),
        duration_input_dim=int(ckpt["duration_input_dim"]),
        macro_input_dim=int(ckpt["macro_input_dim"]),
        macro_gate_in_dim=int(ckpt["macro_gate_in_dim"]),
    ).to(device)
    model.day_memory.populate(
        keys=bridge.day_keys, values=bridge.day_values,
        day_indices=np.arange(len(bridge.dates)),
        train_day_indices=bridge.train_idx,
    )
    model.day_memory.to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def _build_score_matrix(model, bridge, device: torch.device) -> np.ndarray:
    """Forward the trained InVAR model over every test day.

    Returns a (T, N) float32 score matrix with NaN at (day, ticker) pairs
    that are not in the test segment or not active that day. Matches the
    npz ``y_hat`` schema consumed by the Phase 5.5 baseline harness.
    """
    T = bridge.log_returns_1d.shape[0]
    N = bridge.log_returns_1d.shape[1]
    y_hat = np.full((T, N), np.nan, dtype=np.float32)
    with torch.no_grad():
        for t in bridge.test_idx:
            t = int(t)
            try:
                inp = bridge.day_inputs(t)
            except (ValueError, RuntimeError):
                continue
            active = inp["active_indices"].cpu().numpy().astype(np.int64)
            x_window = inp["x_window"].to(device)
            day_query_key = inp["day_query_key"].to(device)
            allowed = inp["allowed_day_indices"].to(device)
            regime_scalars = inp["regime_scalars"].to(device)
            duration_input = inp["duration_input"].to(device)
            macro_input = inp["macro_input"].to(device)
            macro_gate_input = inp["macro_gate_input"].to(device)

            y_active = model(
                x_window,
                day_query_key=day_query_key,
                query_day_idx=t,
                allowed_day_indices=allowed,
                regime_scalars=regime_scalars,
                duration_input=duration_input,
                macro_input=macro_input,
                macro_gate_input=macro_gate_input,
            ).detach().float().cpu().numpy().astype(np.float32)
            y_hat[t, active] = y_active
    return y_hat


def run_one_cell(
    fold: int,
    seed: int,
    output_dir: Path,
    layer1_root: Path,
    panel_end: str,
    two_regime_val: bool,
    bridge=None,
    device: Optional[torch.device] = None,
) -> dict:
    """Evaluate one (fold, seed) cell under the top-25 L/S wrapper.

    Skips silently if the output JSON already exists.
    """
    out_path = output_dir / f"fold{fold}_seed{seed}.json"
    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if bridge is None:
        bridge = _build_bridge(
            fold=fold, panel_end=panel_end,
            two_regime_val=two_regime_val, device=device,
        )

    model = _load_trained_invar(
        fold=fold, seed=seed, output_dir_root=layer1_root,
        bridge=bridge, panel_end=panel_end,
        two_regime_val=two_regime_val, device=device,
    )
    y_hat = _build_score_matrix(model, bridge, device=device)

    # tradable + log_returns: bridge.tradable is (T, N) bool; we pass a
    # numpy view to match the baseline harness signature.
    tradable = np.asarray(bridge.tradable)
    log_returns = bridge.log_returns_1d
    if y_hat.shape != log_returns.shape:
        raise ValueError(
            f"invar_l1 y_hat shape {y_hat.shape} "
            f"!= bridge log_returns shape {log_returns.shape}"
        )

    res_ls = _topk_long_short_portfolio(
        y_hat=y_hat, tradable=tradable, log_returns=log_returns,
        day_indices=list(bridge.test_idx), k=_TOP_K_LS_NATIVE,
    )

    print(
        f"  [invar_l1] fold={fold} seed={seed} "
        f"L/S(k={_TOP_K_LS_NATIVE}) sharpe={res_ls['sharpe_annualised']:+.3f}"
    )

    payload = {
        "baseline": _BASELINE_NAME,
        "universe": "nasdaq100",
        "fold": fold,
        "seed": seed,
        "n_test_days": int(len(bridge.test_idx)),
        "top_k_ls": _TOP_K_LS_NATIVE,
        "top_k_native": _TOP_K_LS_NATIVE,
        "sharpe_ls": res_ls["sharpe_annualised"],
        # Mirror the L-only-native key so the existing baseline rollup
        # in nasdaq100_baseline_eval._rollup picks this row up without
        # special-casing. The wrapper is L/S only (no native long-only
        # protocol for InVAR Layer 1), so we re-use the L/S Sharpe in
        # both fields for schema parity.
        "sharpe_lo_native": res_ls["sharpe_annualised"],
        "methods": {
            f"long_short_top{_TOP_K_LS_NATIVE}_wrapper": res_ls,
        },
        "config": {
            "panel_kind": "nasdaq100",
            "two_regime_val": two_regime_val,
            "panel_end": panel_end,
            "layer1_ckpt_root": str(layer1_root),
            "note": (
                "InVAR Layer 1 (canonical bankless + clpretrain) + "
                "equal-weight top-25 L/S wrapper. No QP, no SAC. "
                "Isolates ranker quality on NDX-100."
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[invar-l1-wrapper] wrote {out_path}")
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "NDX-100 Phase 5.5 row: InVAR Layer 1 + top-25 L/S wrapper "
            "(no QP, no SAC). Isolates ranker quality."
        )
    )
    p.add_argument("--fold", type=int, choices=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int)
    p.add_argument(
        "--layer1-root", type=str,
        default="outputs/nasdaq100/layer1",
        help="Layer-1 output root (must contain _ckpt/foldF_seedS_full.pt).",
    )
    p.add_argument(
        "--output-dir-root", type=str,
        default="outputs/nasdaq100/baselines",
        help="Where to write per-(fold, seed) JSONs.",
    )
    p.add_argument("--panel-end", type=str, default="2025-12-31")
    p.add_argument(
        "--two-regime-val", action="store_true", default=True,
        help="Use canonical fixed val (2017 H2 + 2018 H2).",
    )
    p.add_argument(
        "--sweep-fold", action="store_true",
        help=(
            "Run all 5 seeds (42..46) for the given --fold within a "
            "single bridge build."
        ),
    )
    p.add_argument(
        "--sweep-all", action="store_true",
        help="Run all 5 folds x 5 seeds (25 cells) in one process.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir_root) / _BASELINE_NAME
    layer1_root = Path(args.layer1_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[invar-l1-wrapper] device={device}")

    if args.sweep_all:
        for fold in (1, 2, 3, 4, 5):
            bridge = _build_bridge(
                fold=fold, panel_end=args.panel_end,
                two_regime_val=args.two_regime_val, device=device,
            )
            for seed in (42, 43, 44, 45, 46):
                try:
                    run_one_cell(
                        fold=fold, seed=seed, output_dir=output_dir,
                        layer1_root=layer1_root,
                        panel_end=args.panel_end,
                        two_regime_val=args.two_regime_val,
                        bridge=bridge, device=device,
                    )
                except FileNotFoundError as exc:
                    print(f"[invar-l1-wrapper] WARN fold={fold} seed={seed}: "
                          f"{exc}")
        return 0

    if args.fold is None:
        raise SystemExit(
            "Per-cell mode requires --fold (add --seed for one seed, "
            "--sweep-fold for all 5 seeds, or --sweep-all for all 25 cells)."
        )

    if args.sweep_fold:
        bridge = _build_bridge(
            fold=args.fold, panel_end=args.panel_end,
            two_regime_val=args.two_regime_val, device=device,
        )
        for seed in (42, 43, 44, 45, 46):
            try:
                run_one_cell(
                    fold=args.fold, seed=seed, output_dir=output_dir,
                    layer1_root=layer1_root,
                    panel_end=args.panel_end,
                    two_regime_val=args.two_regime_val,
                    bridge=bridge, device=device,
                )
            except FileNotFoundError as exc:
                print(f"[invar-l1-wrapper] WARN seed={seed}: {exc}")
        return 0

    if args.seed is None:
        raise SystemExit(
            "Per-cell mode requires --seed (or --sweep-fold for all 5 seeds, "
            "or --sweep-all for all 25 cells)."
        )

    run_one_cell(
        fold=args.fold, seed=args.seed, output_dir=output_dir,
        layer1_root=layer1_root,
        panel_end=args.panel_end,
        two_regime_val=args.two_regime_val,
        device=device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
