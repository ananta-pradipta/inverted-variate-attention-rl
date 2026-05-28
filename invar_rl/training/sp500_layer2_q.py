"""S&P 500 Layer 2 Q driver: twin QuantileCritic + CVaR-blend SAC.

Mirrors :mod:`invar_rl.training.sp500_layer2_sia` with two changes:

1. The SAC agent is replaced with
   :class:`invar_rl.layer2_q.sac_q.SACQ`. The SB3 squashed-Gaussian
   actor on the full observation is unchanged; only the twin scalar
   Q-critics are replaced with twin
   :class:`invar_rl.layer2_q.quantile_critic.QuantileCritic` heads,
   and the actor objective is replaced with the mean / CVaR blend.

2. No regime-label / sparse-gate / asymmetric-critic CLI flags; those
   are SIA-specific. The Q driver exposes ``--n-quantiles``,
   ``--alpha-cvar``, and ``--eta-blend`` for the Phase 4 ablation but
   defaults to the Phase 0 plan settings (51 / 0.1 / 0.5).

All other plumbing (Layer 1 ckpt, observation pipeline, val-Sharpe
selector, test rollout) is byte-for-byte equivalent so the Q vs
canonical SAC delta isolates the critic + actor objective change.

CLI::

    python -m invar_rl.training.sp500_layer2_q \\
        --fold 1 --seed 42 \\
        --total-timesteps 20000 \\
        --output-dir-root outputs/sp500/layer2_q/smoke

Phase 1 (smoke): one (fold, seed) cell, default Q hyperparams. Phase 2
is the 25-cell 5-fold-by-5-seed sweep; that requires user approval and
is not run from this driver alone.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.invar import InVARConfig

from invar_rl.common.config import (
    Layer2Config,
    load_layer3_config,
    load_stage3_config,
)
from invar_rl.common.seeding import set_global_seed
from invar_rl.data.lattice_bridge import build_lattice_bridge
from invar_rl.layer1_ranker.canonical_runner import load_trained_invar
from invar_rl.layer2_q.config import QConfig
from invar_rl.layer2_q.sac_q import SACQ
from invar_rl.layer3_control.env import ExposureEnv
from invar_rl.layer3_control.precompute_canonical import (
    precompute_tape_canonical,
)
from invar_rl.training._universe_setup import universe_setup

_TRADING_DAYS: int = 252
_K_WRAPPER: int = 50  # canonical SP500 wrapper K, matches SIA + canonical SAC.
_DEFAULT_OUTPUT_ROOT: str = "outputs/sp500/layer2_q"


def _build_layer2_cfg() -> Layer2Config:
    """Layer 2 config used by the canonical equal-weight wrapper."""
    return Layer2Config(
        estimator="ledoit_wolf",
        factor_rank=10,
        cov_lookback=60,
        risk_aversion=1.0,
        per_name_bound=0.10,
        gross_leverage=1.0,
        topk_k=_K_WRAPPER,
        topk_temperature=1.0,
        topk_temperature_anneal=False,
    )


def _rollout(env: ExposureEnv, agent) -> Tuple[np.ndarray, np.ndarray]:
    """One deterministic rollout; return (returns, exposures)."""
    obs, _ = env.reset(seed=0)
    rets: List[float] = []
    exps: List[float] = []
    while True:
        action, _ = agent.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, _, term, trunc, info = env.step(action)
        rets.append(float(info["strategy_return"]))
        exps.append(float(info["exposure"]))
        if term or trunc:
            break
    return (
        np.asarray(rets, dtype=np.float64),
        np.asarray(exps, dtype=np.float64),
    )


def _pooled_sharpe(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    sd = float(returns.std(ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(returns.mean() / sd * np.sqrt(_TRADING_DAYS))


class _ValSharpeSelector:
    """Keep the in-memory checkpoint with the best validation Sharpe.

    Saves only the SB3 actor + QuantileCritic state_dicts so the restore
    path does NOT round-trip through SB3's ``save`` / ``load`` pickle
    pipeline (which mismatches the SACQ __init__ signature).
    """

    def __init__(self, val_env: ExposureEnv, eval_freq: int) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        self._val_env = val_env
        self._eval_freq = int(max(1, eval_freq))
        self.best_sharpe = -np.inf
        self.best_step = 0
        self.eval_history: List[Dict[str, float]] = []
        self._best_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

        outer = self

        class _Inner(BaseCallback):
            def _on_step(self_inner) -> bool:  # type: ignore[override]
                if self_inner.num_timesteps % outer._eval_freq != 0:
                    return True
                rets, _ = _rollout(outer._val_env, self_inner.model)
                sh = _pooled_sharpe(rets)
                outer.eval_history.append({
                    "step": int(self_inner.num_timesteps),
                    "val_sharpe": float(sh),
                    "val_mean": float(rets.mean()) if rets.size else 0.0,
                    "val_std": (
                        float(rets.std(ddof=1)) if rets.size > 1 else 0.0
                    ),
                })
                if sh > outer.best_sharpe:
                    outer.best_sharpe = float(sh)
                    outer.best_step = int(self_inner.num_timesteps)
                    outer._best_state = {
                        "actor": copy.deepcopy(
                            self_inner.model.actor.state_dict()
                        ),
                        "critic": copy.deepcopy(
                            self_inner.model.critic.state_dict()
                        ),
                        "critic_target": copy.deepcopy(
                            self_inner.model.critic_target.state_dict()
                        ),
                    }
                return True

        self.callback = _Inner()

    def restore_best(self, agent: SACQ) -> bool:
        if self._best_state is None:
            return False
        agent.actor.load_state_dict(self._best_state["actor"])
        agent.critic.load_state_dict(self._best_state["critic"])
        agent.critic_target.load_state_dict(
            self._best_state["critic_target"]
        )
        return True


def _persist_test_outputs(
    tape,
    rets: np.ndarray,
    exps: np.ndarray,
    bridge,
    out_path: Path,
) -> Dict[str, object]:
    """Write the per-test-day exposures + realised returns parquet."""
    n = int(min(tape.days.shape[0], rets.shape[0], exps.shape[0]))
    if n == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=[
            "date", "exposure", "strategy_return", "base_return",
        ]).to_parquet(out_path, index=False)
        return {"n_test_days": 0, "sharpe": 0.0, "mean": 0.0, "std": 0.0}
    day_indices = tape.days[:n].astype(int)
    base_ret = tape.base_return[:n].astype(np.float64)
    dates = np.asarray([str(bridge.dates[int(d)]) for d in day_indices])
    df = pd.DataFrame({
        "date": pd.to_datetime(dates).normalize(),
        "exposure": exps[:n].astype(np.float64),
        "strategy_return": rets[:n].astype(np.float64),
        "base_return": base_ret,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return {
        "n_test_days": int(df.shape[0]),
        "sharpe": float(_pooled_sharpe(rets[:n])),
        "mean": float(rets[:n].mean()),
        "std": float(rets[:n].std(ddof=1)) if n > 1 else 0.0,
    }


def run_one_cell(
    fold: int,
    seed: int,
    ckpt_path: Path,
    layer3_yaml: Path,
    stage3_yaml: Path,
    q_config: QConfig,
    eval_freq: int,
    output_dir_root: Path,
    panel_end: str,
    panel_kind: str = "lattice_native",
    two_regime_val: bool = True,
    device: Optional[torch.device] = None,
    universe_label: str = "sp500",
    long_only: bool = False,
) -> Dict[str, object]:
    """Train the Q SAC for one (fold, seed) cell on the SP500 panel."""
    out_dir = output_dir_root
    out_path = out_dir / f"fold{fold}_seed{seed}.parquet"
    summary_dir = out_dir / "summary"
    summary_path = summary_dir / f"fold{fold}_seed{seed}.json"
    if out_path.exists() and summary_path.exists():
        print(
            f"[layer2_q] {out_path} + summary exist; skipping cell",
            flush=True,
        )
        with open(summary_path) as fh:
            return json.load(fh)

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    set_global_seed(seed)
    layer2 = _build_layer2_cfg()
    layer3 = load_layer3_config(str(layer3_yaml))
    stage3 = load_stage3_config(str(stage3_yaml))

    cfg = InVARConfig(fold=fold, seed=seed)
    cfg.panel_kind = panel_kind
    cfg.two_regime_val = two_regime_val
    cfg.panel_end = panel_end
    cfg.enable_retrieval_bank = False
    bridge = build_lattice_bridge(cfg)
    bundle = load_trained_invar(
        ckpt_path=ckpt_path, bridge=bridge, device=device
    )

    print(
        f"[layer2_q] fold={fold} seed={seed} panel_kind={panel_kind} "
        f"device={device} wrapper_K={_K_WRAPPER} "
        f"n_quantiles={q_config.n_quantiles} "
        f"alpha_cvar={q_config.alpha_cvar} "
        f"eta_blend={q_config.eta_blend} "
        f"actor_hidden={list(q_config.actor_hidden)} "
        f"critic_hidden={list(q_config.critic_hidden)}",
        flush=True,
    )
    t0 = time.time()

    train_tape = precompute_tape_canonical(
        bundle=bundle, bridge=bridge,
        day_indices=list(bridge.train_idx),
        layer2=layer2, stride=stage3.precompute_stride,
        weighting_mode="equal_topk", equal_topk_k=_K_WRAPPER,
        long_only=bool(long_only),
    )
    val_tape = precompute_tape_canonical(
        bundle=bundle, bridge=bridge,
        day_indices=list(bridge.val_idx),
        layer2=layer2, stride=stage3.precompute_stride,
        weighting_mode="equal_topk", equal_topk_k=_K_WRAPPER,
        long_only=bool(long_only),
    )
    test_tape = precompute_tape_canonical(
        bundle=bundle, bridge=bridge,
        day_indices=list(bridge.test_idx),
        layer2=layer2, stride=stage3.precompute_stride,
        weighting_mode="equal_topk", equal_topk_k=_K_WRAPPER,
        long_only=bool(long_only),
    )
    print(
        f"[layer2_q] precompute sizes: "
        f"train={len(train_tape)} val={len(val_tape)} test={len(test_tape)} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    from stable_baselines3.common.monitor import Monitor

    train_base = ExposureEnv(train_tape, layer3, bootstrap_episode=True)
    val_base = ExposureEnv(val_tape, layer3, bootstrap_episode=False)
    test_base = ExposureEnv(test_tape, layer3, bootstrap_episode=False)

    train_env = Monitor(train_base)
    val_env = val_base
    test_env = test_base

    macro_dim = int(train_tape.macro_dim)
    agent = SACQ(
        policy="MlpPolicy",
        env=train_env,
        q_config=q_config,
        seed=int(seed),
        verbose=0,
        device=str(device),
    )

    selector = _ValSharpeSelector(val_env=val_env, eval_freq=eval_freq)
    print(
        f"[layer2_q] training SACQ for {q_config.total_timesteps:,} "
        f"steps (val every {eval_freq:,})",
        flush=True,
    )
    agent.learn(
        total_timesteps=int(q_config.total_timesteps),
        callback=selector.callback,
        progress_bar=False,
    )
    restored = selector.restore_best(agent)
    train_stats = agent.q_train_stats()
    print(
        f"[layer2_q] best val Sharpe={selector.best_sharpe:+.4f} "
        f"at step={selector.best_step} restored={restored}",
        flush=True,
    )
    print(
        f"[layer2_q] train stats: "
        f"critic_loss={train_stats['critic_loss']:.4f} "
        f"actor_loss={train_stats['actor_loss']:+.4f} "
        f"q_mean={train_stats['q_mean']:+.4f} "
        f"q_cvar={train_stats['q_cvar']:+.4f} "
        f"target_q_mean={train_stats['target_q_mean']:+.4f} "
        f"log_prob={train_stats['log_prob']:+.4f} "
        f"ent_coef={train_stats['ent_coef']:.4f}",
        flush=True,
    )

    rets, exps = _rollout(test_env, agent)
    test_stats = _persist_test_outputs(
        tape=test_tape, rets=rets, exps=exps,
        bridge=bridge, out_path=out_path,
    )
    print(
        f"[layer2_q] wrote {out_path} "
        f"(test pooled Sharpe={test_stats['sharpe']:+.4f}, "
        f"n_days={test_stats['n_test_days']})",
        flush=True,
    )

    summary_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "universe": str(universe_label),
        "fold": int(fold),
        "seed": int(seed),
        "model": (
            "InVAR-RL-Q: Layer 1 canonical InVAR + fixed equal-weight L/S "
            f"wrapper (K={_K_WRAPPER} per side) + Layer 2 Q SAC "
            "(twin QuantileCritic + CVaR-blend actor objective)."
        ),
        "panel_kind": panel_kind,
        "two_regime_val": bool(two_regime_val),
        "panel_end": panel_end,
        "wrapper_k": int(_K_WRAPPER),
        "macro_dim": int(macro_dim),
        "q_config": {
            "n_quantiles": int(q_config.n_quantiles),
            "alpha_cvar": float(q_config.alpha_cvar),
            "eta_blend": float(q_config.eta_blend),
            "actor_hidden": list(q_config.actor_hidden),
            "critic_hidden": list(q_config.critic_hidden),
            "total_timesteps": int(q_config.total_timesteps),
            "learning_rate": float(q_config.learning_rate),
            "buffer_size": int(q_config.buffer_size),
            "batch_size": int(q_config.batch_size),
            "gamma": float(q_config.gamma),
            "polyak_tau": float(q_config.polyak_tau),
        },
        "n_train_steps": int(len(train_tape)),
        "n_val_steps": int(len(val_tape)),
        "n_test_steps": int(len(test_tape)),
        "total_timesteps": int(q_config.total_timesteps),
        "eval_freq": int(eval_freq),
        "best_val_sharpe": float(selector.best_sharpe),
        "best_val_step": int(selector.best_step),
        "best_val_restored": bool(restored),
        "val_history": selector.eval_history,
        "methods": {
            "q": {
                "mean_return": float(test_stats["mean"]),
                "volatility": float(test_stats["std"]),
                "test_pooled_sharpe": float(test_stats["sharpe"]),
                "n_test_days": int(test_stats["n_test_days"]),
            },
        },
        "q_train_stats": train_stats,
        "test_pooled_sharpe": float(test_stats["sharpe"]),
        "test_mean_return": float(test_stats["mean"]),
        "test_std_return": float(test_stats["std"]),
        "test_n_days": int(test_stats["n_test_days"]),
        "test_out_path": str(out_path),
        "layer3_yaml": str(layer3_yaml),
        "stage3_yaml": str(stage3_yaml),
        "wall_time_seconds": float(time.time() - t0),
    }
    with open(summary_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"[layer2_q] wrote {summary_path}", flush=True)
    return payload


def _parse_args() -> argparse.Namespace:
    setup = universe_setup("sp500")
    default_config = QConfig()
    p = argparse.ArgumentParser(
        description=(
            "SP500 Layer 2 Q: replace SB3 SAC twin scalar critics with "
            "twin QuantileCritic + CVaR-blend actor objective."
        )
    )
    p.add_argument(
        "--fold", type=int, required=True, choices=[1, 2, 3, 4, 5]
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--total-timesteps", type=int, default=default_config.total_timesteps
    )
    p.add_argument(
        "--n-quantiles", type=int, default=default_config.n_quantiles,
        help="Number of quantile midpoints per critic head.",
    )
    p.add_argument(
        "--alpha-cvar", type=float, default=default_config.alpha_cvar,
        help="Lower-tail level for CVaR statistic in actor objective.",
    )
    p.add_argument(
        "--eta-blend", type=float, default=default_config.eta_blend,
        help="Mean / CVaR blend; 1.0 = mean only, 0.0 = CVaR only.",
    )
    p.add_argument(
        "--eval-freq", type=int, default=2000,
        help="env steps between val Sharpe checkpoint evaluations",
    )
    p.add_argument(
        "--output-dir-root", type=str, default=_DEFAULT_OUTPUT_ROOT,
    )
    p.add_argument(
        "--layer1-ckpt-root", type=str, default=setup.ckpt_root,
    )
    p.add_argument(
        "--layer3", type=str, default="invar_rl/configs/layer3.yaml"
    )
    p.add_argument(
        "--stage3", type=str, default="invar_rl/configs/stage3.yaml"
    )
    p.add_argument("--panel-end", type=str, default="2025-12-31")
    p.add_argument(
        "--panel-kind", type=str, default=setup.panel_kind,
        choices=["lattice_native", "biotech"],
    )
    p.add_argument(
        "--universe-label", type=str, default="sp500",
        help="Universe label written to the summary JSON.",
    )
    p.add_argument(
        "--long-only", action="store_true",
        help=(
            "L/O protocol: precompute tapes with long_only=True "
            "(top-K long-only fully-invested book, w_i >= 0)."
        ),
    )
    p.add_argument(
        "--actor-hidden", type=int, nargs="+", default=None,
        help=(
            "Override actor MLP hidden sizes (e.g. 256 256). Defaults "
            "to QConfig.actor_hidden if omitted."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    ckpt_path = (
        Path(args.layer1_ckpt_root)
        / f"fold{args.fold}_seed{args.seed}_full.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Layer 1 full ckpt missing for fold={args.fold} "
            f"seed={args.seed}: {ckpt_path}"
        )
    out_root = Path(args.output_dir_root)
    out_path = out_root / f"fold{args.fold}_seed{args.seed}.parquet"
    summary_path = (
        out_root / "summary" / f"fold{args.fold}_seed{args.seed}.json"
    )
    if out_path.exists() and summary_path.exists():
        print(
            f"[layer2_q] {out_path} + summary exist; skipping cell",
            flush=True,
        )
        return 0
    q_kwargs = dict(
        n_quantiles=int(args.n_quantiles),
        alpha_cvar=float(args.alpha_cvar),
        eta_blend=float(args.eta_blend),
        total_timesteps=int(args.total_timesteps),
    )
    if args.actor_hidden is not None:
        q_kwargs["actor_hidden"] = [int(x) for x in args.actor_hidden]
    q_config = QConfig(**q_kwargs)
    run_one_cell(
        fold=args.fold,
        seed=args.seed,
        ckpt_path=ckpt_path,
        layer3_yaml=Path(args.layer3),
        stage3_yaml=Path(args.stage3),
        q_config=q_config,
        eval_freq=int(args.eval_freq),
        output_dir_root=out_root,
        panel_end=args.panel_end,
        panel_kind=args.panel_kind,
        universe_label=str(args.universe_label),
        long_only=bool(args.long_only),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
