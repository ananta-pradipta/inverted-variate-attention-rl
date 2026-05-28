"""S&P 500 Layer 2 SIA driver: Sparse Invariant Actor + Full-Info Critic SAC.

Mirrors :mod:`invar_rl.training.sp500_layer3_sac_ablation6` with two
changes:

1. The action is the canonical 1-D exposure scalar. The wrapper is the
   fixed equal-weight L/S top-50 per side, identical to the canonical
   SP500 SAC headline (``stage3_rl_ablation/equal_l2/sac``).
2. The SAC agent is replaced with
   :class:`invar_rl.layer2_sia.sac_sia.SACSIA`. All other plumbing
   (Layer 1 ckpt, observation pipeline, val-Sharpe selector, test rollout)
   is byte-for-byte equivalent so the SIA vs SAC delta isolates the
   actor + auxiliary loss design.

CLI::

    python -m invar_rl.training.sp500_layer2_sia \\
        --fold 1 --seed 42 \\
        --total-timesteps 20000 \\
        --output-dir-root outputs/sp500/layer2_sia/smoke

Phase 1 (smoke): one (fold, seed) cell, default SIA hyperparams. Phase 2
is the 5-cell fold-1 mini-sweep; that requires user approval and is not
run from this driver alone.
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
from invar_rl.layer2_sia.config import SIAConfig
from invar_rl.layer2_sia.env_wrapper import RegimeLabelEnv
from invar_rl.layer2_sia.regime_probs import load_probs_lookup
from invar_rl.layer2_sia.sac_sia import SACSIA
from invar_rl.layer3_control.env import ExposureEnv
from invar_rl.layer3_control.precompute_canonical import (
    precompute_tape_canonical,
)
from invar_rl.training._universe_setup import universe_setup

_TRADING_DAYS: int = 252
_K_WRAPPER: int = 50  # canonical SP500 wrapper K, per Phase 0 plan.


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

    Saves only the SIA actor + SB3 critic state_dicts so the restore path
    does NOT round-trip through SB3's ``save`` / ``load`` pickle pipeline
    (which mismatches the SACSIA __init__ signature).
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
                    "val_std": float(rets.std(ddof=1)) if rets.size > 1 else 0.0,
                })
                if sh > outer.best_sharpe:
                    outer.best_sharpe = float(sh)
                    outer.best_step = int(self_inner.num_timesteps)
                    outer._best_state = {
                        "sia_actor": copy.deepcopy(
                            self_inner.model.sia_actor.state_dict()
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

    def restore_best(self, agent: SACSIA) -> bool:
        if self._best_state is None:
            return False
        agent.sia_actor.load_state_dict(self._best_state["sia_actor"])
        agent.critic.load_state_dict(self._best_state["critic"])
        agent.critic_target.load_state_dict(self._best_state["critic_target"])
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


def _build_day_to_cluster(
    universe: str, fold: int
) -> Dict[int, int]:
    """Load k-means-8 probs and return ``day_idx -> argmax cluster_id``.

    The cache lives at
    ``cache/dr_rl/regime_probs/{universe}/fold{F}/probs.parquet`` and is
    populated by
    :func:`invar_rl.layer2_sia.regime_probs.precompute_all`. The argmax
    over the 8 soft probabilities gives the hard cluster id per day used
    as the group id for the regime-invariance penalty.
    """
    lookup = load_probs_lookup(universe, fold)
    day_to_cluster: Dict[int, int] = {}
    for d, probs in lookup.items():
        day_to_cluster[int(d)] = int(np.argmax(probs))
    return day_to_cluster


def run_one_cell(
    fold: int,
    seed: int,
    ckpt_path: Path,
    layer3_yaml: Path,
    stage3_yaml: Path,
    sia_config: SIAConfig,
    eval_freq: int,
    output_dir_root: Path,
    panel_end: str,
    panel_kind: str = "lattice_native",
    two_regime_val: bool = True,
    device: Optional[torch.device] = None,
    use_regime_label: bool = False,
    universe_label: str = "sp500",
    long_only: bool = False,
) -> Dict[str, object]:
    """Train the SIA SAC for one (fold, seed) cell on the SP500 panel."""
    out_dir = output_dir_root
    out_path = out_dir / f"fold{fold}_seed{seed}.parquet"
    summary_dir = out_dir / "summary"
    summary_path = summary_dir / f"fold{fold}_seed{seed}.json"
    if out_path.exists() and summary_path.exists():
        print(
            f"[layer2_sia] {out_path} + summary exist; skipping cell",
            flush=True,
        )
        with open(summary_path) as fh:
            return json.load(fh)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        f"[layer2_sia] fold={fold} seed={seed} panel_kind={panel_kind} "
        f"device={device} wrapper_K={_K_WRAPPER} "
        f"latent_dim={sia_config.latent_dim} beta_kl={sia_config.beta_kl} "
        f"lambda_gate={sia_config.lambda_gate} "
        f"lambda_inv={sia_config.lambda_inv} "
        f"sparse_gates={sia_config.sparse_gates} "
        f"asymmetric_critic={sia_config.asymmetric_critic}",
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
        f"[layer2_sia] precompute sizes: "
        f"train={len(train_tape)} val={len(val_tape)} test={len(test_tape)} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    from stable_baselines3.common.monitor import Monitor

    train_base = ExposureEnv(train_tape, layer3, bootstrap_episode=True)
    val_base = ExposureEnv(val_tape, layer3, bootstrap_episode=False)
    test_base = ExposureEnv(test_tape, layer3, bootstrap_episode=False)

    if use_regime_label:
        day_to_cluster = _build_day_to_cluster(
            universe=universe_label, fold=int(fold)
        )
        n_with = sum(
            1 for d in train_tape.days.tolist() if int(d) in day_to_cluster
        )
        n_total = int(train_tape.days.shape[0])
        print(
            f"[layer2_sia] regime_label=ON universe={universe_label} "
            f"fold={fold} cache_days={len(day_to_cluster)} "
            f"train_days_with_cluster={n_with}/{n_total}",
            flush=True,
        )
        train_inner = RegimeLabelEnv(
            train_base,
            tape_days=train_tape.days,
            day_to_cluster=day_to_cluster,
        )
        val_inner = RegimeLabelEnv(
            val_base,
            tape_days=val_tape.days,
            day_to_cluster=day_to_cluster,
        )
        test_inner = RegimeLabelEnv(
            test_base,
            tape_days=test_tape.days,
            day_to_cluster=day_to_cluster,
        )
    else:
        train_inner, val_inner, test_inner = (
            train_base, val_base, test_base
        )

    train_env = Monitor(train_inner)
    val_env = val_inner
    test_env = test_inner

    macro_dim = int(train_tape.macro_dim)
    agent = SACSIA(
        policy="MlpPolicy",
        env=train_env,
        sia_config=sia_config,
        macro_dim=macro_dim,
        l1_uncertainty=0,
        regime_lookup=None,
        regime_label=bool(use_regime_label),
        seed=int(seed),
        verbose=0,
        device=str(device),
    )

    selector = _ValSharpeSelector(val_env=val_env, eval_freq=eval_freq)
    print(
        f"[layer2_sia] training SACSIA for {sia_config.total_timesteps:,} "
        f"steps (val every {eval_freq:,})",
        flush=True,
    )
    agent.learn(
        total_timesteps=int(sia_config.total_timesteps),
        callback=selector.callback,
        progress_bar=False,
    )
    restored = selector.restore_best(agent)
    train_stats = agent.sia_train_stats()
    print(
        f"[layer2_sia] best val Sharpe={selector.best_sharpe:+.4f} "
        f"at step={selector.best_step} restored={restored}",
        flush=True,
    )
    print(
        f"[layer2_sia] train stats: "
        f"gate_open_frac={train_stats['gate_open_fraction']:.4f} "
        f"gates=[{train_stats['gate_0']:.3f}, {train_stats['gate_1']:.3f}, "
        f"{train_stats['gate_2']:.3f}, {train_stats['gate_3']:.3f}, "
        f"{train_stats['gate_4']:.3f}] "
        f"exposure_mean={train_stats['exposure_mean']:.3f} "
        f"exposure_std={train_stats['exposure_std']:.3f} "
        f"mu_std={train_stats['mu_std']:.3f}",
        flush=True,
    )

    rets, exps = _rollout(test_env, agent)
    test_stats = _persist_test_outputs(
        tape=test_tape, rets=rets, exps=exps,
        bridge=bridge, out_path=out_path,
    )
    print(
        f"[layer2_sia] wrote {out_path} "
        f"(test pooled Sharpe={test_stats['sharpe']:+.4f}, "
        f"n_days={test_stats['n_test_days']})",
        flush=True,
    )

    summary_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "universe": str(universe_label),
        "fold": int(fold),
        "seed": int(seed),
        "regime_label": bool(use_regime_label),
        "model": (
            "InVAR-RL-SIA: Layer 1 canonical InVAR + fixed equal-weight L/S "
            f"wrapper (K={_K_WRAPPER} per side) + Layer 2 SIA SAC "
            "(sparse invariant actor + full-info SB3 twin-Q critic)."
            + (
                " regime_label=ON (k-means-8 cluster id appended to obs)"
                if use_regime_label else ""
            )
            + (
                " ablation=no_s (sparse_gates=OFF; gates clamped to 1.0)"
                if not bool(sia_config.sparse_gates) else ""
            )
            + (
                " ablation=no_a (asymmetric_critic=OFF; critic on bottleneck)"
                if not bool(sia_config.asymmetric_critic) else ""
            )
        ),
        "panel_kind": panel_kind,
        "two_regime_val": bool(two_regime_val),
        "panel_end": panel_end,
        "wrapper_k": int(_K_WRAPPER),
        "macro_dim": int(macro_dim),
        "sia_config": {
            "latent_dim": int(sia_config.latent_dim),
            "beta_kl": float(sia_config.beta_kl),
            "lambda_gate": float(sia_config.lambda_gate),
            "lambda_inv": float(sia_config.lambda_inv),
            "actor_hidden": list(sia_config.actor_hidden),
            "critic_hidden": list(sia_config.critic_hidden),
            "group_source": str(sia_config.group_source),
            "total_timesteps": int(sia_config.total_timesteps),
            "learning_rate": float(sia_config.learning_rate),
            "buffer_size": int(sia_config.buffer_size),
            "batch_size": int(sia_config.batch_size),
            "gamma": float(sia_config.gamma),
            "polyak_tau": float(sia_config.polyak_tau),
            "sparse_gates": bool(sia_config.sparse_gates),
            "asymmetric_critic": bool(sia_config.asymmetric_critic),
        },
        "n_train_steps": int(len(train_tape)),
        "n_val_steps": int(len(val_tape)),
        "n_test_steps": int(len(test_tape)),
        "total_timesteps": int(sia_config.total_timesteps),
        "eval_freq": int(eval_freq),
        "best_val_sharpe": float(selector.best_sharpe),
        "best_val_step": int(selector.best_step),
        "best_val_restored": bool(restored),
        "val_history": selector.eval_history,
        "methods": {
            "sia": {
                "mean_return": float(test_stats["mean"]),
                "volatility": float(test_stats["std"]),
                "test_pooled_sharpe": float(test_stats["sharpe"]),
                "n_test_days": int(test_stats["n_test_days"]),
            },
        },
        "sia_train_stats": train_stats,
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
    print(f"[layer2_sia] wrote {summary_path}", flush=True)
    return payload


def _parse_args() -> argparse.Namespace:
    setup = universe_setup("sp500")
    p = argparse.ArgumentParser(
        description=(
            "SP500 Layer 2 SIA: replace the canonical SAC actor with a "
            "Sparse Invariant Actor; keep full-info SB3 twin-Q critic."
        )
    )
    p.add_argument(
        "--fold", type=int, required=True, choices=[1, 2, 3, 4, 5]
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--total-timesteps", type=int, default=SIAConfig.total_timesteps
    )
    p.add_argument(
        "--latent-dim", type=int, default=SIAConfig.latent_dim,
        help="Width of the actor's KL-regularised latent z.",
    )
    p.add_argument(
        "--beta-kl", type=float, default=SIAConfig.beta_kl,
        help="Weight on KL(q(z|input) || N(0, I)).",
    )
    p.add_argument(
        "--lambda-gate", type=float, default=SIAConfig.lambda_gate,
        help="Weight on the L1 penalty on per-block sigmoid gates.",
    )
    p.add_argument(
        "--lambda-inv", type=float, default=SIAConfig.lambda_inv,
        help="Weight on the regime-invariance penalty.",
    )
    p.add_argument(
        "--group-source", type=str, default=SIAConfig.group_source,
        help="Source of group ids for the invariance penalty.",
    )
    p.add_argument(
        "--eval-freq", type=int, default=2000,
        help="env steps between val Sharpe checkpoint evaluations",
    )
    p.add_argument(
        "--output-dir-root", type=str, default=setup.sia_output_root,
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
        "--regime-label", action="store_true",
        help=(
            "Wrap envs with RegimeLabelEnv (k-means-8 cluster id appended "
            "to obs tail). Required for the SIA aux_inv term to fire."
        ),
    )
    p.add_argument(
        "--no-sparse-gates", action="store_true",
        help=(
            "Phase 4 no_s ablation: clamp the actor's per-block gates to "
            "constant 1.0 for every input. The KL latent bottleneck and "
            "the regime-invariance penalty still fire; only the per-block "
            "sparse routing is disabled."
        ),
    )
    p.add_argument(
        "--no-asymmetric-critic", action="store_true",
        help=(
            "Phase 4 no_a ablation: rebuild the SB3 twin-Q critic on the "
            "actor's post-gate bottleneck actor_in (1 + 2 + macro_small "
            "+ 4 + 1) instead of the full observation. The critic now "
            "sees exactly what the actor sees."
        ),
    )
    p.add_argument(
        "--universe-label", type=str, default="sp500",
        help=(
            "Label used to locate the cached k-means-8 regime probs at "
            "cache/dr_rl/regime_probs/{universe}/fold{F}/probs.parquet."
        ),
    )
    p.add_argument(
        "--long-only", action="store_true",
        help=(
            "L/O protocol: precompute tapes with long_only=True "
            "(top-K long-only fully-invested book, w_i >= 0)."
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
            f"[layer2_sia] {out_path} + summary exist; skipping cell",
            flush=True,
        )
        return 0
    sia_config = SIAConfig(
        latent_dim=int(args.latent_dim),
        beta_kl=float(args.beta_kl),
        lambda_gate=float(args.lambda_gate),
        lambda_inv=float(args.lambda_inv),
        group_source=str(args.group_source),
        total_timesteps=int(args.total_timesteps),
        sparse_gates=(not bool(args.no_sparse_gates)),
        asymmetric_critic=(not bool(args.no_asymmetric_critic)),
    )
    run_one_cell(
        fold=args.fold,
        seed=args.seed,
        ckpt_path=ckpt_path,
        layer3_yaml=Path(args.layer3),
        stage3_yaml=Path(args.stage3),
        sia_config=sia_config,
        eval_freq=int(args.eval_freq),
        output_dir_root=out_root,
        panel_end=args.panel_end,
        panel_kind=args.panel_kind,
        use_regime_label=bool(args.regime_label),
        universe_label=str(args.universe_label),
        long_only=bool(args.long_only),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
