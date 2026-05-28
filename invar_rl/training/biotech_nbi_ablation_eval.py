"""Biotech NBI Phase 6 four-ablation evaluator.

Direct mirror of :mod:`invar_rl.training.nasdaq100_ablation_eval` for
the biotech NBI universe. One run trains (or rolls) ONE (ablation,
method, fold, seed, protocol) cell and persists the same parquet +
summary JSON pair as :mod:`invar_rl.training.biotech_nbi_layer3_sac`.

Policy P1: precompute tape, observation-strip wrapper, RL agent
factory, env, reward function, and val-Sharpe selector are all
imported from the same modules. Only the universe-keyed branch
(``panel_kind="biotech_nbi"``), the biotech NBI Layer 2 QP
hyperparameters (reused from
:mod:`invar_rl.training.biotech_nbi_layer3_sac`), and the
NBI-appropriate equal-weight K (K=50 per side; see user-confirmed
default) are NBI-specific.

Ablation conditions:

- ``canonical``: identity (no ablation).
- ``random_l1``: Layer 1 InVAR scores replaced with N(0, 1) noise.
- ``equal_l2``: Layer 2 cvxpy QP replaced with equal-weight top-K /
  bottom-K (K=50 per side for biotech NBI; SP500 uses K=50;
  NDX-100 uses K=20).
- ``stripped_l3``: Layer 1 + Layer 2 canonical, but the Layer-1 /
  Layer-2 fields of the RL observation are zeroed via
  :class:`StrippedObservationWrapper`. ``constant_full`` is unaffected
  and is skipped.

Methods: ``recurrent_ppo``, ``feedforward_ppo``, ``sac``,
``constant_full``.

Output schema: per-day exposures + realised portfolio returns parquet
at
``outputs/biotech_nbi/phase6_ablation/{ablation}/{method}/{protocol}/foldF_seedS.parquet``
plus a per-cell summary JSON under the corresponding ``summary``
subdir. The summary JSON also carries the tape base-book mean/vol so
the Layer 1 + Layer 2 base-book Sharpe is recoverable from any RL cell
without re-running constant_full separately.

CLI::

    python -m invar_rl.training.biotech_nbi_ablation_eval \\
        --ablation canonical --method sac --fold 1 --seed 42 \\
        --protocol ls --total-timesteps 20000

Phase 6 acceptance: 4 ablations x 4 methods x 5 folds x 5 seeds x 2
protocols = 800 cells worst case. ``stripped_l3`` skips constant_full
so the actual grid is 30 (ablation, method, protocol) tuples x 25
cells = 750 cells.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.invar import InVARConfig

from invar_rl.baselines.exposure_baselines import ConstantFullExposure
from invar_rl.common.config import (
    Layer2Config,
    load_layer3_config,
    load_stage3_config,
)
from invar_rl.common.seeding import set_global_seed
from invar_rl.data.lattice_bridge import build_lattice_bridge
from invar_rl.layer1_ranker.canonical_runner import load_trained_invar
from invar_rl.layer3_control.env import ExposureEnv
from invar_rl.layer3_control.observation import RiskState
from invar_rl.layer3_control.precompute import EpisodeTape
from invar_rl.layer3_control.precompute_canonical import (
    precompute_tape_canonical,
)
from invar_rl.layer3_control.strip_obs_wrapper import (
    StrippedObservationWrapper,
)

# Biotech NBI Layer 2 QP hyperparameters; identical to the Phase 5
# SAC driver (Policy P1 byte-for-byte from NDX-100). Authoritative
# copy lives in biotech_nbi_layer3_sac.py.
_BIOTECH_NBI_LAYER2: Dict[str, object] = dict(
    estimator="ledoit_wolf",
    factor_rank=8,
    cov_lookback=120,
    risk_aversion=5.0,
    per_name_bound=0.05,
    gross_leverage=1.0,
    topk_k=25,
    topk_temperature=0.5,
    topk_temperature_anneal=True,
)
_TRADING_DAYS = 252
# Per-side K for the equal_l2 ablation on biotech NBI. User-confirmed
# default = 50 per side, matching the SP500 ablation K (the biotech
# NBI active universe is ~270 per day, so 50 / 270 sits between the
# SP500 50/250 fraction and the NDX-100 20/100 fraction).
_EQUAL_L2_K_NBI = 50

_ABLATIONS = ("canonical", "random_l1", "equal_l2", "stripped_l3")
_METHODS = ("recurrent_ppo", "feedforward_ppo", "sac", "constant_full")
_RL_METHODS = ("recurrent_ppo", "feedforward_ppo", "sac")


def _build_layer2_cfg() -> Layer2Config:
    return Layer2Config(**_BIOTECH_NBI_LAYER2)


def _ablation_modes(ablation: str) -> Dict[str, object]:
    """Return (score_mode, weighting_mode, strip_obs) for an ablation."""
    if ablation == "canonical":
        return {"score_mode": "canonical", "weighting_mode": "qp",
                "strip_obs": False}
    if ablation == "random_l1":
        return {"score_mode": "random", "weighting_mode": "qp",
                "strip_obs": False}
    if ablation == "equal_l2":
        return {"score_mode": "canonical", "weighting_mode": "equal_topk",
                "strip_obs": False}
    if ablation == "stripped_l3":
        return {"score_mode": "canonical", "weighting_mode": "qp",
                "strip_obs": True}
    raise ValueError(
        f"unknown ablation {ablation!r}; expected {_ABLATIONS}"
    )


def _pooled_sharpe(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    sd = float(returns.std(ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(returns.mean() / sd * np.sqrt(_TRADING_DAYS))


def _rollout_rl(
    env: ExposureEnv, agent, recurrent: bool
) -> Tuple[np.ndarray, np.ndarray]:
    obs, _ = env.reset(seed=0)
    state, starts = None, np.ones((1,), dtype=bool)
    rets: List[float] = []
    exps: List[float] = []
    while True:
        if recurrent:
            action, state = agent.predict(
                obs, state=state, episode_start=starts, deterministic=True
            )
        else:
            action, _ = agent.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        obs, _, term, trunc, info = env.step(action)
        starts = np.zeros((1,), dtype=bool)
        rets.append(float(info["strategy_return"]))
        exps.append(float(info["exposure"]))
        if term or trunc:
            break
    return (
        np.asarray(rets, dtype=np.float64),
        np.asarray(exps, dtype=np.float64),
    )


def _rollout_constant_full(
    env: ExposureEnv, tape: EpisodeTape, layer3,
) -> Tuple[np.ndarray, np.ndarray]:
    policy = ConstantFullExposure(layer3)
    risk = RiskState(exposure=layer3.exposure_min)
    obs, _ = env.reset(seed=0)
    rets: List[float] = []
    exps: List[float] = []
    t = 0
    while True:
        exp = policy.exposure(tape, min(t, len(tape) - 1), risk)
        action = np.array([exp], dtype=np.float32)
        obs, _, term, trunc, info = env.step(action)
        rets.append(float(info["strategy_return"]))
        exps.append(float(info["exposure"]))
        t += 1
        if term or trunc:
            break
    return (
        np.asarray(rets, dtype=np.float64),
        np.asarray(exps, dtype=np.float64),
    )


class _ValSharpeSelector:
    """SB3 callback that snapshots the best-val-Sharpe checkpoint."""

    def __init__(
        self, val_env: ExposureEnv, eval_freq: int, recurrent: bool,
    ) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        self._val_env = val_env
        self._eval_freq = int(max(1, eval_freq))
        self._recurrent = bool(recurrent)
        self.best_sharpe = -np.inf
        self.best_step = 0
        self.eval_history: List[Dict[str, float]] = []
        self._buffer: Optional[bytes] = None

        outer = self

        class _Inner(BaseCallback):
            def _on_step(self_inner) -> bool:  # type: ignore[override]
                if self_inner.num_timesteps % outer._eval_freq != 0:
                    return True
                rets, _ = _rollout_rl(
                    outer._val_env, self_inner.model,
                    recurrent=outer._recurrent,
                )
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
                    buf = io.BytesIO()
                    self_inner.model.save(buf)
                    outer._buffer = buf.getvalue()
                return True

        self.callback = _Inner()

    def restore_best(self, agent, method: str) -> bool:
        if self._buffer is None:
            return False
        env = agent.get_env()
        buf = io.BytesIO(self._buffer)
        if method == "sac":
            from stable_baselines3 import SAC
            restored = SAC.load(buf, env=env, device=agent.device)
        elif method == "feedforward_ppo":
            from stable_baselines3 import PPO
            restored = PPO.load(buf, env=env, device=agent.device)
        elif method == "recurrent_ppo":
            from sb3_contrib import RecurrentPPO
            restored = RecurrentPPO.load(buf, env=env, device=agent.device)
        else:
            return False
        agent.policy.load_state_dict(restored.policy.state_dict())
        return True


def _wrap_env_if_stripped(
    env: ExposureEnv, strip_obs: bool,
) -> "ExposureEnv | StrippedObservationWrapper":
    if strip_obs:
        return StrippedObservationWrapper(env)
    return env


def _persist_test_outputs(
    tape: EpisodeTape,
    rets: np.ndarray,
    exps: np.ndarray,
    bridge,
    out_path: Path,
) -> Dict[str, object]:
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
    ablation: str,
    method: str,
    fold: int,
    seed: int,
    protocol: str,
    ckpt_path: Path,
    layer3_yaml: Path,
    stage3_yaml: Path,
    total_timesteps: int,
    eval_freq: int,
    output_dir_root: Path,
    panel_end: str,
    panel_kind: str = "biotech_nbi",
    two_regime_val: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Train (or roll) one (ablation, method, fold, seed, protocol) cell."""
    if ablation not in _ABLATIONS:
        raise ValueError(
            f"ablation must be one of {_ABLATIONS}, got {ablation!r}"
        )
    if method not in _METHODS:
        raise ValueError(
            f"method must be one of {_METHODS}, got {method!r}"
        )
    if protocol not in ("ls", "lo"):
        raise ValueError(f"protocol must be 'ls' or 'lo', got {protocol!r}")
    if ablation == "stripped_l3" and method == "constant_full":
        raise ValueError(
            "stripped_l3 ablation is undefined for constant_full "
            "(constant_full does not read the observation); skip this cell"
        )
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    set_global_seed(seed)
    modes = _ablation_modes(ablation)
    layer2 = _build_layer2_cfg()
    layer3 = load_layer3_config(str(layer3_yaml))
    stage3 = load_stage3_config(str(stage3_yaml))

    cfg = InVARConfig(fold=fold, seed=seed)
    cfg.panel_kind = panel_kind
    cfg.two_regime_val = two_regime_val
    cfg.panel_end = panel_end
    cfg.enable_retrieval_bank = False
    bridge = build_lattice_bridge(cfg, device=device)
    bundle = load_trained_invar(
        ckpt_path=ckpt_path, bridge=bridge, device=device
    )

    long_only = (protocol == "lo")
    print(
        f"[biotech_nbi_ablation] ablation={ablation} method={method} "
        f"fold={fold} seed={seed} protocol={protocol} "
        f"long_only={long_only} device={device}",
        flush=True,
    )
    t0 = time.time()

    common_tape_kwargs: Dict[str, object] = dict(
        bundle=bundle, bridge=bridge, layer2=layer2,
        stride=stage3.precompute_stride,
        score_mode=str(modes["score_mode"]),
        weighting_mode=str(modes["weighting_mode"]),
        long_only=long_only,
        equal_topk_k=_EQUAL_L2_K_NBI,
    )
    train_tape = precompute_tape_canonical(
        day_indices=list(bridge.train_idx),
        ablation_seed=seed, **common_tape_kwargs,
    )
    val_tape = precompute_tape_canonical(
        day_indices=list(bridge.val_idx),
        ablation_seed=seed + 1000, **common_tape_kwargs,
    )
    test_tape = precompute_tape_canonical(
        day_indices=list(bridge.test_idx),
        ablation_seed=seed + 2000, **common_tape_kwargs,
    )
    print(
        f"[biotech_nbi_ablation] tape sizes: "
        f"train={len(train_tape)} val={len(val_tape)} test={len(test_tape)} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    selector_payload: Dict[str, object] = {}
    if method in _RL_METHODS:
        from stable_baselines3.common.monitor import Monitor
        from invar_rl.layer3_control.agent import build_agent

        train_inner = ExposureEnv(
            train_tape, layer3, bootstrap_episode=True
        )
        train_inner = _wrap_env_if_stripped(
            train_inner, bool(modes["strip_obs"])
        )
        train_env = Monitor(train_inner)
        val_inner = ExposureEnv(
            val_tape, layer3, bootstrap_episode=False
        )
        val_inner = _wrap_env_if_stripped(
            val_inner, bool(modes["strip_obs"])
        )
        test_inner = ExposureEnv(
            test_tape, layer3, bootstrap_episode=False
        )
        test_inner = _wrap_env_if_stripped(
            test_inner, bool(modes["strip_obs"])
        )

        agent = build_agent(method, train_env, stage3, seed)
        recurrent = (method == "recurrent_ppo")
        selector = _ValSharpeSelector(
            val_env=val_inner, eval_freq=eval_freq, recurrent=recurrent,
        )
        print(
            f"[biotech_nbi_ablation] training {method} for "
            f"{total_timesteps:,} steps (val every {eval_freq:,})",
            flush=True,
        )
        agent.learn(
            total_timesteps=int(total_timesteps),
            callback=selector.callback,
            progress_bar=False,
        )
        restored = selector.restore_best(agent, method)
        print(
            f"[biotech_nbi_ablation] best val Sharpe="
            f"{selector.best_sharpe:+.4f} at step={selector.best_step} "
            f"restored={restored}",
            flush=True,
        )
        rets, exps = _rollout_rl(test_inner, agent, recurrent=recurrent)
        selector_payload = {
            "best_val_sharpe": float(selector.best_sharpe),
            "best_val_step": int(selector.best_step),
            "best_val_restored": bool(restored),
            "val_history": selector.eval_history,
        }
    else:  # constant_full
        test_env = ExposureEnv(
            test_tape, layer3, bootstrap_episode=False,
        )
        rets, exps = _rollout_constant_full(test_env, test_tape, layer3)
        selector_payload = {
            "best_val_sharpe": 0.0,
            "best_val_step": 0,
            "best_val_restored": False,
            "val_history": [],
        }

    out_dir = output_dir_root / ablation / method / protocol
    out_path = out_dir / f"fold{fold}_seed{seed}.parquet"
    test_stats = _persist_test_outputs(
        tape=test_tape, rets=rets, exps=exps, bridge=bridge,
        out_path=out_path,
    )
    print(
        f"[biotech_nbi_ablation] wrote {out_path} "
        f"(test pooled Sharpe={test_stats['sharpe']:+.4f}, "
        f"n_days={test_stats['n_test_days']})",
        flush=True,
    )

    tape_ret = test_tape.base_return.astype(float)
    tape_mean = float(tape_ret.mean()) if tape_ret.size else 0.0
    tape_vol = float(tape_ret.std(ddof=1)) if tape_ret.size > 1 else 0.0

    summary_dir = output_dir_root / ablation / method / protocol / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"fold{fold}_seed{seed}.json"
    payload: Dict[str, object] = {
        "universe": panel_kind,
        "phase": 6,
        "ablation": ablation,
        "method": method,
        "fold": int(fold),
        "seed": int(seed),
        "protocol": protocol,
        "model": (
            f"InVAR-RL Phase 6 ablation={ablation} method={method} "
            f"(score_mode={modes['score_mode']} "
            f"weighting_mode={modes['weighting_mode']} "
            f"strip_obs={modes['strip_obs']})"
        ),
        "panel_kind": panel_kind,
        "two_regime_val": bool(two_regime_val),
        "panel_end": panel_end,
        "long_only": bool(long_only),
        "n_train_steps": int(len(train_tape)),
        "n_val_steps": int(len(val_tape)),
        "n_test_steps": int(len(test_tape)),
        "total_timesteps": int(total_timesteps),
        "eval_freq": int(eval_freq),
        "test_pooled_sharpe": float(test_stats["sharpe"]),
        "test_mean_return": float(test_stats["mean"]),
        "test_std_return": float(test_stats["std"]),
        "test_n_days": int(test_stats["n_test_days"]),
        "test_out_path": str(out_path),
        "tape_constant_full_mean_return": tape_mean,
        "tape_constant_full_volatility": tape_vol,
        "layer2_qp": _BIOTECH_NBI_LAYER2,
        "equal_topk_k": int(_EQUAL_L2_K_NBI),
        "layer3_yaml": str(layer3_yaml),
        "stage3_yaml": str(stage3_yaml),
        "wall_time_seconds": float(time.time() - t0),
        "score_mode": str(modes["score_mode"]),
        "weighting_mode": str(modes["weighting_mode"]),
        "strip_obs": bool(modes["strip_obs"]),
        **selector_payload,
    }
    with open(summary_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"[biotech_nbi_ablation] wrote {summary_path}", flush=True)
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Biotech NBI Phase 6 four-ablation evaluator "
            "(canonical / random_l1 / equal_l2 / stripped_l3 x "
            "recurrent_ppo / feedforward_ppo / sac / constant_full)."
        )
    )
    p.add_argument(
        "--ablation", type=str, required=True, choices=list(_ABLATIONS),
    )
    p.add_argument(
        "--method", type=str, required=True, choices=list(_METHODS),
    )
    p.add_argument(
        "--fold", type=int, required=True, choices=[1, 2, 3, 4, 5],
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--protocol", type=str, required=True, choices=["ls", "lo"],
    )
    p.add_argument("--total-timesteps", type=int, default=20000)
    p.add_argument("--eval-freq", type=int, default=2000)
    p.add_argument(
        "--output-dir-root", type=str,
        default="outputs/biotech_nbi/phase6_ablation",
    )
    p.add_argument(
        "--layer1-ckpt-root", type=str,
        default="outputs/biotech_nbi/layer1/_ckpt",
    )
    p.add_argument(
        "--panel-kind", type=str, default="biotech_nbi",
        choices=["biotech_nbi", "biotech_nbi_enriched"],
        help=(
            "Panel schema. 'biotech_nbi' = original 26-feature "
            "zero-fill panel; 'biotech_nbi_enriched' = 22-feature "
            "biotech-specific panel. Layer 1 ckpts under "
            "--layer1-ckpt-root MUST match the panel (n_features "
            "differs between the two)."
        ),
    )
    p.add_argument(
        "--layer3", type=str, default="invar_rl/configs/layer3.yaml",
    )
    p.add_argument(
        "--stage3", type=str, default="invar_rl/configs/stage3.yaml",
    )
    p.add_argument("--panel-end", type=str, default="2025-12-31")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.ablation == "stripped_l3" and args.method == "constant_full":
        print(
            "[biotech_nbi_ablation] skipping stripped_l3 + constant_full "
            "(undefined combination)", flush=True,
        )
        return 0

    ckpt_path = (
        Path(args.layer1_ckpt_root)
        / f"fold{args.fold}_seed{args.seed}_full.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Layer 1 ckpt missing for fold={args.fold} "
            f"seed={args.seed}: {ckpt_path}; rerun Phase 3 for this cell"
        )

    out_root = Path(args.output_dir_root)
    out_path = (
        out_root / args.ablation / args.method / args.protocol
        / f"fold{args.fold}_seed{args.seed}.parquet"
    )
    summary_path = (
        out_root / args.ablation / args.method / args.protocol / "summary"
        / f"fold{args.fold}_seed{args.seed}.json"
    )
    if out_path.exists() and summary_path.exists():
        print(
            f"[biotech_nbi_ablation] skip {out_path} + summary "
            "(already exist)", flush=True,
        )
        return 0

    run_one_cell(
        ablation=args.ablation,
        method=args.method,
        fold=args.fold,
        seed=args.seed,
        protocol=args.protocol,
        ckpt_path=ckpt_path,
        layer3_yaml=Path(args.layer3),
        stage3_yaml=Path(args.stage3),
        total_timesteps=int(args.total_timesteps),
        eval_freq=int(args.eval_freq),
        output_dir_root=out_root,
        panel_end=args.panel_end,
        panel_kind=args.panel_kind,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
