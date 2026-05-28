"""NASDAQ-100 Phase 5 Layer 3 SAC controller training.

For one ``(fold, seed, protocol)`` cell on the NASDAQ-100 universe,
this driver mirrors :mod:`invar_rl.training.stage3_rl_canonical` but
restricted to the SAC method and parameterised with the NASDAQ-100
Phase 4 Layer 2 mean-variance QP hyperparameters (gamma=5, per-name
cap=0.05, gross=1.0, Ledoit-Wolf 120-day covariance, top-200 names).

The canonical SAC architecture, observation layout, reward function,
and hyperparameters from
:mod:`invar_rl.layer3_control.{agent, env, observation, reward}` are
reused byte-for-byte. The only Phase 5 additions over the canonical
SP500 driver are:

1. A per-checkpoint validation Sharpe selector. Every
   ``--eval-freq`` env steps the agent is rolled out deterministically
   on the 2017-H2 + 2018-H2 validation segment (set by
   ``two_regime_val=True``); the checkpoint with the best pooled
   Sharpe is kept in memory and restored for test inference.
2. Test-segment inference is persisted as a per-day exposures +
   realised portfolio returns parquet at
   ``outputs/nasdaq100/layer3/{protocol}/fold{F}_seed{S}.parquet``.
3. A per-cell summary JSON with test pooled Sharpe is written to
   ``outputs/nasdaq100/layer3/{protocol}/summary/fold{F}_seed{S}.json``.

CLI::

    python -m invar_rl.training.nasdaq100_layer3_sac \\
        --fold F --seed S --protocol {ls|lo} \\
        --total-timesteps 20000 \\
        --output-dir-root outputs/nasdaq100/layer3
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

from invar_rl.common.config import (
    Layer2Config,
    load_layer3_config,
    load_stage3_config,
)
from invar_rl.common.seeding import set_global_seed
from invar_rl.data.lattice_bridge import build_lattice_bridge
from invar_rl.layer1_ranker.canonical_runner import load_trained_invar
from invar_rl.layer3_control.env import ExposureEnv
from invar_rl.layer3_control.precompute import EpisodeTape
from invar_rl.layer3_control.precompute_canonical import (
    precompute_tape_canonical,
)

# NASDAQ-100 Phase 4 Layer 2 QP hyperparameters (must match
# invar_rl.training.nasdaq100_layer2_qp). Ledoit-Wolf is the canonical
# covariance estimator there; the QP solves at gamma=5, per-name cap
# 0.05, gross=1.0, lookback=120, top-200 names.
_NASDAQ100_LAYER2: Dict[str, object] = dict(
    estimator="ledoit_wolf",
    factor_rank=8,            # ignored under ledoit_wolf
    cov_lookback=120,
    risk_aversion=5.0,
    per_name_bound=0.05,
    gross_leverage=1.0,
    topk_k=25,
    topk_temperature=0.5,
    topk_temperature_anneal=True,
)
# Trading-day count used to annualise Sharpe.
_TRADING_DAYS = 252


def _build_layer2_cfg() -> Layer2Config:
    """Construct the NASDAQ-100 Layer 2 config used by the QP precompute."""
    return Layer2Config(**_NASDAQ100_LAYER2)


def _rollout_returns(env: ExposureEnv, agent) -> Tuple[np.ndarray, np.ndarray]:
    """Run one deterministic rollout; return realised returns + exposures."""
    obs, _ = env.reset(seed=0)
    rets: List[float] = []
    exps: List[float] = []
    while True:
        action, _ = agent.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        obs, _, term, trunc, info = env.step(action)
        rets.append(float(info["strategy_return"]))
        exps.append(float(info["exposure"]))
        if term or trunc:
            break
    return np.asarray(rets, dtype=np.float64), np.asarray(exps, dtype=np.float64)


def _pooled_sharpe(returns: np.ndarray) -> float:
    """Annualised Sharpe with sqrt(252) scaling; zero if degenerate."""
    if returns.size < 2:
        return 0.0
    sd = float(returns.std(ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(returns.mean() / sd * np.sqrt(_TRADING_DAYS))


class _ValSharpeSelector:
    """SB3 callback that picks the checkpoint with the best val Sharpe.

    Every ``eval_freq`` env steps, evaluates the current agent
    deterministically on ``val_env`` and, if the pooled Sharpe beats
    the running best, snapshots the agent's parameters into an
    in-memory buffer. After training, :meth:`restore_best` loads the
    snapshot back into the agent.
    """

    def __init__(self, val_env: ExposureEnv, eval_freq: int) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        self._val_env = val_env
        self._eval_freq = int(max(1, eval_freq))
        self.best_sharpe = -np.inf
        self.best_step = 0
        self.eval_history: List[Dict[str, float]] = []
        self._buffer: Optional[bytes] = None

        outer = self

        class _Inner(BaseCallback):
            def _on_step(self_inner) -> bool:  # type: ignore[override]
                if self_inner.num_timesteps % outer._eval_freq != 0:
                    return True
                rets, _ = _rollout_returns(outer._val_env, self_inner.model)
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
                    buf = io.BytesIO()
                    self_inner.model.save(buf)
                    outer._buffer = buf.getvalue()
                return True

        self.callback = _Inner()

    def restore_best(self, agent) -> bool:
        """Load the best-val checkpoint back into ``agent``. Returns success."""
        if self._buffer is None:
            return False
        from stable_baselines3 import SAC

        buf = io.BytesIO(self._buffer)
        env = agent.get_env()
        restored = SAC.load(buf, env=env, device=agent.device)
        agent.policy.load_state_dict(restored.policy.state_dict())
        return True


def _persist_test_outputs(
    tape: EpisodeTape,
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
    protocol: str,
    ckpt_path: Path,
    layer3_yaml: Path,
    stage3_yaml: Path,
    total_timesteps: int,
    eval_freq: int,
    output_dir_root: Path,
    panel_end: str,
    panel_kind: str = "nasdaq100",
    two_regime_val: bool = True,
    weighting_mode: str = "qp",
    equal_topk_k: int = 50,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Train SAC for one cell; persist exposures + summary."""
    if protocol not in ("ls", "lo"):
        raise ValueError(f"protocol must be 'ls' or 'lo', got {protocol!r}")
    if weighting_mode not in ("qp", "equal_topk"):
        raise ValueError(
            f"weighting_mode must be 'qp' or 'equal_topk', "
            f"got {weighting_mode!r}"
        )
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
    bridge = build_lattice_bridge(cfg, device=device)
    bundle = load_trained_invar(
        ckpt_path=ckpt_path, bridge=bridge, device=device
    )

    long_only = (protocol == "lo")
    print(
        f"[nasdaq100_layer3] fold={fold} seed={seed} protocol={protocol} "
        f"long_only={long_only} device={device}",
        flush=True,
    )
    t0 = time.time()
    train_tape = precompute_tape_canonical(
        bundle=bundle, bridge=bridge,
        day_indices=list(bridge.train_idx),
        layer2=layer2, stride=stage3.precompute_stride,
        long_only=long_only,
        weighting_mode=weighting_mode,
        equal_topk_k=int(equal_topk_k),
    )
    val_tape = precompute_tape_canonical(
        bundle=bundle, bridge=bridge,
        day_indices=list(bridge.val_idx),
        layer2=layer2, stride=stage3.precompute_stride,
        long_only=long_only,
        weighting_mode=weighting_mode,
        equal_topk_k=int(equal_topk_k),
    )
    test_tape = precompute_tape_canonical(
        bundle=bundle, bridge=bridge,
        day_indices=list(bridge.test_idx),
        layer2=layer2, stride=stage3.precompute_stride,
        long_only=long_only,
        weighting_mode=weighting_mode,
        equal_topk_k=int(equal_topk_k),
    )
    print(
        f"[nasdaq100_layer3] precompute tape sizes: "
        f"train={len(train_tape)} val={len(val_tape)} test={len(test_tape)} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    train_env = Monitor(
        ExposureEnv(train_tape, layer3, bootstrap_episode=True)
    )
    val_env = ExposureEnv(val_tape, layer3, bootstrap_episode=False)
    test_env = ExposureEnv(test_tape, layer3, bootstrap_episode=False)

    # Build SAC with the canonical agent factory (twin Q, lr=3e-4,
    # batch_size=256, etc.; see invar_rl.layer3_control.agent.build_agent).
    from invar_rl.layer3_control.agent import build_agent
    agent = build_agent("sac", train_env, stage3, seed)

    selector = _ValSharpeSelector(val_env=val_env, eval_freq=eval_freq)
    print(
        f"[nasdaq100_layer3] training SAC for {total_timesteps:,} steps "
        f"(val every {eval_freq:,})",
        flush=True,
    )
    agent.learn(
        total_timesteps=int(total_timesteps),
        callback=selector.callback,
        progress_bar=False,
    )

    restored = selector.restore_best(agent)
    print(
        f"[nasdaq100_layer3] best val Sharpe={selector.best_sharpe:+.4f} "
        f"at step={selector.best_step} restored={restored}",
        flush=True,
    )

    rets, exps = _rollout_returns(test_env, agent)
    out_dir = output_dir_root / protocol
    out_path = out_dir / f"fold{fold}_seed{seed}.parquet"
    test_stats = _persist_test_outputs(
        tape=test_tape, rets=rets, exps=exps,
        bridge=bridge, out_path=out_path,
    )
    print(
        f"[nasdaq100_layer3] wrote {out_path} "
        f"(test pooled Sharpe={test_stats['sharpe']:+.4f}, "
        f"n_days={test_stats['n_test_days']})",
        flush=True,
    )

    summary_dir = output_dir_root / protocol / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"fold{fold}_seed{seed}.json"
    payload = {
        "universe": "nasdaq100",
        "fold": int(fold),
        "seed": int(seed),
        "protocol": protocol,
        "model": "InVAR-RL Phase 5 Layer 3 SAC (canonical SP500 architecture)",
        "panel_kind": panel_kind,
        "two_regime_val": bool(two_regime_val),
        "panel_end": panel_end,
        "long_only": bool(long_only),
        "n_train_steps": int(len(train_tape)),
        "n_val_steps": int(len(val_tape)),
        "n_test_steps": int(len(test_tape)),
        "total_timesteps": int(total_timesteps),
        "eval_freq": int(eval_freq),
        "best_val_sharpe": float(selector.best_sharpe),
        "best_val_step": int(selector.best_step),
        "best_val_restored": bool(restored),
        "val_history": selector.eval_history,
        "test_pooled_sharpe": float(test_stats["sharpe"]),
        "test_mean_return": float(test_stats["mean"]),
        "test_std_return": float(test_stats["std"]),
        "test_n_days": int(test_stats["n_test_days"]),
        "test_out_path": str(out_path),
        "layer2_qp": _NASDAQ100_LAYER2,
        "layer3_yaml": str(layer3_yaml),
        "stage3_yaml": str(stage3_yaml),
        "wall_time_seconds": float(time.time() - t0),
    }
    with open(summary_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"[nasdaq100_layer3] wrote {summary_path}", flush=True)
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NASDAQ-100 Phase 5 Layer 3 SAC controller training."
    )
    p.add_argument("--fold", type=int, required=True,
                   choices=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--protocol", type=str, required=True,
                   choices=["ls", "lo"])
    p.add_argument("--total-timesteps", type=int, default=20000)
    p.add_argument("--eval-freq", type=int, default=2000,
                   help="env steps between val Sharpe checkpoint evaluations")
    p.add_argument("--output-dir-root", type=str,
                   default="outputs/nasdaq100/layer3")
    p.add_argument("--layer1-ckpt-root", type=str,
                   default="outputs/nasdaq100/layer1/_ckpt")
    p.add_argument("--layer3", type=str,
                   default="invar_rl/configs/layer3.yaml")
    p.add_argument("--stage3", type=str,
                   default="invar_rl/configs/stage3.yaml")
    p.add_argument("--panel-end", type=str, default="2025-12-31")
    p.add_argument(
        "--weighting-mode", type=str, default="qp",
        choices=["qp", "equal_topk"],
        help=(
            "Wrapper weighting. 'qp' (default) = canonical QP allocator; "
            "'equal_topk' = fixed equal-weight top-K per side (fair-K "
            "comparison vs ranker baselines)."
        ),
    )
    p.add_argument(
        "--equal-topk-k", type=int, default=50,
        help=(
            "Per-side K for weighting_mode='equal_topk' (ignored under "
            "'qp'). NDX fair-K comparison uses 25."
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
            f"seed={args.seed}: {ckpt_path}; rerun Phase 3 for this cell"
        )
    out_root = Path(args.output_dir_root)
    out_path = out_root / args.protocol / (
        f"fold{args.fold}_seed{args.seed}.parquet"
    )
    summary_path = out_root / args.protocol / "summary" / (
        f"fold{args.fold}_seed{args.seed}.json"
    )
    if out_path.exists() and summary_path.exists():
        print(
            f"[nasdaq100_layer3] {out_path} + summary exist; skipping cell",
            flush=True,
        )
        return 0
    run_one_cell(
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
        weighting_mode=args.weighting_mode,
        equal_topk_k=int(args.equal_topk_k),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
