"""InVAR-RL Stage 3 RL using an EXTERNAL Layer-1 baseline.

The Layer-2 cvxpy MV-QP and the Layer-3 RL controllers (recurrent PPO,
feedforward PPO, SAC) are byte-identical to
``invar_rl.training.stage3_rl_canonical``; the only difference is that
Layer 1 is sourced from a precomputed baseline npz instead of from the
canonical InVAR forward pass. This produces a clean "swap-Layer-1"
whole-stack comparison: the difference in final portfolio Sharpe is
attributable to the Layer-1 ranker because Layer 2 + Layer 3 are held
fixed.

Supported baselines (from ``results/baselines_universal_two_regime_val/``):
master, factorvae, itransformer, stockmixer, dystage, mera, swa_invar.

Per-(fold, seed) JSON at
``invar_rl/results/stage3_rl_external/{baseline}/foldF_seedS.json``.
Schema matches ``stage3_rl_canonical`` so the comparison table in the
paper is paper-grade.

Usage::

    python -m invar_rl.training.stage3_rl_external \
        --baseline master --fold 1 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.invar import InVARConfig

from invar_rl.common.config import (
    load_layer2_config,
    load_layer3_config,
    load_stage3_config,
)
from invar_rl.common.seeding import set_global_seed
from invar_rl.data.lattice_bridge import build_lattice_bridge
from invar_rl.layer1_ranker.external_bundle import load_external_baseline
from invar_rl.layer3_control.agent import RL_METHODS, build_agent
from invar_rl.layer3_control.env import ExposureEnv
from invar_rl.layer3_control.precompute_canonical import (
    precompute_tape_canonical,
)


SUPPORTED_BASELINES = (
    "master", "factorvae", "itransformer", "stockmixer",
    "dystage", "mera", "swa_invar",
)


def _eval_rl_actor(env: ExposureEnv, agent, recurrent: bool) -> Dict:
    obs, _ = env.reset(seed=0)
    state, starts = None, np.ones((1,), dtype=bool)
    rewards: List[float] = []
    rets: List[float] = []
    info: Dict = {}
    while True:
        if recurrent:
            action, state = agent.predict(
                obs, state=state, episode_start=starts, deterministic=True
            )
        else:
            action, _ = agent.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
        starts = np.zeros((1,), dtype=bool)
        rewards.append(float(reward))
        rets.append(float(info["strategy_return"]))
        if term or trunc:
            break
    arr = np.asarray(rets)
    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_return": float(arr.mean()) if arr.size else 0.0,
        "volatility": float(arr.std()) if arr.size else 0.0,
        "final_equity": float(info.get("equity", 1.0)),
        "n_steps": len(rewards),
    }


def run_one_cell(
    baseline: str,
    fold: int,
    seed: int,
    npz_root: str,
    layer2_yaml: Path,
    layer3_yaml: Path,
    stage3_yaml: Path,
    output_dir: Path,
    ckpt_out_dir: Path,
    panel_kind: str,
    panel_end: str,
    two_regime_val: bool,
    methods: List[str],
    device: torch.device,
) -> Dict:
    set_global_seed(seed)
    layer2 = load_layer2_config(str(layer2_yaml))
    layer3 = load_layer3_config(str(layer3_yaml))
    stage3 = load_stage3_config(str(stage3_yaml))

    cfg = InVARConfig(fold=fold, seed=seed)
    cfg.panel_kind = panel_kind
    cfg.two_regime_val = two_regime_val
    cfg.panel_end = panel_end
    bridge = build_lattice_bridge(cfg)
    bundle = load_external_baseline(
        baseline_name=baseline,
        fold=fold,
        seed=seed,
        bridge=bridge,
        npz_root=npz_root,
    )

    train_tape = precompute_tape_canonical(
        bundle=bundle, bridge=bridge,
        day_indices=list(bridge.train_idx),
        layer2=layer2, stride=stage3.precompute_stride,
    )
    val_tape = precompute_tape_canonical(
        bundle=bundle, bridge=bridge,
        day_indices=list(bridge.val_idx),
        layer2=layer2, stride=stage3.precompute_stride,
    )
    test_tape = precompute_tape_canonical(
        bundle=bundle, bridge=bridge,
        day_indices=list(bridge.test_idx),
        layer2=layer2, stride=stage3.precompute_stride,
    )

    from stable_baselines3.common.monitor import Monitor

    ckpt_out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict] = {}
    for method in methods:
        if method not in RL_METHODS:
            raise ValueError(
                f"only RL_METHODS supported: {RL_METHODS}; got {method!r}"
            )
        curve_dir = (
            ckpt_out_dir / f"{method}_curves_{baseline}_f{fold}_s{seed}"
        )
        curve_dir.mkdir(parents=True, exist_ok=True)
        train_env = Monitor(
            ExposureEnv(train_tape, layer3, bootstrap_episode=True),
            filename=str(curve_dir / "monitor"),
        )
        agent = build_agent(method, train_env, stage3, seed)
        agent.learn(total_timesteps=stage3.total_timesteps)
        agent.save(
            str(ckpt_out_dir
                / f"{method}_{baseline}_f{fold}_s{seed}.zip")
        )
        eval_env = ExposureEnv(
            test_tape, layer3, bootstrap_episode=False
        )
        perf = _eval_rl_actor(
            eval_env, agent, recurrent=(method == "recurrent_ppo")
        )
        results[method] = perf

    tape_ret = test_tape.base_return.astype(float)
    tape_mean = float(tape_ret.mean()) if tape_ret.size else 0.0
    tape_vol = float(tape_ret.std()) if tape_ret.size else 0.0

    payload = {
        "baseline": baseline,
        "fold": fold,
        "seed": seed,
        "model": (
            f"InVAR-RL stage3 RL external L1={baseline} "
            f"(canonical L2+L3, swap-L1 comparison)"
        ),
        "n_train_steps": len(train_tape),
        "n_val_steps": len(val_tape),
        "n_test_steps": len(test_tape),
        "tape_constant_full_mean_return": tape_mean,
        "tape_constant_full_volatility": tape_vol,
        "methods": results,
        "config": {
            "panel_kind": panel_kind,
            "two_regime_val": two_regime_val,
            "panel_end": panel_end,
            "precompute_stride": stage3.precompute_stride,
            "total_timesteps": stage3.total_timesteps,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"fold{fold}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[InVAR-RL stage 3 external L1={baseline}] wrote {out_path}")
    for m, perf in results.items():
        print(
            f"  {m:18s} mean_return={perf['mean_return']:+.5f} "
            f"vol={perf['volatility']:.5f} "
            f"final_equity={perf['final_equity']:.4f}"
        )
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="InVAR-RL stage 3 RL with EXTERNAL Layer-1 baseline."
    )
    p.add_argument(
        "--baseline", type=str, required=True,
        choices=list(SUPPORTED_BASELINES),
    )
    p.add_argument("--fold", type=int, required=True, choices=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--npz-root", type=str,
        default="results/baselines_universal_two_regime_val",
    )
    p.add_argument("--layer2", type=str, default="invar_rl/configs/layer2.yaml")
    p.add_argument("--layer3", type=str, default="invar_rl/configs/layer3.yaml")
    p.add_argument("--stage3", type=str, default="invar_rl/configs/stage3.yaml")
    p.add_argument(
        "--output-dir-root", type=str,
        default="invar_rl/results/stage3_rl_external",
    )
    p.add_argument(
        "--methods", type=str,
        default="recurrent_ppo,feedforward_ppo,sac",
    )
    p.add_argument(
        "--panel_kind", type=str, default="lattice_native",
        choices=["biotech", "lattice_native"],
    )
    p.add_argument("--panel_end", type=str, default="2025-12-31")
    p.add_argument("--two_regime_val", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    out_dir = Path(args.output_dir_root) / args.baseline
    ckpt_out_dir = out_dir / "_ckpt"
    print(
        f"[InVAR-RL stage 3 external L1={args.baseline}] "
        f"fold={args.fold} seed={args.seed} methods={methods} device={device}"
    )
    run_one_cell(
        baseline=args.baseline,
        fold=args.fold,
        seed=args.seed,
        npz_root=args.npz_root,
        layer2_yaml=Path(args.layer2),
        layer3_yaml=Path(args.layer3),
        stage3_yaml=Path(args.stage3),
        output_dir=out_dir,
        ckpt_out_dir=ckpt_out_dir,
        panel_kind=args.panel_kind,
        panel_end=args.panel_end,
        two_regime_val=args.two_regime_val,
        methods=methods,
        device=device,
    )


if __name__ == "__main__":
    main()
