"""InVAR-RL Stage 3 RL: train recurrent/feedforward PPO + SAC on canonical L1+L2.

Trains the Layer 3 reinforcement-learning controllers (the methods in
``RL_METHODS``) on top of the canonical InVAR (Layer 1, frozen) and the
mean-variance QP (Layer 2, frozen) precompute tape produced by
:func:`invar_rl.layer3_control.precompute_canonical.precompute_tape_canonical`.

This is the v1 of the InVAR-RL three-layer story: stages 1 and 2 are
the canonical-InVAR-backed evaluations already in
``invar_rl/training/{stage1_rank, stage2_eval}.py``; stage 3 here adds
the RL controller, completing the original three-layer design with
the canonical (not the stripped skeleton) Layer 1.

Per-(fold, seed) JSON is written to
``invar_rl/results/stage3_rl/foldF_seedS.json`` with one sub-dict per
RL method, holding mean_reward, mean_return, volatility,
final_equity, n_steps measured on the test tape.

Usage::

    python -m invar_rl.training.stage3_rl_canonical \
        --fold 1 --seed 42 \
        --layer1-ckpt invar_rl/results/stage1/_ckpt/fold1_seed42_full.pt
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
from invar_rl.layer1_ranker.canonical_runner import load_trained_invar
from invar_rl.layer3_control.agent import RL_METHODS, build_agent
from invar_rl.layer3_control.env import ExposureEnv
from invar_rl.layer3_control.precompute_canonical import (
    precompute_tape_canonical,
)


def _eval_rl_actor(env: ExposureEnv, agent, recurrent: bool) -> Dict:
    """Run one deterministic evaluation episode."""
    obs, _ = env.reset(seed=0)
    state, starts = None, np.ones((1,), dtype=bool)
    rewards: List[float] = []
    rets: List[float] = []
    info: Dict = {}
    while True:
        if recurrent:
            action, state = agent.predict(
                obs,
                state=state,
                episode_start=starts,
                deterministic=True,
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
    fold: int,
    seed: int,
    ckpt_path: Path,
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
    long_only: bool = False,
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
    bundle = load_trained_invar(
        ckpt_path=ckpt_path, bridge=bridge, device=device
    )

    train_tape = precompute_tape_canonical(
        bundle=bundle, bridge=bridge,
        day_indices=list(bridge.train_idx),
        layer2=layer2, stride=stage3.precompute_stride,
        long_only=long_only,
    )
    val_tape = precompute_tape_canonical(
        bundle=bundle, bridge=bridge,
        day_indices=list(bridge.val_idx),
        layer2=layer2, stride=stage3.precompute_stride,
        long_only=long_only,
    )
    test_tape = precompute_tape_canonical(
        bundle=bundle, bridge=bridge,
        day_indices=list(bridge.test_idx),
        layer2=layer2, stride=stage3.precompute_stride,
        long_only=long_only,
    )

    from stable_baselines3.common.monitor import Monitor

    ckpt_out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict] = {}
    for method in methods:
        if method not in RL_METHODS:
            raise ValueError(
                f"stage3_rl only supports RL_METHODS {RL_METHODS}; "
                f"got {method!r}. Use stage3_eval for non-RL baselines."
            )
        curve_dir = ckpt_out_dir / f"{method}_curves_f{fold}_s{seed}"
        curve_dir.mkdir(parents=True, exist_ok=True)
        train_env = Monitor(
            ExposureEnv(
                train_tape, layer3, bootstrap_episode=True
            ),
            filename=str(curve_dir / "monitor"),
        )
        agent = build_agent(method, train_env, stage3, seed)
        agent.learn(total_timesteps=stage3.total_timesteps)
        agent_path = ckpt_out_dir / f"{method}_f{fold}_s{seed}.zip"
        agent.save(str(agent_path))

        eval_env = ExposureEnv(
            test_tape, layer3, bootstrap_episode=False
        )
        perf = _eval_rl_actor(
            eval_env, agent, recurrent=(method == "recurrent_ppo")
        )
        results[method] = perf

    payload = {
        "fold": fold,
        "seed": seed,
        "model": "InVAR-RL stage3 RL (canonical InVAR + QP + L3 RL)",
        "n_train_steps": len(train_tape),
        "n_val_steps": len(val_tape),
        "n_test_steps": len(test_tape),
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
    print(f"[InVAR-RL stage 3 RL] wrote {out_path}")
    for m, perf in results.items():
        print(
            f"  {m:18s} mean_return={perf['mean_return']:+.5f} "
            f"vol={perf['volatility']:.5f} "
            f"final_equity={perf['final_equity']:.4f}"
        )
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="InVAR-RL stage 3 RL: PPO/SAC on canonical L1+L2."
    )
    p.add_argument(
        "--fold", type=int, required=True, choices=[1, 2, 3, 4, 5]
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--layer1-ckpt", type=str, required=True)
    p.add_argument(
        "--layer2", type=str, default="invar_rl/configs/layer2.yaml"
    )
    p.add_argument(
        "--layer3", type=str, default="invar_rl/configs/layer3.yaml"
    )
    p.add_argument(
        "--stage3", type=str, default="invar_rl/configs/stage3.yaml"
    )
    p.add_argument(
        "--output-dir", type=str, default="invar_rl/results/stage3_rl"
    )
    p.add_argument(
        "--ckpt-out-dir",
        type=str,
        default="invar_rl/results/stage3_rl/_ckpt",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="recurrent_ppo,feedforward_ppo,sac",
        help="Comma-separated RL methods from RL_METHODS.",
    )
    p.add_argument(
        "--panel_kind",
        type=str,
        default="lattice_native",
        choices=["biotech", "lattice_native"],
    )
    p.add_argument("--panel_end", type=str, default="2025-12-31")
    p.add_argument("--two_regime_val", action="store_true", default=True)
    p.add_argument(
        "--long-only", action="store_true", default=False,
        help="Use long-only fully-invested QP in Layer 2 (apples-to-apples vs FinRL).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print(
        f"[InVAR-RL stage 3 RL] fold={args.fold} seed={args.seed} "
        f"methods={methods} device={device}"
    )
    run_one_cell(
        fold=args.fold,
        seed=args.seed,
        ckpt_path=Path(args.layer1_ckpt),
        layer2_yaml=Path(args.layer2),
        layer3_yaml=Path(args.layer3),
        stage3_yaml=Path(args.stage3),
        output_dir=Path(args.output_dir),
        ckpt_out_dir=Path(args.ckpt_out_dir),
        panel_kind=args.panel_kind,
        panel_end=args.panel_end,
        two_regime_val=args.two_regime_val,
        methods=methods,
        device=device,
        long_only=args.long_only,
    )


if __name__ == "__main__":
    main()
