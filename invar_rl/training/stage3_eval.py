"""InVAR-RL Stage 3 eval: layer-3 baselines on canonical InVAR + QP.

v0 of stage 3 in the InVAR-RL build. Layer 1 (canonical InVAR) and
Layer 2 (mean-variance QP) are frozen, their per-day outputs are
precomputed into an EpisodeTape via
:func:`invar_rl.layer3_control.precompute_canonical.precompute_tape_canonical`,
and the layer-3 non-RL baselines (constant exposure, volatility
targeting, myopic exposure head) drive the
:class:`invar_rl.layer3_control.env.ExposureEnv` for evaluation.

RL training (recurrent PPO etc) is intentionally deferred: this
entry-point validates the canonical-InVAR-as-layer-1 pipeline through
the layer-3 environment without taking on the stable_baselines3
dependency or the long RL training time. Once this passes on Wulver,
the RL-training entry can be added in a follow-up that reuses
``precompute_tape_canonical``.

Output is per-(fold, seed) JSON at
``invar_rl/results/stage3_eval/foldF_seedS.json`` carrying one
sub-dict per method (``constant_full``, ``vol_target``,
``myopic_head``) with mean_reward, mean_return, volatility,
final_equity, n_steps.

Usage::

    python -m invar_rl.training.stage3_eval \
        --fold 1 --seed 42 \
        --layer1-ckpt invar_rl/results/stage1/_ckpt/fold1_seed42_full.pt \
        --layer2 invar_rl/configs/layer2.yaml \
        --layer3 invar_rl/configs/layer3.yaml \
        --stage3 invar_rl/configs/stage3.yaml \
        --output-dir invar_rl/results/stage3_eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.invar import InVARConfig

from invar_rl.baselines.exposure_baselines import (
    ConstantFullExposure,
    MyopicExposureHead,
    VolatilityTargeting,
)
from invar_rl.common.config import (
    load_layer2_config,
    load_layer3_config,
    load_stage3_config,
)
from invar_rl.common.seeding import set_global_seed
from invar_rl.data.lattice_bridge import build_lattice_bridge
from invar_rl.layer1_ranker.canonical_runner import load_trained_invar
from invar_rl.layer3_control.env import ExposureEnv
from invar_rl.layer3_control.observation import RiskState
from invar_rl.layer3_control.precompute_canonical import (
    precompute_tape_canonical,
)


def _eval_through_env(env: ExposureEnv, actor) -> Dict:
    obs, _ = env.reset(seed=0)
    state, starts = None, np.ones((1,), dtype=bool)
    rewards: List[float] = []
    rets: List[float] = []
    info: Dict = {}
    while True:
        action, state = actor(obs, state, starts)
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


def _policy_actor(policy, tape, layer3):
    step = {"t": 0}
    risk = RiskState(exposure=layer3.exposure_min)

    def actor(obs, state, starts):
        t = step["t"]
        exp = policy.exposure(tape, min(t, len(tape) - 1), risk)
        step["t"] = t + 1
        return np.array([exp], dtype=np.float32), None

    return actor


def run_one_cell(
    fold: int,
    seed: int,
    ckpt_path: Path,
    layer2_yaml: Path,
    layer3_yaml: Path,
    stage3_yaml: Path,
    output_dir: Path,
    panel_kind: str,
    panel_end: str,
    two_regime_val: bool,
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
    bundle = load_trained_invar(
        ckpt_path=ckpt_path, bridge=bridge, device=device
    )

    train_tape = precompute_tape_canonical(
        bundle=bundle,
        bridge=bridge,
        day_indices=list(bridge.train_idx),
        layer2=layer2,
        stride=stage3.precompute_stride,
    )
    val_tape = precompute_tape_canonical(
        bundle=bundle,
        bridge=bridge,
        day_indices=list(bridge.val_idx),
        layer2=layer2,
        stride=stage3.precompute_stride,
    )
    test_tape = precompute_tape_canonical(
        bundle=bundle,
        bridge=bridge,
        day_indices=list(bridge.test_idx),
        layer2=layer2,
        stride=stage3.precompute_stride,
    )

    methods: Dict[str, Dict] = {}
    for name in ("constant_full", "vol_target", "myopic_head"):
        if name == "constant_full":
            policy = ConstantFullExposure(layer3)
        elif name == "vol_target":
            policy = VolatilityTargeting(layer3, stage3)
        elif name == "myopic_head":
            policy = MyopicExposureHead(
                layer3, stage3,
                obs_dim=7 + train_tape.macro_dim,
            )
            policy.fit(train_tape, seed)
        env = ExposureEnv(
            test_tape, layer3, bootstrap_episode=False
        )
        actor = _policy_actor(policy, test_tape, layer3)
        perf = _eval_through_env(env, actor)
        methods[name] = perf

    payload = {
        "fold": fold,
        "seed": seed,
        "model": (
            "InVAR-RL stage3 eval (canonical InVAR + QP + L3 baselines)"
        ),
        "n_train_steps": len(train_tape),
        "n_val_steps": len(val_tape),
        "n_test_steps": len(test_tape),
        "methods": methods,
        "config": {
            "panel_kind": panel_kind,
            "two_regime_val": two_regime_val,
            "panel_end": panel_end,
            "precompute_stride": stage3.precompute_stride,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"fold{fold}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[InVAR-RL stage 3 eval] wrote {out_path}")
    for name, perf in methods.items():
        print(
            f"  {name:14s} mean_return={perf['mean_return']:+.5f} "
            f"vol={perf['volatility']:.5f} "
            f"final_equity={perf['final_equity']:.4f}"
        )
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="InVAR-RL stage 3 eval: L3 baselines on canonical L1+L2."
    )
    p.add_argument("--fold", type=int, required=True, choices=[1, 2, 3, 4, 5])
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
        "--output-dir", type=str, default="invar_rl/results/stage3_eval"
    )
    p.add_argument(
        "--panel_kind",
        type=str,
        default="lattice_native",
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
    print(
        f"[InVAR-RL stage 3 eval] fold={args.fold} seed={args.seed} "
        f"ckpt={args.layer1_ckpt} device={device}"
    )
    run_one_cell(
        fold=args.fold,
        seed=args.seed,
        ckpt_path=Path(args.layer1_ckpt),
        layer2_yaml=Path(args.layer2),
        layer3_yaml=Path(args.layer3),
        stage3_yaml=Path(args.stage3),
        output_dir=Path(args.output_dir),
        panel_kind=args.panel_kind,
        panel_end=args.panel_end,
        two_regime_val=args.two_regime_val,
        device=device,
    )


if __name__ == "__main__":
    main()
