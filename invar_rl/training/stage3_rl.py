"""Stage 3: train the Layer 3 exposure controller and the baselines.

Layers 1 and 2 are frozen. Their per-day outputs are precomputed into an
episode tape; the controller and all baselines see only those detached
values through the environment. No reinforcement-learning signal can reach
Layer 1 or Layer 2: the lower stack is loaded in eval mode with gradients
disabled and is never optimised here.

Trains the recurrent-PPO controller and the RL baselines on the exact-replay
environment across the seed set, evaluates every exposure-control method
(RL and non-RL) on the validation segment, saves learning curves and
checkpoints, and writes a first results table. Stage 4 joint fine-tuning is
not implemented.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from invar_rl.baselines.exposure_baselines import (
    ConstantFullExposure,
    MyopicExposureHead,
    VolatilityTargeting,
)
from invar_rl.common.config import (
    load_base_config,
    load_folds_config,
    load_layer1_config,
    load_layer2_config,
    load_layer3_config,
    load_stage3_config,
)
from invar_rl.common.logging_utils import get_logger
from invar_rl.common.seeding import set_global_seed
from invar_rl.common.splits import WalkForwardSplitter
from invar_rl.data.panel_factory import build_panel, build_splits
from invar_rl.layer1_ranker.invar import INVAR
from invar_rl.layer2_alloc.qp_layer import MeanVarianceQP
from invar_rl.layer3_control.agent import RL_METHODS, build_agent
from invar_rl.layer3_control.env import ExposureEnv
from invar_rl.layer3_control.observation import RiskState
from invar_rl.layer3_control.precompute import precompute_tape

LOGGER = get_logger(__name__)


def _load_frozen_layer1(base, layer1, ckpt_dir, lower_stack, variant,
                        fold, seed, panel) -> INVAR:
    """Load the frozen lower-stack Layer 1 (no gradients are taken here)."""
    model = INVAR(
        layer1.model,
        n_features=panel.n_features,
        lookback=panel.lookback,
        macro_dim=panel.macro_dim,
    )
    if lower_stack == "stage2":
        path = ckpt_dir / f"stage2_{variant}_{fold}_seed{seed}.pt"
    else:
        path = ckpt_dir / f"layer1_{fold}_seed{seed}.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"frozen lower-stack checkpoint not found: {path}. Run the "
            f"prerequisite stage first."
        )
    blob = torch.load(path, map_location="cpu")
    model.load_state_dict(blob["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _eval_through_env(env: ExposureEnv, actor, recurrent: bool) -> Dict:
    """Run one deterministic evaluation episode and summarise it."""
    obs, _ = env.reset(seed=0)
    state, starts = None, np.ones((1,), dtype=bool)
    rewards: List[float] = []
    rets: List[float] = []
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
        "final_equity": float(info["equity"]),
        "n_steps": len(rewards),
    }


def _rl_actor(model, recurrent: bool):
    def actor(obs, state, starts):
        if recurrent:
            a, state = model.predict(
                obs, state=state, episode_start=starts, deterministic=True
            )
            return np.asarray(a, dtype=np.float32), state
        a, _ = model.predict(obs, deterministic=True)
        return np.asarray(a, dtype=np.float32), None

    return actor


def _policy_actor(policy, tape, env_cfg):
    """Adapter so a non-RL ExposurePolicy can drive the same env."""
    step = {"t": 0}
    risk = RiskState(exposure=env_cfg.exposure_min)
    hwm = {"v": 1.0, "eq": 1.0}
    hist: List[float] = []

    def actor(obs, state, starts):
        t = step["t"]
        exp = policy.exposure(tape, min(t, len(tape) - 1), risk)
        step["t"] = t + 1
        return np.array([exp], dtype=np.float32), None

    return actor, risk, hwm, hist


def run(
    base_path: str,
    layer1_path: str,
    layer2_path: str,
    layer3_path: str,
    stage3_path: str,
    folds_path: str,
    seeds: Optional[List[int]],
    fold_names: Optional[List[str]],
) -> Dict:
    base = load_base_config(base_path)
    layer1 = load_layer1_config(layer1_path)
    layer2 = load_layer2_config(layer2_path)
    layer3 = load_layer3_config(layer3_path)
    stage3 = load_stage3_config(stage3_path)
    splits = build_splits(base, folds_path)
    if fold_names:
        splits = [s for s in splits if s.name in set(fold_names)]
    use_seeds = seeds or base.seeds

    ckpt_dir = Path(base.paths.checkpoint_dir)
    curve_dir = Path(base.paths.output_dir) / "stage3_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    raw: List[dict] = []

    for seed in use_seeds:
        for fold in splits:
            set_global_seed(seed)
            panel = build_panel(
                base, seed=seed,
                train_end_index=int(fold.train_idx[-1]),
            )
            frozen = _load_frozen_layer1(
                base, layer1, ckpt_dir, stage3.lower_stack,
                stage3.stage2_variant, fold.name, seed, panel,
            )
            qp = MeanVarianceQP(layer2)
            ts = int(fold.train_idx[0])
            train_tape = precompute_tape(
                frozen, qp, panel, list(fold.train_idx), layer2, ts,
                stride=stage3.precompute_stride,
            )
            val_tape = precompute_tape(
                frozen, qp, panel, list(fold.val_idx), layer2, ts,
                stride=stage3.precompute_stride,
            )

            for method in stage3.methods:
                if method in RL_METHODS:
                    from stable_baselines3.common.monitor import Monitor

                    mdir = curve_dir / f"{method}_{fold.name}_seed{seed}"
                    mdir.mkdir(parents=True, exist_ok=True)
                    train_env = Monitor(
                        ExposureEnv(
                            train_tape, layer3, bootstrap_episode=True
                        ),
                        filename=str(mdir / "monitor"),
                    )
                    agent = build_agent(method, train_env, stage3, seed)
                    agent.learn(total_timesteps=stage3.total_timesteps)
                    agent.save(str(ckpt_dir / f"stage3_{method}_"
                                    f"{fold.name}_seed{seed}"))
                    eval_env = ExposureEnv(
                        val_tape, layer3, bootstrap_episode=False
                    )
                    perf = _eval_through_env(
                        eval_env,
                        _rl_actor(agent, method == "recurrent_ppo"),
                        recurrent=(method == "recurrent_ppo"),
                    )
                else:
                    if method == "constant_full":
                        policy = ConstantFullExposure(layer3)
                    elif method == "vol_target":
                        policy = VolatilityTargeting(layer3, stage3)
                    elif method == "myopic_head":
                        policy = MyopicExposureHead(
                            layer3, stage3,
                            obs_dim=7 + train_tape.macro_dim,
                        )
                        policy.fit(train_tape, seed)
                    else:
                        raise ValueError(f"unknown method {method!r}")
                    eval_env = ExposureEnv(
                        val_tape, layer3, bootstrap_episode=False
                    )
                    actor, *_ = _policy_actor(policy, val_tape, layer3)
                    perf = _eval_through_env(
                        eval_env, actor, recurrent=False
                    )

                LOGGER.info(
                    "method %s fold %s seed %d: mean_reward %.5f "
                    "mean_return %.6f vol %.6f equity %.4f",
                    method, fold.name, seed, perf["mean_reward"],
                    perf["mean_return"], perf["volatility"],
                    perf["final_equity"],
                )
                raw.append(
                    {"method": method, "fold": fold.name,
                     "seed": seed, **perf}
                )

    table: Dict[str, dict] = {}
    for fold in splits:
        table[fold.name] = {}
        for method in stage3.methods:
            cells = [
                r for r in raw
                if r["fold"] == fold.name and r["method"] == method
            ]
            if not cells:
                continue
            table[fold.name][method] = {
                "mean_reward": float(
                    np.mean([c["mean_reward"] for c in cells])
                ),
                "mean_return": float(
                    np.mean([c["mean_return"] for c in cells])
                ),
                "volatility": float(
                    np.mean([c["volatility"] for c in cells])
                ),
                "final_equity": float(
                    np.mean([c["final_equity"] for c in cells])
                ),
                "n_seeds": len(cells),
            }

    out_path = Path(base.paths.output_dir) / "stage3_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({"per_fold": table, "raw": raw}, fh, indent=2)
    LOGGER.info("Stage 3 first results table (per fold, seed-mean):")
    for fname, methods in table.items():
        for method, m in methods.items():
            LOGGER.info(
                "  %s %s: reward %.5f return %.6f vol %.6f equity %.4f",
                fname, method, m["mean_reward"], m["mean_return"],
                m["volatility"], m["final_equity"],
            )
    LOGGER.info("Stage 3 results written to %s", out_path)
    return {"per_fold": table}


def _parse_args() -> argparse.Namespace:
    cfg = Path(__file__).resolve().parents[2] / "configs"
    p = argparse.ArgumentParser(description="Stage 3: RL controller.")
    p.add_argument("--base", default=str(cfg / "base.yaml"))
    p.add_argument("--layer1", default=str(cfg / "layer1.yaml"))
    p.add_argument("--layer2", default=str(cfg / "layer2.yaml"))
    p.add_argument("--layer3", default=str(cfg / "layer3.yaml"))
    p.add_argument("--stage3", default=str(cfg / "stage3.yaml"))
    p.add_argument("--folds", default=str(cfg / "folds.yaml"))
    p.add_argument("--seeds", default=None)
    p.add_argument("--fold-names", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    fold_names = (
        args.fold_names.split(",") if args.fold_names else None
    )
    run(
        args.base, args.layer1, args.layer2, args.layer3,
        args.stage3, args.folds, seeds, fold_names,
    )
    LOGGER.info("Stage 3 training complete")


if __name__ == "__main__":
    main()
