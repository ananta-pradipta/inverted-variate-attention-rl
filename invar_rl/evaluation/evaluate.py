"""Phase 6 orchestrator: full evaluation, CIs, and the decisive analyses.

Loads the frozen lower stack and the saved Stage 3 controllers/baselines,
re-rolls each method deterministically over the validation segment (and the
out-of-distribution stress fold), records the per-step strategy-return and
exposure series, computes the full metric suite per fold with moving-block
bootstrap confidence intervals (emphasis on the OOD fold), and produces the
exposure-trajectory and dissociation analyses plus the recurrent-versus-
feedforward architectural ablation. Heavier ablations (reward-frontier
retrains, QP vs top-k, weakened Layer 1, covariance choice) are separate
costed runs and are not launched here.

Everything is reproducible from the saved checkpoints and the seed set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

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
from invar_rl.evaluation.analyses import (
    dissociation_table,
    exposure_trajectory_figure,
)
from invar_rl.evaluation.bootstrap import moving_block_bootstrap_ci
from invar_rl.evaluation.metrics import (
    annualised_sharpe,
    calmar,
    compute_metrics,
)
from invar_rl.layer1_ranker.invar import INVAR
from invar_rl.layer2_alloc.qp_layer import MeanVarianceQP
from invar_rl.layer3_control.agent import RL_METHODS
from invar_rl.layer3_control.env import ExposureEnv
from invar_rl.layer3_control.precompute import precompute_tape
from invar_rl.training.stage3_rl import _load_frozen_layer1

LOGGER = get_logger(__name__)


def _rollout(env: ExposureEnv, actor, recurrent: bool) -> Tuple[
    np.ndarray, np.ndarray
]:
    """Deterministic episode; return (strategy_return, exposure) series."""
    obs, _ = env.reset(seed=0)
    state, starts = None, np.ones((1,), dtype=bool)
    rets: List[float] = []
    exps: List[float] = []
    while True:
        action, state = actor(obs, state, starts)
        obs, _, term, trunc, info = env.step(action)
        starts = np.zeros((1,), dtype=bool)
        rets.append(float(info["strategy_return"]))
        exps.append(float(info["exposure"]))
        if term or trunc:
            break
    return np.asarray(rets), np.asarray(exps)


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


def _policy_actor(policy, tape):
    step = {"t": 0}
    from invar_rl.layer3_control.observation import RiskState

    risk = RiskState()

    def actor(obs, state, starts):
        t = min(step["t"], len(tape) - 1)
        e = policy.exposure(tape, t, risk)
        step["t"] += 1
        return np.array([e], dtype=np.float32), None

    return actor


def _load_rl(method: str, path: Path):
    if method == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO.load(str(path))
    if method == "feedforward_ppo":
        from stable_baselines3 import PPO

        return PPO.load(str(path))
    from stable_baselines3 import SAC

    return SAC.load(str(path))


def run(
    base_path: str,
    layer1_path: str,
    layer2_path: str,
    layer3_path: str,
    stage3_path: str,
    folds_path: str,
    seeds: Optional[List[int]],
    fold_names: Optional[List[str]] = None,
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
    ckpt = Path(base.paths.checkpoint_dir)
    out_dir = Path(base.paths.output_dir)
    fig_dir = out_dir / "figures"
    need_myopic = "myopic_head" in stage3.methods

    raw: List[dict] = []
    series_store: Dict[Tuple[str, str], List[np.ndarray]] = {}
    traj_store: Dict[Tuple[str, str, int], np.ndarray] = {}

    for seed in use_seeds:
        for fold in splits:
            set_global_seed(seed)
            panel = build_panel(
                base, seed=seed,
                train_end_index=int(fold.train_idx[-1]),
            )
            frozen = _load_frozen_layer1(
                base, layer1, ckpt, stage3.lower_stack,
                stage3.stage2_variant, fold.name, seed, panel,
            )
            qp = MeanVarianceQP(layer2)
            ts = int(fold.train_idx[0])
            val_tape = precompute_tape(
                frozen, qp, panel, list(fold.val_idx), layer2, ts,
                stride=stage3.precompute_stride,
            )
            train_tape = (
                precompute_tape(
                    frozen, qp, panel, list(fold.train_idx), layer2, ts,
                    stride=stage3.precompute_stride,
                )
                if need_myopic
                else None
            )
            for method in stage3.methods:
                env = ExposureEnv(val_tape, layer3,
                                    bootstrap_episode=False)
                if method in RL_METHODS:
                    mpath = ckpt / (
                        f"stage3_{method}_{fold.name}_seed{seed}.zip"
                    )
                    if not mpath.is_file():
                        mpath = ckpt / (
                            f"stage3_{method}_{fold.name}_seed{seed}"
                        )
                    agent = _load_rl(method, mpath)
                    rets, exps = _rollout(
                        env, _rl_actor(agent, method == "recurrent_ppo"),
                        method == "recurrent_ppo",
                    )
                else:
                    if method == "constant_full":
                        pol = ConstantFullExposure(layer3)
                    elif method == "vol_target":
                        pol = VolatilityTargeting(layer3, stage3)
                    else:
                        pol = MyopicExposureHead(
                            layer3, stage3,
                            obs_dim=7 + val_tape.macro_dim,
                        )
                        pol.fit(train_tape, seed)
                    rets, exps = _rollout(
                        env, _policy_actor(pol, val_tape), False
                    )
                m = compute_metrics(
                    rets, exps, val_tape.daily_ic[: rets.size],
                    cvar_level=layer3.cvar_level,
                    horizon=panel.label_horizon,
                )
                raw.append({"method": method, "fold": fold.name,
                            "seed": seed, **m})
                series_store.setdefault((fold.name, method), []).append(
                    rets
                )
                traj_store[(fold.name, method, seed)] = exps
            LOGGER.info(
                "evaluated fold %s seed %d (%d methods)",
                fold.name, seed, len(stage3.methods),
            )

    # Per fold x method seed-mean table, with bootstrap CIs on the OOD fold.
    ood = next((f.name for f in splits if f.is_ood), None)
    table: Dict[str, dict] = {}
    metric_keys = list(raw[0].keys() - {"method", "fold", "seed"})
    for fold in splits:
        table[fold.name] = {}
        for method in stage3.methods:
            cells = [
                r for r in raw
                if r["fold"] == fold.name and r["method"] == method
            ]
            if not cells:
                continue
            entry = {
                k: float(np.mean([c[k] for c in cells]))
                for k in metric_keys
            }
            entry["n_seeds"] = len(cells)
            if fold.name == ood:
                pooled = np.concatenate(
                    series_store[(fold.name, method)]
                )
                for name, fn in (
                    ("sharpe", annualised_sharpe),
                    ("calmar", calmar),
                ):
                    pt, lo, hi = moving_block_bootstrap_ci(
                        pooled, fn, n_boot=1000, block=20, seed=42
                    )
                    entry[f"{name}_ci"] = [lo, hi]
            table[fold.name][method] = entry

    # Decisive analyses on the OOD fold.
    analyses: Dict[str, object] = {}
    if ood and "recurrent_ppo" in stage3.methods:
        rec = traj_store.get((ood, "recurrent_ppo", use_seeds[0]))
        myo = traj_store.get((ood, "myopic_head", use_seeds[0]))
        f_ood = next(f for f in splits if f.name == ood)
        panel0 = build_panel(
            base, seed=use_seeds[0],
            train_end_index=int(f_ood.train_idx[-1]),
        )
        if rec is not None and myo is not None:
            fr = _load_frozen_layer1(
                base, layer1, ckpt, stage3.lower_stack,
                stage3.stage2_variant, ood, use_seeds[0], panel0,
            )
            tape0 = precompute_tape(
                fr, MeanVarianceQP(layer2), panel0,
                list(f_ood.val_idx), layer2, int(f_ood.train_idx[0]),
                stride=stage3.precompute_stride,
            )
            analyses["exposure_trajectory_figure"] = (
                exposure_trajectory_figure(
                    rec, myo, tape0.daily_ic,
                    fig_dir / "exposure_trajectory_ood.png",
                )
            )
        if "myopic_head" in stage3.methods:
            analyses["dissociation"] = dissociation_table(
                table[ood]["recurrent_ppo"],
                table[ood]["myopic_head"],
            )

    # Free architectural ablation: recurrent vs feedforward.
    if {"recurrent_ppo", "feedforward_ppo"} <= set(stage3.methods):
        analyses["recurrence_ablation"] = {
            f.name: {
                "recurrent_calmar":
                    table[f.name]["recurrent_ppo"]["calmar"],
                "feedforward_calmar":
                    table[f.name]["feedforward_ppo"]["calmar"],
                "recurrent_mean_return":
                    table[f.name]["recurrent_ppo"]["mean_return"],
                "feedforward_mean_return":
                    table[f.name]["feedforward_ppo"]["mean_return"],
            }
            for f in splits
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase6_evaluation.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"per_fold": table, "analyses": analyses,
             "ood_fold": ood, "raw": raw},
            fh, indent=2,
        )
    LOGGER.info("Phase 6 evaluation written to %s", out_path)
    for fname, methods in table.items():
        for method, m in methods.items():
            LOGGER.info(
                "  %s %s: ic %.5f sharpe %.4f calmar %.4f mdd %.4f "
                "turnover %.4f", fname, method, m["ic"], m["sharpe"],
                m["calmar"], m["max_drawdown"], m["turnover"],
            )
    if "dissociation" in analyses:
        d = analyses["dissociation"]
        LOGGER.info(
            "Dissociation (OOD %s): delta_ic %.6f delta_calmar %.4f",
            ood, d["delta_ic"], d["delta_calmar"],
        )
    LOGGER.info("Phase 6 evaluation complete")
    return {"per_fold": table, "analyses": analyses}


def _parse_args() -> argparse.Namespace:
    cfg = Path(__file__).resolve().parents[2] / "configs"
    p = argparse.ArgumentParser(description="Phase 6 evaluation.")
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
    LOGGER.info("Phase 6 done")


if __name__ == "__main__":
    main()
