"""InVAR-RL Stage 3 RL with ablations.

Supports three ablations on top of the canonical Stage 3 RL pipeline:

- ``random_l1``: replace Layer 1 (canonical InVAR) with i.i.d. N(0, 1)
  random scores per stock per day. Tests whether the +0.81/+0.85
  Sharpes are reading the canonical InVAR ranker or just the
  QP + RL exposure logic on noise.
- ``equal_l2``: replace Layer 2 (mean-variance QP) with an
  equal-weight top-50 long / bottom-50 short portfolio. Tests how
  much of the lower-stack value is the QP vs naive top-k ranking.
- ``stripped_l3``: keep Layer 1 + Layer 2 canonical, but mask the
  Layer-1/Layer-2 features from the agent's observation (zero out
  score_dispersion, pred_vol, eff_positions, and the macro encoding;
  preserve only the risk-state fields). Tests whether the lift is
  driven by the agent reading the regime signal or just by its own
  trajectory bookkeeping.
- ``none``: identity (no ablation; reproduces stage3_rl_canonical).

Per-(fold, seed) JSON at
``invar_rl/results/stage3_rl_ablation/{ablation}/foldF_seedS.json``
with the same schema as stage 3 RL canonical.

Usage::

    python -m invar_rl.training.stage3_rl_ablation \
        --ablation random_l1 \
        --fold 1 --seed 42 \
        --layer1-ckpt invar_rl/results/stage1/_ckpt/fold1_seed42_full.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
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
from invar_rl.layer3_control.regime_probs import (
    load_fit,
    override_tape_macro_encoding,
    precompute_all as precompute_regime_probs,
)
from invar_rl.layer3_control.strip_obs_wrapper import (
    StrippedObservationWrapper,
)


_ABLATIONS = ("none", "random_l1", "equal_l2", "stripped_l3")


def _persist_daily_tape(
    test_tape, bridge, method: str, perf: Dict, out_path: Path,
) -> None:
    """Phase 1.A: write per-test-day [date, exposure, strategy_return,
    base_return] parquet, mirroring nasdaq100_layer3_sac._persist_test_outputs."""
    rets = np.asarray(perf.get("_strategy_returns", []), dtype=np.float64)
    exps = np.asarray(perf.get("_exposures", []), dtype=np.float64)
    n = int(min(test_tape.days.shape[0], rets.shape[0], exps.shape[0]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["date", "exposure", "strategy_return", "base_return"]
    if n == 0:
        pd.DataFrame(columns=cols).to_parquet(out_path, index=False)
        print(f"[INFO] daily tape (empty) -> {out_path}", flush=True)
        return
    dates = pd.to_datetime(
        [str(bridge.dates[int(d)]) for d in test_tape.days[:n]]
    ).normalize()
    df = pd.DataFrame({
        "date": dates, "exposure": exps[:n],
        "strategy_return": rets[:n],
        "base_return": test_tape.base_return[:n].astype(np.float64),
    })
    df.to_parquet(out_path, index=False)
    print(f"[INFO] daily tape ({method}, n={n}) -> {out_path}", flush=True)


def _ablation_modes(ablation: str) -> Dict[str, str]:
    """Return (score_mode, weighting_mode, strip_obs) for an ablation."""
    if ablation == "none":
        return {
            "score_mode": "canonical",
            "weighting_mode": "qp",
            "strip_obs": False,
        }
    if ablation == "random_l1":
        return {
            "score_mode": "random",
            "weighting_mode": "qp",
            "strip_obs": False,
        }
    if ablation == "equal_l2":
        return {
            "score_mode": "canonical",
            "weighting_mode": "equal_topk",
            "strip_obs": False,
        }
    if ablation == "stripped_l3":
        return {
            "score_mode": "canonical",
            "weighting_mode": "qp",
            "strip_obs": True,
        }
    raise ValueError(
        f"unknown ablation {ablation!r}; expected {_ABLATIONS}"
    )


def _eval_rl_actor(env: ExposureEnv, agent, recurrent: bool) -> Dict:
    obs, _ = env.reset(seed=0)
    state, starts = None, np.ones((1,), dtype=bool)
    rewards: List[float] = []
    rets: List[float] = []
    exps: List[float] = []
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
        exps.append(float(info.get("exposure", 0.0)))
        if term or trunc:
            break
    arr = np.asarray(rets)
    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_return": float(arr.mean()) if arr.size else 0.0,
        "volatility": float(arr.std()) if arr.size else 0.0,
        "final_equity": float(info.get("equity", 1.0)),
        "n_steps": len(rewards),
        # Phase 1.A: raw traces for daily-tape capture (--save-daily-tape).
        "_strategy_returns": rets,
        "_exposures": exps,
    }


def run_one_cell(
    ablation: str,
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
    equal_topk_k: int = 50,
    use_regime_probs: bool = False,
    universe_id: Optional[str] = None,
    kmeans_temperature: float = 1.0,
    save_daily_tape: bool = False,
    daily_tape_dir: Optional[Path] = None,
    warm_start_ckpt: Optional[Path] = None,
    warm_start_timesteps: Optional[int] = None,
    total_timesteps_override: Optional[int] = None,
) -> Dict:
    set_global_seed(seed)
    modes = _ablation_modes(ablation)
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

    if use_regime_probs:
        if not universe_id:
            raise ValueError(
                "use_regime_probs=True requires universe_id to key the "
                "regime_probs cache (e.g. 'sp500', 'nasdaq100')"
            )
        try:
            load_fit(universe_id, fold)
            print(
                f"[InVAR-DR-RL Phase1] regime_probs cache hit: "
                f"universe={universe_id} fold={fold}",
                flush=True,
            )
        except FileNotFoundError:
            print(
                f"[InVAR-DR-RL Phase1] precomputing regime_probs: "
                f"universe={universe_id} fold={fold} "
                f"temperature={kmeans_temperature}",
                flush=True,
            )
            precompute_regime_probs(
                universe=universe_id,
                fold=fold,
                bridge=bridge,
                temperature=float(kmeans_temperature),
            )

    train_tape = precompute_tape_canonical(
        bundle=bundle, bridge=bridge,
        day_indices=list(bridge.train_idx),
        layer2=layer2, stride=stage3.precompute_stride,
        score_mode=modes["score_mode"],
        weighting_mode=modes["weighting_mode"],
        ablation_seed=seed,
        long_only=long_only,
        equal_topk_k=int(equal_topk_k),
    )
    val_tape = precompute_tape_canonical(
        bundle=bundle, bridge=bridge,
        day_indices=list(bridge.val_idx),
        layer2=layer2, stride=stage3.precompute_stride,
        score_mode=modes["score_mode"],
        weighting_mode=modes["weighting_mode"],
        ablation_seed=seed + 1000,
        long_only=long_only,
        equal_topk_k=int(equal_topk_k),
    )
    test_tape = precompute_tape_canonical(
        bundle=bundle, bridge=bridge,
        day_indices=list(bridge.test_idx),
        layer2=layer2, stride=stage3.precompute_stride,
        score_mode=modes["score_mode"],
        weighting_mode=modes["weighting_mode"],
        ablation_seed=seed + 2000,
        long_only=long_only,
        equal_topk_k=int(equal_topk_k),
    )

    if use_regime_probs:
        override_tape_macro_encoding(train_tape, universe_id, fold)
        override_tape_macro_encoding(val_tape, universe_id, fold)
        override_tape_macro_encoding(test_tape, universe_id, fold)
        print(
            f"[InVAR-DR-RL Phase1] tapes overlaid with regime_probs "
            f"(macro_encoding dim now 8, was per-bridge macro_dim)",
            flush=True,
        )

    from stable_baselines3.common.monitor import Monitor

    ckpt_out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict] = {}
    for method in methods:
        if method not in RL_METHODS:
            raise ValueError(
                f"only RL_METHODS supported here: {RL_METHODS}; got {method!r}"
            )
        curve_dir = ckpt_out_dir / f"{method}_curves_f{fold}_s{seed}"
        curve_dir.mkdir(parents=True, exist_ok=True)
        train_inner = ExposureEnv(
            train_tape, layer3, bootstrap_episode=True
        )
        if modes["strip_obs"]:
            train_inner = StrippedObservationWrapper(train_inner)
        train_env = Monitor(train_inner, filename=str(curve_dir / "monitor"))
        if warm_start_ckpt is not None and method == "sac":
            from stable_baselines3 import SAC

            print(
                f"[Option F] warm-start SAC from {warm_start_ckpt} "
                f"(method={method} fold={fold} seed={seed})",
                flush=True,
            )
            agent = SAC.load(str(warm_start_ckpt), env=train_env)
            agent.set_env(train_env)
        elif warm_start_ckpt is not None and method != "sac":
            raise ValueError(
                f"--warm-start-ckpt is only supported for method=sac "
                f"(got method={method!r})"
            )
        else:
            agent = build_agent(method, train_env, stage3, seed)
        if warm_start_timesteps is not None:
            n_timesteps = int(warm_start_timesteps)
        elif total_timesteps_override is not None:
            n_timesteps = int(total_timesteps_override)
        else:
            n_timesteps = stage3.total_timesteps
        agent.learn(total_timesteps=n_timesteps)
        agent.save(
            str(ckpt_out_dir / f"{method}_f{fold}_s{seed}.zip")
        )
        eval_inner = ExposureEnv(
            test_tape, layer3, bootstrap_episode=False
        )
        if modes["strip_obs"]:
            eval_inner = StrippedObservationWrapper(eval_inner)
        perf = _eval_rl_actor(
            eval_inner, agent, recurrent=(method == "recurrent_ppo")
        )
        if save_daily_tape:
            tape_dir = (
                daily_tape_dir
                if daily_tape_dir is not None
                else (ckpt_out_dir.parent / "daily_tape")
            )
            tape_path = tape_dir / method / f"fold{fold}_seed{seed}.parquet"
            _persist_daily_tape(
                test_tape=test_tape, bridge=bridge,
                method=method, perf=perf, out_path=tape_path,
            )
        # Strip the raw per-step traces from the JSON payload (kept only
        # for the daily-tape parquet) so summary JSONs remain compact.
        perf_json = {k: v for k, v in perf.items() if not k.startswith("_")}
        results[method] = perf_json

    # "constant_full" equivalent under the ablation: hold the L1+L2
    # portfolio (no Layer 3 intervention) on the test tape. Recoverable
    # directly from the tape base_return field; recorded here so the
    # stage-2-equivalent ablation number is in the same JSON.
    tape_ret = test_tape.base_return.astype(float)
    tape_mean = float(tape_ret.mean()) if tape_ret.size else 0.0
    tape_vol = float(tape_ret.std()) if tape_ret.size else 0.0
    payload = {
        "ablation": ablation,
        "fold": fold,
        "seed": seed,
        "model": (
            f"InVAR-RL stage3 RL ablation={ablation} "
            f"(score_mode={modes['score_mode']} "
            f"weighting_mode={modes['weighting_mode']} "
            f"strip_obs={modes['strip_obs']} "
            f"long_only={long_only} "
            f"equal_topk_k={equal_topk_k})"
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
            "total_timesteps": (
                int(total_timesteps_override)
                if total_timesteps_override is not None
                else stage3.total_timesteps
            ),
            "score_mode": modes["score_mode"],
            "weighting_mode": modes["weighting_mode"],
            "strip_obs": modes["strip_obs"],
            "long_only": bool(long_only),
            "equal_topk_k": int(equal_topk_k),
            "use_regime_probs": bool(use_regime_probs),
            "universe_id": universe_id,
            "kmeans_temperature": float(kmeans_temperature),
            "warm_start_ckpt": (
                str(warm_start_ckpt) if warm_start_ckpt is not None else ""
            ),
            "warm_start_timesteps": (
                int(warm_start_timesteps)
                if warm_start_timesteps is not None
                else int(stage3.total_timesteps)
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"fold{fold}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[InVAR-RL stage 3 RL ablation={ablation}] wrote {out_path}")
    for m, perf in results.items():
        print(
            f"  {m:18s} mean_return={perf['mean_return']:+.5f} "
            f"vol={perf['volatility']:.5f} "
            f"final_equity={perf['final_equity']:.4f}"
        )
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="InVAR-RL stage 3 RL ablations (random_l1 / equal_l2 / stripped_l3)."
    )
    p.add_argument(
        "--ablation", type=str, required=True, choices=list(_ABLATIONS)
    )
    p.add_argument("--fold", type=int, required=True, choices=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--layer1-ckpt", type=str, required=True)
    p.add_argument("--layer2", type=str, default="invar_rl/configs/layer2.yaml")
    p.add_argument("--layer3", type=str, default="invar_rl/configs/layer3.yaml")
    p.add_argument("--stage3", type=str, default="invar_rl/configs/stage3.yaml")
    p.add_argument(
        "--output-dir-root",
        type=str,
        default="invar_rl/results/stage3_rl_ablation",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "If set, overrides the default output-dir-root/<ablation> "
            "convention with an explicit destination. Useful for "
            "running the same ablation under different K or long_only "
            "configs without colliding with the canonical outputs."
        ),
    )
    p.add_argument(
        "--methods", type=str,
        default="recurrent_ppo,feedforward_ppo,sac",
    )
    p.add_argument(
        "--panel_kind", type=str, default="lattice_native",
        choices=[
            "biotech", "lattice_native", "nasdaq100", "djia30",
            "biotech_nbi", "biotech_nbi_enriched",
        ],
        help=(
            "Panel universe. lattice_native = S&P 500. nasdaq100 / "
            "biotech_nbi_enriched route through the same build_panel "
            "dispatch as the per-universe Layer-1 entrypoints, so the "
            "fixed-K equal_topk wrapper (--equal-topk-k) can be applied "
            "to NDX / NBI for a K-matched comparison vs the ranker "
            "baselines (A3/A4 2026-05-27)."
        ),
    )
    p.add_argument("--panel_end", type=str, default="2025-12-31")
    p.add_argument("--two_regime_val", action="store_true", default=True)
    p.add_argument(
        "--long-only", action="store_true", default=False,
        help=(
            "If set, the precompute tape uses long-only weights "
            "(QP: sum(w)=1, w>=0; equal_topk: equal-weight long top-k "
            "only, no shorts). Mirrors stage3_rl_canonical --long-only "
            "and biotech_nbi_layer3_sac --protocol lo. Used for fair "
            "comparison vs long-only ranker baselines."
        ),
    )
    p.add_argument(
        "--equal-topk-k", type=int, default=50,
        help=(
            "Per-side K for weighting_mode='equal_topk' (set via "
            "--ablation equal_l2). Default 50 reproduces the canonical "
            "SP500 equal_l2 K=50; pass 25 for fair-K comparison vs the "
            "K=25 baseline rankers (fairness audit issue #1)."
        ),
    )
    p.add_argument(
        "--use-regime-probs", action="store_true", default=False,
        help=(
            "InVAR-DR-RL Phase 1 flag. If set, the layer-3 SAC "
            "observation tail is the cached 8-dim k-means-8 soft "
            "assignment over the per-day macro_input (loaded from "
            "cache/dr_rl/regime_probs/{universe-id}/fold{F}/probs.parquet) "
            "instead of the FiLM macro encoding. Zero learned "
            "parameters; deterministic per (universe, fold, seed=42)."
        ),
    )
    p.add_argument(
        "--universe-id", type=str, default=None,
        help=(
            "Cache key for --use-regime-probs (e.g. 'sp500', "
            "'nasdaq100', 'biotech_nbi', 'biotech_nbi_enriched'). "
            "Required when --use-regime-probs is set."
        ),
    )
    p.add_argument(
        "--kmeans-temperature", type=float, default=1.0,
        help=(
            "Softmax temperature for --use-regime-probs (default 1.0). "
            "Ignored when --use-regime-probs is not set."
        ),
    )
    p.add_argument(
        "--save-daily-tape", action="store_true", default=False,
        help=(
            "Robust-InVAR-RL Phase 1.A: persist per-test-day exposures + "
            "realised returns parquet at "
            "outputs/sp500/stage3_rl_ablation/<ablation>/daily_tape/"
            "<method>/foldF_seedS.parquet. Schema matches the NDX "
            "_persist_test_outputs format (date, exposure, "
            "strategy_return, base_return). Off by default to preserve "
            "canonical disk behaviour."
        ),
    )
    p.add_argument(
        "--daily-tape-dir", type=str, default=None,
        help=(
            "Optional override for the --save-daily-tape destination "
            "root. Default = <output-dir>/../daily_tape (sibling of the "
            "ablation-named summary JSON dir)."
        ),
    )
    p.add_argument(
        "--warm-start-ckpt", type=str, default="",
        help=(
            "Option F: path to a pretrained SAC checkpoint (.zip) to "
            "warm-start the cell's SAC training. If empty (default), "
            "trains from scratch (canonical behaviour). Only supported "
            "with --methods sac."
        ),
    )
    p.add_argument(
        "--warm-start-timesteps", type=int, default=0,
        help=(
            "Option F: total_timesteps to use when warm-starting (per "
            "cell finetune). If 0 (default), falls back to the stage3 "
            "config total_timesteps."
        ),
    )
    p.add_argument(
        "--total-timesteps", type=int, default=0,
        help=(
            "HP-sweep override: from-scratch SAC total_timesteps. If 0 "
            "(default), uses stage3.yaml rl.total_timesteps. Ignored "
            "when --warm-start-ckpt is set (warm-start path uses "
            "--warm-start-timesteps instead)."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.output_dir_root) / args.ablation
    ckpt_out_dir = out_dir / "_ckpt"
    print(
        f"[InVAR-RL stage 3 RL ablation={args.ablation}] "
        f"fold={args.fold} seed={args.seed} methods={methods} device={device}"
    )
    tape_dir = (
        Path(args.daily_tape_dir) if args.daily_tape_dir else None
    )
    warm_ckpt = (
        Path(args.warm_start_ckpt) if args.warm_start_ckpt else None
    )
    warm_ts = (
        int(args.warm_start_timesteps)
        if int(args.warm_start_timesteps) > 0
        else None
    )
    total_ts_override = (
        int(args.total_timesteps)
        if int(args.total_timesteps) > 0
        else None
    )
    run_one_cell(
        ablation=args.ablation,
        fold=args.fold,
        seed=args.seed,
        ckpt_path=Path(args.layer1_ckpt),
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
        long_only=bool(args.long_only),
        equal_topk_k=int(args.equal_topk_k),
        use_regime_probs=bool(args.use_regime_probs),
        universe_id=args.universe_id,
        kmeans_temperature=float(args.kmeans_temperature),
        save_daily_tape=bool(args.save_daily_tape),
        daily_tape_dir=tape_dir,
        warm_start_ckpt=warm_ckpt,
        warm_start_timesteps=warm_ts,
        total_timesteps_override=total_ts_override,
    )


if __name__ == "__main__":
    main()
