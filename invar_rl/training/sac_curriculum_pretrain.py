"""Option F: SAC curriculum calm-fold pretrain.

Pretrains a single SAC controller per seed on the concatenated train
segments of the "calm" SP500 folds (F1, F3, F4). The resulting ckpt is
later loaded as the warm-start initialisation for each (fold, seed)
cell in the per-cell finetune stage (see stage3_rl_ablation.py
--warm-start-ckpt).

The tape construction mirrors stage3_rl_ablation --ablation equal_l2
(canonical SAC: top-K equal-weight L/S wrapper at K=50, no QP). The
only deviation from the canonical per-cell training is that the train
tape is the np.concatenate of the F1+F3+F4 train tapes for the same
seed, so the policy sees a broader behaviour distribution before being
fine-tuned on each cell's actual fold.

Usage::

    python -m invar_rl.training.sac_curriculum_pretrain \
        --seed 42 \
        --layer1-ckpt-root invar_rl/results/stage1/_ckpt \
        --output-dir outputs/sac_curriculum/sp500/calm_pretrain
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

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
from invar_rl.layer3_control.env import ExposureEnv
from invar_rl.layer3_control.precompute import EpisodeTape
from invar_rl.layer3_control.precompute_canonical import (
    precompute_tape_canonical,
)


_CALM_FOLDS: Tuple[int, ...] = (1, 3, 4)


def _concat_tapes(tapes: List[EpisodeTape]) -> EpisodeTape:
    """Concatenate a list of EpisodeTape's into one along the time axis.

    The ``days`` field stores global trading-day indices; after
    concatenation it is informational only (the env indexes by step
    position via ``self._t``, not by global day). All other arrays are
    concatenated as-is; the resulting tape has length sum_i len(t_i).
    """
    if not tapes:
        raise ValueError("at least one tape required")
    return EpisodeTape(
        days=np.concatenate([t.days for t in tapes], axis=0),
        score_dispersion=np.concatenate(
            [t.score_dispersion for t in tapes], axis=0
        ),
        macro_encoding=np.concatenate(
            [t.macro_encoding for t in tapes], axis=0
        ),
        pred_vol=np.concatenate([t.pred_vol for t in tapes], axis=0),
        eff_positions=np.concatenate(
            [t.eff_positions for t in tapes], axis=0
        ),
        base_return=np.concatenate(
            [t.base_return for t in tapes], axis=0
        ),
        base_gross=np.concatenate(
            [t.base_gross for t in tapes], axis=0
        ),
        daily_ic=np.concatenate([t.daily_ic for t in tapes], axis=0),
    )


def _build_one_fold_train_tape(
    seed: int,
    fold: int,
    ckpt_path: Path,
    layer2_yaml: Path,
    stage3_yaml: Path,
    panel_kind: str,
    panel_end: str,
    two_regime_val: bool,
    equal_topk_k: int,
    long_only: bool,
    device: torch.device,
) -> EpisodeTape:
    """Build one fold's train tape under equal_l2 wrapper."""
    layer2 = load_layer2_config(str(layer2_yaml))
    stage3 = load_stage3_config(str(stage3_yaml))

    cfg = InVARConfig(fold=fold, seed=seed)
    cfg.panel_kind = panel_kind
    cfg.two_regime_val = two_regime_val
    cfg.panel_end = panel_end
    bridge = build_lattice_bridge(cfg)
    bundle = load_trained_invar(
        ckpt_path=ckpt_path, bridge=bridge, device=device
    )
    print(
        f"[Option F pretrain] seed={seed} fold={fold}: building train "
        f"tape ({len(bridge.train_idx)} days)",
        flush=True,
    )
    tape = precompute_tape_canonical(
        bundle=bundle, bridge=bridge,
        day_indices=list(bridge.train_idx),
        layer2=layer2, stride=stage3.precompute_stride,
        score_mode="canonical",
        weighting_mode="equal_topk",
        ablation_seed=seed,
        long_only=long_only,
        equal_topk_k=int(equal_topk_k),
    )
    print(
        f"[Option F pretrain] seed={seed} fold={fold}: tape len={len(tape)}",
        flush=True,
    )
    return tape


def run_calm_pretrain(
    seed: int,
    layer1_ckpt_root: Path,
    layer2_yaml: Path,
    layer3_yaml: Path,
    stage3_yaml: Path,
    output_dir: Path,
    panel_kind: str,
    panel_end: str,
    two_regime_val: bool,
    equal_topk_k: int,
    long_only: bool,
    total_timesteps: int,
    device: torch.device,
) -> Path:
    set_global_seed(seed)
    layer3 = load_layer3_config(str(layer3_yaml))
    stage3 = load_stage3_config(str(stage3_yaml))

    t_total_start = time.time()
    tapes: List[EpisodeTape] = []
    for fold in _CALM_FOLDS:
        ckpt = layer1_ckpt_root / f"fold{fold}_seed{seed}_full.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"layer-1 ckpt missing for fold={fold} seed={seed}: {ckpt}"
            )
        tape = _build_one_fold_train_tape(
            seed=seed,
            fold=fold,
            ckpt_path=ckpt,
            layer2_yaml=layer2_yaml,
            stage3_yaml=stage3_yaml,
            panel_kind=panel_kind,
            panel_end=panel_end,
            two_regime_val=two_regime_val,
            equal_topk_k=equal_topk_k,
            long_only=long_only,
            device=device,
        )
        tapes.append(tape)

    calm_tape = _concat_tapes(tapes)
    print(
        f"[Option F pretrain] seed={seed} concatenated calm tape: "
        f"len={len(calm_tape)} (F1+F3+F4)",
        flush=True,
    )

    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    output_dir.mkdir(parents=True, exist_ok=True)
    curve_dir = output_dir / f"seed{seed}" / "curves"
    curve_dir.mkdir(parents=True, exist_ok=True)

    train_inner = ExposureEnv(
        calm_tape, layer3, bootstrap_episode=True
    )
    train_env = Monitor(train_inner, filename=str(curve_dir / "monitor"))

    print(
        f"[Option F pretrain] seed={seed} starting SAC.learn "
        f"total_timesteps={total_timesteps} lr={stage3.learning_rate}",
        flush=True,
    )
    t_learn_start = time.time()
    agent = SAC(
        "MlpPolicy", train_env,
        learning_rate=stage3.learning_rate,
        seed=seed,
        verbose=0,
    )
    agent.learn(total_timesteps=int(total_timesteps))
    t_learn_elapsed = time.time() - t_learn_start

    ckpt_dir = output_dir / f"seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out = ckpt_dir / "sac_ckpt.zip"
    agent.save(str(ckpt_out))
    t_total_elapsed = time.time() - t_total_start

    meta = {
        "seed": int(seed),
        "calm_folds": list(_CALM_FOLDS),
        "panel_kind": panel_kind,
        "panel_end": panel_end,
        "two_regime_val": bool(two_regime_val),
        "equal_topk_k": int(equal_topk_k),
        "long_only": bool(long_only),
        "total_timesteps": int(total_timesteps),
        "concat_tape_len": int(len(calm_tape)),
        "per_fold_tape_len": [int(len(t)) for t in tapes],
        "wall_time_learn_s": float(t_learn_elapsed),
        "wall_time_total_s": float(t_total_elapsed),
        "ckpt_path": str(ckpt_out),
    }
    meta_out = ckpt_dir / "meta.json"
    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"[Option F pretrain] seed={seed} DONE: ckpt={ckpt_out} "
        f"learn_s={t_learn_elapsed:.1f} total_s={t_total_elapsed:.1f}",
        flush=True,
    )
    return ckpt_out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Option F: SAC curriculum calm-fold pretrain."
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--layer1-ckpt-root", type=str,
        default="invar_rl/results/stage1/_ckpt",
        help=(
            "Directory containing canonical L1 ckpts named "
            "fold{F}_seed{S}_full.pt."
        ),
    )
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
        "--output-dir", type=str,
        default="outputs/sac_curriculum/sp500/calm_pretrain",
    )
    p.add_argument(
        "--panel_kind", type=str, default="lattice_native",
        choices=["biotech", "lattice_native"],
    )
    p.add_argument("--panel_end", type=str, default="2025-12-31")
    p.add_argument(
        "--two_regime_val", action="store_true", default=True
    )
    p.add_argument(
        "--equal-topk-k", type=int, default=50,
        help="K per side for the top-K equal-weight L/S wrapper.",
    )
    p.add_argument(
        "--long-only", action="store_true", default=False,
    )
    p.add_argument(
        "--total-timesteps", type=int, default=20000,
        help=(
            "SAC.learn total_timesteps for the pretrain stage. "
            "Default 20k matches the canonical per-cell budget; the "
            "tape is ~3x longer so each step is sampled from the wider "
            "calm-folds distribution."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(
        f"[Option F pretrain] seed={args.seed} device={device} "
        f"layer1_ckpt_root={args.layer1_ckpt_root}",
        flush=True,
    )
    run_calm_pretrain(
        seed=int(args.seed),
        layer1_ckpt_root=Path(args.layer1_ckpt_root),
        layer2_yaml=Path(args.layer2),
        layer3_yaml=Path(args.layer3),
        stage3_yaml=Path(args.stage3),
        output_dir=Path(args.output_dir),
        panel_kind=args.panel_kind,
        panel_end=args.panel_end,
        two_regime_val=bool(args.two_regime_val),
        equal_topk_k=int(args.equal_topk_k),
        long_only=bool(args.long_only),
        total_timesteps=int(args.total_timesteps),
        device=device,
    )


if __name__ == "__main__":
    main()
