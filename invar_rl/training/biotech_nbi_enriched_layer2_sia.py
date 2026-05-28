"""Biotech-NBI-enriched Layer 2 SIA driver.

NBI-enriched sibling of :mod:`invar_rl.training.sp500_layer2_sia`.
Identical training pipeline; differs only in:

1. ``_K_WRAPPER = 25`` (canonical NBI-enriched fixed equal-weight L/S
   top-K per side, matching the canonical NBI-enriched SAC L/S K=25
   headline).
2. Default ``--universe-label biotech_nbi_enriched`` and
   ``--panel-kind biotech`` so the k-means-8 regime cache at
   ``cache/dr_rl/regime_probs/biotech_nbi_enriched/fold{F}/probs.parquet``
   is used for the regime-invariance penalty.
3. Default ``--layer1-ckpt-root outputs/biotech_nbi_enriched/layer1/_ckpt``.

CLI::

    python -m invar_rl.training.biotech_nbi_enriched_layer2_sia \\
        --fold 1 --seed 42 \\
        --total-timesteps 20000 \\
        --output-dir-root outputs/biotech_nbi_enriched/layer2_sia/phase3 \\
        --regime-label --beta-kl 1e-4 --lambda-inv <WINNER>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from invar_rl.layer2_sia.config import SIAConfig
from invar_rl.training._universe_setup import universe_setup
from invar_rl.training.sp500_layer2_sia import run_one_cell as _run_one_cell

# Override the per-side L/S wrapper K for NBI-enriched.
_K_WRAPPER_NBI: int = 25


def _parse_args() -> argparse.Namespace:
    setup = universe_setup("biotech_nbi_enriched")
    p = argparse.ArgumentParser(
        description=(
            "Biotech-NBI-enriched Layer 2 SIA: Sparse Invariant Actor + "
            "full-info SB3 twin-Q critic; wrapper K=25 per side."
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
    )
    p.add_argument(
        "--beta-kl", type=float, default=SIAConfig.beta_kl,
    )
    p.add_argument(
        "--lambda-gate", type=float, default=SIAConfig.lambda_gate,
    )
    p.add_argument(
        "--lambda-inv", type=float, default=SIAConfig.lambda_inv,
    )
    p.add_argument(
        "--group-source", type=str, default=SIAConfig.group_source,
    )
    p.add_argument(
        "--eval-freq", type=int, default=2000,
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
    )
    p.add_argument(
        "--no-sparse-gates", action="store_true",
        help="Phase 4 no_s ablation: clamp per-block gates to 1.0.",
    )
    p.add_argument(
        "--no-asymmetric-critic", action="store_true",
        help="Phase 4 no_a ablation: critic on actor's bottleneck.",
    )
    p.add_argument(
        "--universe-label", type=str, default="biotech_nbi_enriched",
    )
    p.add_argument(
        "--long-only", action="store_true",
        help="L/O protocol: precompute tapes with long_only=True.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    import invar_rl.training.sp500_layer2_sia as _sia_mod
    _sia_mod._K_WRAPPER = _K_WRAPPER_NBI

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
            f"[biotech_nbi_enriched_layer2_sia] {out_path} exist; skip",
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _run_one_cell(
        fold=int(args.fold),
        seed=int(args.seed),
        ckpt_path=ckpt_path,
        layer3_yaml=Path(args.layer3),
        stage3_yaml=Path(args.stage3),
        sia_config=sia_config,
        eval_freq=int(args.eval_freq),
        output_dir_root=out_root,
        panel_end=args.panel_end,
        panel_kind=args.panel_kind,
        device=device,
        use_regime_label=bool(args.regime_label),
        universe_label=str(args.universe_label),
        long_only=bool(args.long_only),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
