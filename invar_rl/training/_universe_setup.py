"""Shared universe-setup helper for the InVAR-RL Layer 2 variant drivers.

A single source of truth for the panel kind, the Layer 1 ckpt root, the
default wrapper top-K, the default output roots (UR and SIA), and the
canonical SAC output root (used by reporting scripts to compute the
variant vs canonical SAC delta). The existing canonical SAC drivers do
NOT import from here; this avoids modifying v3 inside the SIA Phase 1
scope. Future SIA drivers (NDX, NBI-enriched) share the same surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class UniverseSetup:
    """Static description of one InVAR-RL universe.

    Attributes:
        name: Canonical universe label used in summary JSON.
        panel_kind: ``lattice_native`` (SP500 / NDX) or ``biotech``.
        ckpt_root: Default Layer 1 ckpt directory; per-cell ckpt files are
            named ``foldF_seedS_full.pt``.
        default_k: Default per-side top-K for the fixed L/S wrapper.
        output_root: Default UR-driver output directory.
        sia_output_root: Default SIA-driver output directory.
        canonical_sac_summary_root: Canonical SAC summary directory; the
            variant report scripts pair variant cells against SAC cells
            by ``(fold, seed)`` using this directory.
    """

    name: str
    panel_kind: str
    ckpt_root: str
    default_k: int
    output_root: str
    sia_output_root: str
    canonical_sac_summary_root: str


_TABLE: Dict[str, UniverseSetup] = {
    "sp500": UniverseSetup(
        name="sp500",
        panel_kind="lattice_native",
        ckpt_root="invar_rl/results/stage1/_ckpt",
        default_k=50,
        output_root="outputs/sp500/layer2_ur",
        sia_output_root="outputs/sp500/layer2_sia",
        canonical_sac_summary_root=(
            "invar_rl/results/stage3_rl"
        ),
    ),
    "nasdaq100": UniverseSetup(
        name="nasdaq100",
        panel_kind="nasdaq100",
        ckpt_root="outputs/nasdaq100/layer1/_ckpt",
        default_k=20,
        output_root="outputs/nasdaq100/layer2_ur",
        sia_output_root="outputs/nasdaq100/layer2_sia",
        canonical_sac_summary_root=(
            "outputs/nasdaq100/layer3"
        ),
    ),
    "biotech_nbi_enriched": UniverseSetup(
        name="biotech_nbi_enriched",
        panel_kind="biotech_nbi_enriched",
        ckpt_root="outputs/biotech_nbi_enriched/layer1/_ckpt",
        default_k=25,
        output_root="outputs/biotech_nbi_enriched/layer2_ur",
        sia_output_root="outputs/biotech_nbi_enriched/layer2_sia",
        canonical_sac_summary_root=(
            "outputs/biotech_nbi_enriched/layer3"
        ),
    ),
}


def universe_setup(universe: str) -> UniverseSetup:
    """Return the :class:`UniverseSetup` for ``universe``.

    Raises:
        KeyError: If ``universe`` is not registered.
    """
    if universe not in _TABLE:
        raise KeyError(
            f"unknown universe {universe!r}; expected one of "
            f"{sorted(_TABLE)}"
        )
    return _TABLE[universe]
