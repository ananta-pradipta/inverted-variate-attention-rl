"""Load and run trained canonical InVAR layer-1 checkpoints.

Loads the per-(fold, seed) full state_dict produced by InVAR-RL Stage 1
(``invar_rl/scripts/wulver/invar_rl_stage1.sbatch`` with
``INVAR_SAVE_FULL_STATE=1``), reinstantiates the canonical
:class:`src.invar.InVAR` model with the right panel-shape arguments,
populates the day-memory bank from the panel data, and exposes a
frozen forward pass that consumes a single day's inputs from a
:class:`invar_rl.data.lattice_bridge.LatticePanelBatch`.

This is the v0 of the InVAR-RL layer-1 forward path: layer 1 is
trained in Stage 1 via the canonical pipeline, then frozen for
stages 2 (portfolio QP) and 3 (RL). Joint decision-focused
finetuning of layer 1 from layer 2's gradient is deferred to
:mod:`invar_rl.training.stage4_joint`.

Usage::

    from invar_rl.data.lattice_bridge import build_lattice_bridge
    from invar_rl.layer1_ranker.canonical_runner import (
        load_trained_invar,
    )

    cfg = ...  # InVARConfig with fold + seed + panel knobs
    bridge = build_lattice_bridge(cfg)
    model = load_trained_invar(
        ckpt_path="invar_rl/results/stage1/_ckpt/fold1_seed42_full.pt",
        bridge=bridge,
    )
    for day_idx in bridge.test_idx:
        inputs = bridge.day_inputs(day_idx)
        scores = model(**inputs_without_extras(inputs))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn

from src.invar import InVAR, InVARConfig

from invar_rl.data.lattice_bridge import LatticePanelBatch


_FORWARD_KEYS = (
    "x_window",
    "day_query_key",
    "query_day_idx",
    "allowed_day_indices",
    "regime_scalars",
    "duration_input",
    "macro_input",
    "macro_gate_input",
)


@dataclass
class TrainedInVARBundle:
    """Frozen canonical InVAR with its source bridge.

    Holds the trained model, the panel batch it was paired against,
    and the panel-shape metadata read from the checkpoint blob (used
    for diagnostics + sanity assertions, not for model construction).

    The model is set to ``eval()`` and its parameters have
    ``requires_grad=False``; the bundle is intended as a frozen scorer
    for stages 2/3.
    """

    model: InVAR
    bridge: LatticePanelBatch
    fold: int
    seed: int
    metadata: Dict[str, Any]

    def forward_day(self, day_idx: int) -> Dict[str, torch.Tensor]:
        """Score one day and return scores + the day's input dict.

        Args:
            day_idx: Trading-day index in ``bridge.dates``.

        Returns:
            A dict with ``scores`` (the canonical InVAR output for the
            day's active cross-section), ``active_indices``,
            ``macro_input`` (forwarded as the regime descriptor for
            layer 3), ``score_dispersion`` (scalar tensor), and ``y``
            (label realised return for layer 2's portfolio sim).
        """
        inputs = self.bridge.day_inputs(day_idx)
        device = next(self.model.parameters()).device
        fwd_kwargs = {}
        for k in _FORWARD_KEYS:
            v = inputs[k]
            if isinstance(v, torch.Tensor):
                fwd_kwargs[k] = v.to(device)
            else:
                fwd_kwargs[k] = v
        with torch.no_grad():
            scores = self.model(**fwd_kwargs)
        score_dispersion = scores.std(unbiased=False)
        return {
            "scores": scores.detach(),
            "active_indices": inputs["active_indices"],
            "macro_input": inputs["macro_input"].detach(),
            "score_dispersion": score_dispersion.detach(),
            "y": inputs["y"],
        }


def load_trained_invar(
    ckpt_path: str | Path,
    bridge: LatticePanelBatch,
    device: Optional[torch.device] = None,
) -> TrainedInVARBundle:
    """Reinstantiate canonical InVAR from a full-state ckpt + populate
    the day-memory bank from the bridge's panel data.

    Args:
        ckpt_path: Path to a ``foldF_seedS_full.pt`` blob saved by the
            canonical pipeline with ``INVAR_SAVE_FULL_STATE=1``.
        bridge: :class:`LatticePanelBatch` for the same (fold, panel)
            configuration the ckpt was trained on. The bridge supplies
            ``day_keys``, ``day_values``, and the train-day index set
            required by ``model.day_memory.populate``.
        device: Optional ``torch.device``. Defaults to CUDA if
            available else CPU.

    Returns:
        :class:`TrainedInVARBundle` ready for inference.
    """
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"canonical InVAR full-state ckpt not found: {ckpt_path}"
        )
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)

    if int(blob["fold"]) != int(bridge.cfg.fold):
        raise ValueError(
            f"ckpt fold {blob['fold']} != bridge fold "
            f"{bridge.cfg.fold}; they must match."
        )
    if blob["panel_kind"] != bridge.cfg.panel_kind:
        raise ValueError(
            f"ckpt panel_kind {blob['panel_kind']!r} != bridge "
            f"panel_kind {bridge.cfg.panel_kind!r}."
        )
    if int(blob["n_features"]) != bridge.n_features:
        raise ValueError(
            f"ckpt n_features {blob['n_features']} != bridge "
            f"n_features {bridge.n_features}."
        )

    cfg = InVARConfig(fold=int(blob["fold"]), seed=int(blob["seed"]))
    cfg.panel_kind = blob["panel_kind"]
    cfg.panel_end = blob["panel_end"]
    cfg.two_regime_val = bool(blob["two_regime_val"])
    cfg.day_value_dim = bridge.day_value_dim
    # F3 (cross-stock attention; legacy ckpts default to False / 4).
    cfg.cross_stock_attn = bool(blob.get("cross_stock_attn", False))
    cfg.cross_stock_heads = int(blob.get("cross_stock_heads", 4))
    assert cfg.enable_retrieval_bank is False, (
        "canonical InVAR is bankless"
    )

    model = InVAR(
        cfg,
        n_features=int(blob["n_features"]),
        day_key_dim=int(blob["day_key_dim"]),
        duration_input_dim=int(blob["duration_input_dim"]),
        macro_input_dim=int(blob["macro_input_dim"]),
        macro_gate_in_dim=int(blob["macro_gate_in_dim"]),
    ).to(device)
    model.day_memory.populate(
        keys=bridge.day_keys,
        values=bridge.day_values,
        day_indices=np.arange(len(bridge.dates)),
        train_day_indices=bridge.train_idx,
    )
    model.day_memory.to(device)
    incompat = model.load_state_dict(blob["state_dict"], strict=True)
    assert not incompat.missing_keys and not incompat.unexpected_keys, (
        f"strict load failed: missing={incompat.missing_keys}, "
        f"unexpected={incompat.unexpected_keys}"
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return TrainedInVARBundle(
        model=model,
        bridge=bridge,
        fold=int(blob["fold"]),
        seed=int(blob["seed"]),
        metadata={
            "panel_kind": blob["panel_kind"],
            "panel_end": blob["panel_end"],
            "two_regime_val": bool(blob["two_regime_val"]),
            "n_features": int(blob["n_features"]),
            "day_key_dim": int(blob["day_key_dim"]),
            "duration_input_dim": int(blob["duration_input_dim"]),
            "macro_input_dim": int(blob["macro_input_dim"]),
            "macro_gate_in_dim": int(blob["macro_gate_in_dim"]),
        },
    )


__all__ = [
    "TrainedInVARBundle",
    "load_trained_invar",
]
