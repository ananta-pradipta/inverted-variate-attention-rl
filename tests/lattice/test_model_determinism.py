"""Determinism test: same seed -> bitwise-identical outputs on CPU.

Per spec section 6.10 acceptance gate.
"""
from __future__ import annotations

import torch

from src.lattice.model.lattice import LATTICE, LatticeConfig
from tests.lattice.test_model_shapes import _make_inputs


def test_lattice_determinism() -> None:
    """Two forward passes on the same inputs with the same seed must match."""
    torch.manual_seed(42)
    inputs = _make_inputs(B=2, N=200)

    torch.manual_seed(42)
    model_a = LATTICE(LatticeConfig())
    model_a.eval()
    with torch.no_grad():
        y_a, _ = model_a(**inputs)

    torch.manual_seed(42)
    model_b = LATTICE(LatticeConfig())
    model_b.eval()
    with torch.no_grad():
        y_b, _ = model_b(**inputs)

    assert torch.equal(y_a, y_b), (
        f"Bitwise mismatch: max abs diff = "
        f"{(y_a - y_b).abs().max().item():.6e}"
    )
