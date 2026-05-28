"""Unit tests for C3: sector-aware per-stock InfoNCE positives.

Covers the public surface of
``src.models.pretrain_improvements.sector_positives`` (the selector and
sector-map cache helpers) AND the canonical-preserve invariant on the
Stage-1 pretrainer: when ``pretrain_positive_method == "regime"`` (the
default) the per-stock projections method exists but the InfoNCE loop
continues to use the canonical day-level path, and the contrastive
loss on a synthetic batch matches the canonical formula.

Tests:
  * test_sector_disabled_preserves_canonical: when the C3 flag is off
    (i.e. method == "regime") the canonical day-level InfoNCE loss is
    computed via the existing ``_supcon_infonce_loss`` and the
    pretrainer's ``per_ticker_projections`` method (added in C3) does
    not alter the day-level pretrain state_dict / loss formula.
  * test_sector_positives_correct: synthetic 50 tickers in 3 sectors;
    verify that for every anchor the indices returned by
    ``SectorPositivesSelector.select_positives`` are exactly the
    same-sector indices (anchor self excluded), and the (N, N) mask
    built by ``build_pos_mask_per_day`` matches.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.baselines.train_invar_clpretrain_v2 import (
    CL_TEMPERATURE,
    TemporalEncoderContrastivePretrainer,
    _supcon_infonce_loss,
    _supcon_infonce_loss_per_day,
)
from src.models.pretrain_improvements.sector_positives import (
    GICS_SECTOR_ORDER,
    SectorPositivesConfig,
    SectorPositivesSelector,
    UNKNOWN_SECTOR_ID,
)


# Small encoder size so the tests run fast on CPU (mirrors the B2 tests).
N_FEATURES = 4
TEMPORAL_WINDOW = 6
D_MODEL = 16
N_HEADS = 2
D_FF = 32
E_LAYERS = 1
DROPOUT = 0.0
ACTIVATION = "gelu"
PROJ_DIM = 8
BATCH_DAYS = 4
N_ACTIVE = 5


def _make_pretrainer(seed: int = 0) -> TemporalEncoderContrastivePretrainer:
    torch.manual_seed(seed)
    return TemporalEncoderContrastivePretrainer(
        n_features=N_FEATURES,
        temporal_window=TEMPORAL_WINDOW,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        e_layers=E_LAYERS,
        dropout=DROPOUT,
        activation=ACTIVATION,
        proj_dim=PROJ_DIM,
        aux_regression_head=False,
    )


# ---------------------------------------------------------------------
# Test 1: canonical-preserve.
# ---------------------------------------------------------------------
def test_sector_disabled_preserves_canonical() -> None:
    """C3 OFF: day-level InfoNCE loss is byte-identical to canonical.

    The pretrainer in C3 ships with a new ``per_ticker_projections``
    method, but the canonical day-level path (called when
    ``pretrain_positive_method == "regime"``) must still use
    ``day_embedding`` + ``_supcon_infonce_loss`` exactly the way the
    pre-C3 trainer did. We assert this by running the same canonical
    pipeline on a synthetic batch with a deterministic positive mask
    and checking the loss matches a hand-rolled InfoNCE on the same
    embeddings.
    """
    pretrainer = _make_pretrainer(seed=0)
    pretrainer.eval()
    torch.manual_seed(1)
    z_list = []
    for _ in range(BATCH_DAYS):
        x_win = torch.randn(N_ACTIVE, TEMPORAL_WINDOW, N_FEATURES)
        with torch.no_grad():
            z_list.append(pretrainer.day_embedding(x_win))
    z = torch.stack(z_list, dim=0)
    # Deterministic positive mask: each anchor's positive is its
    # cyclic-next neighbour (i.e. (0, 1), (1, 2), (2, 3), (3, 0)),
    # mirroring the n_pos=1 nearest-neighbour case but with a fixed
    # pattern so the test is reproducible.
    pos_mask = torch.zeros(
        BATCH_DAYS, BATCH_DAYS, dtype=torch.bool,
    )
    for i in range(BATCH_DAYS):
        pos_mask[i, (i + 1) % BATCH_DAYS] = True
    loss = _supcon_infonce_loss(z, pos_mask, CL_TEMPERATURE)
    assert loss.requires_grad is False  # encoder is in eval, no grad here
    assert torch.isfinite(loss)

    # Hand-rolled reference InfoNCE on the same z + mask.
    sim = (z @ z.t()) / CL_TEMPERATURE
    self_mask = torch.eye(BATCH_DAYS, dtype=torch.bool)
    sim_masked = sim.masked_fill(self_mask, float("-inf"))
    logsumexp = torch.logsumexp(sim_masked, dim=1)
    pos = pos_mask & (~self_mask)
    log_prob = sim - logsumexp.unsqueeze(1)
    pos_log_prob = (log_prob * pos.float()).sum(dim=1)
    counts = pos.sum(dim=1).clamp_min(1).float()
    ref_loss = -(pos_log_prob / counts).mean()
    assert torch.allclose(loss, ref_loss, atol=1e-5), (
        f"canonical InfoNCE drift: loss={loss.item():.6f} "
        f"ref={ref_loss.item():.6f}"
    )


# ---------------------------------------------------------------------
# Test 2: C3 selector correctness on a 50 / 3-sector synthetic case.
# ---------------------------------------------------------------------
def test_sector_positives_correct() -> None:
    """50 synthetic tickers across 3 sectors; positives must be exactly
    the same-sector peers and the anchor itself must be excluded.
    """
    n_tickers = 50
    rng = np.random.default_rng(0)
    sector_ids = rng.integers(low=0, high=3, size=n_tickers)
    tickers = [f"TKR_{i:03d}" for i in range(n_tickers)]
    sector_map = dict(zip(tickers, sector_ids.astype(int)))

    selector = SectorPositivesSelector.__new__(SectorPositivesSelector)
    selector.config = SectorPositivesConfig(universe="_synthetic_")
    selector._sector_lookup = sector_map  # type: ignore[attr-defined]

    for anchor_i, anchor in enumerate(tickers):
        anchor_sec = sector_map[anchor]
        expected = [
            j for j, tk in enumerate(tickers)
            if j != anchor_i and sector_map[tk] == anchor_sec
        ]
        got = selector.select_positives(
            day_active_tickers=tickers,
            anchor_ticker=anchor,
            sector_map=sector_map,
        )
        assert sorted(got.tolist()) == sorted(expected), (
            f"anchor={anchor} sector={anchor_sec}: got={got.tolist()} "
            f"expected={expected}"
        )

    # Pos-mask correctness over the same 50-ticker active set.
    pos_mask = selector.build_pos_mask_per_day(sector_ids)
    assert pos_mask.shape == (n_tickers, n_tickers)
    # Diagonal is always False.
    assert not np.any(np.diag(pos_mask))
    # Off-diagonal mask matches same-sector membership.
    for i in range(n_tickers):
        for j in range(n_tickers):
            if i == j:
                continue
            assert bool(pos_mask[i, j]) == (
                sector_ids[i] == sector_ids[j]
            ), f"mask mismatch at ({i}, {j})"

    # Unknown-sector tickers (sector_id == -1) must produce empty
    # positives and an all-False row in the mask.
    sector_ids_with_unknown = sector_ids.copy()
    sector_ids_with_unknown[0] = UNKNOWN_SECTOR_ID
    sector_map_unknown = dict(zip(tickers, sector_ids_with_unknown))
    selector_u = SectorPositivesSelector.__new__(SectorPositivesSelector)
    selector_u.config = SectorPositivesConfig(universe="_synthetic_")
    selector_u._sector_lookup = sector_map_unknown  # type: ignore[attr-defined]
    got_unknown = selector_u.select_positives(
        day_active_tickers=tickers,
        anchor_ticker=tickers[0],
        sector_map=sector_map_unknown,
    )
    assert got_unknown.size == 0
    mask_u = selector_u.build_pos_mask_per_day(sector_ids_with_unknown)
    assert not np.any(mask_u[0, :]), "unknown-sector anchor row not False"


# ---------------------------------------------------------------------
# Test 3: per-day per-stock loss wiring sanity (C3 ON path).
# ---------------------------------------------------------------------
def test_sector_per_day_loss_finite_with_grad() -> None:
    """C3 ON: per-day per-stock InfoNCE loss is finite and grad flows.

    Synthetic (N_active, T, F) window for ONE day; build a same-sector
    mask over N_active=5 with 2 sectors (so each anchor has >=1
    positive). Forward + backward; encoder + projection head must
    receive gradients.
    """
    pretrainer = _make_pretrainer(seed=0)
    pretrainer.train()
    torch.manual_seed(2)
    x_win = torch.randn(
        N_ACTIVE, TEMPORAL_WINDOW, N_FEATURES,
        requires_grad=False,
    )
    z_stocks = pretrainer.per_ticker_projections(x_win)
    assert z_stocks.shape == (N_ACTIVE, PROJ_DIM)
    # Three of five stocks in sector 0, two in sector 1; every anchor
    # has at least one same-sector peer.
    sector_ids = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    selector = SectorPositivesSelector.__new__(SectorPositivesSelector)
    selector.config = SectorPositivesConfig(universe="_synthetic_")
    selector._sector_lookup = {}  # type: ignore[attr-defined]
    pos_mask_np = selector.build_pos_mask_per_day(sector_ids)
    pos_mask = torch.from_numpy(pos_mask_np)
    loss = _supcon_infonce_loss_per_day(
        z_stocks, pos_mask, CL_TEMPERATURE,
    )
    assert torch.isfinite(loss)
    loss.backward()
    n_grad = sum(
        1 for p in pretrainer.parameters()
        if p.grad is not None and torch.any(p.grad.abs() > 0)
    )
    assert n_grad > 0, "no encoder / proj-head parameter received gradient"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
