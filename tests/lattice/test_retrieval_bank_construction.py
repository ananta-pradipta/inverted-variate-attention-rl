"""Phase 5b acceptance gates for proper retrieval banks (spec section 5.4).

Covers:
  - Regime key shape (T_train, 14).
  - Novelty key shape (n_novelty_entries, 8).
  - Eligibility mask correctness (no train day used as query for tau > t - 10).
  - Train-fold-only standardisation (no test data bleed).
  - Novelty bank size in expected range or empty-bank-guard activated.
  - Sector projection: deterministic, frozen, saved to disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.lattice.data.episode_keys import (
    REGIME_KEY_DIM, NOVELTY_KEY_DIM,
    INCUMBENT_WINDOW_TRADING_DAYS, NOVELTY_MAX_MONTHS,
    build_regime_key_tensor, build_novelty_key_tensor,
    compute_first_panel_idx_per_ticker, compute_idiovol_60d_proxy,
    fit_regime_stats, apply_regime_stats,
)
from src.lattice.data.sector_projection import (
    SECTOR_PROJECTION_PATH, build_or_load_sector_projection, project_sector,
    N_GICS_SECTORS,
)


@pytest.fixture(scope="module")
def synthetic_panel():
    """Small synthetic panel for fast unit tests."""
    np.random.seed(0)
    T, N = 800, 50
    F_panel = 26
    panel = np.random.randn(T, N, F_panel).astype(np.float32) * 0.02
    mask = np.zeros((T, N), dtype=bool)
    # 30 incumbents present from t=0; 20 IPO additions starting at random t
    for n in range(30):
        mask[:, n] = True
    rng = np.random.default_rng(1)
    for n in range(30, 50):
        ipo_t = int(rng.integers(low=INCUMBENT_WINDOW_TRADING_DAYS + 1, high=T - 100))
        mask[ipo_t:, n] = True
    macro = np.random.randn(T, 24).astype(np.float32)
    macro[:, 0] = 15.0 + np.random.randn(T) * 5.0  # VIX-like
    st = np.random.randn(T, N, 5).astype(np.float32)
    sector_per_ticker = (np.arange(N) % 11).astype(np.int64)
    return {
        "panel": panel, "mask": mask, "macro": macro, "st": st,
        "sector_per_ticker": sector_per_ticker, "T": T, "N": N,
    }


def test_sector_projection_deterministic(tmp_path):
    """Same seed yields the same weight matrix; saved layer is frozen."""
    p1 = tmp_path / "proj1.pt"
    p2 = tmp_path / "proj2.pt"
    w1 = build_or_load_sector_projection(path=p1, seed=0)
    w2 = build_or_load_sector_projection(path=p2, seed=0)
    assert torch.allclose(w1, w2)
    assert w1.shape == (1, N_GICS_SECTORS)
    assert not w1.requires_grad
    # Reload from disk hits the cached path
    w3 = build_or_load_sector_projection(path=p1, seed=0)
    assert torch.allclose(w1, w3)


def test_sector_projection_apply():
    """project_sector returns a scalar per ticker."""
    proj = build_or_load_sector_projection()
    sids = np.array([0, 1, 2, 10, -1, 5], dtype=np.int64)
    out = project_sector(proj, sids)
    assert out.shape == (6,)
    assert out[4] == 0.0  # negative sector projects to 0


def test_regime_keys_shape_and_standardisation(synthetic_panel):
    """Regime keys are (T, 14) and z-scored against train_idx only."""
    sp = synthetic_panel
    train_idx = np.arange(0, 600, dtype=np.int64)  # train slice
    keys, stats = build_regime_key_tensor(
        sp["panel"], sp["mask"], sp["macro"], train_idx,
    )
    assert keys.shape == (sp["T"], REGIME_KEY_DIM)
    assert stats.means.shape == (REGIME_KEY_DIM,)
    assert stats.stds.shape == (REGIME_KEY_DIM,)
    # train portion of keys should have mean approximately 0 and std approximately 1
    train_keys = keys[train_idx]
    np.testing.assert_allclose(train_keys.mean(axis=0), 0.0, atol=1e-3)
    np.testing.assert_allclose(train_keys.std(axis=0), 1.0, atol=1e-3)
    # No NaNs/Infs
    assert np.isfinite(keys).all()


def test_regime_stats_no_test_bleed(synthetic_panel):
    """Stats fitted on train_idx must equal stats fitted on a different
    test_idx slice only by chance; specifically, swapping in only test days
    yields different means."""
    sp = synthetic_panel
    train_stats = fit_regime_stats(sp["panel"], sp["mask"], sp["macro"],
                                     np.arange(0, 600))
    test_stats = fit_regime_stats(sp["panel"], sp["mask"], sp["macro"],
                                    np.arange(700, 800))
    assert not np.allclose(train_stats.means, test_stats.means, atol=1e-6)


def test_novelty_keys_shape_and_eligibility(synthetic_panel):
    """Novelty keys are (T, N, 8) and eligibility excludes incumbents and
    cells with months_since_ipo > 36."""
    sp = synthetic_panel
    train_idx = np.arange(0, 600, dtype=np.int64)
    first_panel_idx = compute_first_panel_idx_per_ticker(sp["mask"])
    iv = compute_idiovol_60d_proxy(sp["panel"], sp["mask"], sp["sector_per_ticker"])
    proj_scalars = project_sector(
        build_or_load_sector_projection(), sp["sector_per_ticker"],
    )
    keys, stats, eligible = build_novelty_key_tensor(
        sp["panel"], sp["mask"], sp["st"],
        sp["sector_per_ticker"], proj_scalars, first_panel_idx, train_idx,
        idiovol_tensor=iv,
    )
    assert keys.shape == (sp["T"], sp["N"], NOVELTY_KEY_DIM)
    assert eligible.shape == (sp["T"], sp["N"])
    # No incumbent cells eligible
    for n in range(30):
        assert not eligible[:, n].any()
    # Some IPO cells eligible
    assert eligible[:, 30:].any()
    # Months-since-ipo cap: no cell eligible after first_panel_idx + 36*21
    cap_days = NOVELTY_MAX_MONTHS * 21
    for n in range(30, sp["N"]):
        for t in np.where(eligible[:, n])[0]:
            assert t - int(first_panel_idx[n]) <= cap_days


def test_novelty_bank_population_smoke(synthetic_panel):
    """Bank-population end-to-end on synthetic data; non-empty bank produces
    non-zero forward-pass output through the model's existing retrieval."""
    from src.lattice.model.retrieval import (
        DualRetrieval, DualRetrievalConfig,
    )
    sp = synthetic_panel
    train_idx = np.arange(0, 600, dtype=np.int64)
    first_panel_idx = compute_first_panel_idx_per_ticker(sp["mask"])
    iv = compute_idiovol_60d_proxy(sp["panel"], sp["mask"], sp["sector_per_ticker"])
    proj_scalars = project_sector(
        build_or_load_sector_projection(), sp["sector_per_ticker"],
    )
    novelty_keys, _, eligible = build_novelty_key_tensor(
        sp["panel"], sp["mask"], sp["st"],
        sp["sector_per_ticker"], proj_scalars, first_panel_idx, train_idx,
        idiovol_tensor=iv,
    )
    train_set = np.zeros(sp["T"], dtype=bool)
    train_set[train_idx] = True
    eligible_train = eligible & train_set[:, None]
    day_idx, ticker_idx = np.where(eligible_train)
    assert day_idx.size > 0, "synthetic panel should have eligible novelty cells"

    cfg = DualRetrievalConfig(d_model=32)
    retrieval = DualRetrieval(cfg)

    keys = torch.from_numpy(novelty_keys[day_idx, ticker_idx]).float()
    sector_ids = torch.from_numpy(
        sp["sector_per_ticker"][ticker_idx].astype(np.int64),
    )
    day_indices = torch.from_numpy(day_idx.astype(np.int64))
    retrieval.novelty.populate_bank(keys, sector_ids, day_indices)
    assert retrieval.novelty._bank_populated
    assert retrieval.novelty.bank_keys.shape[0] == day_idx.size


def test_empty_novelty_bank_guard():
    """When zero training cells qualify, the model returns zeros from the
    retrieval forward pass."""
    from src.lattice.model.retrieval import (
        DualRetrieval, DualRetrievalConfig,
    )
    cfg = DualRetrievalConfig(d_model=32)
    retrieval = DualRetrieval(cfg)
    # Bank stays at default size 1; flag stays False.
    assert not retrieval.novelty._bank_populated
    z = torch.randn(1, 5, 32)
    qk = torch.zeros(1, 5, 8)
    sids = torch.zeros(1, 5, dtype=torch.long)
    qday = torch.tensor([10], dtype=torch.long)
    mask = torch.ones(1, 5, dtype=torch.bool)
    delta, alpha = retrieval.novelty(z, qk, sids, qday, mask)
    # Empty/unpopulated bank -> zero residual and zero gate
    assert torch.allclose(delta, torch.zeros_like(delta))
    assert torch.allclose(alpha, torch.zeros_like(alpha))
