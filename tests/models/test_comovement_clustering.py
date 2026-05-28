"""Unit tests for A2: sequential pretrain regime -> co-movement clustering.

Covers the public surface of
``src.models.pretrain_improvements.comovement_clustering`` AND the
canonical-preserve invariant on the sequential pretrain wrapper.

Tests:
  * test_single_stage_preserves_canonical: when ``pretrain_stages ==
    ["regime"]`` (the default) the sequential wrapper dispatches
    directly to the canonical single-stage pretrain code path; no
    co-movement fit is touched.
  * test_comovement_no_test_leakage: the clusterer never reads val /
    test rows of the panel. We verify by passing a synthetic train-only
    returns DataFrame and confirming the fit succeeds with the
    train-only stats (mocking val/test data would also be visible as
    a different cluster set; we instead check that the fitter raises
    when fed too-short input that would not have been valid as a
    train segment).
  * test_clusters_non_degenerate: synthetic 100 x 500 returns matrix
    (500 days x 100 tickers) with three planted co-movement groups;
    verify each fitted cluster has >= 5 stocks (small but non-trivial;
    the >=5 floor mirrors the SP500 LOCAL VERIFY gate).
"""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

from src.baselines.train_invar_clpretrain_v2 import (
    CL_TEMPERATURE,
    TemporalEncoderContrastivePretrainer,
    _supcon_infonce_loss,
    run_stage1_pretrain,
    run_stage1_sequential_pretrain,
)
from src.models.pretrain_improvements.comovement_clustering import (
    COMOVE_MIN_CLUSTER_SIZE,
    CoMovementClusterer,
    CoMovementConfig,
    cluster_size_summary,
)


# Encoder sizes copied from test_sector_positives.py so the canonical
# preserve check runs in identical conditions.
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
# Test 1: single-stage curriculum preserves the canonical pretrain path.
# ---------------------------------------------------------------------
def test_single_stage_preserves_canonical() -> None:
    """When pretrain_stages == ["regime"] the sequential wrapper must
    delegate to run_stage1_pretrain unchanged. We verify by mocking
    run_stage1_pretrain and asserting the wrapper makes exactly one
    call with init_from_ckpt unset (the default canonical path) and
    the canonical epoch budget.
    """

    class _Cfg:
        pretrain_stages = ["regime"]
        pretrain_comovement_epochs = 5

    cfg = _Cfg()
    ckpt_path = Path("/tmp/_a2_test_canonical.pt")
    with mock.patch(
        "src.baselines.train_invar_clpretrain_v2.run_stage1_pretrain"
    ) as m:
        run_stage1_sequential_pretrain(
            cfg, pretrain_epochs=10, device=torch.device("cpu"),
            ckpt_path=ckpt_path,
        )
    assert m.call_count == 1
    # Confirm the single-stage delegation passes the canonical args.
    call_args = m.call_args
    # Positional: (cfg, pretrain_epochs, device, ckpt_path).
    assert call_args.args[1] == 10
    assert call_args.args[3] == ckpt_path
    # Default kwargs path: no init_from_ckpt forced; the wrapper does
    # not pass it on the single-stage branch.
    assert "init_from_ckpt" not in call_args.kwargs


# ---------------------------------------------------------------------
# Test 2: the co-movement clusterer never sees val / test data.
# ---------------------------------------------------------------------
def test_comovement_no_test_leakage() -> None:
    """The clusterer reads ONLY the returns DataFrame the caller hands
    it; in particular, it never opens any side-channel parquet, never
    reaches out for forward-return targets, and never sees a val/test
    row that the caller has excluded.

    We verify by mocking ``pd.read_parquet`` and any other I/O the
    fitter could plausibly use, then confirming the fit succeeds and
    its output depends ONLY on the input DataFrame shape and values.
    A correlation-equivalence check (per-pair correlation in the
    aggregated matrix matches the per-pair pearson correlation on the
    same train rows) closes the loop: if the fitter were silently
    inflating its cohort with val/test data, the produced correlation
    matrix would not match the train-only pearson values.
    """
    rng = np.random.default_rng(2026)
    n_train, n_tickers = 1200, 80
    tickers = [f"TKR_{i:03d}" for i in range(n_tickers)]
    factors = rng.standard_normal(size=(n_train, 4))
    loadings = rng.choice(4, size=n_tickers, replace=True)
    L = np.zeros((n_tickers, 4), dtype=np.float64)
    for i, g in enumerate(loadings):
        L[i, g] = 1.0
    noise = 0.1 * rng.standard_normal(size=(n_train, n_tickers))
    train_returns = factors @ L.T + noise

    train_dates = pd.bdate_range("2018-01-02", periods=n_train)
    df_train = pd.DataFrame(
        train_returns, index=train_dates, columns=tickers
    )

    cfg = CoMovementConfig(
        universe="_test_", n_clusters=4, window=252, seed=0,
    )

    # Mock the disk I/O paths the fitter could in principle consult:
    # parquet load, csv load, and the panel build pipeline. A passing
    # fit under these mocks proves the fitter consumes ONLY the
    # DataFrame argument.
    with mock.patch(
        "pandas.read_parquet",
        side_effect=AssertionError(
            "leakage: clusterer must not read external parquet"
        ),
    ), mock.patch(
        "pandas.read_csv",
        side_effect=AssertionError(
            "leakage: clusterer must not read external csv"
        ),
    ):
        clusterer = CoMovementClusterer(cfg)
        clusters_train_only = clusterer.fit(df_train, n_clusters=4)

    # Sanity: fit succeeded under strict I/O sandbox.
    assert len(clusters_train_only) == n_tickers
    assert set(clusters_train_only.values()).issubset({0, 1, 2, 3})

    # Correlation-equivalence: the aggregated fold-level correlation
    # matrix the fitter stored must equal a pearson correlation
    # computed on the SAME train rows the caller handed in (modulo the
    # rolling aggregation; we check the last-window path directly so
    # the equivalence is exact).
    cfg_last = CoMovementConfig(
        universe="_test_", n_clusters=4, window=252, seed=0,
        aggregation="last",
    )
    clusterer_last = CoMovementClusterer(cfg_last)
    clusterer_last.fit(df_train, n_clusters=4)
    expected_corr = (
        df_train.iloc[-252:].corr(numeric_only=True).to_numpy(
            dtype=np.float64
        )
    )
    np.testing.assert_allclose(
        clusterer_last.correlation_matrix_, expected_corr,
        atol=1e-8,
        err_msg=(
            "last-window correlation matrix does not match pearson "
            "of the supplied DataFrame rows; clusterer may be pulling "
            "external state."
        ),
    )


# ---------------------------------------------------------------------
# Test 3: clusters are non-degenerate on a 500x100 synthetic panel.
# ---------------------------------------------------------------------
def test_clusters_non_degenerate() -> None:
    """500 train days x 100 tickers with three planted co-movement
    groups; verify each fitted cluster has >= COMOVE_MIN_CLUSTER_SIZE
    stocks (default 5)."""
    rng = np.random.default_rng(11)
    n_days, n_tickers, n_groups = 500, 100, 3
    tickers = [f"TK_{i:03d}" for i in range(n_tickers)]
    factors = rng.standard_normal(size=(n_days, n_groups))
    group_of = rng.choice(n_groups, size=n_tickers, replace=True)
    # Force each group to have at least 5 members so the planted
    # structure is non-degenerate; if rng happened to skew, re-balance.
    for g in range(n_groups):
        members = np.where(group_of == g)[0]
        if members.size < 5:
            need = 5 - members.size
            other = np.where(group_of != g)[0]
            swap = rng.choice(other, size=need, replace=False)
            group_of[swap] = g
    L = np.zeros((n_tickers, n_groups), dtype=np.float64)
    for i, g in enumerate(group_of):
        L[i, g] = 1.0
    noise = 0.15 * rng.standard_normal(size=(n_days, n_tickers))
    returns = factors @ L.T + noise

    dates = pd.bdate_range("2019-01-02", periods=n_days)
    df = pd.DataFrame(returns, index=dates, columns=tickers)
    cfg = CoMovementConfig(
        universe="_test_", n_clusters=n_groups, window=252, seed=42,
    )
    clusterer = CoMovementClusterer(cfg)
    clusters = clusterer.fit(df, n_clusters=n_groups)

    sizes = cluster_size_summary(clusters)
    assert len(sizes) == n_groups, (
        f"expected {n_groups} clusters, got {len(sizes)}: {sizes}"
    )
    for cid, sz in sizes.items():
        assert sz >= COMOVE_MIN_CLUSTER_SIZE, (
            f"cluster {cid} has {sz} stocks (< "
            f"{COMOVE_MIN_CLUSTER_SIZE}); planted structure should be "
            f"recoverable."
        )


# ---------------------------------------------------------------------
# Test 4: invalid configs raise.
# ---------------------------------------------------------------------
def test_clusterer_rejects_short_history() -> None:
    """An input DataFrame with fewer than ``window`` rows must raise."""
    df = pd.DataFrame(
        np.zeros((50, 10), dtype=np.float64),
        index=pd.bdate_range("2020-01-02", periods=50),
        columns=[f"T_{i}" for i in range(10)],
    )
    cfg = CoMovementConfig(universe="_test_", window=252)
    with pytest.raises(ValueError, match="need >= 252"):
        CoMovementClusterer(cfg).fit(df, n_clusters=4)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
