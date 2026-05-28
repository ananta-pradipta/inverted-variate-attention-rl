"""Tests that the synthetic panel satisfies the data contract and is seeded."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from invar_rl.common.config import load_base_config
from invar_rl.data.panel_dataset import make_day_loader
from invar_rl.data.synthetic import SyntheticPanel

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _panel(seed: int) -> SyntheticPanel:
    base = load_base_config(CONFIG_DIR / "base.yaml")
    return SyntheticPanel(base.synthetic, seed=seed)


def test_contract_shapes_are_consistent() -> None:
    panel = _panel(42)
    day = 500
    tickers, features = panel.feature_window(day)
    n_active = len(tickers)

    assert features.shape == (n_active, panel.lookback, panel.n_features)
    assert panel.macro_vector(day).shape == (panel.macro_dim,)

    raw, z = panel.forward_label(day)
    mask = panel.tradable_mask(day)
    assert raw.shape == (n_active,)
    assert z.shape == (n_active,)
    assert mask.shape == (n_active,)
    assert mask.dtype == bool

    _, r1 = panel.realized_returns(day, horizon=1)
    _, r5 = panel.realized_returns(day, horizon=panel.label_horizon)
    assert r1.shape == (n_active,)
    assert r5.shape == (n_active,)


def test_membership_varies_over_time() -> None:
    panel = _panel(42)
    early = len(panel.active_tickers(panel.lookback))
    late = len(panel.active_tickers(1000))
    assert late >= early
    assert late <= load_base_config(
        CONFIG_DIR / "base.yaml"
    ).synthetic.n_tickers


def test_within_day_label_is_zscored() -> None:
    panel = _panel(42)
    raw, z = panel.forward_label(400)
    mask = panel.tradable_mask(400)
    valid = mask & np.isfinite(z)
    assert valid.sum() >= 2
    assert abs(float(z[valid].mean())) < 1e-6
    assert abs(float(z[valid].std()) - 1.0) < 1e-6


def test_end_of_panel_labels_are_masked() -> None:
    panel = _panel(42)
    last = len(panel.trading_days()) - 1
    mask = panel.tradable_mask(last)
    assert not mask.any()


def test_seeding_is_deterministic_and_seed_sensitive() -> None:
    a1 = _panel(42).feature_window(300)[1]
    a2 = _panel(42).feature_window(300)[1]
    b = _panel(43).feature_window(300)[1]
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, b)


def test_panel_dataloader_yields_expected_tensors() -> None:
    panel = _panel(42)
    loader = make_day_loader(panel, day_indices=range(100, 110))
    batch = next(iter(loader))
    n_active = len(batch["tickers"])
    assert batch["features"].shape == (
        n_active,
        panel.lookback,
        panel.n_features,
    )
    assert batch["macro"].shape == (panel.macro_dim,)
    assert batch["mask"].dtype.__str__() == "torch.bool"
