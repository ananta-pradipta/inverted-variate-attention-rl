"""Tests for the walk-forward and embargo splitter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from invar_rl.common.config import (
    FoldsConfig,
    FoldSpec,
    load_folds_config,
)
from invar_rl.common.splits import EmbargoViolation, WalkForwardSplitter

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_configured_folds_split_cleanly() -> None:
    cfg = load_folds_config(CONFIG_DIR / "folds.yaml")
    splitter = WalkForwardSplitter(cfg)
    trading_days = list(range(1520))
    folds = splitter.split(trading_days)

    assert len(folds) == len(cfg.folds)
    for fold in folds:
        train = set(fold.train_idx.tolist())
        val = set(fold.val_idx.tolist())
        test = set(fold.test_idx.tolist())
        # No overlap between any two segments.
        assert not (train & val)
        assert not (train & test)
        assert not (val & test)
        # Indices stay within the available day range.
        assert fold.test_idx.max() < len(trading_days)
        # Embargo respected at both boundaries.
        assert fold.val_idx.min() - fold.train_idx.max() - 1 >= cfg.embargo_days
        assert fold.test_idx.min() - fold.val_idx.max() - 1 >= cfg.embargo_days


def test_exactly_one_ood_fold_flagged() -> None:
    cfg = load_folds_config(CONFIG_DIR / "folds.yaml")
    splitter = WalkForwardSplitter(cfg)
    folds = splitter.split(list(range(1520)))
    ood = [f for f in folds if f.is_ood]
    assert len(ood) == 1
    assert ood[0].name == "fold_3_stress"
    assert splitter.ood_fold(list(range(1520))).name == "fold_3_stress"


def test_embargo_violation_is_detected() -> None:
    bad = FoldsConfig(
        embargo_days=5,
        folds=[
            FoldSpec(
                name="bad",
                train=(0, 100),
                val=(103, 150),  # gap is 2 days, below the embargo
                test=(160, 200),
                ood=True,
            )
        ],
    )
    splitter = WalkForwardSplitter(bad)
    with pytest.raises(EmbargoViolation):
        splitter.split(list(range(220)))


def test_out_of_bounds_fold_is_rejected() -> None:
    cfg = FoldsConfig(
        embargo_days=5,
        folds=[
            FoldSpec(
                name="oob",
                train=(0, 100),
                val=(106, 150),
                test=(156, 999),
                ood=True,
            )
        ],
    )
    splitter = WalkForwardSplitter(cfg)
    with pytest.raises(ValueError):
        splitter.split(list(range(300)))


def test_missing_ood_flag_is_rejected_at_config_time() -> None:
    with pytest.raises(ValueError):
        FoldsConfig(
            embargo_days=5,
            folds=[
                FoldSpec("a", (0, 10), (16, 20), (26, 30), ood=False),
                FoldSpec("b", (0, 30), (36, 40), (46, 50), ood=False),
            ],
        )
