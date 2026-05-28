"""Walk-forward and embargo splitter.

Given the ordered list of trading days and a fold configuration, produce the
train, validation, and test day-index ranges per fold, enforcing a
configurable multi-day embargo between every adjacent segment. Exactly one
fold is flagged as the out-of-distribution stress fold and that flag is
carried through so downstream code never selects on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from invar_rl.common.config import FoldsConfig, FoldSpec


@dataclass(frozen=True)
class FoldSplit:
    """Materialised index arrays for one fold."""

    name: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    is_ood: bool

    @property
    def train_bounds(self) -> tuple[int, int]:
        return int(self.train_idx[0]), int(self.train_idx[-1])

    @property
    def val_bounds(self) -> tuple[int, int]:
        return int(self.val_idx[0]), int(self.val_idx[-1])

    @property
    def test_bounds(self) -> tuple[int, int]:
        return int(self.test_idx[0]), int(self.test_idx[-1])


class EmbargoViolation(ValueError):
    """Raised when adjacent segments are closer than the embargo."""


class WalkForwardSplitter:
    """Builds and validates walk-forward folds."""

    def __init__(self, folds_config: FoldsConfig) -> None:
        """Initialise the splitter.

        Args:
            folds_config: The validated fold configuration.
        """
        self._cfg = folds_config

    @property
    def embargo_days(self) -> int:
        return self._cfg.embargo_days

    def _check_fold(self, n_days: int, fold: FoldSpec) -> None:
        """Validate one fold against the available day count and embargo."""
        segments = [fold.train, fold.val, fold.test]
        for label, (start, end) in zip(("train", "val", "test"), segments):
            if start < 0 or end >= n_days:
                raise ValueError(
                    f"fold {fold.name}: {label} range {(start, end)} is "
                    f"outside the available day index [0, {n_days - 1}]"
                )

        ordered = [
            ("train", fold.train),
            ("val", fold.val),
            ("test", fold.test),
        ]
        for (prev_label, prev), (next_label, nxt) in zip(ordered, ordered[1:]):
            if nxt[0] <= prev[1]:
                raise ValueError(
                    f"fold {fold.name}: {next_label} starts at {nxt[0]} which "
                    f"overlaps {prev_label} ending at {prev[1]}"
                )
            gap = nxt[0] - prev[1] - 1
            if gap < self._cfg.embargo_days:
                raise EmbargoViolation(
                    f"fold {fold.name}: gap between {prev_label} and "
                    f"{next_label} is {gap} days, embargo requires at least "
                    f"{self._cfg.embargo_days}"
                )

    def split(self, trading_days: Sequence) -> List[FoldSplit]:
        """Materialise all folds as index arrays.

        Args:
            trading_days: The ordered list of trading days. Only its length
                is used; identifiers may be integers or dates.

        Returns:
            One ``FoldSplit`` per configured fold, in configuration order.

        Raises:
            ValueError: If a fold range is out of bounds or segments overlap.
            EmbargoViolation: If adjacent segments violate the embargo.
        """
        n_days = len(trading_days)
        splits: List[FoldSplit] = []
        for fold in self._cfg.folds:
            self._check_fold(n_days, fold)
            splits.append(
                FoldSplit(
                    name=fold.name,
                    train_idx=np.arange(fold.train[0], fold.train[1] + 1),
                    val_idx=np.arange(fold.val[0], fold.val[1] + 1),
                    test_idx=np.arange(fold.test[0], fold.test[1] + 1),
                    is_ood=fold.ood,
                )
            )
        return splits

    def ood_fold(self, trading_days: Sequence) -> FoldSplit:
        """Return the single out-of-distribution stress fold."""
        for fold in self.split(trading_days):
            if fold.is_ood:
                return fold
        raise ValueError("no out-of-distribution fold was flagged")
