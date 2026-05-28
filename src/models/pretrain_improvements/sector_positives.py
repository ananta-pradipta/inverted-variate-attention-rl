"""C3 (2026-05-27): Sector-aware positives for Stage 1 InfoNCE pretrain.

Replaces the canonical L2-nearest-neighbour DAY-level positive selector
in ``src.baselines.train_invar_clpretrain_v2.run_stage1_pretrain`` with
a PER-STOCK selector whose positives are same-sector stocks within the
same trading day's active set. Inductive bias: sector-coherence rather
than macro regime-coherence (canonical) or HMM regime-coherence (B1).

Design
------
The canonical Stage-1 pretrain (see ``train_invar_clpretrain_v2``)
encodes a B-day batch and mean-pools the per-ticker (N_active, d) last-
step encoder outputs into a single (d,) DAY embedding, then projects
through a SimCLR head to (proj,) and runs a SupCon InfoNCE across the
B days in the batch (positives = N_pos nearest in-batch days by L2 in
a 14-d regime fingerprint; negatives = remaining in-batch days).

C3 changes the granularity and the positive rule, NOT the encoder or
the projection head:

  * Per day t in the batch, the encoder still produces per-ticker
    (N_active(t), d) last-step outputs. We DO NOT mean-pool. Instead
    every active stock i is projected through the SAME SimCLR head into
    proj-space and L2-normalised.

  * For each day's active cross-section, positives for anchor stock i =
    OTHER same-day stocks in the SAME sector as i. Negatives = OTHER
    same-day stocks in DIFFERENT sectors. Self is excluded.

  * The loss is a per-day SupCon InfoNCE summed over days in the batch
    and averaged over anchors that actually have at least one positive
    (a day with all stocks in the same sector contributes 0 anchors;
    a stock from a singleton sector that day contributes 0). The
    temperature ``tau`` is reused from the canonical pretrain.

Cross-day mixing is intentionally OFF in C3: the spec wants positives
from "the same day's active set", so the negatives are also restricted
to the same day to keep the contrastive signal a pure sector classifier
inside a day's macro state. This is the cleanest "sector-coherence
without macro confound" comparison vs the canonical regime-coherence
selector.

Sector source
-------------
SP500 uses ``data/processed/sp500_sector_map.csv`` (11 GICS sectors,
550 tickers) plus a small hard-coded supplement for ~50 recent additions
not yet in the CSV (Snowflake, Datadog, Crowdstrike, Uber, etc.) so the
lattice_native 600-ticker panel reaches >=95% coverage. The merged map
is cached at ``cache/sector_labels/<universe>.parquet`` with columns
``ticker`` (str) and ``sector_id`` (int, 0-based; -1 for unknown).

Forbidden inputs: future returns, the target panel y, val days, test
days. The sector map is a STATIC business-attribute lookup with no
temporal leakage. Stage 1 still operates on TRAIN-day windows only.

Canonical-preserve invariant: callers MUST only invoke this module
when the caller-side flag ``pretrain_positive_method == "sector"``.
With ``"regime"`` (the canonical default) the run does not touch this
file and the pretrain loop runs the kmeans/HMM day-level selector.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Hard-coded GICS supplement for tickers absent from the CSV snapshot.
# These are well-known recent S&P 500 additions / spin-offs; the GICS
# sector assignment is the public, well-documented sector classification
# at each company's index inclusion. Used ONLY to lift coverage above
# the 95% LOCAL VERIFY gate; CSV remains the source of truth where
# present.
# ---------------------------------------------------------------------
GICS_SECTOR_SUPPLEMENT: dict[str, str] = {
    "ABNB": "Consumer Discretionary",
    "AMTM": "Industrials",
    "APO": "Financials",
    "APP": "Information Technology",
    "ARES": "Financials",
    "AXON": "Industrials",
    "BG": "Consumer Staples",
    "BLDR": "Industrials",
    "BX": "Financials",
    "COIN": "Financials",
    "CRH": "Materials",
    "CRWD": "Information Technology",
    "CVNA": "Consumer Discretionary",
    "DASH": "Consumer Discretionary",
    "DDOG": "Information Technology",
    "DECK": "Consumer Discretionary",
    "DELL": "Information Technology",
    "EME": "Industrials",
    "ERIE": "Financials",
    "EXE": "Energy",
    "FICO": "Information Technology",
    "FIX": "Industrials",
    "GDDY": "Information Technology",
    "GEHC": "Health Care",
    "GEV": "Industrials",
    "HOOD": "Financials",
    "HUBB": "Industrials",
    "IBKR": "Financials",
    "JBL": "Information Technology",
    "KKR": "Financials",
    "KVUE": "Consumer Staples",
    "LII": "Industrials",
    "LULU": "Consumer Discretionary",
    "PANW": "Information Technology",
    "PLTR": "Information Technology",
    "PODD": "Health Care",
    "Q": "Information Technology",
    "SMCI": "Information Technology",
    "SNDK": "Information Technology",
    "SOLS": "Information Technology",
    "SOLV": "Health Care",
    "TKO": "Communication Services",
    "TPL": "Energy",
    "TTD": "Communication Services",
    "UBER": "Industrials",
    "VLTO": "Industrials",
    "VST": "Utilities",
    "WDAY": "Information Technology",
    "WSM": "Consumer Discretionary",
    "XYZ": "Financials",
}


# Canonical 11-sector GICS string-to-id mapping. Stable order so the
# cached parquet ``sector_id`` column is reproducible across runs.
GICS_SECTOR_ORDER: list[str] = [
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
]
UNKNOWN_SECTOR_ID: int = -1


# ---------------------------------------------------------------------
# Disk cache helpers.
# ---------------------------------------------------------------------
def _cache_dir() -> Path:
    return Path("cache/sector_labels")


def sector_cache_path(universe: str) -> Path:
    """Path to the cached per-universe sector parquet."""
    return _cache_dir() / f"{universe}.parquet"


def build_sp500_sector_map(
    raw_csv: str = "data/processed/sp500_sector_map.csv",
    out_parquet: Optional[Path] = None,
) -> pd.DataFrame:
    """Merge the SP500 sector CSV with the supplement and persist parquet.

    Returns the cached DataFrame with columns ``ticker`` (str) and
    ``sector_id`` (int; -1 if unknown). Always overwrites the cache.
    """
    raw = pd.read_csv(raw_csv)
    if not {"ticker", "sector"}.issubset(raw.columns):
        raise ValueError(
            f"SP500 sector CSV {raw_csv} missing 'ticker'/'sector' cols; "
            f"got {list(raw.columns)}"
        )
    rows: list[tuple[str, str]] = list(
        zip(raw["ticker"].astype(str), raw["sector"].astype(str))
    )
    # Supplement: only fill tickers not in the CSV (CSV is source of
    # truth where present).
    csv_set = set(raw["ticker"].astype(str))
    for tk, sec in GICS_SECTOR_SUPPLEMENT.items():
        if tk not in csv_set:
            rows.append((tk, sec))
    sector_to_id = {s: i for i, s in enumerate(GICS_SECTOR_ORDER)}
    out_rows: list[tuple[str, int]] = []
    for tk, sec in rows:
        out_rows.append((tk, int(sector_to_id.get(sec, UNKNOWN_SECTOR_ID))))
    df = pd.DataFrame(out_rows, columns=["ticker", "sector_id"])
    df = df.drop_duplicates(subset=["ticker"], keep="first")
    df = df.reset_index(drop=True)
    out_path = (
        out_parquet
        if out_parquet is not None
        else sector_cache_path("sp500")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def load_sector_map(universe: str) -> pd.DataFrame:
    """Load the cached per-universe sector map. Build SP500 if missing."""
    path = sector_cache_path(universe)
    if not path.exists():
        if universe == "sp500":
            print(
                f"[INFO] SectorPositives: {path} missing; building from "
                f"data/processed/sp500_sector_map.csv + supplement."
            )
            return build_sp500_sector_map(out_parquet=path)
        raise FileNotFoundError(
            f"sector parquet not found for universe={universe}: {path}. "
            f"Build it (only SP500 auto-build is wired today)."
        )
    return pd.read_parquet(path)


def map_tickers_to_sector_ids(
    tickers: list[str],
    universe: str = "sp500",
) -> np.ndarray:
    """Map a list of tickers to (N,) int64 sector ids.

    Unknown tickers (not in the cache) get :data:`UNKNOWN_SECTOR_ID`.
    """
    df = load_sector_map(universe)
    lookup = dict(zip(df["ticker"].astype(str), df["sector_id"].astype(int)))
    return np.array(
        [int(lookup.get(str(tk), UNKNOWN_SECTOR_ID)) for tk in tickers],
        dtype=np.int64,
    )


def coverage_fraction(
    tickers: list[str],
    universe: str = "sp500",
) -> float:
    """Fraction of tickers with a known sector_id in the cache."""
    if len(tickers) == 0:
        return 0.0
    ids = map_tickers_to_sector_ids(tickers, universe=universe)
    return float((ids != UNKNOWN_SECTOR_ID).sum()) / float(len(tickers))


# ---------------------------------------------------------------------
# Per-day positive selector.
# ---------------------------------------------------------------------
@dataclass
class SectorPositivesConfig:
    """Hyperparameters for the C3 selector.

    Attributes:
        universe: cache key under ``cache/sector_labels/<id>.parquet``.
        min_positives_per_anchor: minimum #same-sector peers for an
            anchor to participate in the per-day InfoNCE. Default 1.
    """

    universe: str = "sp500"
    min_positives_per_anchor: int = 1


class SectorPositivesSelector:
    """For each anchor stock, positives = same-sector stocks in the day.

    Use:
        selector = SectorPositivesSelector(SectorPositivesConfig(
            universe="sp500"))
        same = selector.select_positives(
            day_active_tickers=["AAPL", "MSFT", "JPM", "BAC"],
            anchor_ticker="AAPL",
        )
        # -> indices into day_active_tickers of stocks sharing AAPL's
        #    GICS sector, anchor self excluded.

    For the Stage-1 batch loop we use :meth:`build_pos_mask_per_day`
    which returns the (N_active, N_active) bool same-sector mask for a
    whole day in one call.
    """

    def __init__(self, config: Optional[SectorPositivesConfig] = None) -> None:
        self.config = config or SectorPositivesConfig()
        self._sector_lookup: dict[str, int] = {}
        df = load_sector_map(self.config.universe)
        self._sector_lookup = dict(
            zip(df["ticker"].astype(str), df["sector_id"].astype(int))
        )

    # ------------------------------------------------------------------
    def sector_id_of(self, ticker: str) -> int:
        """Return the GICS sector id for ``ticker`` (-1 if unknown)."""
        return int(self._sector_lookup.get(str(ticker), UNKNOWN_SECTOR_ID))

    # ------------------------------------------------------------------
    def select_positives(
        self,
        day_active_tickers: list[str],
        anchor_ticker: str,
        sector_map: Optional[dict[str, int]] = None,
    ) -> np.ndarray:
        """Return indices into ``day_active_tickers`` of same-sector peers.

        Self is excluded. Returns an empty array if the anchor's sector
        is unknown (-1) or no other stock shares it.

        Args:
            day_active_tickers: active tickers on the anchor's trading day.
            anchor_ticker: anchor symbol.
            sector_map: optional override (ticker -> sector_id) for
                testing; defaults to the cached map.

        Returns:
            (k,) int64 numpy array of indices.
        """
        lookup = sector_map or self._sector_lookup
        anchor_sec = int(lookup.get(str(anchor_ticker), UNKNOWN_SECTOR_ID))
        if anchor_sec == UNKNOWN_SECTOR_ID:
            return np.empty(0, dtype=np.int64)
        pos: list[int] = []
        for i, tk in enumerate(day_active_tickers):
            if str(tk) == str(anchor_ticker):
                continue
            sec_i = int(lookup.get(str(tk), UNKNOWN_SECTOR_ID))
            if sec_i == anchor_sec:
                pos.append(i)
        return np.array(pos, dtype=np.int64)

    # ------------------------------------------------------------------
    def build_pos_mask_per_day(
        self,
        day_active_sector_ids: np.ndarray,
    ) -> np.ndarray:
        """Build the (N, N) bool same-sector mask for one day's active set.

        Anchor self excluded on the diagonal. Anchors with sector id -1
        get an all-False row (no positives, no negatives, the loss
        skips them).

        Args:
            day_active_sector_ids: (N,) int sector ids for the active
                tickers on this day, in the SAME order the encoder
                processes them.

        Returns:
            (N, N) bool numpy array.
        """
        s = np.asarray(day_active_sector_ids, dtype=np.int64)
        n = int(s.shape[0])
        same = (s[:, None] == s[None, :])
        known = (s[:, None] != UNKNOWN_SECTOR_ID) & (
            s[None, :] != UNKNOWN_SECTOR_ID
        )
        eye = np.eye(n, dtype=bool)
        return (same & known) & (~eye)


__all__ = [
    "GICS_SECTOR_ORDER",
    "GICS_SECTOR_SUPPLEMENT",
    "SectorPositivesConfig",
    "SectorPositivesSelector",
    "UNKNOWN_SECTOR_ID",
    "build_sp500_sector_map",
    "coverage_fraction",
    "load_sector_map",
    "map_tickers_to_sector_ids",
    "sector_cache_path",
]
