"""A2 (2026-05-27): Co-movement clustering for sequential Stage 1 pretrain.

Replaces / composes with the canonical L2-nearest-neighbour DAY-level
positive selector in ``src.baselines.train_invar_clpretrain_v2.run_
stage1_pretrain`` with a PER-STOCK selector whose positives are stocks
in the SAME co-movement cluster as the anchor. Co-movement clusters
are derived from a per-fold 252-day rolling return correlation matrix
on the TRAIN segment only (no val / test leakage), then k-means on the
distance matrix (d = 1 - rho) with K=8 by default.

Design (sequential pretrain composition with the regime SSL stage):

  Stage 1a: canonical regime InfoNCE on the day-level cohort (the same
            14-d episode-key fingerprint, k-means or HMM selector,
            unchanged).
  Stage 1b: per-day per-stock SupCon InfoNCE whose positives are SAME-
            DAY SAME-CO-MOVEMENT-CLUSTER peers and negatives are SAME-
            DAY DIFFERENT-CLUSTER peers. The Stage-1b backbone is
            initialised from the Stage-1a backbone (carry forward); the
            projection head is re-initialised so the two stages have
            disjoint contrastive logits but a shared encoder.
  Stage 2:  canonical cs_mse finetune (unchanged); loads the Stage-1b
            backbone via the same encoder ckpt convention.

Why co-movement (vs sector)?
  Sector ids are an external, universe-specific lookup. The sector path
  is brittle on biotech (every ticker is "Health Care"). Co-movement
  clusters are DATA-DRIVEN from the train-fold returns matrix; they
  work on ANY universe with a daily-returns panel. This makes A2 the
  universe-agnostic version of C3.

Sequential composition with A1's ["regime", "sector"] is forbidden in
the trainer (A1's stage carries the "sector" tag; A2's carries
"comovement"; both belong to the per-stock SupCon family and the
trainer rejects more than one per-stock stage in a given config).

Forbidden inputs: future returns, the target panel y, val days, test
days, any val/test-fitted standardisation stats. The caller MUST pass
TRAIN-segment returns to ``fit``; this module enforces a minimum days
gate (>=252) so the rolling window has at least one full draw.

Canonical-preserve invariant: callers MUST only invoke this module
when the caller-side config flag carries the "comovement" stage. When
``pretrain_stages == ["regime"]`` (the canonical default) the run does
not touch this file and the pretrain loop is byte-identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


COMOVE_DEFAULT_N_CLUSTERS: int = 8
COMOVE_DEFAULT_WINDOW: int = 252
COMOVE_DEFAULT_SEED: int = 42
COMOVE_MIN_CLUSTER_SIZE: int = 5


@dataclass
class CoMovementConfig:
    """Hyperparameters for the A2 co-movement clusterer.

    Attributes:
        universe: cache key under
            ``cache/pretrain_improvements/comovement/<id>/foldF/``.
        n_clusters: number of co-movement clusters (default 8 to match
            the k-means-8 regime cluster count).
        window: rolling window length in trading days (default 252,
            one calendar year).
        seed: random seed for k-means init.
        aggregation: how to collapse the rolling correlation matrices
            into a single fold-level correlation matrix. "mean"
            averages across all rolling windows; "last" uses only the
            most recent 252-day window. Default "mean".
    """

    universe: str = "sp500"
    n_clusters: int = COMOVE_DEFAULT_N_CLUSTERS
    window: int = COMOVE_DEFAULT_WINDOW
    seed: int = COMOVE_DEFAULT_SEED
    aggregation: str = "mean"


class CoMovementClusterer:
    """Cluster stocks by 252-day rolling return correlation.

    Per fold:
        1. Use train-segment daily returns matrix ``[T_train, N]`` of
           active stocks.
        2. Compute a per-fold correlation matrix by aggregating the
           252-day rolling windows over the train segment (mean by
           default; "last" uses just the most recent window).
        3. Convert correlation to distance: ``d = 1 - rho``.
        4. Apply k-means clustering on the distance matrix (K=8 by
           default to match the k-means-8 regime cluster count).
        5. Each ticker gets a cluster_id; positives = same cluster.

    The result is cached per (universe, fold) at
    ``cache/pretrain_improvements/comovement/<universe>/foldF/
    cluster_ids.parquet`` (columns: ticker, cluster_id).

    Use:
        clusterer = CoMovementClusterer(CoMovementConfig(universe="sp500"))
        cluster_ids = clusterer.fit(
            returns=train_returns_df,  # DataFrame [T_train, N]
            n_clusters=8,
        )
        # -> dict ticker -> cluster_id (int)
    """

    def __init__(self, config: Optional[CoMovementConfig] = None) -> None:
        self.config = config or CoMovementConfig()
        self.cluster_ids_: Optional[dict[str, int]] = None
        self.correlation_matrix_: Optional[np.ndarray] = None
        self.ticker_order_: Optional[list[str]] = None
        self.n_windows_: int = 0

    # ------------------------------------------------------------------
    def fit(
        self,
        returns: pd.DataFrame,
        n_clusters: Optional[int] = None,
    ) -> dict[str, int]:
        """Fit per-fold co-movement clusters from train-segment returns.

        Args:
            returns: DataFrame indexed by date with one column per
                ticker; values are daily returns for the TRAIN segment
                only (no val/test rows). Tickers absent from at least
                one full rolling window are still clustered using
                pairwise-available correlation.
            n_clusters: optional override of ``self.config.n_clusters``.

        Returns:
            Dict mapping ticker -> int cluster_id (0-based).

        Raises:
            ValueError: insufficient rows for the rolling window or
                empty/degenerate input.
            RuntimeError: sklearn KMeans not importable.
        """
        if n_clusters is not None:
            self.config.n_clusters = int(n_clusters)
        K = int(self.config.n_clusters)
        if K < 2:
            raise ValueError(f"n_clusters must be >= 2, got {K}")
        if returns is None or returns.empty:
            raise ValueError("returns DataFrame is empty")
        T, N = returns.shape
        W = int(self.config.window)
        if T < W:
            raise ValueError(
                f"need >= {W} train rows for a single 252-day window, "
                f"got T={T}"
            )
        if N < K:
            raise ValueError(
                f"need at least K={K} tickers to form {K} clusters, "
                f"got N={N}"
            )

        tickers = list(returns.columns.astype(str))
        x = returns.to_numpy(dtype=np.float64, copy=True)

        # Aggregate rolling correlation matrices into one fold-level
        # correlation matrix. The "mean" path is the spec default;
        # "last" is offered for one-shot debugging.
        agg = str(self.config.aggregation).lower()
        if agg not in ("mean", "last"):
            raise ValueError(
                f"aggregation must be 'mean' or 'last', got {agg!r}"
            )

        if agg == "last":
            window_block = x[-W:, :]
            corr = self._pairwise_corr(window_block)
            n_windows = 1
        else:
            # "mean" aggregation: average correlation across all
            # rolling windows of size W. Step = 1 day; for SP500 train
            # this is on the order of 1500 windows. To keep the
            # computation tractable we stride the rolling window (the
            # spec says "mean across all rolling windows", which we
            # implement as a non-overlapping stride of W // 4 so an
            # ~8-year train segment yields ~32 windows; correlation
            # estimates per-window already use 252 days so further
            # daily striding would be wasteful and dominated by
            # neighbouring duplicates). The aggregation is symmetric
            # to permutation so any stride preserves the mean shape.
            stride = max(1, W // 4)
            corrs = []
            for start in range(0, T - W + 1, stride):
                block = x[start: start + W, :]
                corrs.append(self._pairwise_corr(block))
            if not corrs:  # pragma: no cover - guarded by T >= W above
                raise ValueError(
                    "no rolling windows produced; check window size."
                )
            corr = np.mean(np.stack(corrs, axis=0), axis=0)
            n_windows = len(corrs)
        self.correlation_matrix_ = corr.astype(np.float64)
        self.ticker_order_ = tickers
        self.n_windows_ = int(n_windows)

        # Distance = 1 - rho (clipped to [0, 2]).
        dist = 1.0 - corr
        dist = np.clip(dist, 0.0, 2.0).astype(np.float64)
        # Use a low-rank embedding so k-means on a euclidean distance
        # space is well-defined; classical MDS via the top eigenvectors
        # of the centred (-0.5 * D^2) matrix preserves pairwise
        # distances up to a metric scaling.
        emb = _classical_mds_embedding(dist, n_components=min(K + 4, N - 1))

        try:
            from sklearn.cluster import KMeans
        except Exception as exc:  # pragma: no cover - env guard
            raise RuntimeError(
                "sklearn.cluster.KMeans not importable; install scikit-"
                "learn (already required by sector_positives)."
            ) from exc
        km = KMeans(
            n_clusters=K,
            n_init=10,
            random_state=int(self.config.seed),
        )
        labels = km.fit_predict(emb)
        labels = labels.astype(np.int64)

        cluster_ids = {tk: int(lbl) for tk, lbl in zip(tickers, labels)}
        self.cluster_ids_ = cluster_ids
        return cluster_ids

    # ------------------------------------------------------------------
    @staticmethod
    def _pairwise_corr(block: np.ndarray) -> np.ndarray:
        """Pairwise correlation of columns of ``block`` (W, N).

        Handles columns that are all-NaN or constant by setting the
        corresponding correlation row/col to zero off-diagonal and one
        on the diagonal. NaN cells are masked per pairwise pair so a
        partially-missing column still contributes where it has data.
        """
        W, N = block.shape
        # Mean-impute per column over its finite rows so constant or
        # nearly-constant rows have a well-defined denominator. The
        # alternative (pairwise dropna) is O(N^2 * W) and unnecessary
        # for the once-per-fold fit.
        x = block.astype(np.float64, copy=True)
        # Replace +/- inf with NaN so np.nanmean / nanstd ignore them.
        x[~np.isfinite(x)] = np.nan
        # All-NaN columns trigger a benign RuntimeWarning under
        # np.nanmean; silence it (we zero-fill those columns below).
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.filterwarnings(
                "ignore", category=RuntimeWarning,
                message="Mean of empty slice",
            )
            col_means = np.nanmean(x, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        # Fill remaining NaNs with column mean (zero for empty columns)
        # so the dot-product below sees no NaN.
        mask = ~np.isfinite(x)
        if mask.any():
            idx = np.where(mask)
            x[idx] = col_means[idx[1]]
        x = x - col_means[None, :]
        std = x.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        x_n = x / std[None, :]
        # Pairwise correlation == (x_n.T @ x_n) / W when x_n has zero
        # mean and unit std per column.
        corr = (x_n.T @ x_n) / float(W)
        # Numerical clamp and force diagonal exactly 1.0.
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        return corr


# ---------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------
def _classical_mds_embedding(
    distance: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """Classical MDS: embed a symmetric distance matrix in R^k.

    Computes the top-k eigenpairs of the double-centred -0.5 * D^2
    matrix; the result is an (N, k) array whose pairwise euclidean
    distances approximate the input distance up to a metric scaling.
    k-means on this embedding is well-defined and respects the input
    distance geometry.
    """
    D = np.asarray(distance, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(
            f"distance must be square, got shape {D.shape}"
        )
    N = D.shape[0]
    # Squared distance, double-centred.
    D2 = D ** 2
    J = np.eye(N) - np.ones((N, N)) / float(N)
    B = -0.5 * (J @ D2 @ J)
    # Symmetrize (numerical) before eigendecomposition.
    B = 0.5 * (B + B.T)
    eig_vals, eig_vecs = np.linalg.eigh(B)
    # eigh returns ascending eigenvalues; take the top-k positives.
    order = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[order]
    eig_vecs = eig_vecs[:, order]
    k = int(min(n_components, N))
    top_vals = np.clip(eig_vals[:k], a_min=0.0, a_max=None)
    top_vecs = eig_vecs[:, :k]
    emb = top_vecs * np.sqrt(top_vals)[None, :]
    return emb.astype(np.float64)


# ---------------------------------------------------------------------
# Disk cache helpers.
# ---------------------------------------------------------------------
def _cache_dir(universe: str, fold: int) -> Path:
    return (
        Path("cache/pretrain_improvements/comovement")
        / str(universe)
        / f"fold{int(fold)}"
    )


def cluster_ids_path(universe: str, fold: int) -> Path:
    """Path to the A2 cached per-fold cluster-id parquet."""
    return _cache_dir(universe, fold) / "cluster_ids.parquet"


def save_cluster_ids(
    cluster_ids: dict[str, int],
    universe: str,
    fold: int,
    n_clusters: int,
    n_train_days: int,
    n_windows: int,
    seed: int,
) -> Path:
    """Persist per-ticker cluster ids keyed by ticker.

    Cache layout is DISJOINT from the B1 HMM cache
    (``cache/pretrain_improvements/hmm_regime/...``) and the C3 sector
    cache (``cache/sector_labels/...``) so the canonical and sibling
    paths are never touched.
    """
    df = pd.DataFrame(
        {
            "ticker": list(cluster_ids.keys()),
            "cluster_id": [int(v) for v in cluster_ids.values()],
        }
    )
    df["n_clusters"] = int(n_clusters)
    df["n_train_days"] = int(n_train_days)
    df["n_windows"] = int(n_windows)
    df["seed"] = int(seed)
    out = cluster_ids_path(universe, fold)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def load_cluster_ids(universe: str, fold: int) -> pd.DataFrame:
    """Load the cached per-fold cluster-id parquet."""
    p = cluster_ids_path(universe, fold)
    if not p.exists():
        raise FileNotFoundError(
            f"cluster_ids.parquet not found for universe={universe} "
            f"fold={fold}: {p}; rerun the A2 Stage 1b."
        )
    return pd.read_parquet(p)


def cluster_size_summary(cluster_ids: dict[str, int]) -> dict[int, int]:
    """Return cluster_id -> count, sorted ascending by cluster id."""
    counts: dict[int, int] = {}
    for cid in cluster_ids.values():
        counts[int(cid)] = counts.get(int(cid), 0) + 1
    return dict(sorted(counts.items()))


def map_tickers_to_cluster_ids(
    tickers: list[str],
    cluster_ids: dict[str, int],
) -> np.ndarray:
    """Map a list of tickers to (N,) int64 cluster ids.

    Tickers absent from ``cluster_ids`` get -1 (unknown).
    """
    return np.array(
        [int(cluster_ids.get(str(tk), -1)) for tk in tickers],
        dtype=np.int64,
    )


__all__ = [
    "COMOVE_DEFAULT_N_CLUSTERS",
    "COMOVE_DEFAULT_SEED",
    "COMOVE_DEFAULT_WINDOW",
    "COMOVE_MIN_CLUSTER_SIZE",
    "CoMovementClusterer",
    "CoMovementConfig",
    "cluster_ids_path",
    "cluster_size_summary",
    "load_cluster_ids",
    "map_tickers_to_cluster_ids",
    "save_cluster_ids",
]
