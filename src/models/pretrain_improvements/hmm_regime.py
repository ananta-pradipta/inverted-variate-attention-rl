"""B1 (2026-05-27): Gaussian HMM regime labeler for clpretrain positives.

Replaces the canonical L2-nearest-neighbour positive selector inside
``src/baselines/train_invar_clpretrain_v2.run_stage1_pretrain`` with a
SOFT regime posterior. The day-level macro / regime fingerprint
(``build_episode_keys``) is unchanged; only the way "same regime"
positives are picked changes:

  canonical (kmeans path == day_keys L2 nearest-neighbours):
      positives = top n_pos in-batch days with smallest
      ``cdist(keys_b, keys_b)`` to the anchor in standardised
      14-d episode-key space.

  B1 hmm path:
      1. Fit a temporal Gaussian HMM (or GMM fallback) on the
         standardised 14-d episode keys of the TRAIN segment only
         (``pretrain_idx == train_idx``); ``n_states`` latent regimes.
      2. ``predict_proba`` over all days the pretrain loop will touch
         (= train days; val/test never read) and cache them at
         ``cache/pretrain_improvements/hmm_regime/<universe>/foldF/
         posteriors.parquet`` keyed by day_idx.
      3. Inside the InfoNCE batch loop, positives for anchor i = the
         in-batch days j (j != i) whose posterior cosine similarity to
         the anchor's posterior is >= ``positive_threshold`` (default
         0.7). Ties are broken by similarity descending; if no j
         crosses the threshold the anchor is skipped (matches the
         canonical SupCon "anchor with no positive" handling in
         ``_supcon_infonce_loss``).

Library preference:
  - ``hmmlearn.GaussianHMM`` if importable (true HMM with learnt
    transition matrix);
  - ``sklearn.mixture.GaussianMixture`` fallback (no transition matrix
    => I.I.D. soft assignment over days, similar in spirit to
    soft-k-means but with full-covariance Gaussian components). The
    fallback path is logged with a [WARN] so the experiment trail is
    clear.

Forbidden inputs: future returns, the target panel y, val days, test
days, any val/test-fitted standardisation stats. The caller is
responsible for passing TRAIN-day standardised keys to ``fit``; this
module enforces that ``n_states < n_train_days`` and that
``n_train_days >= 10`` so the EM has something to bite on.

Canonical-preserve invariant: callers MUST only invoke this module
when the caller-side flag ``pretrain_regime_method == "hmm"``. With
``"kmeans"`` (the canonical default) the run does not touch this file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


HMM_DEFAULT_N_STATES: int = 4
HMM_DEFAULT_POSITIVE_THRESHOLD: float = 0.7
HMM_DEFAULT_SEED: int = 42
HMM_DEFAULT_MAX_ITER: int = 100
HMM_DEFAULT_TOL: float = 1.0e-3


def _backend_available() -> tuple[bool, bool]:
    """Return (hmmlearn_available, sklearn_gmm_available)."""
    try:  # pragma: no cover - exercised on Wulver, may be missing locally
        import hmmlearn  # noqa: F401
        hmm_ok = True
    except Exception:
        hmm_ok = False
    try:
        from sklearn.mixture import GaussianMixture  # noqa: F401
        gmm_ok = True
    except Exception:
        gmm_ok = False
    return hmm_ok, gmm_ok


@dataclass
class HMMRegimeConfig:
    """Hyperparameters for the B1 HMM regime labeler.

    Attributes:
        n_states: number of latent regimes (default 4; spec range 3-5).
        positive_threshold: cosine-similarity floor for SupCon positives
            in posterior space (default 0.7).
        seed: random seed for the EM init (default 42).
        max_iter: EM max iterations (default 100).
        tol: EM convergence tolerance (default 1.0e-3).
        prefer_hmmlearn: if True (default), use hmmlearn.GaussianHMM
            when importable; else fall back to sklearn GaussianMixture.
    """

    n_states: int = HMM_DEFAULT_N_STATES
    positive_threshold: float = HMM_DEFAULT_POSITIVE_THRESHOLD
    seed: int = HMM_DEFAULT_SEED
    max_iter: int = HMM_DEFAULT_MAX_ITER
    tol: float = HMM_DEFAULT_TOL
    prefer_hmmlearn: bool = True


class HMMRegimeLabeler:
    """Gaussian HMM (or GMM fallback) over the InVAR regime fingerprint.

    Fit-once / predict-many. Holds the chosen backend (``"hmmlearn"`` or
    ``"gmm"``), the fitted model, and a small summary blob for the
    cache header. The fingerprint passed at ``fit`` time MUST already
    be standardised with TRAIN-day stats only (the caller in
    ``src/baselines/train_invar_clpretrain_v2.py`` already does this
    for the canonical L2 selector; B1 reuses the exact same array).
    """

    def __init__(self, config: Optional[HMMRegimeConfig] = None) -> None:
        self.config = config or HMMRegimeConfig()
        self.backend: Optional[str] = None
        self.model = None
        self.n_train_days: int = 0
        self.n_features: int = 0
        self.converged: bool = False

    # ------------------------------------------------------------------
    def fit(
        self,
        macro_features: np.ndarray,
        n_states: Optional[int] = None,
    ) -> "HMMRegimeLabeler":
        """Fit the HMM (preferred) or a Gaussian mixture (fallback).

        Args:
            macro_features: (n_train_days, n_features) float array of
                TRAIN-day standardised regime fingerprints.
            n_states: optional override of ``self.config.n_states``.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: shape or size sanity violations.
            RuntimeError: neither backend importable.
        """
        if macro_features.ndim != 2:
            raise ValueError(
                f"macro_features must be 2D, got shape "
                f"{macro_features.shape}"
            )
        if n_states is not None:
            self.config.n_states = int(n_states)
        n_states_ = int(self.config.n_states)
        if n_states_ < 2:
            raise ValueError(
                f"n_states must be >= 2, got {n_states_}"
            )
        n_days, n_feat = macro_features.shape
        if n_days < max(10, n_states_):
            raise ValueError(
                f"need >= {max(10, n_states_)} train days, got {n_days}"
            )
        x = np.asarray(macro_features, dtype=np.float64)
        self.n_train_days = n_days
        self.n_features = n_feat

        hmm_ok, gmm_ok = _backend_available()
        if self.config.prefer_hmmlearn and hmm_ok:
            from hmmlearn.hmm import GaussianHMM
            model = GaussianHMM(
                n_components=n_states_,
                covariance_type="full",
                n_iter=int(self.config.max_iter),
                tol=float(self.config.tol),
                random_state=int(self.config.seed),
                init_params="stmc",
                params="stmc",
            )
            # hmmlearn expects (T, n_features) as a single sequence and
            # a lengths=[T] kwarg so it does not chop into mini-seqs.
            model.fit(x, lengths=[n_days])
            self.backend = "hmmlearn"
            self.model = model
            self.converged = bool(getattr(model.monitor_, "converged", True))
            print(
                f"[INFO] HMMRegimeLabeler: hmmlearn GaussianHMM fit ok "
                f"(n_states={n_states_} n_features={n_feat} "
                f"n_train_days={n_days} converged={self.converged})"
            )
            return self
        if not gmm_ok:
            raise RuntimeError(
                "Neither hmmlearn nor sklearn.mixture.GaussianMixture "
                "is importable. Install hmmlearn or scikit-learn."
            )
        # Fallback: GMM. Document explicitly.
        from sklearn.mixture import GaussianMixture
        model = GaussianMixture(
            n_components=n_states_,
            covariance_type="full",
            max_iter=int(self.config.max_iter),
            tol=float(self.config.tol),
            random_state=int(self.config.seed),
            init_params="kmeans",
        )
        model.fit(x)
        self.backend = "gmm"
        self.model = model
        self.converged = bool(model.converged_)
        print(
            f"[WARN] HMMRegimeLabeler: hmmlearn not importable; using "
            f"sklearn GaussianMixture FALLBACK (no learnt transition "
            f"matrix; positives are still posterior-weighted but the "
            f"temporal structure of the regime sequence is i.i.d.). "
            f"n_states={n_states_} n_features={n_feat} "
            f"n_train_days={n_days} converged={self.converged}"
        )
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, macro_features: np.ndarray) -> np.ndarray:
        """Posterior over latent regimes for each input day.

        Args:
            macro_features: (n_days, n_features) float array. n_features
                must match the value seen at ``fit`` time.

        Returns:
            (n_days, n_states) float64 array; each row sums to 1.

        Raises:
            RuntimeError: model not fitted yet.
            ValueError: feature-dim mismatch or NaN / non-finite output.
        """
        if self.model is None:
            raise RuntimeError(
                "HMMRegimeLabeler.predict_proba called before fit"
            )
        if macro_features.ndim != 2:
            raise ValueError(
                f"macro_features must be 2D, got shape "
                f"{macro_features.shape}"
            )
        if macro_features.shape[1] != self.n_features:
            raise ValueError(
                f"feature-dim mismatch: fit n_features={self.n_features}, "
                f"got {macro_features.shape[1]}"
            )
        x = np.asarray(macro_features, dtype=np.float64)
        if self.backend == "hmmlearn":
            # GaussianHMM.predict_proba returns (T, n_components)
            # forward-backward smoothed posteriors.
            posteriors = self.model.predict_proba(x)
        elif self.backend == "gmm":
            posteriors = self.model.predict_proba(x)
        else:  # pragma: no cover - defensive
            raise RuntimeError(
                f"unknown backend: {self.backend!r}"
            )
        posteriors = np.asarray(posteriors, dtype=np.float64)
        if not np.all(np.isfinite(posteriors)):
            raise ValueError(
                "HMMRegimeLabeler.predict_proba produced non-finite "
                "posteriors; check fit-time data for NaNs."
            )
        # Numerical safety: renormalise rows (hmmlearn/GMM occasionally
        # return rows that sum to 1 +/- 1e-9; we enforce exact unity).
        row_sums = posteriors.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise ValueError(
                "HMMRegimeLabeler.predict_proba produced a row with "
                "zero mass."
            )
        posteriors = posteriors / row_sums
        return posteriors


# ---------------------------------------------------------------------
# Disk cache helpers.
# ---------------------------------------------------------------------
def _cache_dir(universe: str, fold: int) -> Path:
    return (
        Path("cache/pretrain_improvements/hmm_regime")
        / str(universe)
        / f"fold{int(fold)}"
    )


def posteriors_path(universe: str, fold: int) -> Path:
    """Path to the B1 cached per-day posteriors parquet."""
    return _cache_dir(universe, fold) / "posteriors.parquet"


def save_posteriors(
    day_indices: np.ndarray,
    posteriors: np.ndarray,
    universe: str,
    fold: int,
    backend: str,
    n_states: int,
    n_train_days: int,
    seed: int,
) -> Path:
    """Persist per-day posteriors keyed by day_idx.

    Cache layout is DISJOINT from the k-means-8 cache used by L2 SIA
    (``cache/dr_rl/regime_probs/...``) so the canonical path is never
    touched.
    """
    if posteriors.ndim != 2 or posteriors.shape[1] != int(n_states):
        raise ValueError(
            f"posteriors shape mismatch: got {posteriors.shape}, "
            f"expected (n, {n_states})"
        )
    if posteriors.shape[0] != day_indices.shape[0]:
        raise ValueError(
            f"day_indices/posteriors row mismatch: "
            f"{day_indices.shape[0]} vs {posteriors.shape[0]}"
        )
    df = pd.DataFrame({
        "day_idx": np.asarray(day_indices, dtype=np.int64),
    })
    for k in range(int(n_states)):
        df[f"prob_{k}"] = posteriors[:, k].astype(np.float64)
    df["backend"] = str(backend)
    df["n_states"] = int(n_states)
    df["n_train_days"] = int(n_train_days)
    df["seed"] = int(seed)
    out = posteriors_path(universe, fold)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def load_posteriors(universe: str, fold: int) -> pd.DataFrame:
    """Load the cached per-day posteriors parquet."""
    p = posteriors_path(universe, fold)
    if not p.exists():
        raise FileNotFoundError(
            f"posteriors.parquet not found for universe={universe} "
            f"fold={fold}: {p}; rerun the B1 pretrain Stage 1."
        )
    return pd.read_parquet(p)


def cosine_similarity_matrix(posteriors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity over rows of a posterior matrix.

    Args:
        posteriors: (n, n_states) float array; rows assumed to sum to 1
            but the cosine path L2-normalises explicitly.

    Returns:
        (n, n) float64 matrix; diagonal is 1.0.
    """
    x = np.asarray(posteriors, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1.0e-12, 1.0, norms)
    xn = x / norms
    return xn @ xn.T


def build_positive_mask(
    posteriors_batch: np.ndarray,
    positive_threshold: float = HMM_DEFAULT_POSITIVE_THRESHOLD,
) -> np.ndarray:
    """Build the SupCon positive mask from posterior cosine similarity.

    Args:
        posteriors_batch: (B, n_states) per-day posteriors for the
            in-batch days.
        positive_threshold: cosine similarity floor (default 0.7); pairs
            (i, j != i) with similarity strictly above this become
            positives. The diagonal is forced False.

    Returns:
        (B, B) bool array.
    """
    sim = cosine_similarity_matrix(posteriors_batch)
    eye = np.eye(sim.shape[0], dtype=bool)
    pos = (sim >= float(positive_threshold)) & (~eye)
    return pos


__all__ = [
    "HMMRegimeConfig",
    "HMMRegimeLabeler",
    "HMM_DEFAULT_N_STATES",
    "HMM_DEFAULT_POSITIVE_THRESHOLD",
    "HMM_DEFAULT_SEED",
    "build_positive_mask",
    "cosine_similarity_matrix",
    "load_posteriors",
    "posteriors_path",
    "save_posteriors",
]
