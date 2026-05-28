"""Post-hoc k-means-8 soft assignment over macro_input (SIA group source).

For each (universe, fold), fits ``KMeans(n_clusters=8, random_state=42)``
on the per-day ``macro_input`` rows from the TRAINING window only (no
val or test contamination). Per-day soft assignment is then the softmax
of negative distances to the 8 centroids, with a temperature knob.

Zero learned parameters, deterministic per (universe, fold, seed=42).
SIA uses the per-day hard argmax over these 8 clusters as the group id
inside the actor auxiliary regime-invariance penalty. The SIA observation
is unchanged (still uses the canonical macro_encoding column).

Ported from the InVAR-DR-RL Phase 1 commit (e039508) into the SIA module
space so this branch does not depend on layer3_control's regime_probs
(which lives on a different branch lineage). The on-disk cache layout
is shared, so a single ``cache/dr_rl/regime_probs/{universe}/fold{F}/``
directory can serve both code paths.

Caches:
  - centroids + scale: ``cache/dr_rl/regime_probs/{universe}/fold{F}/kmeans_fit.npz``
  - per-day probabilities: ``cache/dr_rl/regime_probs/{universe}/fold{F}/probs.parquet``
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


_N_CLUSTERS: int = 8
_KMEANS_SEED: int = 42
_DEFAULT_TEMPERATURE: float = 1.0


def fit_kmeans_8(
    macro_inputs_train: np.ndarray,
    temperature: float = _DEFAULT_TEMPERATURE,
    random_state: int = _KMEANS_SEED,
) -> Dict[str, object]:
    """Fit a KMeans-8 over training-window per-day macro inputs.

    Args:
        macro_inputs_train: (n_train_days, macro_dim) float array of
            standardised macro vectors taken from the bridge's training
            window only.
        temperature: Softmax temperature; default 1.0. Lower values
            sharpen the soft assignment toward a one-hot; higher values
            spread it.
        random_state: KMeans initialisation seed (fixed at 42 for
            reproducibility).

    Returns:
        Dict with:
          - ``centroids``: (8, macro_dim) cluster centroids
          - ``temperature``: float
          - ``macro_dim``: int
          - ``n_train_days``: int
          - ``inertia``: float (final KMeans inertia)
    """
    if macro_inputs_train.ndim != 2:
        raise ValueError(
            f"macro_inputs_train must be 2D, got shape "
            f"{macro_inputs_train.shape}"
        )
    if macro_inputs_train.shape[0] < _N_CLUSTERS:
        raise ValueError(
            f"need at least {_N_CLUSTERS} training days, got "
            f"{macro_inputs_train.shape[0]}"
        )
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    km = KMeans(
        n_clusters=_N_CLUSTERS,
        random_state=int(random_state),
        n_init=10,
    )
    km.fit(macro_inputs_train.astype(np.float64))
    return {
        "centroids": km.cluster_centers_.astype(np.float64),
        "temperature": float(temperature),
        "macro_dim": int(macro_inputs_train.shape[1]),
        "n_train_days": int(macro_inputs_train.shape[0]),
        "inertia": float(km.inertia_),
    }


def compute_regime_probs(
    macro_input_day: np.ndarray,
    fitted: Dict[str, object],
) -> np.ndarray:
    """Soft k-means assignment for one day's macro_input.

    Args:
        macro_input_day: (macro_dim,) float vector for one day.
        fitted: Output of :func:`fit_kmeans_8`.

    Returns:
        (8,) float64 softmax probabilities, sums to 1.
    """
    centroids = np.asarray(fitted["centroids"], dtype=np.float64)
    temperature = float(fitted["temperature"])
    if macro_input_day.shape != (centroids.shape[1],):
        raise ValueError(
            f"macro_input_day shape {macro_input_day.shape} does not "
            f"match centroids {centroids.shape}"
        )
    x = macro_input_day.astype(np.float64)
    diff = centroids - x[None, :]
    dist = np.linalg.norm(diff, axis=1)
    logits = -dist / temperature
    logits -= logits.max()  # numerical stability
    exp = np.exp(logits)
    return (exp / exp.sum()).astype(np.float64)


def compute_regime_probs_batch(
    macro_inputs: np.ndarray,
    fitted: Dict[str, object],
) -> np.ndarray:
    """Vectorised batch version of :func:`compute_regime_probs`.

    Args:
        macro_inputs: (n_days, macro_dim) float array.
        fitted: Output of :func:`fit_kmeans_8`.

    Returns:
        (n_days, 8) float64 softmax probabilities; each row sums to 1.
    """
    centroids = np.asarray(fitted["centroids"], dtype=np.float64)
    temperature = float(fitted["temperature"])
    x = macro_inputs.astype(np.float64)
    if x.shape[1] != centroids.shape[1]:
        raise ValueError(
            f"macro_inputs.shape[1]={x.shape[1]} does not match "
            f"centroids.shape[1]={centroids.shape[1]}"
        )
    # (n_days, 8) of distances.
    diff = x[:, None, :] - centroids[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    logits = -dist / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float64)


def _cache_dir(universe: str, fold: int) -> Path:
    return (
        Path("cache/dr_rl/regime_probs")
        / str(universe)
        / f"fold{int(fold)}"
    )


def _fit_path(universe: str, fold: int) -> Path:
    return _cache_dir(universe, fold) / "kmeans_fit.npz"


def _probs_path(universe: str, fold: int) -> Path:
    return _cache_dir(universe, fold) / "probs.parquet"


def save_fit(fitted: Dict[str, object], universe: str, fold: int) -> Path:
    """Persist KMeans-8 fit + scale to cache/dr_rl/regime_probs/.../kmeans_fit.npz."""
    out = _fit_path(universe, fold)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        centroids=np.asarray(fitted["centroids"], dtype=np.float64),
        temperature=np.asarray([float(fitted["temperature"])], dtype=np.float64),
        macro_dim=np.asarray([int(fitted["macro_dim"])], dtype=np.int64),
        n_train_days=np.asarray([int(fitted["n_train_days"])], dtype=np.int64),
        inertia=np.asarray([float(fitted["inertia"])], dtype=np.float64),
    )
    return out


def load_fit(universe: str, fold: int) -> Dict[str, object]:
    """Load the cached KMeans-8 fit; raises FileNotFoundError if missing."""
    p = _fit_path(universe, fold)
    if not p.exists():
        raise FileNotFoundError(
            f"kmeans_fit.npz not found for universe={universe} fold={fold}: "
            f"{p}; run precompute_all first"
        )
    blob = np.load(p)
    return {
        "centroids": blob["centroids"].astype(np.float64),
        "temperature": float(blob["temperature"][0]),
        "macro_dim": int(blob["macro_dim"][0]),
        "n_train_days": int(blob["n_train_days"][0]),
        "inertia": float(blob["inertia"][0]),
    }


def save_probs(
    day_indices: Sequence[int],
    dates: Sequence[str],
    probs: np.ndarray,
    universe: str,
    fold: int,
) -> Path:
    """Persist per-day regime_probs to a single parquet keyed by date.

    Args:
        day_indices: Per-row global trading-day indices.
        dates: Per-row ISO date strings (or pandas-parseable).
        probs: (n, 8) softmax probabilities.
        universe: e.g. "sp500".
        fold: 1..5.

    Returns:
        Path to the written parquet.
    """
    if probs.shape[1] != _N_CLUSTERS:
        raise ValueError(
            f"probs must have {_N_CLUSTERS} columns, got {probs.shape[1]}"
        )
    if len(day_indices) != probs.shape[0]:
        raise ValueError(
            f"day_indices length {len(day_indices)} != probs rows "
            f"{probs.shape[0]}"
        )
    df = pd.DataFrame({
        "day_idx": np.asarray(day_indices, dtype=np.int64),
        "date": pd.to_datetime(np.asarray(dates)).normalize(),
    })
    for k in range(_N_CLUSTERS):
        df[f"prob_{k}"] = probs[:, k].astype(np.float64)
    out = _probs_path(universe, fold)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def load_probs(universe: str, fold: int) -> pd.DataFrame:
    """Load the cached per-day regime_probs parquet as a DataFrame."""
    p = _probs_path(universe, fold)
    if not p.exists():
        raise FileNotFoundError(
            f"probs.parquet not found for universe={universe} "
            f"fold={fold}: {p}; run precompute_all first"
        )
    return pd.read_parquet(p)


def load_probs_lookup(universe: str, fold: int) -> Dict[int, np.ndarray]:
    """Load the per-day regime_probs as a day_idx -> (8,) lookup dict."""
    df = load_probs(universe, fold)
    cols = [f"prob_{k}" for k in range(_N_CLUSTERS)]
    arr = df[cols].to_numpy(dtype=np.float64)
    return {int(d): arr[i] for i, d in enumerate(df["day_idx"].to_numpy())}


def precompute_all(
    universe: str,
    fold: int,
    bridge,
    temperature: float = _DEFAULT_TEMPERATURE,
    extra_days: Optional[Iterable[int]] = None,
) -> Dict[str, object]:
    """Fit KMeans-8 on the training window and persist all-day regime_probs.

    Args:
        universe: One of {"sp500", "nasdaq100", "biotech_nbi",
            "biotech_nbi_enriched"} (caller's responsibility to use a
            consistent label).
        fold: 1..5.
        bridge: A ``LatticePanelBatch``-like object exposing
            ``macro_arr`` (T, macro_dim), ``train_idx``, ``val_idx``,
            ``test_idx``, and ``dates`` (T-long iterable).
        temperature: Softmax temperature.
        extra_days: Optional extra global day indices to include in the
            persisted probs (e.g. warmup days the SAC env may touch).

    Returns:
        Dict with ``fitted`` (KMeans output), ``n_days`` (probs rows),
        ``fit_path``, ``probs_path``.
    """
    macro = np.asarray(bridge.macro_arr, dtype=np.float64)
    train_idx = np.asarray(bridge.train_idx, dtype=np.int64)
    if train_idx.size < _N_CLUSTERS:
        raise ValueError(
            f"fold {fold}: train window has only {train_idx.size} days, "
            f"need >= {_N_CLUSTERS}"
        )
    train_macro = macro[train_idx]
    fitted = fit_kmeans_8(train_macro, temperature=temperature)

    all_days: List[int] = []
    all_days.extend(int(d) for d in bridge.train_idx)
    all_days.extend(int(d) for d in bridge.val_idx)
    all_days.extend(int(d) for d in bridge.test_idx)
    if extra_days is not None:
        all_days.extend(int(d) for d in extra_days)
    # Deduplicate while preserving order.
    seen = set()
    ordered: List[int] = []
    for d in all_days:
        if d in seen:
            continue
        seen.add(d)
        if 0 <= d < macro.shape[0]:
            ordered.append(d)
    day_arr = np.asarray(ordered, dtype=np.int64)
    macro_subset = macro[day_arr]
    probs = compute_regime_probs_batch(macro_subset, fitted)

    dates = [str(bridge.dates[int(d)]) for d in day_arr]
    fit_path = save_fit(fitted, universe=universe, fold=fold)
    probs_path = save_probs(
        day_indices=day_arr,
        dates=dates,
        probs=probs,
        universe=universe,
        fold=fold,
    )
    return {
        "fitted": fitted,
        "n_days": int(day_arr.shape[0]),
        "fit_path": str(fit_path),
        "probs_path": str(probs_path),
    }


def regime_probs_for_day_indices(
    universe: str,
    fold: int,
    day_indices: Sequence[int],
) -> np.ndarray:
    """Convenience: stack regime_probs for a sequence of day indices.

    Returns (len(day_indices), 8) float64 array. Missing days raise.
    """
    lookup = load_probs_lookup(universe, fold)
    rows: List[np.ndarray] = []
    for d in day_indices:
        d_int = int(d)
        if d_int not in lookup:
            raise KeyError(
                f"day_idx={d_int} not in cached regime_probs for "
                f"universe={universe} fold={fold}; re-run precompute_all "
                f"with extra_days"
            )
        rows.append(lookup[d_int])
    return np.asarray(rows, dtype=np.float64)


def override_tape_macro_encoding(tape, universe: str, fold: int):
    """Overwrite an EpisodeTape's macro_encoding with cached regime_probs.

    Phase 1 (Option B) integration helper. The tape is built by the
    canonical precompute pipeline with the FiLM-style macro encoding
    in ``macro_encoding``. This helper replaces that column with the
    cached 8-dim k-means soft assignment, keyed by ``tape.days``.

    Args:
        tape: An :class:`EpisodeTape` or any object with a writable
            ``macro_encoding`` attribute and a ``days`` attribute that
            indexes the cached probs.
        universe: e.g. "sp500".
        fold: 1..5.

    Returns:
        The same tape object (mutated in place) for chaining.

    Raises:
        KeyError: if any day in the tape is missing from the cache.
            Re-run :func:`precompute_all` with ``extra_days=tape.days``.
    """
    lookup = load_probs_lookup(universe, fold)
    days = np.asarray(tape.days, dtype=np.int64)
    rows = np.zeros((days.shape[0], _N_CLUSTERS), dtype=np.float64)
    for i, d in enumerate(days):
        d_int = int(d)
        if d_int not in lookup:
            raise KeyError(
                f"day_idx={d_int} not in cached regime_probs for "
                f"universe={universe} fold={fold}; rerun precompute_all"
            )
        rows[i] = lookup[d_int]
    tape.macro_encoding = rows
    return tape


def override_variable_k_tape(tape, universe: str, fold: int):
    """Same as :func:`override_tape_macro_encoding` for a VariableKTape.

    A :class:`VariableKTape` wraps an EpisodeTape under ``.episode``; we
    forward to the inner tape so the SP500 Ablation 6 driver can use the
    same helper.
    """
    if hasattr(tape, "episode"):
        override_tape_macro_encoding(tape.episode, universe, fold)
        return tape
    return override_tape_macro_encoding(tape, universe, fold)


__all__ = [
    "fit_kmeans_8",
    "compute_regime_probs",
    "compute_regime_probs_batch",
    "save_fit",
    "load_fit",
    "save_probs",
    "load_probs",
    "load_probs_lookup",
    "precompute_all",
    "regime_probs_for_day_indices",
]
