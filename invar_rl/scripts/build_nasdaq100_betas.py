"""Build the NASDAQ-100 per-(day, ticker) rolling macro-beta panel.

Mirrors ``src.v2.data.rolling_macro_betas.build_rolling_betas`` byte-for-byte
on the NASDAQ-100 universe. Writes to
``data/processed/nasdaq100_rolling_betas.parquet`` with the SAME
``ROLLING_BETA_COLS`` schema so the canonical InVAR's
``betas_to_tensor`` consumes it without any code change.

Factor sources are the universal macro parquet at
``data/nasdaq100/macro.parquet`` which carries 32 of the same 32
universe-shared columns as the S&P 500 macro parquet (byte-identical
on the overlap; see ``reports/nasdaq100/phase_2_report.md``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PRICES_PATH = REPO_ROOT / "data" / "nasdaq100" / "prices.parquet"
MACRO_PATH = REPO_ROOT / "data" / "nasdaq100" / "macro.parquet"
OUT_PATH = REPO_ROOT / "data" / "processed" / "nasdaq100_rolling_betas.parquet"

PANEL_START = "2014-09-01"
PANEL_END = "2025-12-31"
WINDOW_60 = 60
WINDOW_120 = 120
MIN_OBS_60 = 30
MIN_OBS_120 = 60
RIDGE_ALPHA = 1e-3


def _ridge_solve(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Solve ridge OLS for one window.

    Byte-identical to ``src.v2.data.rolling_macro_betas._ridge_solve``.
    """
    if x.shape[0] < x.shape[1] + 1 or y.std() < 1e-9:
        return np.full(x.shape[1], np.nan, dtype=np.float32)
    xtx = x.T @ x + alpha * np.eye(x.shape[1])
    try:
        beta = np.linalg.solve(xtx, x.T @ y)
    except np.linalg.LinAlgError:
        return np.full(x.shape[1], np.nan, dtype=np.float32)
    return beta.astype(np.float32)


def _rolling_betas_for_ticker(
    ticker_ret: np.ndarray, factor_mat: np.ndarray,
    window: int, min_obs: int, alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling beta for one ticker. Byte-identical to upstream helper."""
    t_total, f = factor_mat.shape
    betas = np.full((t_total, f), 0.0, dtype=np.float32)
    valid = np.zeros(t_total, dtype=np.float32)
    for t in range(window - 1, t_total):
        lo = t - window + 1
        y_win = ticker_ret[lo: t + 1]
        x_win = factor_mat[lo: t + 1]
        ok = ~np.isnan(y_win) & ~np.isnan(x_win).any(axis=1)
        n_obs = int(ok.sum())
        valid[t] = n_obs / float(window)
        if n_obs < min_obs:
            continue
        b = _ridge_solve(x_win[ok], y_win[ok], alpha)
        if not np.isnan(b).any():
            betas[t] = b
    return betas, valid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=PANEL_START)
    parser.add_argument("--end", default=PANEL_END)
    args = parser.parse_args()

    if not PRICES_PATH.exists():
        print(f"ERROR: missing {PRICES_PATH}", flush=True); return 1
    if not MACRO_PATH.exists():
        print(f"ERROR: missing {MACRO_PATH}", flush=True); return 1

    raw = pd.read_parquet(PRICES_PATH).copy()
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    raw["ticker"] = raw["ticker"].astype(str).str.upper()
    raw = raw[
        (raw["date"] >= pd.Timestamp(args.start))
        & (raw["date"] <= pd.Timestamp(args.end))
    ].sort_values(["ticker", "date"]).reset_index(drop=True)
    raw["log_return"] = raw.groupby("ticker", sort=False)["close"].transform(
        lambda s: np.log(s / s.shift(1))
    )
    print(
        f"[nasdaq100_betas] prices: {len(raw):,} rows; "
        f"{raw['ticker'].nunique()} tickers",
        flush=True,
    )

    macro = pd.read_parquet(MACRO_PATH)
    macro.index = pd.to_datetime(macro.index).normalize()
    panel_dates = pd.DatetimeIndex(
        sorted(set(raw["date"]).union(set(macro.index)))
    )
    panel_dates = panel_dates[
        (panel_dates >= pd.Timestamp(args.start))
        & (panel_dates <= pd.Timestamp(args.end))
    ]
    macro_aligned = macro.reindex(panel_dates).ffill(limit=5)
    print(
        f"[nasdaq100_betas] panel dates: {len(panel_dates)}; "
        f"macro cols: {macro_aligned.shape[1]}",
        flush=True,
    )

    xbi_1d = macro_aligned["xbi_ret_1d"].to_numpy(dtype=np.float32)
    qqq_close = (macro_aligned["qqq_ret_5d"] / 5.0).to_numpy(dtype=np.float32)
    spy_close = (macro_aligned["spy_ret_5d"] / 5.0).to_numpy(dtype=np.float32)
    rate_shock_1d = macro_aligned["dgs10"].diff().to_numpy(dtype=np.float32)
    credit_shock_1d = macro_aligned["hy_spread"].diff().to_numpy(dtype=np.float32)
    factor_mat = np.stack(
        [xbi_1d, qqq_close, spy_close, rate_shock_1d, credit_shock_1d], axis=1
    )

    rows = []
    tickers = sorted(raw["ticker"].unique())
    for i, tk in enumerate(tickers, 1):
        sub = raw[raw["ticker"] == tk].set_index("date").reindex(panel_dates)
        ret = sub["log_return"].to_numpy(dtype=np.float32)
        b60, v60 = _rolling_betas_for_ticker(
            ret, factor_mat, WINDOW_60, MIN_OBS_60, RIDGE_ALPHA,
        )
        b120, v120 = _rolling_betas_for_ticker(
            ret, factor_mat[:, [0, 3, 4]],
            WINDOW_120, MIN_OBS_120, RIDGE_ALPHA,
        )
        out = pd.DataFrame({
            "date": panel_dates, "ticker": tk,
            "rolling_xbi_beta_60d": b60[:, 0],
            "rolling_qqq_beta_60d": b60[:, 1],
            "rolling_spy_beta_60d": b60[:, 2],
            "rolling_rate_beta_60d": b60[:, 3],
            "rolling_credit_beta_60d": b60[:, 4],
            "rolling_xbi_beta_120d": b120[:, 0],
            "rolling_rate_beta_120d": b120[:, 1],
            "rolling_credit_beta_120d": b120[:, 2],
            "beta_valid_ratio_60d": v60,
            "beta_valid_ratio_120d": v120,
        })
        rows.append(out)
        if i % 30 == 0:
            print(f"  [{i}/{len(tickers)}] {tk} betas done", flush=True)
    long = pd.concat(rows, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(OUT_PATH, index=False)
    print(
        f"[nasdaq100_betas] wrote {OUT_PATH}: shape={long.shape}, "
        f"tickers={long['ticker'].nunique()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
