"""Cumulative-return chart (RQ5) for the InVAR-RL universal paper.

RAG-STAR-style figure: the five macro-stratified folds are concatenated
left-to-right and the within-fold cumulative return (compounded daily
strategy return, averaged over the 5 seeds) is plotted, resetting at each
fold boundary (folds are independent walk-forward tests). Lines: InVAR-RL
long-short (A4), InVAR-RL long-only (A3), and the no-SAC InVAR-L1 baseline.

Data source: per-cell daily tapes (strategy_return / base_return) on the
S&P 500 panel, aggregated to invar_rl/results/cumret_sp500.csv via the
Wulver helper (whole-stack baselines store no daily series, so they cannot
appear on a daily curve).

Output (both venues):
  drafts/universal_RL_aaai/figures/cumret_chart.pdf
  drafts/universal_RL_kdd/figures/cumret_chart.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = "invar_rl/results/cumret_sp500.csv"
FOLD_LABELS = {1: "F1 2020", 2: "F2 2021-22", 3: "F3 2022-23",
               4: "F4 2024", 5: "F5 2025"}
STYLE = [
    ("InVAR-RL long-short (A4)", "#15396b", 2.4, "-"),
    ("InVAR-RL long-only (A3)", "#3a76c0", 1.7, "--"),
    ("InVAR-L1 (no SAC, L/S)", "#9a9a9a", 1.3, "-"),
]


def main() -> None:
    df = pd.read_csv(SRC)
    folds = sorted(df["fold"].unique())
    fig, ax = plt.subplots(figsize=(10, 3.6))

    # Concatenate folds left-to-right; record boundaries.
    offset = 0
    bounds = [0]
    xmap = {}
    for fold in folds:
        n = int(df[df["fold"] == fold]["t"].max()) + 1
        xmap[fold] = np.arange(offset, offset + n)
        offset += n
        bounds.append(offset)

    ax.axhline(0, color="black", lw=0.6, zorder=1)
    for name, color, lw, ls in STYLE:
        xs, ys = [], []
        for fold in folds:
            seg = df[df["fold"] == fold][name].to_numpy()
            xs.append(xmap[fold][:len(seg)])
            ys.append(seg)
        ax.plot(np.concatenate(xs), np.concatenate(ys), color=color,
                lw=lw, ls=ls, label=name, zorder=5 if lw > 2 else 3,
                solid_capstyle="round")

    # F2 rate-stress shading + fold separators and labels.
    ax.axvspan(bounds[1], bounds[2], color="#d62728", alpha=0.06, zorder=0)
    y_top = ax.get_ylim()[1]
    ax.set_ylim(top=y_top + 0.06 * (y_top - ax.get_ylim()[0]))
    for b in bounds[1:-1]:
        ax.axvline(b, color="grey", lw=0.6, ls=":", zorder=1)
    for fold in folds:
        mid = (xmap[fold][0] + xmap[fold][-1]) / 2
        ax.text(mid, ax.get_ylim()[1], FOLD_LABELS[fold], ha="center",
                va="top", fontsize=7.5, color="#333333")

    ax.set_ylabel("Cumulative return (per fold)", fontsize=9)
    ax.set_xlabel("fold-sequential trading-day index "
                  "(folds are independent walk-forward tests; "
                  "cumulative return resets per fold)", fontsize=7.5)
    ax.set_xticks([])
    ax.margins(x=0.01)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=7.5, loc="lower right", ncol=1, framealpha=0.92,
              borderpad=0.5)
    plt.tight_layout()

    for out_dir in [Path("drafts/universal_RL_aaai/figures"),
                    Path("drafts/universal_RL_kdd/figures")]:
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "cumret_chart.pdf", bbox_inches="tight")
    plt.close()
    print("Saved cumret_chart.pdf to 2 dirs")


if __name__ == "__main__":
    main()
