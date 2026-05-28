"""Results-comparison figure for the InVAR-RL universal paper.

Two panels, both from the long-only Sharpe numbers reported in Table~2
(tab:tableB) of the paper:

  (a) Overall annualised Sharpe by model across the three universes
      (S&P 500, NASDAQ-100, biotech NBI), showing InVAR-RL wins each.
  (b) Per-fold S&P 500 Sharpe for InVAR-RL vs the two whole-stack RL
      baselines and the best ranker, with the F2 rate-stress fold shaded,
      showing InVAR-RL's shallow F2 drawdown where FinRL/StockFormer collapse.

Numbers are the paper's own reported values (no recompute), so this figure
visualises Table 2 rather than introducing new results.

Output (both venues):
  drafts/universal_RL_aaai/figures/results_chart.pdf
  drafts/universal_RL_kdd/figures/results_chart.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SYS = "InVAR-RL"
SYS_COLOR = "#15396b"

# (a) Overall long-only Sharpe per universe (Table 2 "Overall" column).
MODELS = ["FactorVAE", "MASTER", "StockMixer", "DySTAGE",
          "FinRL", "StockFormer", SYS]
OVERALL = {
    "S&P 500":   [0.26, 0.39, 0.28, 0.02, 0.48, 0.10, 0.85],
    "NASDAQ-100": [0.53, 0.64, 0.52, 0.50, 0.15, 0.80, 0.84],
    "biotech NBI": [0.35, 0.45, 0.31, 0.39, 0.13, 0.49, 0.62],
}

# (b) Per-fold S&P 500 long-only Sharpe (Table 2 SP500 block).
FOLDS = ["F1", "F2", "F3", "F4", "F5"]
PERFOLD = {
    "MASTER":      [0.10, -0.84, 0.87, 1.02, 0.82],
    "FinRL":       [1.09, -1.15, 0.79, -0.20, 1.88],
    "StockFormer": [1.15, -1.34, 0.58, 1.29, -1.17],
    SYS:           [1.27, -0.31, 1.01, 1.21, 1.07],
}
PERFOLD_COLOR = {
    "MASTER": "#c0712f", "FinRL": "#3f8f43",
    "StockFormer": "#8a6fc0", SYS: SYS_COLOR,
}


def main() -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 3.6),
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})

    # ---- Panel (a): grouped bars, overall Sharpe across universes ----
    universes = list(OVERALL.keys())
    n_models = len(MODELS)
    x = np.arange(len(universes))
    width = 0.8 / n_models
    base_colors = ["#9a9a9a", "#c0712f", "#7a9ec0", "#3f8f43",
                   "#b07aa1", "#8a6fc0", SYS_COLOR]
    for j, m in enumerate(MODELS):
        vals = [OVERALL[u][j] for u in universes]
        off = (j - (n_models - 1) / 2) * width
        is_sys = m == SYS
        axL.bar(x + off, vals, width,
                color=base_colors[j],
                edgecolor="black" if is_sys else "none",
                linewidth=1.1 if is_sys else 0.0,
                label=m, zorder=3 if is_sys else 2)
    axL.axhline(0, color="black", lw=0.6)
    axL.set_xticks(x)
    axL.set_xticklabels(universes, fontsize=8)
    axL.set_ylabel("Overall annualised Sharpe", fontsize=8)
    axL.set_title("(a) Long-only Sharpe across universes", fontsize=9)
    axL.tick_params(axis="y", labelsize=7)
    axL.grid(axis="y", alpha=0.25, zorder=0)
    axL.legend(fontsize=6.5, ncol=2, loc="upper left", framealpha=0.9)

    # ---- Panel (b): per-fold SP500 Sharpe, F2 shaded ----
    xf = np.arange(len(FOLDS))
    axR.axvspan(0.5, 1.5, color="#d62728", alpha=0.10, zorder=0)
    base_ls = {"MASTER": (0, (5, 2)), "FinRL": (0, (3, 1, 1, 1)),
               "StockFormer": (0, (1, 1.5))}
    for m, vals in PERFOLD.items():
        is_sys = m == SYS
        axR.plot(xf, vals, marker="o" if is_sys else "s", ms=5 if is_sys else 4,
                 lw=2.8 if is_sys else 1.2,
                 ls="-" if is_sys else base_ls.get(m, (0, (5, 2))),
                 color=PERFOLD_COLOR[m],
                 zorder=6 if is_sys else 3,
                 markeredgecolor="white", markeredgewidth=0.5,
                 label=m)
    axR.axhline(0, color="black", lw=0.6)
    axR.set_xticks(xf)
    axR.set_xticklabels(FOLDS, fontsize=8)
    axR.set_ylabel("Annualised Sharpe", fontsize=8)
    axR.set_title("(b) S&P 500 per-fold Sharpe (F2 = rate stress)",
                  fontsize=9)
    axR.tick_params(axis="y", labelsize=7)
    axR.grid(axis="y", alpha=0.25, zorder=0)
    axR.annotate("F2 rate stress", xy=(1, -1.0), xytext=(1.6, -1.25),
                 fontsize=6.5, color="#d62728",
                 ha="left", va="center")
    axR.legend(fontsize=6.5, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    for out_dir in [Path("drafts/universal_RL_aaai/figures"),
                    Path("drafts/universal_RL_kdd/figures")]:
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "results_chart.pdf", bbox_inches="tight")
    plt.close()
    print("Saved results_chart.pdf to 2 dirs")


if __name__ == "__main__":
    main()
