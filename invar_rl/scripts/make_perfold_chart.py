"""Per-fold cumulative-return comparison (RQ5) for the InVAR-RL paper.

Line chart of per-fold cumulative return (PR; final equity minus one,
averaged over the 5 seeds) across the five macro-stratified folds, on the
S&P 500 long-only protocol. One headline InVAR-RL variant (long-only A3)
is drawn solid/bold against every long-only baseline drawn dashed/dotted.

All numbers are computed from the per-cell artifacts (baselines: final
equity in the result JSONs; InVAR-RL: the layer-3 daily tapes on Wulver),
so this figure visualises the long-only protocol at fold granularity.
(Whole-stack and ranker baselines store no daily series, so a smooth
within-fold cumulative curve is not available for them; per-fold PR is.)

Output (both venues):
  drafts/universal_RL_aaai/figures/perfold_chart.pdf
  drafts/universal_RL_kdd/figures/perfold_chart.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FOLDS = ["F1\nCOVID", "F2\nrate-stress", "F3\npost-stress",
         "F4\nAI rally", "F5\nfed-cut"]

# Per-fold cumulative return (PR), S&P 500 long-only, mean over 5 seeds.
OURS = ("InVAR-RL (A3, ours)", [0.621, -0.138, 0.198, 0.242, 0.064])
BASELINES = [
    ("MASTER",      [0.059, -0.215, 0.235, 0.195, 0.070], "#c0712f", (0, (5, 2)),    "s"),
    ("FactorVAE",   [0.097, -0.220, 0.189, 0.085, 0.070], "#3f8f43", (0, (5, 2)),    "^"),
    ("StockMixer",  [0.165, -0.199, 0.083, 0.068, 0.093], "#7a9ec0", (0, (1, 1.5)),  "D"),
    ("DySTAGE",     [0.039, -0.197, 0.116, 0.106, -0.017], "#b07aa1", (0, (1, 1.5)), "v"),
    ("FinRL A2C",   [0.329, -0.231, 0.117, -0.040, 0.151], "#9c4221", (0, (3, 1, 1, 1)), "P"),
    ("StockFormer", [0.074, -0.147, 0.037, 0.110, -0.038], "#6f6f6f", (0, (3, 1, 1, 1)), "X"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    x = np.arange(len(FOLDS))
    ax.axvspan(0.5, 1.5, color="#d62728", alpha=0.07, zorder=0)
    ax.axhline(0, color="black", lw=0.7, zorder=1)

    for name, vals, color, ls, mk in BASELINES:
        ax.plot(x, vals, color=color, lw=1.2, ls=ls, marker=mk, ms=4,
                alpha=0.9, zorder=3, label=name,
                markeredgecolor="white", markeredgewidth=0.4)
    ax.plot(x, OURS[1], color="#15396b", lw=2.8, ls="-", marker="o", ms=6,
            zorder=6, label=OURS[0],
            markeredgecolor="white", markeredgewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(FOLDS, fontsize=8)
    ax.set_ylabel("Cumulative return (per fold)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.margins(x=0.04)
    ax.annotate("F2 rate stress", xy=(1, ax.get_ylim()[0]),
                xytext=(1, -0.30), fontsize=7, color="#d62728", ha="center")
    ax.legend(fontsize=7.5, loc="upper right", ncol=2, framealpha=0.92,
              borderpad=0.5)
    plt.tight_layout()

    for out_dir in [Path("drafts/universal_RL_aaai/figures"),
                    Path("drafts/universal_RL_kdd/figures")]:
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "perfold_chart.pdf", bbox_inches="tight")
    plt.close()
    print("Saved perfold_chart.pdf to 2 dirs")


if __name__ == "__main__":
    main()
