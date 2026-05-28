"""The three decisive analyses for the out-of-distribution stress fold.

1. Exposure-trajectory figure: exposure over time for the recurrent
   controller versus the myopic supervised head, with the rolling daily IC,
   to make visible whether the controller cuts exposure into a regime change
   before realised IC degrades.
2. Dissociation test: change in IC versus change in Calmar for the
   controller relative to the myopic head. The supporting result is a
   near-zero IC change with a clearly positive Calmar change.
3. Reward-frontier: terminal wealth against maximum drawdown as the
   drawdown-penalty weight is swept.

Figures use the non-interactive Agg backend so they render headless on
Wulver.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def exposure_trajectory_figure(
    recurrent_exposure: np.ndarray,
    myopic_exposure: np.ndarray,
    daily_ic: np.ndarray,
    out_path: str | Path,
) -> str:
    """Plot OOD-fold exposure paths plus rolling IC; save and return path."""
    fig, ax1 = plt.subplots(figsize=(10, 4))
    t = np.arange(recurrent_exposure.size)
    ax1.plot(t, recurrent_exposure, label="recurrent controller",
             color="C0")
    ax1.plot(t, myopic_exposure[: t.size], label="myopic head",
             color="C1", linestyle="--")
    ax1.set_xlabel("trading day (OOD fold)")
    ax1.set_ylabel("exposure")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    w = min(20, max(2, daily_ic.size // 10))
    roll_ic = np.convolve(
        daily_ic[: t.size], np.ones(w) / w, mode="same"
    )
    ax2.plot(t, roll_ic, color="C3", alpha=0.5,
             label="rolling daily IC")
    ax2.set_ylabel("rolling daily IC")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return str(out)


def dissociation_table(
    controller_metrics: Dict[str, float],
    myopic_metrics: Dict[str, float],
) -> Dict[str, float]:
    """Change in IC and Calmar of the controller relative to the myopic head.

    The paper-supporting pattern is delta_ic near zero with delta_calmar
    clearly positive: the controller leaves the signal untouched but
    improves risk-adjusted return.
    """
    return {
        "controller_ic": controller_metrics["ic"],
        "myopic_ic": myopic_metrics["ic"],
        "delta_ic": controller_metrics["ic"] - myopic_metrics["ic"],
        "controller_calmar": controller_metrics["calmar"],
        "myopic_calmar": myopic_metrics["calmar"],
        "delta_calmar": (
            controller_metrics["calmar"] - myopic_metrics["calmar"]
        ),
    }


def reward_frontier_figure(
    points: Sequence[Tuple[float, float, float]],
    out_path: str | Path,
) -> str:
    """Plot terminal wealth vs max drawdown over drawdown-penalty sweep.

    Args:
        points: Iterable of (drawdown_penalty, terminal_wealth, max_dd).
        out_path: Output image path.
    """
    pts = sorted(points, key=lambda p: p[0])
    pen = [p[0] for p in pts]
    tw = [p[1] for p in pts]
    mdd = [p[2] for p in pts]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mdd, tw, marker="o", color="C0")
    for p, x, y in zip(pen, mdd, tw):
        ax.annotate(f"lambda={p:g}", (x, y),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8)
    ax.set_xlabel("maximum drawdown")
    ax.set_ylabel("terminal wealth")
    ax.set_title("Reward frontier (drawdown-penalty sweep)")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return str(out)
