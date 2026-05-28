"""Reconstruct ranker cumulative-return series from the published RAG-STAR chart.

Context: the universal-panel ranker prediction npz
(results/baselines_universal_two_regime_val/{master,factorvae,stockmixer,
itransformer} and results/rag_star_universe_v2_no_ipo) are no longer
retained on any machine. The only surviving artifact is the rendered
cumulative-return figure (drafts/universal_paper_{aaai,kdd}/figures/
cumulative_returns.pdf), produced by the upstream
scripts/cumulative_returns_universal.py.

That figure is a VECTOR PDF, so each model curve is stored as an exact
polyline (not a rasterized image). This script digitizes those polylines
and inverts the axis calibration to recover the per-model cumulative
series the chart plots:
  panel (a): cumulative excess return, dollar-neutral long-short top/bottom 25
  panel (b): cumulative return, long top-quintile book (+ eq-weight benchmark)
both on a non-overlapping 5-day rebalance grid, concatenated across the
five macro folds with an equity reset at each fold boundary.

IMPORTANT (what this is and is NOT):
  - This recovers the PLOTTED CURVE DATA (a 1-D cumulative series per model),
    to high precision (stored coordinates + linear axis inversion).
  - It does NOT recover the raw predictions npz (y_hat[T, ~600 tickers],
    y_true[T, ~600]). The chart is a many-to-one projection of that matrix;
    it is not invertible. Per-ticker scores are unrecoverable.
  - Models present are exactly those on the chart: RAG-STAR, MASTER,
    FactorVAE, iTransformer, StockMixer. DySTAGE is NOT on this chart.
  - This is the RAG-STAR universal 600-ticker / two-regime-val panel,
    long-short and long-quintile books. Confirm protocol/universe match
    before splicing into the InVAR-RL universal_RL Figure 8 (which is the
    SP500 long-only PR protocol).

Outputs:
  invar_rl/results/ranker_curves_from_chart.csv   (long format)
  invar_rl/results/ranker_curves_from_chart.npz   (per-model arrays)
"""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
import numpy as np

PDF = "drafts/universal_paper_aaai/figures/cumulative_returns.pdf"

# Stroke colors as emitted by cumulative_returns_universal.py (matplotlib
# normalizes the hex to these RGB triples in the PDF).
COLORS = {
    "RAG-STAR":     (0.122, 0.231, 0.42),   # #1f3b6b
    "MASTER":       (0.612, 0.259, 0.129),  # #9c4221
    "FactorVAE":    (0.18, 0.49, 0.196),    # #2e7d32
    "iTransformer": (0.486, 0.227, 0.929),  # #7c3aed
    "StockMixer":   (0.4, 0.4, 0.4),        # #666666
    "Universe":     (0.0, 0.0, 0.0),        # eq-weight benchmark (panel b)
}

# Axis calibration read from the PDF text spans (tick label centers).
PANEL_X0 = {"a": 57.5, "b": 520.1}   # px at step 0
PX_PER_STEP = 1.656                  # both panels (50 steps over 82.8 px)
Y0_PX = {"a": 126.3, "b": 107.3}     # px at value 0
PX_PER_20 = {"a": 33.35, "b": 45.0}  # px per 20 units of cumulative %
PANEL_SPLIT_X = 470                  # x < split => panel a, else panel b

# Fold boundaries in step units (from the grey dotted separators).
FOLD_BOUNDS = [0, 52, 104, 155, 207, 231]


def _near(a, b, tol=0.01):
    return all(abs(x - y) < tol for x, y in zip(a, b))


def _xstep(px, panel):
    return (px - PANEL_X0[panel]) / PX_PER_STEP


def _yval(py, panel):
    return (Y0_PX[panel] - py) * (20.0 / PX_PER_20[panel])


def extract_curves(pdf_path: str) -> dict:
    """Return {(panel, model): (steps, values)} digitized from the PDF."""
    doc = fitz.open(pdf_path)
    pg = doc[0]
    out: dict = {}
    for d in pg.get_drawings():
        col = d.get("color")
        if col is None:
            continue
        model = next((m for m, c in COLORS.items()
                      if _near(tuple(col), c)), None)
        if model is None:
            continue
        pts = []
        for it in d["items"]:
            if it[0] == "l":
                pts.extend([(it[1].x, it[1].y), (it[2].x, it[2].y)])
            elif it[0] == "m":
                pts.append((it[1].x, it[1].y))
        if len(pts) < 20:
            continue
        panel = "a" if d["rect"].x0 < PANEL_SPLIT_X else "b"
        arr = np.array([(_xstep(x, panel), _yval(y, panel)) for x, y in pts])
        # Deduplicate by step (segment endpoints repeat) and sort.
        arr = arr[np.argsort(arr[:, 0])]
        _, uniq = np.unique(np.round(arr[:, 0], 3), return_index=True)
        arr = arr[np.sort(uniq)]
        key = (panel, model)
        if key not in out or len(arr) > len(out[key][0]):
            out[key] = (arr[:, 0], arr[:, 1])
    return out


def main() -> None:
    curves = extract_curves(PDF)
    rows = ["panel,book,model,fold,step,cum_pct"]
    npz: dict = {}
    book = {"a": "long_short_excess", "b": "long_quintile"}
    for (panel, model), (steps, vals) in sorted(curves.items()):
        npz[f"{panel}__{model}__step"] = steps
        npz[f"{panel}__{model}__cum_pct"] = vals
        for s, v in zip(steps, vals):
            fold = next((i + 1 for i in range(5)
                         if FOLD_BOUNDS[i] <= s < FOLD_BOUNDS[i + 1]), 5)
            rows.append(f"{panel},{book[panel]},{model},{fold},"
                        f"{s:.2f},{v:.3f}")

    out_csv = Path("invar_rl/results/ranker_curves_from_chart.csv")
    out_npz = Path("invar_rl/results/ranker_curves_from_chart.npz")
    out_csv.write_text("\n".join(rows) + "\n")
    np.savez(out_npz, **npz)

    print(f"Wrote {out_csv} ({len(rows) - 1} rows) and {out_npz}")
    print("Recovered (panel, model): end-of-series cumulative %")
    for (panel, model), (steps, vals) in sorted(curves.items()):
        print(f"  {panel} {book[panel]:18s} {model:12s} "
              f"n={len(steps):3d}  end={vals[-1]:6.2f}%")


if __name__ == "__main__":
    main()
