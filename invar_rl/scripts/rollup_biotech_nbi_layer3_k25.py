"""Roll up biotech NBI Phase 5 Layer 3 SAC K=25 results.

Byte-for-byte mirror of :mod:`invar_rl.scripts.rollup_biotech_nbi_layer3`
except the input root is ``outputs/biotech_nbi/layer3_k25/{ls,lo}/`` and
the output report lands at
``reports/biotech_nbi/phase_5_layer3_sharpe_k25.md``.

The K=25 run uses an equal-weight long top-25 / short bottom-25 wrapper
in place of the MV-QP, giving an apples-to-apples K against the Phase
5.5 baseline rankers (which all use a top-25 long / bottom-25 short
equal-weight book).

Usage::

    PYTHONPATH=$PWD python3 -m invar_rl.scripts.rollup_biotech_nbi_layer3_k25
"""
from __future__ import annotations

from pathlib import Path

from invar_rl.scripts.rollup_biotech_nbi_layer3 import (
    _day_stream_pooled_sharpe,
    _load_protocol,
    _per_cell_mean_sharpe,
    _render_markdown,
)


def main() -> int:
    root = Path("outputs/biotech_nbi/layer3_k25")
    ls_cells = _load_protocol(root / "ls")
    lo_cells = _load_protocol(root / "lo")
    out_path = Path("reports/biotech_nbi/phase_5_layer3_sharpe_k25.md")
    _render_markdown(
        ls_cells, lo_cells, out_path,
        title=(
            "Biotech NBI Phase 5: Layer 3 SAC Sharpe rollup "
            "(K=25 per side, equal-weight wrapper, fair comparison "
            "vs Phase 5.5 baselines)"
        ),
        parquet_root="outputs/biotech_nbi/layer3_k25",
    )
    print(
        f"[rollup-k25] wrote {out_path}: "
        f"ls cells={len(ls_cells)} lo cells={len(lo_cells)}"
    )
    if ls_cells:
        ls_pc_mean, ls_pc_std = _per_cell_mean_sharpe(ls_cells)
        print(
            f"[rollup-k25] LS per-cell mean Sharpe (PRIMARY): "
            f"{ls_pc_mean:+.4f} +/- {ls_pc_std:.4f}"
        )
        print(
            f"[rollup-k25] LS day-stream pooled Sharpe (secondary): "
            f"{_day_stream_pooled_sharpe(ls_cells):+.4f}"
        )
    if lo_cells:
        lo_pc_mean, lo_pc_std = _per_cell_mean_sharpe(lo_cells)
        print(
            f"[rollup-k25] LO per-cell mean Sharpe (PRIMARY): "
            f"{lo_pc_mean:+.4f} +/- {lo_pc_std:.4f}"
        )
        print(
            f"[rollup-k25] LO day-stream pooled Sharpe (secondary): "
            f"{_day_stream_pooled_sharpe(lo_cells):+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
