"""Self-verifying rollup of portfolio-return (PR) and annual-return (AR)
metrics to accompany the Sharpe ratios in the headline result tables.

Method (faithful, no fabrication): for every candidate result source we
extract the 25 per-cell (Sharpe, final_equity, n_steps) triples, compute
the per-cell-mean Sharpe (the table's pooling convention), and match it
against the Sharpe value already reported in the paper table. Only when
the computed Sharpe reproduces the reported value (within tol) do we trust
and emit that source's PR and AR. Cells that fail to reproduce are flagged,
never guessed.

  PR (portfolio/cumulative return) = mean_cells(final_equity - 1)
  AR (annualised return)           = mean_cells(final_equity^(252/n) - 1)

Usage:
  PYTHONPATH=$PWD python3 invar_rl/scripts/rollup_portfolio_metrics.py
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

TRADING_DAYS = 252
TOL = 0.03  # |SR_calc - SR_table| acceptable for a verified match

# Reported Overall Sharpe from the paper tables (tab:tableB long-only,
# tab:tableLS long-short). Keyed (universe, protocol, method).
TARGETS: Dict[Tuple[str, str, str], float] = {
    # long-only (Table B)
    ("sp500", "lo", "factorvae"): 0.26, ("sp500", "lo", "master"): 0.39,
    ("sp500", "lo", "stockmixer"): 0.28, ("sp500", "lo", "dystage"): 0.02,
    ("sp500", "lo", "finrl"): 0.48, ("sp500", "lo", "stockformer"): 0.10,
    ("sp500", "lo", "invarrl"): 0.85,
    ("nasdaq100", "lo", "factorvae"): 0.53, ("nasdaq100", "lo", "master"): 0.64,
    ("nasdaq100", "lo", "stockmixer"): 0.52, ("nasdaq100", "lo", "dystage"): 0.50,
    ("nasdaq100", "lo", "finrl"): 0.15, ("nasdaq100", "lo", "stockformer"): 0.80,
    ("nasdaq100", "lo", "invarrl"): 0.84,
    ("biotech", "lo", "factorvae"): 0.35, ("biotech", "lo", "master"): 0.45,
    ("biotech", "lo", "stockmixer"): 0.31, ("biotech", "lo", "dystage"): 0.39,
    ("biotech", "lo", "finrl"): 0.13, ("biotech", "lo", "stockformer"): 0.49,
    ("biotech", "lo", "invarrl"): 0.62,
    # long-short (Table LS)
    ("sp500", "ls", "factorvae"): 0.31, ("sp500", "ls", "master"): 0.58,
    ("sp500", "ls", "stockmixer"): 0.45, ("sp500", "ls", "dystage"): 0.04,
    ("sp500", "ls", "invarrl"): 1.03,
    ("nasdaq100", "ls", "factorvae"): 1.53, ("nasdaq100", "ls", "master"): 1.43,
    ("nasdaq100", "ls", "stockmixer"): 1.09, ("nasdaq100", "ls", "dystage"): 1.25,
    ("nasdaq100", "ls", "invarrl"): 1.19,
    ("biotech", "ls", "factorvae"): 1.61, ("biotech", "ls", "master"): 1.50,
    ("biotech", "ls", "stockmixer"): 0.78, ("biotech", "ls", "dystage"): 1.24,
    ("biotech", "ls", "invarrl"): 1.54,
}

RANKERS = {"factorvae", "master", "stockmixer", "dystage"}


def universe_of(path: str) -> str:
    if "nasdaq100" in path:
        return "nasdaq100"
    if "biotech_nbi" in path:
        return "biotech"
    return "sp500"


def annual_return(fe: float, n: int) -> float:
    return fe ** (TRADING_DAYS / n) - 1.0 if n > 0 and fe > 0 else float("nan")


def cell_metrics(sr: float, fe: float, n: int) -> Tuple[float, float, float]:
    return sr, fe - 1.0, annual_return(fe, n)


def _walk_json(obj) -> List[Tuple[str, float, float, int]]:
    """Return (key, sharpe, final_equity, n_steps) for every method dict."""
    out = []
    if isinstance(obj, dict):
        if "sharpe_annualised" in obj and "final_equity" in obj:
            out.append(("", float(obj["sharpe_annualised"]),
                        float(obj["final_equity"]),
                        int(obj.get("n_steps", obj.get("n_test_days", 0)))))
        for k, v in obj.items():
            for (sub, s, fe, n) in _walk_json(v):
                out.append((f"{k}.{sub}".rstrip("."), s, fe, n))
    return out


def json_groups() -> Dict[Tuple[str, str], List[Tuple[float, float, int]]]:
    """Map (source_dir, method_subkey) -> list of per-cell (sr, fe, n)."""
    groups: Dict[Tuple[str, str], List[Tuple[float, float, int]]] = {}
    patterns = [
        "invar_rl/results/baselines_long_only/*/fold*_seed*.json",
        "invar_rl/results/native_ranker_baselines/*/fold*_seed*.json",
        "invar_rl/results/finrl_faithful/*/fold*_seed*.json",
        "invar_rl/results/stockformer_faithful/*/fold*_seed*.json",
        "invar_rl/results/whole_stack_rl*/**/fold*_seed*.json",
        "outputs/*/baselines/*/fold*_seed*.json",
    ]
    for pat in patterns:
        for f in glob.glob(pat, recursive=True):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            sub = d.get("methods", d.get("perf", d))
            for (key, s, fe, n) in _walk_json({"_": sub}):
                gk = (os.path.dirname(f), key)
                groups.setdefault(gk, []).append((s, fe, n))
    return groups


def parquet_groups() -> Dict[Tuple[str, str], List[Tuple[float, float, int]]]:
    import pandas as pd
    groups: Dict[Tuple[str, str], List[Tuple[float, float, int]]] = {}
    for f in glob.glob("outputs/*/**/fold*_seed*.parquet", recursive=True):
        if "layer3" not in f:
            continue
        try:
            df = pd.read_parquet(f, columns=["strategy_return"])
        except Exception:
            continue
        r = df["strategy_return"].to_numpy()
        if r.size < 2:
            continue
        sd = r.std(ddof=1)
        sr = float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-12 else 0.0
        fe = float(np.prod(1.0 + r))
        groups.setdefault((os.path.dirname(f), "tape"), []).append(
            (sr, fe, int(r.size)))
    return groups


def best_match(groups, universe: str, protocol: str, method: str,
               target_sr: float):
    """Pick the candidate group whose per-cell-mean SR matches target_sr."""
    best = None
    for (src, key), cells in groups.items():
        if universe_of(src) != universe:
            continue
        path = f"{src}/{key}".lower()
        # method filter
        if method in RANKERS:
            if method not in path:
                continue
        elif method == "invarrl":
            if "layer3" not in path:
                continue
        else:  # finrl / stockformer
            if method not in path:
                continue
        # protocol filter
        is_lo = any(t in path for t in ("long_only", "/lo/", "lo_native",
                                        "_lo.", "native"))
        is_ls = any(t in path for t in ("long_short", "/ls/", "topk_ls",
                                        "wrapper", "sac_ls"))
        if protocol == "lo" and is_ls and not is_lo:
            continue
        if protocol == "ls" and is_lo and not is_ls:
            continue
        if len(cells) < 20:
            continue
        srs = np.array([c[0] for c in cells])
        sr_mean = float(np.nanmean(srs))
        diff = abs(sr_mean - target_sr)
        cand = (diff, sr_mean, cells, src, key)
        if best is None or diff < best[0]:
            best = cand
    return best


def main() -> None:
    groups = {**json_groups(), **parquet_groups()}
    print(f"scanned {len(groups)} candidate source groups\n")
    header = f"{'universe':10} {'prot':4} {'method':11} {'SRtab':>6} {'SRcalc':>7} {'PR':>7} {'AR':>7}  match  source"
    print(header)
    print("-" * len(header))
    results = {}
    for (u, p, m), sr_t in sorted(TARGETS.items()):
        b = best_match(groups, u, p, m, sr_t)
        if b is None:
            print(f"{u:10} {p:4} {m:11} {sr_t:6.2f} {'--':>7} {'--':>7} {'--':>7}  MISS   (no candidate)")
            continue
        diff, sr_c, cells, src, key = b
        prs = [c[1] - 1.0 for c in cells]
        ars = [annual_return(c[1], c[2]) for c in cells]
        pr, ar = float(np.nanmean(prs)), float(np.nanmean(ars))
        ok = "OK" if diff <= TOL else "FAIL"
        results[(u, p, m)] = (sr_c, pr, ar, ok, f"{src}::{key}")
        short = src.replace("invar_rl/results/", "").replace("outputs/", "")
        print(f"{u:10} {p:4} {m:11} {sr_t:6.2f} {sr_c:7.3f} {pr:7.3f} {ar:7.3f}  {ok:5}  {short}::{key}")
    n_ok = sum(1 for v in results.values() if v[3] == "OK")
    print(f"\nverified {n_ok}/{len(TARGETS)} cells")


if __name__ == "__main__":
    main()
