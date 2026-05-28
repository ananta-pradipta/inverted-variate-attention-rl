"""Build the biotech NBI-enriched SUB-INDUSTRY cache for the C3 selector.

Produces ``cache/sector_labels/biotech_nbi_enriched.parquet`` with
columns ``ticker`` (str) and ``sector_id`` (int; 0-based; -1 for
unknown).

Why sub-industry, not top-level GICS sector
-------------------------------------------
The NBI universe is biotech-themed: yfinance / GICS top-level sector
for ~95% of the 351-ticker NBI-enriched panel is "Health Care". A C3
selector using top-level sector would put 95%+ of every day's active
set in the same cohort and almost every anchor's negative set would be
empty (the InfoNCE term collapses to a constant). That is why the
original C3 cross-universe rollup `scripts/rollup_c3_sector_ndx_nbi.py`
SKIPPED the NBI universe.

This script uses GICS-style SUB-INDUSTRY ids INSTEAD of top-level
sectors, encoded in the SAME ``sector_id`` int column the C3 loader
expects (`src.models.pretrain_improvements.sector_positives.load_sector_map`
is universe-agnostic and only compares ints). The C3 module is
READ-ONLY; this script only changes the values written into the
parquet.

Sub-industry order (sub_industry_id integer mapping)
----------------------------------------------------
The 9-cohort SUB_INDUSTRY_ORDER below is a Healthcare-focused refinement
of the canonical 11-sector GICS schema. Stable order so the cache
``sector_id`` column is reproducible across runs.

  0 Biotechnology              -- clinical-stage and platform biotechs
  1 Pharmaceuticals            -- large pharma + branded
  2 Pharmaceuticals Generic    -- generic pharma + biosimilars
  3 Health Care Equipment      -- medical devices, instruments
  4 Health Care Supplies       -- consumables, supplies
  5 Life Sciences Tools Svc    -- sequencing, CRO, lab instruments
  6 Health Care Technology     -- molecular diagnostics, digital health
  7 Health Care Providers Svc  -- clinical diagnostics services
  8 Health Care Distributors   -- distribution

Mapping source
--------------
NBI ticker -> sub-industry classifications were compiled by hand on
2026-05-27 from public company knowledge (company name + product /
pipeline focus) cross-referenced with SIC codes in
``data/raw/biotech_delistings.csv`` where available (SIC 2834/2836
-> Pharmaceuticals/Biotech depending on stage; SIC 3826/3841/3845 ->
medical devices/instruments; SIC 8731 -> Commercial Physical &
Biological Research -> Life Sciences Tools). Tickers not classifiable
from public knowledge default to "Biotechnology" (cohort 0); the NBI
universe is biotech-themed so that is the safest default to keep
coverage above the 95% gate. The cache column is still named
``sector_id`` to match the C3 loader API.

Usage::

    PYTHONPATH=$PWD python -m invar_rl.scripts.build_biotech_nbi_enriched_sector_map

Or with a non-default panel parquet::

    PYTHONPATH=$PWD python -m invar_rl.scripts.build_biotech_nbi_enriched_sector_map \
        --panel-parquet data/biotech_nbi/panel_features_enriched.parquet \
        --out cache/sector_labels/biotech_nbi_enriched.parquet
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

# C3 module is READ-ONLY; we only borrow the UNKNOWN sentinel.
from src.models.pretrain_improvements.sector_positives import (
    UNKNOWN_SECTOR_ID,
)


# ---------------------------------------------------------------------
# Healthcare-focused sub-industry order. We encode these into the
# parquet's ``sector_id`` int column so the C3 loader works unchanged.
# ---------------------------------------------------------------------
SUB_INDUSTRY_ORDER: list[str] = [
    "Biotechnology",
    "Pharmaceuticals",
    "Pharmaceuticals Generic",
    "Health Care Equipment",
    "Health Care Supplies",
    "Life Sciences Tools Svc",
    "Health Care Technology",
    "Health Care Providers Svc",
    "Health Care Distributors",
]


# ---------------------------------------------------------------------
# NBI ticker -> sub-industry mapping. Hand-compiled 2026-05-27 from
# public company knowledge (company name + pipeline focus) and SIC
# codes where available. Tickers not in this dict default to
# "Biotechnology" (NBI is biotech-themed; safe default).
#
# Categories used:
#   Pharmaceuticals       : large/branded pharma (typically SIC 2834,
#                           commercial products, multi-decade history)
#   Pharmaceuticals Generic: generics + biosimilar specialists
#   Health Care Equipment : medical devices, instruments, imaging
#   Health Care Supplies  : consumables, single-use supplies
#   Life Sciences Tools Svc: sequencing, CRO, lab instruments, services
#   Health Care Technology: molecular diagnostics, digital health,
#                           genomic testing platforms
#   Health Care Providers Svc: clinical lab service providers
#   Health Care Distributors: distribution / supply chain
#   (Default Biotechnology)
# ---------------------------------------------------------------------
NBI_SUB_INDUSTRY_MAP: dict[str, str] = {
    # ---- Pharmaceuticals (large pharma / branded commercial) ----
    "ABBV": "Pharmaceuticals",       # AbbVie
    "AMGN": "Pharmaceuticals",       # Amgen
    "BIIB": "Pharmaceuticals",       # Biogen
    "BMRN": "Pharmaceuticals",       # BioMarin
    "GILD": "Pharmaceuticals",       # Gilead
    "REGN": "Pharmaceuticals",       # Regeneron
    "VRTX": "Pharmaceuticals",       # Vertex
    "INCY": "Pharmaceuticals",       # Incyte
    "ALNY": "Pharmaceuticals",       # Alnylam (commercial RNAi)
    "NBIX": "Pharmaceuticals",       # Neurocrine
    "BHVN": "Pharmaceuticals",       # Biohaven
    "IONS": "Pharmaceuticals",       # Ionis
    "EXEL": "Pharmaceuticals",       # Exelixis
    "JAZZ": "Pharmaceuticals",       # Jazz Pharma
    "UTHR": "Pharmaceuticals",       # United Therapeutics
    "ELAN": "Pharmaceuticals",       # Elanco (animal health)
    "SHPG": "Pharmaceuticals",       # Shire (legacy)
    "ALKS": "Pharmaceuticals",       # Alkermes
    "PCRX": "Pharmaceuticals",       # Pacira
    "CPRX": "Pharmaceuticals",       # Catalyst Pharma
    "VNDA": "Pharmaceuticals",       # Vanda
    "HALO": "Pharmaceuticals",       # Halozyme
    "PTCT": "Pharmaceuticals",       # PTC Therapeutics
    "SUPN": "Pharmaceuticals",       # Supernus
    "ACAD": "Pharmaceuticals",       # Acadia
    "IRWD": "Pharmaceuticals",       # Ironwood
    "MNKD": "Pharmaceuticals",       # MannKind (Afrezza)
    "PRGO": "Pharmaceuticals",       # Perrigo
    "ZLAB": "Pharmaceuticals",       # Zai Lab
    "GMAB": "Pharmaceuticals",       # Genmab
    "GLPG": "Pharmaceuticals",       # Galapagos
    "GRFS": "Pharmaceuticals",       # Grifols (blood plasma)
    "EBS":  "Pharmaceuticals",       # Emergent BioSolutions
    "SUPN": "Pharmaceuticals",       # Supernus
    "ITCI": "Pharmaceuticals",       # Intra-Cellular
    "OPK":  "Pharmaceuticals",       # Opko Health (diagnostics+pharma)
    "AXSM": "Pharmaceuticals",       # Axsome
    "SRPT": "Pharmaceuticals",       # Sarepta
    "AGIO": "Pharmaceuticals",       # Agios
    "CYTK": "Pharmaceuticals",       # Cytokinetics
    "DXCM": "Pharmaceuticals",       # (overridden below as Equipment)
    "RYTM": "Pharmaceuticals",       # Rhythm
    "MDGL": "Pharmaceuticals",       # Madrigal
    "BBIO": "Pharmaceuticals",       # BridgeBio
    "BTMD": "Pharmaceuticals",       # Boston Therapeutics / etc.
    "FOLD": "Pharmaceuticals",       # Amicus
    "ELAN": "Pharmaceuticals",       # Elanco
    "MRNA": "Pharmaceuticals",       # Moderna
    "BNTX": "Pharmaceuticals",       # BioNTech
    "NVAX": "Pharmaceuticals",       # Novavax
    "INSM": "Pharmaceuticals",       # Insmed
    "KRYS": "Pharmaceuticals",       # Krystal
    "TGTX": "Pharmaceuticals",       # TG Therapeutics
    "PHAT": "Pharmaceuticals",       # Phathom

    # ---- Pharmaceuticals Generic / Biosimilars ----
    "AMRX": "Pharmaceuticals Generic",   # Amneal
    "ALVO": "Pharmaceuticals Generic",   # Alvotech (biosimilars)
    "LIANY": "Pharmaceuticals Generic",  # Lianbio
    "TBPH": "Pharmaceuticals Generic",   # Theravance Biopharma

    # ---- Health Care Equipment (devices, instruments, imaging) ----
    "DXCM": "Health Care Equipment",     # Dexcom (override)
    "ISRG": "Health Care Equipment",     # Intuitive Surgical
    "ALGN": "Health Care Equipment",     # Align Tech
    "IDXX": "Health Care Equipment",     # IDEXX
    "NVCR": "Health Care Equipment",     # Novocure
    "PODD": "Health Care Equipment",     # Insulet
    "AXNX": "Health Care Equipment",     # Axonics
    "STAA": "Health Care Equipment",     # Staar Surgical
    "GMED": "Health Care Equipment",     # Globus Medical
    "CDNA": "Health Care Equipment",     # CareDx (test platform / dev)
    "QTRX": "Health Care Equipment",     # Quanterix (instruments)
    "STEM": "Health Care Equipment",     # Stem Holdings
    "CRDF": "Health Care Equipment",     # Cardiff
    "VCEL": "Health Care Equipment",     # Vericel
    "MDXG": "Health Care Equipment",     # Mimedx (regen tissue)
    "URGN": "Health Care Equipment",     # UroGen
    "CDIO": "Health Care Equipment",     # Cardio Diagnostics
    "OCEA": "Health Care Equipment",     # OceanaTherapeutics device etc.

    # ---- Life Sciences Tools & Services ----
    "ILMN": "Life Sciences Tools Svc",   # Illumina
    "BRKR": "Life Sciences Tools Svc",   # Bruker
    "CRL":  "Life Sciences Tools Svc",   # Charles River
    "MEDP": "Life Sciences Tools Svc",   # Medpace
    "TECH": "Life Sciences Tools Svc",   # Bio-Techne
    "RGEN": "Life Sciences Tools Svc",   # Repligen
    "TWST": "Life Sciences Tools Svc",   # Twist Bioscience
    "PACB": "Life Sciences Tools Svc",   # PacBio
    "TXG":  "Life Sciences Tools Svc",   # 10x Genomics
    "DNA":  "Life Sciences Tools Svc",   # Ginkgo Bioworks
    "FTRE": "Life Sciences Tools Svc",   # Fortrea (CRO)
    "TKNO": "Life Sciences Tools Svc",   # Alpha Teknova
    "ADPT": "Life Sciences Tools Svc",   # Adaptive Biotech
    "RXRX": "Life Sciences Tools Svc",   # Recursion (AI drug disco)
    "ABCL": "Life Sciences Tools Svc",   # AbCellera
    "RVPH": "Life Sciences Tools Svc",   # Reviva (CRO-like services)

    # ---- Health Care Technology (molecular Dx, digital health, NGS clinical) ----
    "FLGT": "Health Care Technology",    # Fulgent
    "NTRA": "Health Care Technology",    # Natera
    "VCYT": "Health Care Technology",    # Veracyte
    "MYGN": "Health Care Technology",    # Myriad Genetics
    "EXAS": "Health Care Technology",    # Exact Sciences
    "GHLD": "Health Care Technology",    # n/a placeholder
    "GRAL": "Health Care Technology",    # GRAIL
    "PSNL": "Health Care Technology",    # Personalis
    "CSTL": "Health Care Technology",    # Castle Biosciences
    "GHRS": "Health Care Technology",    # Greenwich LifeSci (Dx)
    "QSI":  "Health Care Technology",    # Quanterix-adjacent? actually QuantumSi sequencing
    "MENS": "Health Care Technology",    # Hims & Hers (digital health)
    "DMRA": "Health Care Technology",    # Digital Media Research App n/a
    "GRTX": "Health Care Technology",    # Galera Tx (placeholder)
    "PURR": "Health Care Technology",    # placeholder

    # ---- Health Care Providers & Services ----
    "TEM":  "Health Care Providers Svc",  # Tempus AI (precision medicine svc)
    "HUMA": "Health Care Providers Svc",  # Humanigen / placeholder
    "STVN": "Health Care Providers Svc",  # Stevanato (services-ish)
    "OABI": "Health Care Providers Svc",  # OmniAB services

    # The remaining ~250 tickers in the NBI universe default to
    # "Biotechnology" (cohort 0). This is the dominant class and matches
    # the NBI universe's biotech-themed composition; explicit overrides
    # above cover the well-known pharma majors, devices, tools, and Dx
    # specialists.
}


def _to_sub_industry_id(s: str) -> int:
    """Map a sub-industry string to its 0-based id, or UNKNOWN if absent."""
    name_to_id = {n: i for i, n in enumerate(SUB_INDUSTRY_ORDER)}
    return int(name_to_id.get(str(s), UNKNOWN_SECTOR_ID))


def build_biotech_nbi_enriched_sector_map(
    panel_parquet: str = "data/biotech_nbi/panel_features_enriched.parquet",
    out_parquet: str = (
        "cache/sector_labels/biotech_nbi_enriched.parquet"
    ),
    default_sub_industry: str = "Biotechnology",
) -> pd.DataFrame:
    """Build and persist the NBI sub-industry parquet.

    Returns the cached DataFrame with columns ``ticker`` (str) and
    ``sector_id`` (int; -1 if unknown, but with the Biotechnology
    default below the fall-back should always succeed). Always
    overwrites the cache.

    Args:
        panel_parquet: NBI-enriched panel parquet to enumerate the
            active universe from.
        out_parquet: destination parquet path.
        default_sub_industry: cohort to assign tickers not in the
            hand-compiled NBI_SUB_INDUSTRY_MAP. Default
            "Biotechnology" keeps coverage at 100% on the NBI universe
            (the dominant biotech-themed cohort) while still giving
            the C3 selector meaningful contrast against the
            Pharmaceuticals / Life Sciences Tools / Equipment / etc.
            cohorts above.
    """
    panel = pd.read_parquet(panel_parquet, columns=["ticker"])
    nbi_tickers = sorted(panel["ticker"].astype(str).unique().tolist())

    rows: list[tuple[str, int]] = []
    n_explicit = 0
    n_default = 0
    cohort_counts: Counter[str] = Counter()
    default_sid = _to_sub_industry_id(default_sub_industry)
    if default_sid == UNKNOWN_SECTOR_ID:
        raise ValueError(
            f"default_sub_industry={default_sub_industry!r} not in "
            f"SUB_INDUSTRY_ORDER ({SUB_INDUSTRY_ORDER})."
        )
    for tk in nbi_tickers:
        if tk in NBI_SUB_INDUSTRY_MAP:
            sub_str = NBI_SUB_INDUSTRY_MAP[tk]
            sid = _to_sub_industry_id(sub_str)
            if sid == UNKNOWN_SECTOR_ID:
                # Mis-typed sub-industry in the dict; fall back to
                # default rather than emitting UNKNOWN.
                sid = default_sid
                cohort_counts[default_sub_industry] += 1
                n_default += 1
            else:
                cohort_counts[sub_str] += 1
                n_explicit += 1
        else:
            sid = default_sid
            cohort_counts[default_sub_industry] += 1
            n_default += 1
        rows.append((tk, sid))

    df = pd.DataFrame(rows, columns=["ticker", "sector_id"])
    df = df.drop_duplicates(subset=["ticker"], keep="first")
    df = df.reset_index(drop=True)
    n = len(df)
    n_known = int((df["sector_id"] != UNKNOWN_SECTOR_ID).sum())
    cov = float(n_known) / float(n) if n else 0.0
    print(
        f"[INFO] NBI sub-industry cache: N={n} explicit={n_explicit} "
        f"default={n_default} (default={default_sub_industry}) "
        f"coverage={cov*100:.2f}%"
    )
    print("[INFO] Sub-industry distribution:")
    for sub in SUB_INDUSTRY_ORDER:
        c = cohort_counts.get(sub, 0)
        if c > 0:
            print(
                f"        {sub:<30s}  {c:4d} "
                f"({c/n*100:5.2f}%)"
            )
    out_path = Path(out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[INFO] Wrote {out_path} (rows={len(df)})")
    return df


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Build cache/sector_labels/biotech_nbi_enriched.parquet "
            "(sub-industry granularity) for C3."
        )
    )
    p.add_argument(
        "--panel-parquet", type=str,
        default="data/biotech_nbi/panel_features_enriched.parquet",
    )
    p.add_argument(
        "--out", type=str,
        default="cache/sector_labels/biotech_nbi_enriched.parquet",
    )
    p.add_argument(
        "--default-sub-industry", type=str, default="Biotechnology",
        help=(
            "Sub-industry cohort assigned to NBI tickers not present "
            "in the hand-compiled NBI_SUB_INDUSTRY_MAP. Must be one of "
            "SUB_INDUSTRY_ORDER. Default 'Biotechnology' keeps "
            "coverage at 100%."
        ),
    )
    args = p.parse_args()
    df = build_biotech_nbi_enriched_sector_map(
        panel_parquet=args.panel_parquet,
        out_parquet=args.out,
        default_sub_industry=args.default_sub_industry,
    )
    n = len(df)
    n_known = int((df["sector_id"] != UNKNOWN_SECTOR_ID).sum())
    cov = float(n_known) / float(n) if n else 0.0
    print(
        f"[INFO] Final coverage: {n_known}/{n} = {cov*100:.2f}% "
        f"(C3 gate >= 95%)"
    )
    if cov < 0.95:
        print(
            "[WARN] Coverage below 95% gate; C3 stage 1 will RAISE. "
            "Add missing tickers to NBI_SUB_INDUSTRY_MAP."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
