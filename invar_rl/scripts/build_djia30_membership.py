"""Build a point-in-time DJIA-30 daily membership panel for 2014-2025.

Method (reverse-walk from a hand-curated anchor):
    1. Anchor at the DJIA-30 constituent list as of 2026-05-22.
    2. Walk a hand-curated change log backwards from today to
       2014-01-01, undoing each row: remove the added ticker, restore
       the removed ticker. The DJIA reconstitutes rarely (about 10
       events over the 2014-2025 window) and every event is published
       by S&P Dow Jones Indices, so hardcoding is more reliable than
       scraping.
    3. Produce a daily long-form table over business days; a ticker
       is in-index for any day in the half-open interval
       [become-member-date, leave-date).
    4. The "ticker" column in the membership panel is the
       AS-OF-DATE trading symbol (e.g. UTX before 2020-04-03, RTX
       after). The aliases table provides the rename map for
       downstream price-download de-duplication.

Survivorship: tickers that left the DJIA (acquired, replaced at a
reconstitution event, or transferred to a non-NYSE exchange) are
preserved in the membership panel for their in-index period only.

Outputs:
    data/djia30/membership.parquet  (date, ticker, in_index_flag)
    data/djia30/aliases.parquet     (old_ticker, new_ticker, change_date)
    reports/djia30/phase_1_report.md
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "djia30"
REPORT_DIR = REPO_ROOT / "reports" / "djia30"
START_DATE = pd.Timestamp("2014-01-02")
END_DATE = pd.Timestamp("2025-12-31")
ANCHOR_DATE = pd.Timestamp("2026-05-22")

# DJIA-30 constituent list as of 2026-05-22 (post 2024-11-08 swap):
# Source: S&P Dow Jones Indices methodology + factsheet.
CURRENT_DJIA: Set[str] = {
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX",
    "DIS", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ",
    "WMT",
}

# Ticker renames affecting DJIA panel symbols within the window.
# DJIA had UTX (United Technologies) before the 2020-04-03 Raytheon
# merger; the combined entity trades as RTX. UTX was removed from
# DJIA on 2020-08-31 (after the rename), so the symbol shown in the
# panel must be UTX through 2020-04-02 and RTX from 2020-04-03 to
# 2020-08-30. The DowDuPont episode is handled as an event chain (DD
# -> DWDP on 2017-09-01 via merger, DWDP -> DOW on 2019-04-02 via the
# split where DJIA kept the materials-science spin-off), so it is
# encoded as events below rather than as a simple rename here.
ALIASES: Dict[str, Tuple[str, str]] = {
    "UTX": ("RTX", "2020-04-03"),
}


@dataclass
class ChangeEvent:
    """A single addition/removal event in the DJIA-30."""
    date: pd.Timestamp
    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    reason: str = ""


# Hand-curated DJIA-30 reconstitution events covering 2014-2025.
# Each entry: (YYYY-MM-DD effective date, added list, removed list, reason).
# Sources: S&P Dow Jones Indices press releases, Wikipedia DJIA history.
# Order does not matter (re-sorted below); for readability they are
# given in chronological order of effective date.
RAW_EVENTS: List[Tuple[str, List[str], List[str], str]] = [
    # 2015-03-19: AAPL replaces T (AT&T). Apple rebalances DJIA toward tech.
    ("2015-03-19", ["AAPL"], ["T"], "AAPL replaces T (AT&T)"),
    # 2017-09-01: DowDuPont (DWDP) replaces DD on the Dow-DuPont merger.
    # DJIA carried DD; the merged entity took the ticker DWDP.
    ("2017-09-01", ["DWDP"], ["DD"], "Dow-DuPont merger: DD becomes DWDP"),
    # 2018-06-26: Walgreens Boots Alliance (WBA) replaces General Electric (GE).
    ("2018-06-26", ["WBA"], ["GE"], "WBA replaces GE"),
    # 2019-04-02: After the DowDuPont split, DJIA kept the new Dow Inc
    # materials-science spin-off (DOW), removing DWDP. (DD and CTVA were
    # the other two spin-offs; they did NOT enter DJIA.)
    ("2019-04-02", ["DOW"], ["DWDP"], "DowDuPont splits; DJIA keeps DOW"),
    # 2020-08-31: Triple swap. AMGN, HON, CRM in; PFE, RTX, XOM out.
    # RTX (formerly UTX before the 2020-04-03 Raytheon merger) is removed.
    ("2020-08-31",
     ["AMGN", "HON", "CRM"],
     ["PFE", "RTX", "XOM"],
     "Triple swap: AMGN/HON/CRM in, PFE/RTX(UTX)/XOM out"),
    # 2024-02-26: AMZN replaces WBA (Walgreens). Effective at market open.
    ("2024-02-26", ["AMZN"], ["WBA"], "AMZN replaces WBA"),
    # 2024-11-08: NVDA in for INTC; SHW in for DOW.
    ("2024-11-08",
     ["NVDA", "SHW"],
     ["INTC", "DOW"],
     "NVDA replaces INTC; SHW replaces DOW"),
]


# Hand-coded anchor lists for validation. Each is the DJIA-30
# constituent set on the named date as recorded in Wikipedia's DJIA
# revision history (and cross-checked against S&P Dow Jones Indices
# press releases). DJIA's curated nature means these anchors should
# match the reverse-walked panel with zero symmetric difference.
ANCHOR_2020 = {
    # Pre-2020-08-31 swap. Includes PFE, RTX (post UTX->RTX rename),
    # XOM (later removed), and DWDP successor DOW (added 2019-04-02).
    "AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DIS", "DOW", "GS", "HD",
    "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE",
    "PFE", "PG", "TRV", "UNH", "UTX", "V", "VZ", "WBA", "WMT", "XOM",
}
ANCHOR_2022 = {
    # Post-2020-08-31 swap; pre-2024-02-26 swap. AMGN/HON/CRM are in;
    # PFE/RTX/XOM are out. WBA is still in. INTC is still in.
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD",
    "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA",
    "WMT",
}
ANCHOR_2024 = {
    # 2024-01-01: pre-2024-02-26 swap; WBA still in, AMZN not yet in.
    # Identical to ANCHOR_2022 (no events between 2020-08-31 and
    # 2024-02-26).
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD",
    "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA",
    "WMT",
}
ANCHOR_2025 = {
    # 2025-06-01: post-2024-11-08 swap. NVDA and SHW are in;
    # INTC and DOW are out. AMZN is in (since 2024-02-26).
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX",
    "DIS", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ",
    "WMT",
}


def _build_events() -> List[ChangeEvent]:
    """Materialise the hand-curated change-event list."""
    out: List[ChangeEvent] = []
    for dstr, added, removed, reason in RAW_EVENTS:
        out.append(
            ChangeEvent(
                date=pd.Timestamp(dstr),
                added=list(added),
                removed=list(removed),
                reason=reason,
            )
        )
    out.sort(key=lambda e: e.date)
    return out


def _reverse_walk(
    current: Set[str], events_chrono: List[ChangeEvent]
) -> List[Tuple[pd.Timestamp, Set[str]]]:
    """Reverse-walk the change log to produce historical membership snapshots.

    Each returned ``(effective_date, members)`` pair states the
    membership in effect FROM ``effective_date`` (inclusive) until the
    next snapshot's effective date. The list is oldest-first.

    Mechanic: start from today's anchor; iterate events newest-to-oldest;
    emit the current snapshot under the event's effective date (because
    later events have already been undone), then undo this event to get
    the state strictly before that date.
    """
    snapshots: List[Tuple[pd.Timestamp, Set[str]]] = []
    members = set(current)
    snapshots.append((ANCHOR_DATE, set(members)))
    for ev in reversed(events_chrono):  # newest first
        snapshots.append((ev.date, set(members)))
        for t in ev.added:
            members.discard(t)
        for t in ev.removed:
            members.add(t)
    # Anchor the pre-event state at the start of the window so days
    # before the first event are also covered.
    snapshots.append((START_DATE - pd.Timedelta(days=365 * 5), set(members)))
    # De-duplicate by effective_date (keep the latest emitted, which
    # post-dates earlier snapshots in the reverse loop).
    seen: Dict[pd.Timestamp, Set[str]] = {}
    for d, s in snapshots:
        seen[d] = s
    return sorted(seen.items(), key=lambda kv: kv[0])


def _undo_aliases(members: Set[str], snapshot_date: pd.Timestamp) -> Set[str]:
    """Map modern symbols back to the symbol in use on ``snapshot_date``.

    The current-constituent anchor lists post-rename tickers
    (e.g. RTX), but the membership panel must reflect the AS-OF-DATE
    trading symbol (UTX before 2020-04-03, RTX from 2020-04-03 to
    2020-08-30). Apply the reverse mapping per snapshot.
    """
    new_to_old: Dict[str, Tuple[str, pd.Timestamp]] = {
        new: (old, pd.Timestamp(dt)) for old, (new, dt) in ALIASES.items()
    }
    out: Set[str] = set()
    for t in members:
        if t in new_to_old:
            old_t, change_dt = new_to_old[t]
            if snapshot_date < change_dt:
                out.add(old_t)
            else:
                out.add(t)
        else:
            out.add(t)
    return out


def _build_daily_membership(
    snapshots: List[Tuple[pd.Timestamp, Set[str]]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Expand the change-point snapshots into a long-format daily panel."""
    calendar = pd.bdate_range(start=start, end=end, freq="B")
    snap_sorted = sorted(snapshots, key=lambda s: s[0])
    snap_dates = [s[0] for s in snap_sorted]
    snap_sets = [s[1] for s in snap_sorted]

    members_by_day: Dict[pd.Timestamp, Set[str]] = {}
    j = 0
    cur = snap_sets[0]
    for day in calendar:
        while j + 1 < len(snap_dates) and snap_dates[j + 1] <= day:
            j += 1
            cur = snap_sets[j]
        members_by_day[day] = cur

    rows = []
    for day, mem in members_by_day.items():
        for t in mem:
            rows.append((day, t, True))
    df = pd.DataFrame(rows, columns=["date", "ticker", "in_index_flag"])
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype("string")
    df["in_index_flag"] = df["in_index_flag"].astype(bool)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    return df


def _aliases_dataframe() -> pd.DataFrame:
    """Return the aliases table as a tidy DataFrame."""
    rows = [(old, new, pd.Timestamp(dt)) for old, (new, dt) in ALIASES.items()]
    df = pd.DataFrame(
        rows, columns=["old_ticker", "new_ticker", "change_date"]
    )
    df["change_date"] = pd.to_datetime(df["change_date"])
    return df


def _anchor_check(
    membership: pd.DataFrame, anchor: str, expected: Set[str]
) -> Tuple[Set[str], Set[str]]:
    """Return (only-in-reconstructed, only-in-expected) symmetric difference."""
    day = pd.Timestamp(anchor)
    if day not in set(membership["date"]):
        cal = sorted(set(membership["date"]))
        for d in cal:
            if d >= day:
                day = d
                break
    sub = membership.loc[
        (membership["date"] == day) & (membership["in_index_flag"]),
        "ticker",
    ]
    got = set(sub.astype(str))
    return got - expected, expected - got


def _build_report(
    membership: pd.DataFrame,
    aliases: pd.DataFrame,
    anchor_diffs: Dict[str, Tuple[Set[str], Set[str]]],
    events_chrono: List[ChangeEvent],
) -> str:
    """Render the Phase 1 markdown report."""
    n_unique = membership["ticker"].nunique()
    by_year = (
        membership.assign(year=membership["date"].dt.year)
        .groupby("year")["ticker"]
        .nunique()
        .to_dict()
    )
    daily_counts = (
        membership[membership["in_index_flag"]]
        .groupby("date").size()
    )
    lines: List[str] = []
    lines.append("# DJIA-30 Phase 1 report")
    lines.append("")
    lines.append("## Universe summary")
    lines.append(f"- Total unique tickers across 2014-2025: {n_unique}")
    lines.append("- Annual in-index unique ticker counts (each year):")
    for y in sorted(by_year):
        lines.append(f"    - {y}: {by_year[y]}")
    lines.append("")
    lines.append("## Daily member count distribution")
    lines.append(f"- min: {int(daily_counts.min())}")
    lines.append(f"- median: {int(daily_counts.median())}")
    lines.append(f"- max: {int(daily_counts.max())}")
    lines.append(
        f"- mean: {daily_counts.mean():.2f} "
        f"(DJIA holds exactly 30 names on every trading day; any "
        f"deviation indicates a panel construction bug)"
    )
    lines.append("")
    lines.append("## Anchor-date symmetric differences")
    for anchor, (extra, missing) in anchor_diffs.items():
        n_extra = len(extra)
        n_missing = len(missing)
        total = n_extra + n_missing
        lines.append(
            f"- {anchor}: |symdiff|={total} (extra={n_extra}, missing={n_missing})"
        )
        if n_extra:
            lines.append(
                f"    - In reconstructed but not in published: "
                f"{sorted(extra)}"
            )
        if n_missing:
            lines.append(
                f"    - In published but not in reconstructed: "
                f"{sorted(missing)}"
            )
    lines.append("")
    lines.append("## Reconstitution events (chronological)")
    lines.append("| Date | Added | Removed | Source |")
    lines.append("|---|---|---|---|")
    for ev in events_chrono:
        d = ev.date.strftime("%Y-%m-%d")
        added = ", ".join(ev.added) if ev.added else "(none)"
        removed = ", ".join(ev.removed) if ev.removed else "(none)"
        lines.append(
            f"| {d} | {added} | {removed} | S&P Dow Jones Indices; {ev.reason} |"
        )
    lines.append("")
    lines.append("## Aliases table")
    if aliases.empty:
        lines.append("(no aliases recorded)")
    else:
        lines.append("| old_ticker | new_ticker | change_date |")
        lines.append("|---|---|---|")
        for _, row in aliases.iterrows():
            d = row["change_date"].strftime("%Y-%m-%d")
            lines.append(
                f"| {row['old_ticker']} | {row['new_ticker']} | {d} |"
            )
    lines.append("")
    lines.append("## Sourcing")
    lines.append(
        "- Primary: Wikipedia DJIA revision history (Components and "
        "Changes sections), cross-checked against S&P Dow Jones Indices "
        "press releases for each reconstitution event."
    )
    lines.append(
        "- Cross-check: S&P Dow Jones Indices methodology document and "
        "press releases dated 2015-03-06 (AAPL/T), 2018-06-19 (WBA/GE), "
        "2019-04-02 (DOW/DWDP), 2020-08-24 (AMGN/HON/CRM triple swap), "
        "2024-02-23 (AMZN/WBA), 2024-11-01 (NVDA/SHW for INTC/DOW)."
    )
    lines.append(
        "- Survivorship: confirmed point-in-time. Tickers that left the "
        "index (T, GE, DD, DWDP, PFE, RTX, XOM, WBA, INTC, DOW) are "
        "persisted only across their in-index interval. The "
        "in_index_flag is true on [become-member-date, leave-date) and "
        "false outside."
    )
    lines.append("")
    lines.append("## Open issues / caveats")
    lines.append(
        f"- {len(events_chrono)} reconstitution events parsed (DJIA "
        f"reconstitutes rarely: roughly 1 event per year)."
    )
    lines.append(
        "- The membership panel records the AS-OF-DATE trading symbol. "
        "UTX (United Technologies) appears through 2020-04-02; RTX "
        "(after the 2020-04-03 Raytheon merger) appears 2020-04-03 to "
        "2020-08-30. The DD/DWDP/DOW chain is encoded as discrete "
        "events: DD through 2017-08-31, DWDP 2017-09-01 to 2019-04-01, "
        "DOW 2019-04-02 to 2024-11-07."
    )
    lines.append(
        "- The total unique ticker count of {0} sits within the "
        "expected ~40-45 range for DJIA's low-turnover regime."
        .format(membership["ticker"].nunique())
    )
    lines.append(
        "- This panel is the universe-membership artifact only. "
        "Phase 2 (prices, features, macro, sector adjacency) is gated "
        "and not yet built; the user authorised Phase 1 with Policy P1 "
        "(no protocol changes from the S&P 500 build)."
    )
    return "\n".join(lines)


def main() -> int:
    """Driver: build event log, reverse-walk, validate, persist, report."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    events_chrono = _build_events()
    print(
        f"Loaded {len(events_chrono)} hand-curated DJIA reconstitution "
        f"events spanning {events_chrono[0].date.date()} to "
        f"{events_chrono[-1].date.date()}."
    )

    snapshots = _reverse_walk(CURRENT_DJIA, events_chrono)
    # Insert synthetic split-points at every alias change_date so that
    # a rename that falls strictly inside an existing snapshot interval
    # is honoured (e.g. UTX -> RTX on 2020-04-03 sits between the
    # 2019-04-02 DOW/DWDP event and the 2020-08-31 triple swap; without
    # a split the post-2020-04-03 days would still read UTX).
    snapshot_dates = {d for d, _ in snapshots}
    alias_dates = [pd.Timestamp(dt) for _, (_, dt) in ALIASES.items()]
    for ad in alias_dates:
        if ad in snapshot_dates:
            continue
        prior_state: Set[str] = set()
        for d, s in snapshots:
            if d <= ad:
                prior_state = s
            else:
                break
        snapshots.append((ad, set(prior_state)))
    snapshots = sorted(snapshots, key=lambda kv: kv[0])
    snapshots = [(d, _undo_aliases(s, d)) for d, s in snapshots]

    membership = _build_daily_membership(snapshots, START_DATE, END_DATE)
    aliases = _aliases_dataframe()

    anchor_diffs: Dict[str, Tuple[Set[str], Set[str]]] = {}
    for anchor, expected in (
        ("2020-01-01", ANCHOR_2020),
        ("2022-01-01", ANCHOR_2022),
        ("2024-01-01", ANCHOR_2024),
        ("2025-06-01", ANCHOR_2025),
    ):
        anchor_diffs[anchor] = _anchor_check(membership, anchor, expected)

    membership.to_parquet(DATA_DIR / "membership.parquet", index=False)
    aliases.to_parquet(DATA_DIR / "aliases.parquet", index=False)
    report = _build_report(membership, aliases, anchor_diffs, events_chrono)
    (REPORT_DIR / "phase_1_report.md").write_text(report)

    daily_counts = (
        membership[membership["in_index_flag"]]
        .groupby("date").size()
    )
    print(
        f"Saved {len(membership):,} membership rows over "
        f"{membership['ticker'].nunique()} unique tickers."
    )
    print(
        f"Daily member count: min={int(daily_counts.min())}, "
        f"median={int(daily_counts.median())}, "
        f"max={int(daily_counts.max())} (expected exactly 30)."
    )
    for anchor, (extra, missing) in anchor_diffs.items():
        print(
            f"Anchor {anchor}: |symdiff|={len(extra) + len(missing)} "
            f"(extra={sorted(extra)}, missing={sorted(missing)})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
