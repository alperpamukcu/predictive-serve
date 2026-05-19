"""
Append the API-Tennis recent-results CSV onto matches_clean.csv so the
feature pipeline (elo / form / sets / build_features) and the model
training step pick up matches that haven't yet been published by
tennis-data.co.uk.

The merge:
    1. Reads `data/processed/matches_clean.csv` (canonical history)
    2. Reads `data/processed/recent_results_apitennis.csv` (live API supplement)
    3. Maps each API player name to the canonical history spelling so the
       same player isn't tracked under two distinct keys (e.g. "Sinner J."
       vs "Jannik Sinner").
    4. Drops any API row whose (date, canonical-A, canonical-B) already
       exists in the history.
    5. Appends, sorts by date, writes back to matches_clean.csv.

Idempotent. Designed to be re-run nightly.
"""
from __future__ import annotations

import sys
from collections import Counter

import pandas as pd

from src.data.schema import MATCH_COLUMNS
from src.utils.config import PROCESSED_DIR
from src.utils.player_meta import build_history_index, canonical_parts, resolve_history_name


HISTORY_PATH = PROCESSED_DIR / "matches_clean.csv"
API_PATH = PROCESSED_DIR / "recent_results_apitennis.csv"

# A genuine ATP main-tour match is between two players who BOTH have a
# real history on tour. Qualifying / ITF / junior players have ~0 prior
# main-tour matches, so requiring this cleanly drops the qualifying-week
# pollution (e.g. Roland Garros qualifying labelled "French Open" with
# everyone sitting at base Elo 1500).
MIN_HISTORY_MATCHES = 20

# tennis-data.co.uk and api-tennis.com disagree on a handful of tournament
# names. Normalise the API spelling onto the historical one so the same
# event isn't split in the Tournaments tab.
TOURNAMENT_ALIASES = {
    "rome": "Internazionali BNL d'Italia",
    "internazionali bnl d'italia": "Internazionali BNL d'Italia",
    "madrid": "Madrid Masters",
    "mutua madrid open": "Madrid Masters",
    "monte carlo": "Monte Carlo Masters",
    "rolex monte-carlo masters": "Monte Carlo Masters",
    "indian wells": "Indian Wells Masters",
    "miami": "Miami Masters",
    "miami open": "Miami Masters",
    "canada": "Canada Masters",
    "cincinnati": "Cincinnati Masters",
    "shanghai": "Shanghai Masters",
    "paris": "Paris Masters",
    "us open": "US Open",
    "french open": "French Open",
    "australian open": "Australian Open",
    "wimbledon": "Wimbledon",
}


def _normalise_tourney(name: str) -> str:
    key = str(name or "").strip().lower()
    return TOURNAMENT_ALIASES.get(key, str(name or "").strip())


def _canonical_pair_key(date_str: str, a: str, b: str) -> str:
    da = canonical_parts(str(a))[1] or str(a).lower()
    db = canonical_parts(str(b))[1] or str(b).lower()
    pair = "|".join(sorted((da, db)))
    return f"{date_str}|{pair}"


def main() -> int:
    if not HISTORY_PATH.exists():
        print(f"[merge] history missing: {HISTORY_PATH}")
        return 1
    if not API_PATH.exists():
        print(f"[merge] no API supplement at {API_PATH} — nothing to merge.")
        return 0

    history = pd.read_csv(HISTORY_PATH)
    api = pd.read_csv(API_PATH)
    if api.empty:
        print("[merge] API supplement is empty.")
        return 0

    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    api["date"] = pd.to_datetime(api["date"], errors="coerce")
    api = api.dropna(subset=["date", "playerA", "playerB"])
    history = history.dropna(subset=["date"])

    # Map API player names onto the history spelling so the same player
    # carries one identity across Elo / form histograms.
    names = pd.concat([history["playerA"], history["playerB"]]).dropna().astype(str).unique().tolist()
    by_init_sur, by_surname = build_history_index(names)
    api["playerA"] = api["playerA"].astype(str).apply(lambda n: resolve_history_name(n, by_init_sur, by_surname) or n)
    api["playerB"] = api["playerB"].astype(str).apply(lambda n: resolve_history_name(n, by_init_sur, by_surname) or n)
    api["playerA_norm"] = api["playerA"].astype(str).str.strip().str.lower()
    api["playerB_norm"] = api["playerB"].astype(str).str.strip().str.lower()

    # Build a set of existing (date, canonical-pair) keys to dedup against.
    hist_keys = {
        _canonical_pair_key(
            str(r["date"].date() if hasattr(r["date"], "date") else r["date"]),
            r["playerA"],
            r["playerB"],
        )
        for _, r in history.iterrows()
    }
    api_keys = api.apply(
        lambda r: _canonical_pair_key(str(r["date"].date()), r["playerA"], r["playerB"]),
        axis=1,
    )
    new_mask = ~api_keys.isin(hist_keys)
    new_rows = api[new_mask].copy()
    print(
        f"[merge] history rows: {len(history):,} | API supplement: {len(api):,} | "
        f"new after dedup: {len(new_rows):,}"
    )

    if new_rows.empty:
        return 0

    # --- Drop qualifying / ITF / junior pollution ------------------------
    # Count how many historical main-tour matches each canonical player has.
    hist_counts: Counter = Counter()
    for col in ("playerA", "playerB"):
        for n in history[col].dropna().astype(str):
            k = canonical_parts(n)
            if k[1]:
                hist_counts[k] += 1

    def _established(name: str) -> bool:
        return hist_counts.get(canonical_parts(str(name)), 0) >= MIN_HISTORY_MATCHES

    before = len(new_rows)
    keep_mask = new_rows.apply(
        lambda r: _established(r["playerA"]) and _established(r["playerB"]), axis=1
    )
    dropped = new_rows[~keep_mask]
    new_rows = new_rows[keep_mask].copy()
    print(
        f"[merge] tour-level filter: kept {len(new_rows)}/{before} "
        f"(dropped {before - len(new_rows)} qualifying/ITF-level matches; "
        f"both players need >= {MIN_HISTORY_MATCHES} prior main-tour matches)"
    )
    if not dropped.empty:
        drop_tourneys = dropped["tourney"].value_counts().head(6).to_dict()
        print(f"[merge]   dropped by tournament: {drop_tourneys}")

    if new_rows.empty:
        print("[merge] nothing left to merge after the tour-level filter.")
        return 0

    # Normalise tournament names so the API spelling lines up with history.
    new_rows["tourney"] = new_rows["tourney"].apply(_normalise_tourney)

    # Concatenate, sort by date, write back. Keep schema column order.
    combined = pd.concat([history, new_rows], ignore_index=True, sort=False)
    combined = combined.sort_values("date").reset_index(drop=True)
    combined = combined.reindex(columns=MATCH_COLUMNS, fill_value=None)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    combined.to_csv(HISTORY_PATH, index=False, encoding="utf-8")
    print(f"[merge] wrote {HISTORY_PATH} (rows={len(combined):,}); latest date {combined['date'].max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
