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

import pandas as pd

from src.data.schema import MATCH_COLUMNS
from src.utils.config import PROCESSED_DIR
from src.utils.player_meta import build_history_index, canonical_parts, resolve_history_name


HISTORY_PATH = PROCESSED_DIR / "matches_clean.csv"
API_PATH = PROCESSED_DIR / "recent_results_apitennis.csv"


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
