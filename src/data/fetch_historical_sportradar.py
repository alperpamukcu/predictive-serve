from __future__ import annotations

import datetime as dt
from pathlib import Path
import sys

import pandas as pd

from src.utils.config import PROJECT_ROOT, RAW_DIR
from src.utils.env import try_load_dotenv, getenv, getenv_int
from src.integrations.sportradar_tennis import SportradarTennisConfig, fetch_daily_summaries


OUT_PATH = RAW_DIR / "sportradar_historical.csv"


def _clean(s) -> str:
    return "" if s is None else " ".join(str(s).replace("\u00a0", " ").split()).strip()


def _extract_matches(payload: dict) -> list[dict]:
    """
    Best-effort extract completed matches from daily summaries payload.
    We store a minimal set of fields so we can compare quality/coverage.
    """
    out: list[dict] = []
    events: list[dict] = []

    summaries = payload.get("summaries")
    if isinstance(summaries, list):
        for item in summaries:
            if not isinstance(item, dict):
                continue
            se = item.get("sport_event")
            if not isinstance(se, dict):
                continue
            merged = dict(se)
            ses = item.get("sport_event_status")
            if isinstance(ses, dict):
                st = ses.get("status")
                if st:
                    merged["status"] = st
                wid = ses.get("winner_id")
                if wid:
                    merged["winner_id"] = wid
            events.append(merged)
    else:
        events = [e for e in (payload.get("sport_events") or []) if isinstance(e, dict)]

    for ev in events:
        if not isinstance(ev, dict):
            continue
        status = _clean(ev.get("status"))
        # Keep only events that look finished; Sportradar uses statuses like "closed"
        if status and status.lower() not in {"closed", "ended", "finished"}:
            continue

        comps = ev.get("competitors") or []
        if not isinstance(comps, list) or len(comps) < 2:
            continue
        names = []
        quals = []
        for c in comps:
            if not isinstance(c, dict):
                continue
            nm = _clean(c.get("name"))
            q = _clean(c.get("qualifier"))
            if nm:
                names.append(nm)
                quals.append(q)
        if len(names) < 2:
            continue

        scheduled = _clean(ev.get("scheduled")) or _clean(ev.get("start_time")) or ""
        date = scheduled[:10] if scheduled else ""

        # Tournament name
        tourn = ""
        t = ev.get("tournament") or {}
        if isinstance(t, dict):
            tourn = _clean(t.get("name"))

        # Surface / extended metadata live here frequently.
        ctx = ev.get("sport_event_context") or {}
        if isinstance(ctx, dict) and not tourn:
            comp = ctx.get("competition")
            if isinstance(comp, dict):
                tourn = _clean(comp.get("name"))
            if not tourn:
                season = ctx.get("season")
                if isinstance(season, dict):
                    tourn = _clean(season.get("name"))

        # Round
        rnd = ""
        tr = ev.get("tournament_round") or {}
        if isinstance(tr, dict):
            rnd = _clean(tr.get("name")) or _clean(tr.get("type"))
        if not rnd and isinstance(ctx, dict):
            cr = ctx.get("round")
            if isinstance(cr, dict):
                rnd = _clean(cr.get("name"))
                if not rnd and cr.get("number") is not None:
                    rnd = f"Round {cr.get('number')}"

        # Surface (often in sport_event_context)
        surface = ""
        if isinstance(ctx, dict):
            srf = ctx.get("surface")
            if isinstance(srf, dict):
                surface = _clean(srf.get("name"))
            elif isinstance(srf, str):
                surface = _clean(srf)

        # Winner/Loser (best-effort): some payloads include "winner_id"
        winner = ""
        loser = ""
        winner_id = ev.get("winner_id")
        if isinstance(winner_id, str) and winner_id:
            for c in comps:
                if isinstance(c, dict) and _clean(c.get("id")) == winner_id:
                    winner = _clean(c.get("name"))
            for c in comps:
                if isinstance(c, dict) and _clean(c.get("id")) != winner_id:
                    loser = _clean(c.get("name"))
        if not winner or not loser:
            # fallback: keep competitors but leave winner/loser empty (still useful for coverage)
            winner = ""
            loser = ""

        out.append(
            {
                "date": date,
                "tourney": tourn,
                "surface": surface,
                "round": rnd,
                "winner": winner,
                "loser": loser,
                "competitor_1": names[0],
                "competitor_2": names[1],
                "sport_event_id": _clean(ev.get("id")),
                "status": status,
                "scheduled": scheduled,
            }
        )
    return out


def main() -> int:
    try_load_dotenv(PROJECT_ROOT)

    api_key = getenv("SPORTRADAR_TENNIS_API_KEY")
    if not api_key:
        print("[sr-hist] ERROR: SPORTRADAR_TENNIS_API_KEY is not set.")
        return 2

    access_level = getenv("SPORTRADAR_TENNIS_ACCESS_LEVEL", "trial") or "trial"
    lang = getenv("SPORTRADAR_TENNIS_LANG", "en") or "en"

    # We intentionally keep this bounded; this is for A/B quality checks, not full backfill.
    days = getenv_int("SR_HIST_DAYS", 30)
    days = max(1, min(days, 365))
    end = dt.date.today()
    start = end - dt.timedelta(days=days - 1)

    cfg = SportradarTennisConfig(api_key=api_key, access_level=access_level, language=lang)

    print(f"[sr-hist] Fetching daily summaries for {days} days: {start}..{end}")
    rows: list[dict] = []
    for i in range(days):
        day = start + dt.timedelta(days=i)
        try:
            payload = fetch_daily_summaries(cfg, day)
            rows.extend(_extract_matches(payload))
        except Exception as e:
            print(f"[sr-hist] WARNING: {day} failed: {e}")

    if not rows:
        print("[sr-hist] No rows extracted. Check access level/package.")
        return 1

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["sport_event_id"], keep="first")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"[sr-hist] Saved: {OUT_PATH} (rows={len(df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

