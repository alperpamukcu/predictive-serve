from pathlib import Path
import argparse
import json
from datetime import date, timedelta

import pandas as pd

from src.integrations.sportradar_client import SportradarClient


def canonicalize_daily(payload: dict) -> list[dict]:
    rows = []
    for s in payload.get("summaries", []):
        ev = s.get("sport_event", {}) or {}
        ctx = ev.get("sport_event_context", {}) or {}
        comp = ctx.get("competition", {}) or {}
        season = ctx.get("season", {}) or {}
        rnd = ctx.get("round", {}) or {}
        mode = ctx.get("mode", {}) or {}
        venue = ev.get("venue", {}) or {}

        # MVP: singles odaklı (doubles team/players node işi uzatır)
        if comp.get("type") and comp.get("type") != "singles":
            continue

        competitors = ev.get("competitors", []) or []
        if len(competitors) != 2:
            continue

        home = next((c for c in competitors if c.get("qualifier") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("qualifier") == "away"), competitors[1])

        rows.append({
            "sr_event_id": ev.get("id"),
            "start_time_utc": ev.get("start_time"),
            "start_time_confirmed": ev.get("start_time_confirmed"),
            "estimated": ev.get("estimated"),
            "category_name": (ctx.get("category") or {}).get("name"),
            "competition_id": comp.get("id"),
            "competition_name": comp.get("name"),
            "competition_level": comp.get("level"),
            "competition_gender": comp.get("gender"),
            "season_id": season.get("id"),
            "season_name": season.get("name"),
            "round_name": rnd.get("name"),
            "round_number": rnd.get("number"),
            "best_of": mode.get("best_of"),
            "venue_timezone": venue.get("timezone"),
            "venue_name": venue.get("name"),

            "sr_competitorA_id": home.get("id"),
            "playerA_name": home.get("name"),
            "playerA_country": home.get("country_code"),
            "playerA_seed": home.get("seed"),

            "sr_competitorB_id": away.get("id"),
            "playerB_name": away.get("name"),
            "playerB_country": away.get("country_code"),
            "playerB_seed": away.get("seed"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--access_level", default="trial", choices=["trial", "production"])
    ap.add_argument("--lang", default="en")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (default: today)")
    args = ap.parse_args()

    start = date.fromisoformat(args.start) if args.start else date.today()
    out_raw = Path("data/upcoming/raw")
    out_raw.mkdir(parents=True, exist_ok=True)

    all_rows = []
    client = SportradarClient(access_level=args.access_level, language=args.lang)

    for i in range(args.days):
        d = start + timedelta(days=i)
        d_str = d.isoformat()
        payload = client.daily_summaries(d_str)

        # raw cache
        (out_raw / f"{d_str}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        # canonical rows
        all_rows.extend(canonicalize_daily(payload))

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["sr_event_id"])
    out_csv = Path("data/upcoming/upcoming_sportradar.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved: {out_csv}  rows={len(df)}")


if __name__ == "__main__":
    main()
