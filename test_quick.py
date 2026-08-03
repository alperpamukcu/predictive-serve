"""
Lightweight smoke test — verifies that the source tree imports cleanly and
the leakage-safe feature selector is wired correctly. Generated artifacts
(CSV/PKL) are intentionally NOT required because they're built by the
pipeline at runtime.
"""

import datetime as dt
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict


def _finished_event(day: dt.date, first: str, second: str) -> Dict[str, Any]:
    """One finished ATP singles match shaped like an API-Tennis fixture."""
    return {
        "event_date": day.strftime("%Y-%m-%d"),
        "event_type_type": "Atp Singles",
        "tournament_name": "Test Open",
        "tournament_round": "Final",
        "tournament_surface": "Hard",
        "event_first_player": first,
        "event_second_player": second,
        "event_winner": "First Player",
        "event_status": "Finished",
        "scores": [
            {"set_number": "Set 1", "score_first": "6", "score_second": "4"},
            {"set_number": "Set 2", "score_first": "6", "score_second": "3"},
        ],
    }


def test_partial_fetch_never_truncates() -> list[str]:
    """Regression guard for the silent partial-refresh path.

    ``fetch_recent_results_apitennis.main()`` splits its window into 14-day
    API calls. If one of those calls fails and the run still writes the CSV
    and exits 0, the daily refresh commits a silently truncated dataset and
    retrains the published model on it. Simulate a mid-window failure and
    assert we neither report success nor replace the existing file.
    """
    from src.data.schema import MATCH_COLUMNS
    import src.data.fetch_recent_results_apitennis as recent

    errors: list[str] = []
    today = dt.date.today()
    calls = {"n": 0}

    def flaky_get_fixtures(cfg, date_start, date_stop, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return [
                _finished_event(date_start, "J. Doe", "R. Roe"),
                _finished_event(date_start, "A. Poe", "M. Moe"),
            ]
        raise RuntimeError("simulated upstream 502 Bad Gateway")

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "recent_results_apitennis.csv"
        # Stand-in for the dataset already committed to the repo.
        before = "\n".join(
            [",".join(MATCH_COLUMNS)]
            + [
                f"{today - dt.timedelta(days=i)},Test Open,Hard,Final,J. Doe,R. Roe,"
                ",,,,6-4 6-3,Completed,api-tennis,M,A,j. doe,r. roe,,"
                for i in range(1, 6)
            ]
        ) + "\n"
        out_path.write_text(before, encoding="utf-8")

        saved = (recent.OUT_PATH, recent.get_fixtures, os.environ.copy())
        recent.OUT_PATH = out_path
        recent.get_fixtures = flaky_get_fixtures
        # 21 days forces two 14-day chunks, so the second one can fail.
        os.environ["API_TENNIS_KEY"] = "unit-test-key"
        os.environ["RECENT_RESULTS_PAST_DAYS"] = "21"
        try:
            rc = recent.main()
        finally:
            recent.OUT_PATH, recent.get_fixtures = saved[0], saved[1]
            os.environ.clear()
            os.environ.update(saved[2])

        after = out_path.read_text(encoding="utf-8")

    if calls["n"] < 2:
        errors.append(
            f"test setup: expected >=2 chunks so one could fail, got {calls['n']}"
        )
    if rc == 0:
        errors.append(
            "fetch_recent_results_apitennis.main() returned 0 after a chunk failed — "
            "the daily refresh cannot tell the window was incomplete"
        )
    if after != before:
        errors.append(
            "fetch_recent_results_apitennis.main() overwrote the existing CSV "
            f"({len(before.splitlines())} lines -> {len(after.splitlines())}) "
            "from an incomplete fetch"
        )
    return errors


def test_basic() -> bool:
    print("=" * 60)
    print("Predictive Serve — quick smoke test")
    print("=" * 60)

    errors: list[str] = []

    # 1) Core imports
    try:
        from src.utils.config import DATA_DIR, MODELS_DIR, PROCESSED_DIR, PROJECT_ROOT  # noqa: F401
        from src.utils.feature_utils import (
            LEAKY_MARKET_COLS,
            META_COLS,
            select_model_features,
        )
        from src.utils.surface import guess_surface_from_tournament
        from src.data import cleaning, fetch_data, preprocess, schema  # noqa: F401
        from src.features import build_features, elo, form  # noqa: F401
        from src.models import score_all_matches, train_best  # noqa: F401
        from src.integrations.api_tennis import (
            ApiTennisConfig,
            consensus_decimal_moneyline,
            get_fixtures,
        )  # noqa: F401
        print("[OK] All source modules import cleanly.")
    except Exception as e:  # pragma: no cover
        print(f"[FAIL] Import error: {e}")
        errors.append(str(e))
        return False

    # 2) Leakage guard sanity
    market_examples = ["oddsA", "oddsB", "pA_market", "logit_pA_market"]
    columns = ["eloA", "eloB", "elo_diff"] + market_examples + ["form_winrateA_5"]
    selected = select_model_features(columns, include_market=False)
    leaked = [c for c in market_examples if c in selected]
    if leaked:
        msg = f"select_model_features still emits market columns: {leaked}"
        print(f"[FAIL] {msg}")
        errors.append(msg)
    else:
        print(f"[OK] LEAKY_MARKET_COLS ({len(LEAKY_MARKET_COLS)}) excluded from training set.")

    # 3) Surface inference sanity
    cases = {
        "Roland Garros": "Clay",
        "Wimbledon": "Grass",
        "US Open": "Hard",
        "Madrid": "Clay",
    }
    for tour, expected in cases.items():
        got = guess_surface_from_tournament(tour)
        if got != expected:
            errors.append(f"Surface inference {tour} -> {got} (expected {expected})")
            print(f"[FAIL] Surface {tour} -> {got} (expected {expected})")
        else:
            print(f"[OK] Surface inference {tour} -> {got}")

    # 4) Partial API refresh must not truncate the committed dataset
    try:
        partial_errors = test_partial_fetch_never_truncates()
        if partial_errors:
            for msg in partial_errors:
                print(f"[FAIL] {msg}")
            errors.extend(partial_errors)
        else:
            print("[OK] Partial API-Tennis fetch leaves the existing dataset intact.")
    except Exception as e:
        errors.append(f"partial-fetch guard: {e}")
        print(f"[FAIL] partial-fetch guard: {e}")

    # 5) Streamlit app parses
    try:
        import ast

        ast.parse(open("streamlit_app.py", encoding="utf-8").read())
        print("[OK] streamlit_app.py parses.")
    except Exception as e:
        errors.append(f"streamlit_app.py parse: {e}")
        print(f"[FAIL] streamlit_app.py parse: {e}")

    print("=" * 60)
    if errors:
        print(f"[FAIL] {len(errors)} issue(s):")
        for err in errors:
            print(f"  - {err}")
        return False
    print("[OK] All structural checks passed.")
    return True


if __name__ == "__main__":
    sys.exit(0 if test_basic() else 1)
