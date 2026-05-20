"""
Lightweight smoke test — verifies that the source tree imports cleanly and
the leakage-safe feature selector is wired correctly. Generated artifacts
(CSV/PKL) are intentionally NOT required because they're built by the
pipeline at runtime.
"""

import sys


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

    # 4) Streamlit app parses
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
