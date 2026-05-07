# src/utils/feature_utils.py

from pathlib import Path
from typing import List, Set


# Market-derived columns. These are *consequences* of bookmaker prices and
# encode the answer the model is trying to beat. They MUST NOT enter the
# main "pure" model — otherwise `edge = p_model - pA_market` becomes
# meaningless and validation metrics overstate skill.
LEAKY_MARKET_COLS: Set[str] = {
    "oddsA", "oddsB",
    "pA_market", "pB_market",
    "p_diff", "logit_pA_market",
    "has_market",
}

# Meta columns kept alongside features for filtering / display.
META_COLS: Set[str] = {"date", "surface", "playerA", "playerB", "y"}


def load_feature_list(path: Path) -> List[str]:
    """Read the persisted feature column list (one name per line)."""
    with Path(path).open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def select_model_features(columns: List[str], *, include_market: bool = False) -> List[str]:
    """
    Pick model-input columns from a dataset's full column list.

    Default: drop meta columns AND every market-derived column. Set
    ``include_market=True`` only for explicitly market-aware experiments
    (never for the production "pure" model).
    """
    drop: Set[str] = set(META_COLS)
    if not include_market:
        drop |= LEAKY_MARKET_COLS
    return [c for c in columns if c not in drop]
