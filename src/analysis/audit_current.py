from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from src.utils.config import PROCESSED_DIR, MODELS_DIR


@dataclass(frozen=True)
class EvalResult:
    n_val: int
    logloss: float
    brier: float
    acc: float
    n_market: int
    market_logloss: Optional[float]
    market_brier: Optional[float]
    market_acc: Optional[float]


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def eval_saved_model() -> EvalResult:
    df = pd.read_csv(PROCESSED_DIR / "train_dataset.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    val = df[df["date"].dt.year >= 2022].copy()

    feature_cols = _read_lines(MODELS_DIR / "feature_columns.txt")
    model = load(MODELS_DIR / "logreg_final.pkl")
    imputer = load(MODELS_DIR / "imputer_final.pkl")

    X = val[feature_cols].copy()
    y = val["y"].astype(int).values
    X_imp = imputer.transform(X)
    p = model.predict_proba(X_imp)[:, 1]

    res = EvalResult(
        n_val=len(val),
        logloss=float(log_loss(y, p)),
        brier=float(brier_score_loss(y, p)),
        acc=float(accuracy_score(y, (p >= 0.5).astype(int))),
        n_market=0,
        market_logloss=None,
        market_brier=None,
        market_acc=None,
    )

    if "pA_market" in val.columns:
        pm = pd.to_numeric(val["pA_market"], errors="coerce").astype(float).values
        # Prefer explicit availability indicator when present
        if "has_market" in val.columns:
            m = val["has_market"].astype(int).values == 1
        else:
            m = ~np.isnan(pm)
        if m.any():
            res = EvalResult(
                n_val=res.n_val,
                logloss=res.logloss,
                brier=res.brier,
                acc=res.acc,
                n_market=int(m.sum()),
                market_logloss=float(log_loss(y[m], pm[m])),
                market_brier=float(brier_score_loss(y[m], pm[m])),
                market_acc=float(accuracy_score(y[m], (pm[m] >= 0.5).astype(int))),
            )

    return res


def main() -> None:
    base = PROCESSED_DIR
    files = [
        "matches_allyears.csv",
        "matches_clean.csv",
        "matches_with_elo_form_sets.csv",
        "train_dataset.csv",
        "all_predictions.csv",
        "fixtures_upcoming.csv",
    ]
    print("=== Dataset sizes ===")
    for f in files:
        p = base / f
        if not p.exists():
            print(f"- {f}: MISSING")
            continue
        try:
            df = pd.read_csv(p, low_memory=False)
            print(f"- {f}: rows={len(df):,} cols={len(df.columns)}")
        except Exception as e:
            print(f"- {f}: ERROR reading ({e})")

    print("\n=== Saved model evaluation (val: year>=2022) ===")
    res = eval_saved_model()
    print(f"- val_rows: {res.n_val:,}")
    print(f"- model_logloss: {res.logloss:.6f}")
    print(f"- model_brier:   {res.brier:.6f}")
    print(f"- model_acc:     {res.acc:.6f}")
    if res.market_logloss is not None:
        print(f"- market_rows:   {res.n_market:,}")
        print(f"- market_logloss:{res.market_logloss:.6f}")
        print(f"- market_brier:  {res.market_brier:.6f}")
        print(f"- market_acc:    {res.market_acc:.6f}")


if __name__ == "__main__":
    main()

