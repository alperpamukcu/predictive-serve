# src/analysis/metrics.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    accuracy_score,
)


@dataclass
class OverallMetrics:
    model_logloss: float
    model_brier: float
    model_acc: float

    market_logloss: float | None = None
    market_brier: float | None = None
    market_acc: float | None = None

    n_matches: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_matches": self.n_matches,
            "model_logloss": self.model_logloss,
            "model_brier": self.model_brier,
            "model_accuracy": self.model_acc,
            "market_logloss": self.market_logloss,
            "market_brier": self.market_brier,
            "market_accuracy": self.market_acc,
        }


def compute_overall_metrics(df: pd.DataFrame) -> OverallMetrics:
    """
    val_predictions.csv (veya onun filtresi) üzerinden
    model vs bahis şirketi metriklerini hesaplar.

    Beklenen kolonlar:
      - y          : gerçek sonuç (0/1)
      - p_model    : modelin A kazanır olasılığı
      - pA_market  : marketin A kazanır implied probability'si
    """
    if df.empty:
        raise ValueError("compute_overall_metrics: Boş DataFrame verildi.")

    if "y" not in df or "p_model" not in df:
        raise ValueError("compute_overall_metrics: 'y' ve 'p_model' kolonları zorunlu.")

    y_true = df["y"].astype(int).values
    p_model = df["p_model"].astype(float).values

    # --- Model metrikleri ---
    model_logloss = log_loss(y_true, p_model)
    model_brier = brier_score_loss(y_true, p_model)
    model_acc = accuracy_score(y_true, (p_model >= 0.5).astype(int))

    # --- Market metrikleri (varsa) ---
    if "pA_market" in df.columns:
        p_market = df["pA_market"].astype(float).values
        market_logloss = log_loss(y_true, p_market)
        market_brier = brier_score_loss(y_true, p_market)
        market_acc = accuracy_score(y_true, (p_market >= 0.5).astype(int))
    else:
        market_logloss = None
        market_brier = None
        market_acc = None

    return OverallMetrics(
        model_logloss=model_logloss,
        model_brier=model_brier,
        model_acc=model_acc,
        market_logloss=market_logloss,
        market_brier=market_brier,
        market_acc=market_acc,
        n_matches=len(df),
    )
