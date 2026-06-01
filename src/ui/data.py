"""Streamlit-cached data loaders for the Predictive Serve UI.

Pulled out of streamlit_app.py so the entry point stays focused on
layout. Every loader is wrapped in ``@st.cache_data`` (or
``@st.cache_resource`` for the model artefacts) so the page only pays
the disk-IO cost on the first user request.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
from joblib import load as joblib_load

from src.utils.config import MODELS_DIR, PROCESSED_DIR
from src.utils.feature_utils import load_feature_list

PRED_PATH = PROCESSED_DIR / "all_predictions.csv"
HISTORY_PATH = PROCESSED_DIR / "matches_with_elo_form.csv"
FIXTURES_PATH = PROCESSED_DIR / "fixtures_upcoming.csv"
METRICS_PATH = MODELS_DIR / "metrics.json"
MODEL_PATH = MODELS_DIR / "logreg_final.pkl"
IMPUTER_PATH = MODELS_DIR / "imputer_final.pkl"
FEATURE_COLS_PATH = MODELS_DIR / "feature_columns.txt"


def _coerce_str(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                .str.replace(" ", " ", regex=False)
                .str.strip()
            )
    return df


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    if not PRED_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(PRED_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = _coerce_str(df, ["surface", "playerA", "playerB"])
    return df


@st.cache_data(show_spinner=False)
def load_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(HISTORY_PATH, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "tourney" in df.columns and "tournament" not in df.columns:
        df = df.rename(columns={"tourney": "tournament"})
    df = _coerce_str(df, ["surface", "round", "tournament", "playerA", "playerB"])
    return df


@st.cache_data(show_spinner=False)
def load_real_fixtures() -> pd.DataFrame:
    if not FIXTURES_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(FIXTURES_PATH)
    if "match_date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"match_date": "date"})
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    if "tourney" in df.columns and "tournament" not in df.columns:
        df = df.rename(columns={"tourney": "tournament"})
    df = _coerce_str(df, ["tournament", "surface", "round", "playerA", "playerB"])
    for c in ("oddsA", "oddsB"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_resource(show_spinner=False)
def load_artifacts() -> Tuple[Any, Any, List[str]]:
    if not (MODEL_PATH.exists() and IMPUTER_PATH.exists() and FEATURE_COLS_PATH.exists()):
        return None, None, []
    model = joblib_load(MODEL_PATH)
    imputer = joblib_load(IMPUTER_PATH)
    feature_cols = load_feature_list(FEATURE_COLS_PATH)
    return model, imputer, feature_cols


@st.cache_data(show_spinner=False)
def load_metrics_json() -> Dict[str, Any]:
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
