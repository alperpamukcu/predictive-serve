
import pandas as pd
from pathlib import Path
import numpy as np
import sys

# Add src to path if needed (though we can just mock the structure)

PRED_PATH = Path("data/processed/all_predictions.csv")
HIST_PATH = Path("data/processed/matches_with_elo_form_sets.csv")

def clean_text(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).replace("\u00a0", " ").strip()
    return " ".join(s.split())

def normalize_predictions(df):
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Simulate the production code exactly
    if "tourney" in df.columns and "tournament" not in df.columns:
        df = df.rename(columns={"tourney": "tournament"})
            
    return df

def normalize_history(df):
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        
    if "tourney" in df.columns and "tournament" not in df.columns:
        df = df.rename(columns={"tourney": "tournament"})
            
    return df

def load_and_merge():
    print("--- DEBUG START ---")
    
    # 1. Load Preds
    df = pd.read_csv(PRED_PATH)
    df = normalize_predictions(df)
    print(f"Preds loaded. Columns: {df.columns.tolist()}")
    print(f"Preds 'tournament' exists? {'tournament' in df.columns}")
    if 'tournament' in df.columns:
        print(f"Preds 'tournament' nulls: {df['tournament'].isna().sum()}/{len(df)}")

    # 2. Logic Check
    needs_merge = "tournament" not in df.columns or df["tournament"].isna().all()
    print(f"Needs merge? {needs_merge}")
    
    if needs_merge:
        # 3. Load History
        hdf = pd.read_csv(HIST_PATH)
        hdf = normalize_history(hdf)
        print(f"History loaded. Columns: {hdf.columns.tolist()}")
        print(f"History 'tournament' exists? {'tournament' in hdf.columns}")
        
        # 4. Merge
        cols_to_merge = ["date", "playerA", "playerB"]
        cols_to_pull = ["tournament", "round"]
        av_cols = [c for c in cols_to_pull if c in hdf.columns and c not in df.columns]
        print(f"Available cols to pull: {av_cols}")
        
        if av_cols:
             hdf = hdf.drop_duplicates(subset=cols_to_merge)
             merged = df.merge(hdf[cols_to_merge + av_cols], on=cols_to_merge, how="left")
             print(f"Merge done. Shape: {merged.shape}")
             print(f"Merged 'tournament' nulls: {merged['tournament'].isna().sum()}")
             
             # 5. Check 'safe_unique' equivalent
             vals = merged["tournament"].dropna().astype(str).map(clean_text)
             vals = vals[vals != ""]
             unique_vals = sorted(vals.unique().tolist())
             print(f"Unique Tournaments found: {len(unique_vals)}")
             print(f"Sample: {unique_vals[:5]}")
        else:
            print("No cols to pull.")
    
    print("--- DEBUG END ---")

if __name__ == "__main__":
    load_and_merge()
