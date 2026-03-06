
import pandas as pd
from pathlib import Path
import numpy as np

# Mocking the paths based on previous file lists
PRED_PATH = Path("data/processed/all_predictions.csv")
HIST_PATH = Path("data/processed/matches_with_elo_form_sets.csv")

def clean_text(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).replace("\u00a0", " ").strip()
    return " ".join(s.split())

def normalize_predictions(df):
    df = df.copy()
    if "match_date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"match_date": "date"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "player_1" in df.columns and "playerA" not in df.columns:
        df = df.rename(columns={"player_1": "playerA"})
    if "player_2" in df.columns and "playerB" not in df.columns:
        df = df.rename(columns={"player_2": "playerB"})

    for c in ["playerA", "playerB", "surface", "tournament", "round"]:
        if c in df.columns:
            df[c] = df[c].map(clean_text)
            
    if "tourney" in df.columns and "tournament" not in df.columns:
        df = df.rename(columns={"tourney": "tournament"})
            
    return df

def normalize_history(df):
    df = df.copy()
    if "match_date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"match_date": "date"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        
    for c in ["playerA", "playerB", "surface", "tournament", "round"]:
        if c in df.columns:
            df[c] = df[c].map(clean_text)
    return df

def test_merge():
    print("Loading predictions...")
    if not PRED_PATH.exists():
        print(f"Error: {PRED_PATH} not found.")
        return
    df = pd.read_csv(PRED_PATH)
    print(f"Pred shape original: {df.shape}")
    print(f"Pred columns: {df.columns.tolist()}")
    
    df = normalize_predictions(df)
    
    print("Loading history...")
    if not HIST_PATH.exists():
        print(f"Error: {HIST_PATH} not found.")
        return
    hdf = pd.read_csv(HIST_PATH)
    print(f"History shape: {hdf.shape}")
    print(f"History columns: {hdf.columns.tolist()}")
    
    hdf = normalize_history(hdf)
    
    # Logic from app
    if "tournament" not in df.columns or df["tournament"].isna().all():
        print("Trigerring merge logic...")
        cols_to_merge = ["date", "playerA", "playerB"]
        cols_to_pull = ["tournament", "round"]
        
        # Check availability
        # History has 'tourney' likely, check normalization
        print(f"History 'tournament' present? {'tournament' in hdf.columns}")
        if 'tourney' in hdf.columns and 'tournament' not in hdf.columns:
             print("History has 'tourney' but not 'tournament', fixing...")
             hdf = hdf.rename(columns={'tourney': 'tournament'})

        av_cols = [c for c in cols_to_pull if c in hdf.columns and c not in df.columns]
        print(f"Columns to pull: {av_cols}")
        
        if av_cols:
            hdf = hdf.drop_duplicates(subset=cols_to_merge)
            
            # Print sample to check merge keys
            print("\nSample Pred Keys:")
            print(df[cols_to_merge].head())
            print("\nSample Hist Keys:")
            print(hdf[cols_to_merge].head())
            
            merged = df.merge(hdf[cols_to_merge + av_cols], on=cols_to_merge, how="left")
            print(f"\nMerged shape: {merged.shape}")
            print(f"Merged 'tournament' null count: {merged['tournament'].isna().sum()}")
            print(f"Merged 'tournament' unique (first 10): {merged['tournament'].dropna().unique()[:10]}")
            
            if merged['tournament'].isna().sum() == len(merged):
                print("MERGE FAILED: All tournaments are clean.")
        else:
            print("No columns to pull found.")
    else:
        print("Tournament column already exists and has data.")

if __name__ == "__main__":
    test_merge()
