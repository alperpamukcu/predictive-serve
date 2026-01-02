import joblib
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Fix path to allow importing src
sys.path.append(str(Path.cwd()))

from src.utils.config import MODELS_DIR

def inspect():
    model_path = MODELS_DIR / "logreg_final.pkl"
    print(f"Loading model from: {model_path}")
    
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    col_path = MODELS_DIR / "feature_columns.txt"
    if not col_path.exists():
        print("Feature columns file not found!")
        return
        
    cols = col_path.read_text(encoding="utf-8").splitlines()
    print(f"Loaded {len(cols)} feature names.")

    # Check if Pipeline
    if hasattr(model, "named_steps"):
        print("Model is a Pipeline.")
        scaler = model.named_steps['standardscaler']
        clf = model.named_steps['logisticregression']
    else:
        print("Model is NOT a Pipeline (Raw LogisticRegression?).")
        # In this case, we can't inspect scaler easily, assume raw
        clf = model
        scaler = None

    # Find elo_diff
    target = "elo_diff"
    if target in cols:
        idx = cols.index(target)
        print(f"\n--- Analysis for '{target}' (Index {idx}) ---")
        
        if scaler:
            print(f"Scaler Mean : {scaler.mean_[idx]:.4f}")
            print(f"Scaler Scale: {scaler.scale_[idx]:.4f}")
        
        # clf.coef_ is usually shape (1, n_features)
        c = clf.coef_[0][idx]
        print(f"Coefficient : {c:.6f}")
        
        # Calculate impact of -755
        raw_val = -755.0
        if scaler:
            scaled_val = (raw_val - scaler.mean_[idx]) / scaler.scale_[idx]
            print(f"Raw Input   : {raw_val}")
            print(f"Scaled Input: {scaled_val:.4f}")
            print(f"Logit Impact: {scaled_val * c:.4f}")
        else:
             print(f"Logit Impact: {raw_val * c:.4f}")
             
    else:
        print(f"'{target}' NOT found in feature columns!")

    # Top features
    print("\n--- Top 5 Features by Weight ---")
    coefs = pd.Series(clf.coef_[0], index=cols)
    print(coefs.abs().sort_values(ascending=False).head(5))
    print(coefs[coefs.abs().sort_values(ascending=False).head(5).index])

if __name__ == "__main__":
    inspect()
