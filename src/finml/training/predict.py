from __future__ import annotations

from pathlib import Path

import pandas as pd


def predict(
    *,
    model_path: str | Path,
    X: pd.DataFrame,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns: DataFrame with columns:
      symbol, ts, pred
    """
    import joblib

    model = joblib.load(model_path)
    p = model.predict(X.values)

    out = meta[["symbol", "ts"]].copy()
    out["pred"] = p
    out = out.drop_duplicates(subset=["symbol", "ts"], keep="last")
    return out