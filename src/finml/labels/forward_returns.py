from __future__ import annotations

import pandas as pd


def build_forward_return_labels(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """
    Input: MarketFrame (symbol, ts, close, ...)
    Output: LabelFrame (symbol, ts, y_fwdret_{h})
    """
    df = df.sort_values(["symbol", "ts"]).copy()
    df = df.drop_duplicates(subset=["symbol", "ts"], keep="last")
    g = df.groupby("symbol", sort=False)

    out = df[["symbol", "ts"]].copy()
    for h in horizons:
        out[f"y_fwdret_{h}"] = g["close"].shift(-h) / df["close"] - 1.0

    return out