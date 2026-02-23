from __future__ import annotations

import numpy as np
import pandas as pd


def build_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: MarketFrame with columns at least:
      symbol, ts, open, high, low, close, volume

    Output: FeatureFrame:
      symbol, ts, f_*
    """
    df = df.sort_values(["symbol", "ts"]).copy()
    df = df.drop_duplicates(subset=["symbol", "ts"], keep="last")
    g = df.groupby("symbol", sort=False)

    # returns
    df["f_ret_1d"] = g["close"].pct_change(1)
    df["f_logret_1d"] = np.log(df["close"] / g["close"].shift(1))
    df["f_ret_5d"] = g["close"].pct_change(5)

    # volatility (no cross-symbol leakage)
    df["f_vol_20d"] = (
        g["f_logret_1d"]
        .rolling(20)
        .std()
        .reset_index(level=0, drop=True)
    )

    # momentum
    df["f_mom_20d"] = g["close"].pct_change(20)
    df["f_mom_60d"] = g["close"].pct_change(60)

    # ranges
    df["f_hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["f_oc_range"] = (df["close"] - df["open"]) / df["open"]

    # volume z-score (20d)
    vol_mean = g["volume"].rolling(20).mean().reset_index(level=0, drop=True)
    vol_std = g["volume"].rolling(20).std().reset_index(level=0, drop=True).replace(0, np.nan)
    df["f_vol_z_20d"] = (df["volume"] - vol_mean) / vol_std

    keep = ["symbol", "ts"] + [c for c in df.columns if c.startswith("f_")]
    return df[keep]