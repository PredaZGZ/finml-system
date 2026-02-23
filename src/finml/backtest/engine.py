from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_rank_ls(
    preds: pd.DataFrame,
    market: pd.DataFrame,
    *,
    long_q: float = 0.1,
    short_q: float = 0.1,
    fee_bps: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Daily cross-sectional long/short:
      - each day rank by pred
      - long top q, short bottom q
      - equal weights, dollar neutral
      - execute same-day close-to-close return proxy (ret_1d)
    """
    preds = preds.sort_values(["symbol", "ts"]).drop_duplicates(["symbol", "ts"])
    m = market.sort_values(["symbol", "ts"]).drop_duplicates(["symbol", "ts"])

    g = m.groupby("symbol", sort=False)
    m["ret_1d"] = g["close"].pct_change(1)

    df = preds.merge(m[["symbol", "ts", "ret_1d"]], on=["symbol", "ts"], how="left")
    df = df.dropna(subset=["pred", "ret_1d"]).copy()

    # cross-sectional selection each day
    df["pos"] = 0.0
    for ts, day in df.groupby("ts"):
        n = len(day)
        if n < 20:
            continue

        idx = day.sort_values("pred").index
        k_short = max(1, int(np.floor(short_q * n)))
        k_long = max(1, int(np.floor(long_q * n)))

        df.loc[idx[:k_short], "pos"] = -1.0 / k_short
        df.loc[idx[-k_long:], "pos"] = 1.0 / k_long

    # turnover costs
    df["pos_prev"] = df.groupby("symbol", sort=False)["pos"].shift(1).fillna(0.0)
    df["turnover"] = (df["pos"] - df["pos_prev"]).abs()

    fee = fee_bps / 1e4
    df["cost"] = fee * df["turnover"]
    # PnL realized at t is from the position held from t-1 to t
    df["pnl_sym"] = df["pos_prev"] * df["ret_1d"] - df["cost"]

    daily = df.groupby("ts", as_index=False).agg(pnl=("pnl_sym", "sum"))
    daily = daily.sort_values("ts").reset_index(drop=True)
    daily["equity"] = (1.0 + daily["pnl"]).cumprod()

    pnl = daily["pnl"].to_numpy()
    mean = float(np.mean(pnl))
    std = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else float("nan")
    sharpe = float((mean / std) * np.sqrt(252)) if std and std > 0 else float("nan")

    dd = daily["equity"] / daily["equity"].cummax() - 1.0
    max_dd = float(dd.min()) if len(dd) else float("nan")

    metrics = {
        "n_days": int(len(daily)),
        "mean_daily": mean,
        "vol_daily": std,
        "sharpe_252": sharpe,
        "max_drawdown": max_dd,
        "fee_bps": fee_bps,
        "long_q": long_q,
        "short_q": short_q,
    }
    return daily, metrics