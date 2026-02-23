from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_sign_strategy(
    preds: pd.DataFrame,
    market: pd.DataFrame,
    *,
    fee_bps: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Very simple backtest:
      - position = sign(pred) per symbol (long/short)
      - daily return uses next-day close-to-close return (already in market)
      - costs: fee_bps paid on turnover (abs(delta position))

    Returns:
      equity curve dataframe (ts, pnl, equity)
      metrics dict
    """
    preds = preds.sort_values(["symbol", "ts"]).copy()

    m = market.sort_values(["symbol", "ts"]).copy()
    g = m.groupby("symbol", sort=False)
    m["ret_1d"] = g["close"].pct_change(1)

    df = preds.merge(m[["symbol", "ts", "ret_1d"]], on=["symbol", "ts"], how="left")
    df = df.dropna(subset=["ret_1d"]).copy()

    # positions
    df["pos"] = np.sign(df["pred"]).astype(float)

    # turnover costs
    df["pos_prev"] = df.groupby("symbol", sort=False)["pos"].shift(1).fillna(0.0)
    df["turnover"] = (df["pos"] - df["pos_prev"]).abs()

    fee = fee_bps / 1e4
    df["cost"] = fee * df["turnover"]
    df["pnl_sym"] = df["pos"] * df["ret_1d"] - df["cost"]

    # equal-weight across symbols per day
    daily = df.groupby("ts", as_index=False).agg(pnl=("pnl_sym", "mean"))
    daily = daily.sort_values("ts").reset_index(drop=True)
    daily["equity"] = (1.0 + daily["pnl"]).cumprod()

    # metrics
    pnl = daily["pnl"].to_numpy()
    mean = float(np.nanmean(pnl))
    std = float(np.nanstd(pnl, ddof=1)) if len(pnl) > 1 else float("nan")
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
    }

    return daily, metrics