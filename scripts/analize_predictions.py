from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from finml.data.io import read_parquet


RUN_ID = "20260223T212353Z"


def sharpe_252(pnl: pd.Series) -> float:
    x = pnl.dropna().to_numpy()
    if x.size < 2:
        return float("nan")
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd == 0:
        return float("nan")
    return float((mu / sd) * np.sqrt(252))


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.dropna()
    if eq.empty:
        return float("nan")
    dd = eq / eq.cummax() - 1.0
    return float(dd.min())

def get_latest_run_id() -> str:
    runs_dir = Path("reports/runs")

    if not runs_dir.exists():
        raise RuntimeError("No runs directory found")

    run_ids = [
        p.name
        for p in runs_dir.iterdir()
        if p.is_dir()
    ]

    if not run_ids:
        raise RuntimeError("No runs found")

    return sorted(run_ids)[-1]


def main() -> None:
    run_id = get_latest_run_id()

    print(f"\n=== Analyzing run: {run_id} ===")

    run_dir = Path("reports/runs") / run_id

    preds = read_parquet(run_dir / "predictions")
    labels = read_parquet("data/processed/labels")
    equity = read_parquet(run_dir / "equity_curve")

    df = preds.merge(labels, on=["symbol", "ts"], how="inner")
    df = df.drop_duplicates(subset=["symbol", "ts"], keep="last")
    df = df.dropna(subset=["pred", "y_fwdret_21"]).copy()

    # 1) core signal quality
    corr = float(df["pred"].corr(df["y_fwdret_21"]))
    dir_acc = float(((df["pred"] > 0) == (df["y_fwdret_21"] > 0)).mean())

    print("\n=== Signal quality (OOS) ===")
    print("Correlation pred vs real:", corr)
    print("Direction accuracy:", dir_acc)

    print("\n=== By symbol (corr) ===")
    print(df.groupby("symbol").apply(lambda x: x["pred"].corr(x["y_fwdret_21"])).sort_values(ascending=False))

    # 2) bucket test (most important)
    # Use per-day buckets if you have many tickers; with few tickers global is ok.
    df["bucket"] = pd.qcut(df["pred"], 5, labels=False, duplicates="drop")
    bucket = df.groupby("bucket")["y_fwdret_21"].mean()

    print("\n=== Bucket test (mean realized fwd 21d return) ===")
    print(bucket)

    # 3) show best/worst predictions
    print("\n=== Top 15 signals (pred) ===")
    print(df.sort_values("pred", ascending=False)[["ts", "symbol", "pred", "y_fwdret_21"]].head(15))

    print("\n=== Bottom 15 signals (pred) ===")
    print(df.sort_values("pred", ascending=True)[["ts", "symbol", "pred", "y_fwdret_21"]].head(15))

    # 4) equity curve + pnl stats
    equity = equity.sort_values("ts").copy()
    print("\n=== Backtest summary ===")
    print("n_days:", len(equity))
    print("Sharpe_252:", sharpe_252(equity["pnl"]))
    print("Max DD:", max_drawdown(equity["equity"]))
    print("Final equity:", float(equity["equity"].iloc[-1]))

    # plots
    plt.figure()
    plt.plot(equity["ts"], equity["equity"])
    plt.title(f"Equity curve (run {RUN_ID})")
    plt.xlabel("ts")
    plt.ylabel("equity")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.hist(equity["pnl"].dropna().to_numpy(), bins=50)
    plt.title("Daily PnL distribution")
    plt.xlabel("pnl")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

    # optional: load metrics.json if present
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print("\n=== Stored metrics.json ===")
        print(json.loads(metrics_path.read_text()))


if __name__ == "__main__":
    main()