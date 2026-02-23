from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from finml.data.io import read_parquet, write_parquet
from finml.ingestion.market import ingest_market
from finml.ingestion.providers.base import MarketRequest
from finml.ingestion.providers.yahoo import YahooProvider
from finml.features.market_features import build_market_features
from finml.labels.forward_returns import build_forward_return_labels
from finml.training.dataset import build_dataset
from finml.training.train import train_ridge_baseline
from finml.training.predict import predict
from finml.backtest.engine import backtest_rank_ls


RUN_ROOT = Path("reports/runs")
MODELS_ROOT = Path("models/runs")


def load_universe(path: str) -> list[str]:
    import pandas as pd

    df = pd.read_csv(path)
    return (
        df["symbol"]
        .astype(str)
        .str.strip()
        .dropna()
        .drop_duplicates()
        .tolist()
    )


def chunks(xs: list[str], n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def step_ingest(universe: list[str]) -> None:
    print("== ingest ==")

    provider = YahooProvider()

    for batch in chunks(universe, 50):
        req = MarketRequest(
            symbols=batch,
            start="2018-01-01",
            end="2026-01-01",
            interval="1d",
        )
        res = ingest_market(provider, req, out_dir="data/processed/market")
        print(res)


def step_features() -> None:
    print("== features ==")

    df = read_parquet("data/processed/market")
    feats = build_market_features(df)

    write_parquet(
        feats,
        "data/processed/features/market",
        partition_cols=["symbol"],
    )

    print("features:", feats.shape)


def step_labels() -> None:
    print("== labels ==")

    df = read_parquet("data/processed/market")
    labels = build_forward_return_labels(df, horizons=[21])

    write_parquet(
        labels,
        "data/processed/labels",
        partition_cols=["symbol"],
    )

    print("labels:", labels.shape)


def step_train() -> str:
    print("== train ==")

    ds = build_dataset(
        label_col="y_fwdret_21",
        train_end="2022-12-31",
        val_end="2023-12-31",
    )

    res = train_ridge_baseline(
        ds.X_train,
        ds.y_train,
        ds.X_val,
        ds.y_val,
        alpha=1.0,
        out_root=MODELS_ROOT,
    )

    print(res)
    return res.run_id


def step_backtest(run_id: str) -> None:
    print("== backtest ==")

    model_path = MODELS_ROOT / run_id / "model.joblib"

    ds = build_dataset(
        label_col="y_fwdret_21",
        train_end="2022-12-31",
        val_end="2023-12-31",
    )

    preds = predict(
        model_path=model_path,
        X=ds.X_test,
        meta=ds.meta_test,
    )

    market = read_parquet("data/processed/market")

    equity, metrics = backtest_rank_ls(
        preds,
        market,
        long_q=0.1,
        short_q=0.1,
        fee_bps=1.0,
    )

    out_dir = RUN_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    write_parquet(preds, out_dir / "predictions", partition_cols=["symbol"])
    write_parquet(equity, out_dir / "equity_curve")

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    print(metrics)


def main() -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    universe = load_universe("configs/universe_us_large.csv")

    step_ingest(universe)
    step_features()
    step_labels()

    run_id = step_train()

    step_backtest(run_id)

    print("done:", run_id)


if __name__ == "__main__":
    main()