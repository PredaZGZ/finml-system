from __future__ import annotations

import json
from pathlib import Path

from finml.data.io import read_parquet, write_parquet
from finml.training.dataset import build_dataset
from finml.training.predict import predict
from finml.backtest.engine import backtest_rank_ls


def main() -> None:
    run_id = "20260223T224653Z"
    model_path = Path("models/runs") / run_id / "model.joblib"

    ds = build_dataset(
        label_col="y_fwdret_21",
        train_end="2022-12-31",
        val_end="2023-12-31",
    )

    # Predict on test only (OOS)
    preds = predict(model_path=model_path, X=ds.X_test, meta=ds.meta_test)

    # Load market for realized returns
    market = read_parquet("data/processed/market")

    equity, metrics = backtest_rank_ls(preds, market, long_q=0.1, short_q=0.1, fee_bps=1.0)

    # Save
    out_dir = Path("reports/runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    write_parquet(preds, out_dir / "predictions", partition_cols=["symbol"])
    write_parquet(equity, out_dir / "equity_curve")
    (out_dir / "backtest_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(metrics)


if __name__ == "__main__":
    main()