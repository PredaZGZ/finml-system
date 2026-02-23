from __future__ import annotations

import pandas as pd

from finml.data.io import read_parquet


RUN_ID = "20260223T212353Z"


def main() -> None:

    preds = read_parquet(f"reports/runs/{RUN_ID}/predictions")
    labels = read_parquet("data/processed/labels")

    df = preds.merge(
        labels,
        on=["symbol", "ts"],
        how="inner",
    )

    df = df.dropna(subset=["pred", "y_fwdret_21"])

    df["error"] = df["pred"] - df["y_fwdret_21"]

    df["correct_direction"] = (
        (df["pred"] > 0) & (df["y_fwdret_21"] > 0)
        |
        (df["pred"] < 0) & (df["y_fwdret_21"] < 0)
    )

    df = df.sort_values("ts")

    print("\n=== Últimas 20 predicciones ===\n")
    print(
        df[
            [
                "ts",
                "symbol",
                "pred",
                "y_fwdret_21",
                "error",
                "correct_direction",
            ]
        ].tail(20)
    )

    print("\n=== Métricas ===\n")

    accuracy = df["correct_direction"].mean()
    corr = df["pred"].corr(df["y_fwdret_21"])

    print("Direction accuracy:", accuracy)
    print("Correlation:", corr)

    print("\n=== Por símbolo ===\n")

    print(
        df.groupby("symbol")[["pred", "y_fwdret_21"]]
        .corr()
        .iloc[0::2, -1]
    )


if __name__ == "__main__":
    main()