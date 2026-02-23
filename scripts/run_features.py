from __future__ import annotations

from finml.data.io import read_parquet, write_parquet
from finml.features.market_features import build_market_features


def main() -> None:
    df = read_parquet("data/processed/market")
    features = build_market_features(df)

    write_parquet(
        features,
        "data/processed/features/market",
        partition_cols=["symbol"],
    )

    print(f"Wrote features: shape={features.shape}")


if __name__ == "__main__":
    main()