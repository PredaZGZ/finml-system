from __future__ import annotations

from finml.data.io import read_parquet, write_parquet
from finml.labels.forward_returns import build_forward_return_labels


def main() -> None:
    mkt = read_parquet("data/processed/market")
    labels = build_forward_return_labels(mkt, horizons=[5, 21])

    write_parquet(labels, "data/processed/labels", partition_cols=["symbol"])
    print(f"Wrote labels: shape={labels.shape}")


if __name__ == "__main__":
    main()