from __future__ import annotations

from finml.ingestion.market import ingest_market
from finml.ingestion.providers.base import MarketRequest
from finml.ingestion.providers.yahoo import YahooProvider


def main() -> None:
    req = MarketRequest(
        symbols=["AAPL", "MSFT", "SPY"],
        start="2018-01-01",
        end="2026-01-01",
        interval="1d",
    )

    res = ingest_market(YahooProvider(), req, out_dir="data/processed/market")
    print(res)


if __name__ == "__main__":
    main()
