from __future__ import annotations

from pathlib import Path

import pandas as pd

from finml.ingestion.market import ingest_market
from finml.ingestion.providers.base import MarketRequest
from finml.ingestion.providers.yahoo import YahooProvider


def load_universe(path: str | Path) -> list[str]:
    df = pd.read_csv(path)
    syms = df["symbol"].astype(str).str.strip().tolist()
    syms = [s for s in syms if s and s != "nan"]
    # dedupe manteniendo orden
    seen = set()
    out = []
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def chunks(xs: list[str], n: int) -> list[list[str]]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]


def main() -> None:
    symbols = load_universe("configs/universe_us_large.csv")

    provider = YahooProvider()

    for batch in chunks(symbols, 50):  # 25–100; 50 suele ser estable
        req = MarketRequest(
            symbols=batch,
            start="2018-01-01",
            end="2026-01-01",
            interval="1d",
        )
        res = ingest_market(provider, req, out_dir="data/processed/market")
        print(res)


if __name__ == "__main__":
    main()