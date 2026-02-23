from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from finml.data.io import write_parquet
from finml.data.schemas import MARKET_SCHEMA
from finml.ingestion.providers.base import MarketProvider, MarketRequest


@dataclass(frozen=True)
class MarketIngestionResult:
    provider: str
    symbols: list[str]
    n_rows_raw: int
    n_rows_canonical: int
    out_dir: str


def normalize_market_df(raw: pd.DataFrame, *, provider: str) -> pd.DataFrame:
    """
    Map provider-shaped DataFrame to canonical MarketFrame columns:
      symbol, ts, open, high, low, close, volume (+ optional adj_close, provider)

    This function only renames/selects columns and attaches provenance.
    Type normalization, UTC conversion, sorting, and dedupe happens in schema.validate().
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume", "provider"])

    df = raw.copy()

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={c: c.lower() for c in df.columns})

    # Common mappings
    rename = {
        "date": "ts",
        "datetime": "ts",
        "timestamp": "ts",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj close": "adj_close",   # yfinance after lower()
        "adj_close": "adj_close",
        "volume": "volume",
        "symbol": "symbol",
    }
    df = df.rename(columns=rename)

    required = {"symbol", "ts", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Provider '{provider}' missing required columns after normalize: {missing}. "
            f"Columns present: {sorted(df.columns.tolist())}"
        )

    df["provider"] = provider

    keep = ["symbol", "ts", "open", "high", "low", "close", "volume", "provider"]
    if "adj_close" in df.columns:
        keep.append("adj_close")

    return df[keep]


def ingest_market(
    provider: MarketProvider,
    req: MarketRequest,
    *,
    out_dir: str | Path = "data/processed/market",
) -> MarketIngestionResult:
    raw = provider.fetch_market(req)
    n_raw = int(len(raw))

    canonical = normalize_market_df(raw, provider=provider.name)
    print(canonical.dtypes)
    print(canonical["provider"].head())
    canonical = MARKET_SCHEMA.validate(canonical, strict=False)
    canonical = canonical.drop_duplicates(subset=["symbol", "ts"], keep="last")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Partition by symbol to keep reads cheap later
    write_parquet(canonical, out_path, partition_cols=["symbol"])

    return MarketIngestionResult(
        provider=provider.name,
        symbols=req.symbols,
        n_rows_raw=n_raw,
        n_rows_canonical=int(len(canonical)),
        out_dir=str(out_path),
    )
