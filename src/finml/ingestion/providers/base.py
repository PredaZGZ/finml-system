from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class MarketRequest:
    symbols: list[str]
    start: str | None = None   # "YYYY-MM-DD"
    end: str | None = None     # "YYYY-MM-DD"
    interval: str = "1d"       # provider-dependent (yfinance: "1d","1h","5m",...)


class MarketProvider(Protocol):
    name: str

    def fetch_market(self, req: MarketRequest) -> pd.DataFrame:
        """
        Return a provider-shaped DataFrame for all symbols.
        Must contain:
          - a timestamp column (Date/Datetime/ts) and
          - a symbol column (symbol)
          - OHLCV columns (Open/High/Low/Close/Volume) in some form
        """
        ...
