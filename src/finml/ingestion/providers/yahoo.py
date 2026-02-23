from __future__ import annotations

import pandas as pd

from finml.ingestion.providers.base import MarketRequest


class YahooProvider:
    name = "yahoo"

    def fetch_market(self, req: MarketRequest) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as e:
            raise RuntimeError("Missing dependency: yfinance") from e

        if not req.symbols:
            return pd.DataFrame()

        df = yf.download(
            tickers=req.symbols,
            start=req.start,
            end=req.end,
            interval=req.interval,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )

        if df is None or len(df) == 0:
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []

        # MultiIndex columns when multiple tickers
        if isinstance(df.columns, pd.MultiIndex):
            tickers_present = set(df.columns.get_level_values(0))
            for sym in req.symbols:
                if sym not in tickers_present:
                    continue
                part = df[sym].copy()
                part = part.reset_index()  # index is Date/Datetime
                part["symbol"] = sym
                frames.append(part)
        else:
            # Single ticker
            part = df.copy().reset_index()
            part["symbol"] = req.symbols[0]
            frames.append(part)

        out = pd.concat(frames, ignore_index=True)

        # Standardize timestamp column name
        if "Date" in out.columns:
            out = out.rename(columns={"Date": "ts"})
        elif "Datetime" in out.columns:
            out = out.rename(columns={"Datetime": "ts"})

        return out
