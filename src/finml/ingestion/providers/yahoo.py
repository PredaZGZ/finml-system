from __future__ import annotations

import time
import pandas as pd

from finml.ingestion.providers.base import MarketRequest


class YahooProvider:
    name = "yahoo"

    def _download(self, symbols: list[str], req: MarketRequest) -> pd.DataFrame:
        import yfinance as yf
        return yf.download(
            tickers=symbols,
            start=req.start,
            end=req.end,
            interval=req.interval,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )

    def fetch_market(self, req: MarketRequest) -> pd.DataFrame:
        try:
            import yfinance as yf  # noqa: F401
        except ImportError as e:
            raise RuntimeError("Missing dependency: yfinance") from e

        if not req.symbols:
            return pd.DataFrame()

        df = self._download(req.symbols, req)

        frames: list[pd.DataFrame] = []

        def add_one(sym: str, part: pd.DataFrame) -> None:
            part = part.copy().reset_index()
            part["symbol"] = sym
            frames.append(part)

        if isinstance(df.columns, pd.MultiIndex):
            present = set(df.columns.get_level_values(0))
            missing = [s for s in req.symbols if s not in present]
            for sym in req.symbols:
                if sym in present:
                    add_one(sym, df[sym])

            # fallback a individual
            for sym in missing:
                time.sleep(0.2)
                one = self._download([sym], req)
                if one is None or len(one) == 0:
                    continue
                add_one(sym, one)

        else:
            # single ticker response
            add_one(req.symbols[0], df)

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True)

        if "Date" in out.columns:
            out = out.rename(columns={"Date": "ts"})
        elif "Datetime" in out.columns:
            out = out.rename(columns={"Datetime": "ts"})

        return out