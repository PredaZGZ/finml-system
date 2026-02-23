from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def write_parquet(
    df: pd.DataFrame,
    path: str | Path,
    *,
    partition_cols: list[str] | None = None,
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if partition_cols:
        df.to_parquet(path, index=False, engine="pyarrow", partition_cols=partition_cols)
    else:
        (path / "data.parquet").parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path / "data.parquet", index=False, engine="pyarrow")


def read_parquet(
    path: str | Path,
    *,
    symbols: Iterable[str] | None = None,
) -> pd.DataFrame:
    path = Path(path)
    filters = None
    if symbols is not None:
        filters = [("symbol", "in", list(symbols))]
    return pd.read_parquet(path, engine="pyarrow", filters=filters)
