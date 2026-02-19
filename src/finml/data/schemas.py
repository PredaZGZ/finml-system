from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd


# Helpers (shared)
def _missing_cols(df: pd.DataFrame, cols: Sequence[str]) -> list[str]:
    return [c for c in cols if c not in df.columns]


def _ensure_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = _missing_cols(df, required)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _ensure_no_dupes(df: pd.DataFrame, key_cols: Sequence[str]) -> None:
    if df.duplicated(subset=list(key_cols)).any():
        dup = df[df.duplicated(subset=list(key_cols), keep=False)].sort_values(list(key_cols))
        raise ValueError(
            f"Duplicate rows found for key columns {list(key_cols)}.\n"
            f"Example duplicates:\n{dup.head(20)}"
        )


def _ensure_sorted(df: pd.DataFrame, by: Sequence[str]) -> None:
    # Enforce stable deterministic ordering (required for rolling/shift/merge_asof)
    sorted_idx = df.sort_values(list(by), kind="mergesort").index
    if not sorted_idx.equals(df.index):
        raise ValueError(f"DataFrame is not sorted by {list(by)}")


def _to_utc_datetime(series: pd.Series) -> pd.Series:
    """
    Normalize timestamps to tz-aware UTC.
    - If tz-naive: assume already UTC and localize.
    - If tz-aware: convert to UTC.
    """
    s = pd.to_datetime(series, errors="raise")
    # pandas stores tz on dtype; for tz-naive, .dt.tz is None
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize("UTC")
    else:
        s = s.dt.tz_convert("UTC")
    return s


def _cast_string(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string")


def _cast_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="raise")


def _strict_unknown_cols(df: pd.DataFrame, allowed: set[str]) -> None:
    unknown = [c for c in df.columns if c not in allowed]
    if unknown:
        raise ValueError(f"Unknown columns (strict mode): {unknown}")


def _enforce_nonnegative(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            if (df[c] < 0).any():
                bad = df[df[c] < 0][["symbol", c]].head(10)
                raise ValueError(f"Negative values found in column '{c}'. Example:\n{bad}")


# Market (OHLCV) schema
REQUIRED_MARKET_COLS: tuple[str, ...] = (
    "symbol",
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

OPTIONAL_MARKET_COLS: tuple[str, ...] = (
    "adj_close",  # adjusted close if provider supplies it
    "vwap",
    "bid",
    "ask",
    "spread",
    "iv",         # implied vol
    "rate",       # rate used (if embedded)
    "fx",         # fx rate used (if embedded)
    "provider",   # provenance (optional)
)


@dataclass(frozen=True)
class MarketSchema:
    """
    Canonical OHLCV contract.

    Key:
      (symbol, ts)

    Invariants:
      - ts tz-aware UTC
      - unique on key
      - sorted by key
      - numeric columns are numeric
    """

    required_cols: tuple[str, ...] = REQUIRED_MARKET_COLS
    optional_cols: tuple[str, ...] = OPTIONAL_MARKET_COLS
    key_cols: tuple[str, ...] = ("symbol", "ts")

    def validate(self, df: pd.DataFrame, *, strict: bool = False) -> pd.DataFrame:
        _ensure_columns(df, self.required_cols)

        allowed = set(self.required_cols) | set(self.optional_cols)
        if strict:
            _strict_unknown_cols(df, allowed)

        out = df.copy()

        _cast_string(out, ["symbol", "provider"])
        out["ts"] = _to_utc_datetime(out["ts"])

        _cast_numeric(out, ["open", "high", "low", "close", "volume"])
        _cast_numeric(out, self.optional_cols)

        _enforce_nonnegative(out, ["volume"])

        # Key checks
        _ensure_no_dupes(out, self.key_cols)

        # Deterministic ordering
        out = out.sort_values(list(self.key_cols), kind="mergesort").reset_index(drop=True)
        _ensure_sorted(out, self.key_cols)

        return out


# Fundamentals schema (availability-based)
REQUIRED_FUND_COLS: tuple[str, ...] = (
    "symbol",
    "period_end",    # accounting period end date
    "available_ts",  # when market can know it (filing/earnings publish time)
)

OPTIONAL_FUND_COLS: tuple[str, ...] = (
    # income statement
    "revenue",
    "gross_profit",
    "operating_income",
    "ebitda",
    "net_income",
    # balance sheet
    "assets",
    "liabilities",
    "equity",
    "debt",
    "cash",
    # share counts
    "shares_basic",
    "shares_diluted",
    # metadata
    "fiscal_year",
    "fiscal_quarter",
    "currency",
    "provider",
    "filing_type",
)


@dataclass(frozen=True)
class FundamentalsSchema:
    """
    Canonical fundamentals contract.

    Key:
      (symbol, available_ts)

    Invariants:
      - available_ts tz-aware UTC (anti-leakage anchor)
      - unique + sorted on key
      - period_end is date (informational; not used for alignment)
    """

    required_cols: tuple[str, ...] = REQUIRED_FUND_COLS
    optional_cols: tuple[str, ...] = OPTIONAL_FUND_COLS
    key_cols: tuple[str, ...] = ("symbol", "available_ts")

    def validate(self, df: pd.DataFrame, *, strict: bool = False) -> pd.DataFrame:
        _ensure_columns(df, self.required_cols)

        allowed = set(self.required_cols) | set(self.optional_cols)
        if strict:
            _strict_unknown_cols(df, allowed)

        out = df.copy()

        _cast_string(out, ["symbol", "currency", "provider", "filing_type"])
        out["available_ts"] = _to_utc_datetime(out["available_ts"])
        out["period_end"] = pd.to_datetime(out["period_end"], errors="raise").dt.date

        # numeric casts
        numeric_cols = (
            "revenue",
            "gross_profit",
            "operating_income",
            "ebitda",
            "net_income",
            "assets",
            "liabilities",
            "equity",
            "debt",
            "cash",
            "shares_basic",
            "shares_diluted",
            "fiscal_year",
            "fiscal_quarter",
        )
        _cast_numeric(out, [c for c in numeric_cols if c in out.columns])

        # Key checks + ordering
        _ensure_no_dupes(out, self.key_cols)
        out = out.sort_values(list(self.key_cols), kind="mergesort").reset_index(drop=True)
        _ensure_sorted(out, self.key_cols)

        return out


# Corporate actions schema (splits/dividends)
REQUIRED_CA_COLS: tuple[str, ...] = (
    "symbol",
    "effective_ts",  # when it takes effect in market data
    "action_type",   # "split" | "dividend" | "other"
)

OPTIONAL_CA_COLS: tuple[str, ...] = (
    # for splits: split ratio (e.g., 2.0 for 2-for-1), or numerator/denominator
    "split_ratio",
    "split_numerator",
    "split_denominator",
    # for dividends: cash dividend per share
    "dividend_cash",
    # metadata
    "provider",
)


@dataclass(frozen=True)
class CorporateActionsSchema:
    """
    Canonical corporate actions contract (optional for MVP).

    Key:
      (symbol, effective_ts, action_type) ideally unique.

    Invariants:
      - effective_ts tz-aware UTC
      - sorted by (symbol, effective_ts, action_type)
    """

    required_cols: tuple[str, ...] = REQUIRED_CA_COLS
    optional_cols: tuple[str, ...] = OPTIONAL_CA_COLS
    key_cols: tuple[str, ...] = ("symbol", "effective_ts", "action_type")

    def validate(self, df: pd.DataFrame, *, strict: bool = False) -> pd.DataFrame:
        _ensure_columns(df, self.required_cols)

        allowed = set(self.required_cols) | set(self.optional_cols)
        if strict:
            _strict_unknown_cols(df, allowed)

        out = df.copy()

        _cast_string(out, ["symbol", "action_type", "provider"])
        out["effective_ts"] = _to_utc_datetime(out["effective_ts"])

        _cast_numeric(
            out,
            [
                "split_ratio",
                "split_numerator",
                "split_denominator",
                "dividend_cash",
            ],
        )

        # Basic validation of action_type domain
        valid = {"split", "dividend", "other"}
        bad = set(out["action_type"].dropna().unique()) - valid
        if bad:
            raise ValueError(f"Invalid action_type values: {sorted(bad)} (valid: {sorted(valid)})")

        _ensure_no_dupes(out, self.key_cols)
        out = out.sort_values(list(self.key_cols), kind="mergesort").reset_index(drop=True)
        _ensure_sorted(out, self.key_cols)

        return out


# Convention helpers (features/labels)
def ensure_feature_convention(df: pd.DataFrame) -> None:
    """
    Optional convention:
      - Must have key cols (symbol, ts)
      - Feature columns should start with 'f_'
    """
    _ensure_columns(df, ("symbol", "ts"))
    feature_cols = [c for c in df.columns if c not in ("symbol", "ts")]
    bad = [c for c in feature_cols if not c.startswith("f_")]
    if bad:
        raise ValueError(f"Feature columns must start with 'f_'. Bad: {bad[:20]}")


def ensure_label_convention(df: pd.DataFrame) -> None:
    """
    Optional convention:
      - Must have key cols (symbol, ts)
      - Label columns should start with 'y_'
    """
    _ensure_columns(df, ("symbol", "ts"))
    label_cols = [c for c in df.columns if c not in ("symbol", "ts")]
    bad = [c for c in label_cols if not c.startswith("y_")]
    if bad:
        raise ValueError(f"Label columns must start with 'y_'. Bad: {bad[:20]}")


# Factory (singletons for importing)
MARKET_SCHEMA = MarketSchema()
FUNDAMENTALS_SCHEMA = FundamentalsSchema()
CORP_ACTIONS_SCHEMA = CorporateActionsSchema()


# Provider mapping validation utility
def validate_provider_mapping(mapping: Mapping[str, str], required_out_cols: Sequence[str]) -> None:
    """
    Helper to validate a provider->canonical column mapping.
    mapping: {provider_col: canonical_col}
    required_out_cols: canonical cols that must be produced.
    """
    produced = set(mapping.values())
    missing = [c for c in required_out_cols if c not in produced]
    if missing:
        raise ValueError(f"Provider mapping does not produce required canonical columns: {missing}")
