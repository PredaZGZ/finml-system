from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from finml.data.io import read_parquet


@dataclass(frozen=True)
class DatasetSplits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    meta_train: pd.DataFrame
    meta_val: pd.DataFrame
    meta_test: pd.DataFrame


def build_dataset(
    *,
    features_path: str = "data/processed/features/market",
    labels_path: str = "data/processed/labels",
    label_col: str = "y_fwdret_21",
    train_end: str = "2022-12-31",
    val_end: str = "2023-12-31",
) -> DatasetSplits:
    feats = read_parquet(features_path)
    labels = read_parquet(labels_path)

    df = feats.merge(labels, on=["symbol", "ts"], how="inner").sort_values(["symbol", "ts"])

    feature_cols = [c for c in df.columns if c.startswith("f_")]
    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' not found. Available labels: {[c for c in df.columns if c.startswith('y_')]}")

    # drop rows where label is missing (last horizons) or features missing
    df = df.dropna(subset=feature_cols + [label_col]).reset_index(drop=True)

    meta = df[["symbol", "ts"]].copy()
    X = df[feature_cols].copy()
    y = df[label_col].copy()

    # time split
    ts = pd.to_datetime(meta["ts"], utc=True)
    train_mask = ts <= pd.Timestamp(train_end, tz="UTC")
    val_mask = (ts > pd.Timestamp(train_end, tz="UTC")) & (ts <= pd.Timestamp(val_end, tz="UTC"))
    test_mask = ts > pd.Timestamp(val_end, tz="UTC")

    return DatasetSplits(
        X_train=X.loc[train_mask].reset_index(drop=True),
        y_train=y.loc[train_mask].reset_index(drop=True),
        X_val=X.loc[val_mask].reset_index(drop=True),
        y_val=y.loc[val_mask].reset_index(drop=True),
        X_test=X.loc[test_mask].reset_index(drop=True),
        y_test=y.loc[test_mask].reset_index(drop=True),
        meta_train=meta.loc[train_mask].reset_index(drop=True),
        meta_val=meta.loc[val_mask].reset_index(drop=True),
        meta_test=meta.loc[test_mask].reset_index(drop=True),
    )