from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrainResult:
    run_id: str
    model_path: str
    metrics_path: str


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    e = y_true - y_pred
    return float(np.sqrt(np.mean(e * e)))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def train_ridge_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    alpha: float = 1.0,
    out_root: str | Path = "models/runs",
) -> TrainResult:
    try:
        from sklearn.linear_model import Ridge
    except ImportError as e:
        raise RuntimeError("Missing dependency: scikit-learn") from e

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train.values, y_train.values)

    pred_train = model.predict(X_train.values)
    pred_val = model.predict(X_val.values)

    metrics = {
        "alpha": alpha,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "rmse_train": _rmse(y_train.values, pred_train),
        "rmse_val": _rmse(y_val.values, pred_val),
        "ic_train": _corr(y_train.values, pred_train),
        "ic_val": _corr(y_val.values, pred_val),
        "feature_names": list(X_train.columns),
    }

    # save model
    import joblib

    model_path = out_dir / "model.joblib"
    joblib.dump(model, model_path)

    # save metrics
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return TrainResult(
        run_id=run_id,
        model_path=str(model_path),
        metrics_path=str(metrics_path),
    )