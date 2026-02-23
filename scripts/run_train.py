from __future__ import annotations

from finml.training.dataset import build_dataset
from finml.training.train import train_ridge_baseline


def main() -> None:
    ds = build_dataset(
        label_col="y_fwdret_21",
        train_end="2022-12-31",
        val_end="2023-12-31",
    )

    res = train_ridge_baseline(
        ds.X_train,
        ds.y_train,
        ds.X_val,
        ds.y_val,
        alpha=1.0,
        out_root="models/runs",
    )

    print(res)


if __name__ == "__main__":
    main()