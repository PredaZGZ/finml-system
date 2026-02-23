from __future__ import annotations

from finml.training.dataset import build_dataset


def main() -> None:
    ds = build_dataset(
        label_col="y_fwdret_21",
        train_end="2022-12-31",
        val_end="2023-12-31",
    )

    print("train:", ds.X_train.shape, ds.y_train.shape)
    print("val:  ", ds.X_val.shape, ds.y_val.shape)
    print("test: ", ds.X_test.shape, ds.y_test.shape)


if __name__ == "__main__":
    main()