from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoresearch_timeseries_agent.data import CsvDatasetConfig, make_csv_dataset


def test_csv_loader_happy_path(tmp_path: Path) -> None:
    csv_path = _write_example_csv(tmp_path)
    config = CsvDatasetConfig(
        path=str(csv_path),
        timestamp_column="timestamp",
        target_column="value",
        feature_columns=["feature_a", "feature_b"],
        input_length=8,
        forecast_horizon=4,
        train_fraction=0.6,
        val_fraction=0.2,
    )

    splits = make_csv_dataset(config)

    assert splits.train.X.shape[1:] == (8, 3)
    assert splits.train.y.shape[1] == 4
    assert splits.metadata["source"] == "csv"
    assert splits.metadata["target_column"] == "value"
    assert splits.metadata["selected_feature_columns"] == ["feature_a", "feature_b"]
    np.testing.assert_array_equal(splits.raw_series[:3, 0], np.array([10.0, 11.0, 12.0]))


def test_csv_loader_inferrs_numeric_features(tmp_path: Path) -> None:
    csv_path = _write_example_csv(tmp_path)
    config = CsvDatasetConfig(
        path=str(csv_path),
        timestamp_column="timestamp",
        target_column="value",
        feature_columns=None,
        input_length=8,
        forecast_horizon=4,
        train_fraction=0.6,
        val_fraction=0.2,
    )

    splits = make_csv_dataset(config)

    assert splits.metadata["selected_feature_columns"] == [
        "feature_a",
        "feature_b",
        "feature_c",
    ]
    assert splits.raw_series.shape[1] == 4


def test_csv_loader_missing_target_column_error(tmp_path: Path) -> None:
    csv_path = _write_example_csv(tmp_path)
    config = CsvDatasetConfig(
        path=str(csv_path),
        target_column="missing",
        input_length=8,
        forecast_horizon=4,
        train_fraction=0.6,
        val_fraction=0.2,
    )

    with pytest.raises(ValueError, match="Missing target column"):
        make_csv_dataset(config)


def test_csv_loader_insufficient_rows_error(tmp_path: Path) -> None:
    csv_path = _write_example_csv(tmp_path, n_rows=10)
    config = CsvDatasetConfig(
        path=str(csv_path),
        timestamp_column="timestamp",
        target_column="value",
        input_length=8,
        forecast_horizon=4,
        train_fraction=0.6,
        val_fraction=0.2,
    )

    with pytest.raises(ValueError, match="insufficient rows"):
        make_csv_dataset(config)


def test_csv_loader_rejects_nonnumeric_requested_features(tmp_path: Path) -> None:
    csv_path = _write_example_csv(tmp_path)
    config = CsvDatasetConfig(
        path=str(csv_path),
        timestamp_column="timestamp",
        target_column="value",
        feature_columns=["feature_a", "label"],
        input_length=8,
        forecast_horizon=4,
        train_fraction=0.6,
        val_fraction=0.2,
    )

    with pytest.raises(ValueError, match="nonnumeric"):
        make_csv_dataset(config)


def _write_example_csv(tmp_path: Path, *, n_rows: int = 80) -> Path:
    time = np.arange(n_rows, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="h"),
            "value": 10.0 + time,
            "feature_a": 2.0 * time,
            "feature_b": np.sin(time / 3.0),
            "feature_c": np.cos(time / 5.0),
            "label": ["ignored"] * n_rows,
        }
    )
    path = tmp_path / "local_timeseries.csv"
    frame.iloc[::-1].to_csv(path, index=False)
    return path
