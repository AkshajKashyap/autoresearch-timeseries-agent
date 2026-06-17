from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas.api.types import is_numeric_dtype

from autoresearch_timeseries_agent.data.synthetic import FloatArray, TimeSeriesDatasetSplits
from autoresearch_timeseries_agent.data.windowing import WindowedDataset, create_windows


@dataclass(frozen=True)
class CsvDatasetConfig:
    source: str = "csv"
    path: str = "data/raw/example_timeseries.csv"
    timestamp_column: str | None = None
    target_column: str = "value"
    feature_columns: list[str] | None = None
    input_length: int = 48
    forecast_horizon: int = 24
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    split_strategy: str = "chronological"
    seed: int = 42


def make_csv_dataset(config: CsvDatasetConfig) -> TimeSeriesDatasetSplits:
    """Load a local CSV file and return train/val/test forecasting windows."""

    _validate_static_config(config)
    frame = _read_csv(config.path)
    if config.timestamp_column is not None:
        _validate_columns(frame, [config.timestamp_column], role="timestamp")
        frame = frame.sort_values(config.timestamp_column).reset_index(drop=True)

    selected_feature_columns = _select_feature_columns(frame, config)
    ordered_columns = [config.target_column, *selected_feature_columns]
    _validate_numeric_columns(frame, ordered_columns)
    _validate_no_missing(frame, ordered_columns)

    series = frame.loc[:, ordered_columns].to_numpy(dtype=np.float64)
    _validate_row_count(series.shape[0], config)
    metadata = _metadata(config, frame, selected_feature_columns)

    if config.split_strategy == "blocked_shuffle":
        return _make_blocked_shuffle_dataset(series, config, metadata)

    train_raw, val_raw, test_raw = _split_series(series, config)
    return TimeSeriesDatasetSplits(
        train=create_windows(
            train_raw,
            input_length=config.input_length,
            forecast_horizon=config.forecast_horizon,
        ),
        val=create_windows(
            val_raw,
            input_length=config.input_length,
            forecast_horizon=config.forecast_horizon,
        ),
        test=create_windows(
            test_raw,
            input_length=config.input_length,
            forecast_horizon=config.forecast_horizon,
        ),
        raw_series=series,
        metadata=metadata,
    )


def _read_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        msg = f"CSV dataset file does not exist: {csv_path}"
        raise FileNotFoundError(msg)
    frame = pd.read_csv(csv_path)
    if frame.empty:
        msg = f"CSV dataset is empty: {csv_path}"
        raise ValueError(msg)
    return frame


def _validate_static_config(config: CsvDatasetConfig) -> None:
    if config.source != "csv":
        msg = f"source must be 'csv'; got {config.source!r}"
        raise ValueError(msg)
    if config.split_strategy not in {"chronological", "blocked_shuffle"}:
        msg = (
            "split_strategy must be 'chronological' or 'blocked_shuffle'; "
            f"got {config.split_strategy!r}"
        )
        raise ValueError(msg)
    if config.input_length <= 0:
        msg = f"input_length must be positive; got {config.input_length}"
        raise ValueError(msg)
    if config.forecast_horizon <= 0:
        msg = f"forecast_horizon must be positive; got {config.forecast_horizon}"
        raise ValueError(msg)
    if not 0.0 < config.train_fraction < 1.0:
        msg = f"train_fraction must be between 0 and 1; got {config.train_fraction}"
        raise ValueError(msg)
    if not 0.0 <= config.val_fraction < 1.0:
        msg = f"val_fraction must be in [0, 1); got {config.val_fraction}"
        raise ValueError(msg)
    if config.train_fraction + config.val_fraction >= 1.0:
        msg = (
            "train_fraction + val_fraction must be less than 1; "
            f"got {config.train_fraction + config.val_fraction}"
        )
        raise ValueError(msg)


def _select_feature_columns(frame: pd.DataFrame, config: CsvDatasetConfig) -> list[str]:
    _validate_columns(frame, [config.target_column], role="target")

    if config.feature_columns is not None:
        _validate_columns(frame, config.feature_columns, role="feature")
        duplicates = {column for column in config.feature_columns if column == config.target_column}
        if duplicates:
            msg = (
                "feature_columns must not include target_column because the target is "
                f"inserted first internally; got {sorted(duplicates)}"
            )
            raise ValueError(msg)
        return list(config.feature_columns)

    excluded = {config.target_column}
    if config.timestamp_column is not None:
        excluded.add(config.timestamp_column)
    return [
        column
        for column in frame.columns
        if column not in excluded and is_numeric_dtype(frame[column])
    ]


def _validate_columns(frame: pd.DataFrame, columns: list[str], *, role: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        msg = f"Missing {role} column(s) in CSV dataset: {missing}"
        raise ValueError(msg)


def _validate_numeric_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    nonnumeric = [
        column for column in columns if not is_numeric_dtype(frame[column])
    ]
    if nonnumeric:
        msg = f"CSV target/features must be numeric; nonnumeric column(s): {nonnumeric}"
        raise ValueError(msg)


def _validate_no_missing(frame: pd.DataFrame, columns: list[str]) -> None:
    missing_counts = frame.loc[:, columns].isna().sum()
    missing = {
        column: int(count)
        for column, count in missing_counts.items()
        if int(count) > 0
    }
    if missing:
        msg = f"CSV target/features contain missing values: {missing}"
        raise ValueError(msg)


def _validate_row_count(row_count: int, config: CsvDatasetConfig) -> None:
    min_required = config.input_length + config.forecast_horizon
    if row_count < min_required:
        msg = (
            "CSV dataset has insufficient rows: row_count must be at least "
            f"input_length + forecast_horizon ({min_required}); got {row_count}"
        )
        raise ValueError(msg)


def _split_series(
    series: FloatArray,
    config: CsvDatasetConfig,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    n_timesteps = series.shape[0]
    train_end = int(n_timesteps * config.train_fraction)
    val_end = train_end + int(n_timesteps * config.val_fraction)
    split_lengths = {
        "train": train_end,
        "val": val_end - train_end,
        "test": n_timesteps - val_end,
    }

    min_required = config.input_length + config.forecast_horizon
    too_short = [
        f"{name}={length}" for name, length in split_lengths.items() if length < min_required
    ]
    if too_short:
        msg = (
            "Each CSV split must contain at least input_length + forecast_horizon "
            f"timesteps ({min_required}); got {', '.join(too_short)}"
        )
        raise ValueError(msg)

    return series[:train_end], series[train_end:val_end], series[val_end:]


def _make_blocked_shuffle_dataset(
    series: FloatArray,
    config: CsvDatasetConfig,
    metadata: dict[str, Any],
) -> TimeSeriesDatasetSplits:
    windows = create_windows(
        series,
        input_length=config.input_length,
        forecast_horizon=config.forecast_horizon,
    )
    num_windows = windows.X.shape[0]
    train_count = int(num_windows * config.train_fraction)
    val_count = int(num_windows * config.val_fraction)
    test_count = num_windows - train_count - val_count
    if min(train_count, val_count, test_count) <= 0:
        msg = (
            "blocked_shuffle produced an empty CSV split; increase rows or adjust "
            f"fractions (train={train_count}, val={val_count}, test={test_count})"
        )
        raise ValueError(msg)

    indices = np.random.default_rng(config.seed).permutation(num_windows)
    train_idx = indices[:train_count]
    val_idx = indices[train_count : train_count + val_count]
    test_idx = indices[train_count + val_count :]
    return TimeSeriesDatasetSplits(
        train=_subset_windows(windows, train_idx),
        val=_subset_windows(windows, val_idx),
        test=_subset_windows(windows, test_idx),
        raw_series=series,
        metadata=metadata,
    )


def _subset_windows(dataset: WindowedDataset, indices: NDArray[np.integer]) -> WindowedDataset:
    return WindowedDataset(X=dataset.X[indices], y=dataset.y[indices])


def _metadata(
    config: CsvDatasetConfig,
    frame: pd.DataFrame,
    selected_feature_columns: list[str],
) -> dict[str, Any]:
    return {
        "source": "csv",
        "path": config.path,
        "row_count": int(frame.shape[0]),
        "columns": list(frame.columns),
        "timestamp_column": config.timestamp_column,
        "target_column": config.target_column,
        "feature_columns": config.feature_columns,
        "selected_feature_columns": selected_feature_columns,
        "ordered_columns": [config.target_column, *selected_feature_columns],
        "n_features": len(selected_feature_columns) + 1,
        "input_length": config.input_length,
        "forecast_horizon": config.forecast_horizon,
        "split_strategy": config.split_strategy,
    }
