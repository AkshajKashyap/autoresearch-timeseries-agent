from typing import TypeAlias

from autoresearch_timeseries_agent.data.csv_loader import CsvDatasetConfig, make_csv_dataset
from autoresearch_timeseries_agent.data.synthetic import (
    SyntheticDatasetConfig,
    TimeSeriesDatasetSplits,
    generate_synthetic_series,
    make_synthetic_dataset,
)
from autoresearch_timeseries_agent.data.windowing import WindowedDataset, create_windows

DatasetConfig: TypeAlias = SyntheticDatasetConfig | CsvDatasetConfig


def make_dataset(config: DatasetConfig) -> TimeSeriesDatasetSplits:
    if isinstance(config, SyntheticDatasetConfig):
        return make_synthetic_dataset(config)
    if isinstance(config, CsvDatasetConfig):
        return make_csv_dataset(config)
    msg = f"Unsupported dataset config type: {type(config).__name__}"
    raise TypeError(msg)


__all__ = [
    "CsvDatasetConfig",
    "DatasetConfig",
    "SyntheticDatasetConfig",
    "TimeSeriesDatasetSplits",
    "WindowedDataset",
    "create_windows",
    "generate_synthetic_series",
    "make_csv_dataset",
    "make_dataset",
    "make_synthetic_dataset",
]
