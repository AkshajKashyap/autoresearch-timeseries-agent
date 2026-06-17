from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from autoresearch_timeseries_agent.data.windowing import WindowedDataset, create_windows


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    source: str = "synthetic"
    mode: str = "linear"
    split_strategy: str = "chronological"
    n_timesteps: int = 600
    n_features: int = 4
    input_length: int = 48
    forecast_horizon: int = 24
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    seed: int = 42


@dataclass(frozen=True)
class TimeSeriesDatasetSplits:
    train: WindowedDataset
    val: WindowedDataset
    test: WindowedDataset
    raw_series: FloatArray
    metadata: dict[str, Any] = field(default_factory=dict)


def generate_synthetic_series(config: SyntheticDatasetConfig) -> FloatArray:
    """Generate deterministic correlated multivariate time-series data."""

    _validate_config(config)
    rng = np.random.default_rng(config.seed)
    time = np.arange(config.n_timesteps, dtype=np.float64)

    trend = 0.015 * time
    daily = 2.0 * np.sin(2.0 * np.pi * time / 24.0)
    slow = 0.8 * np.cos(2.0 * np.pi * time / 96.0)
    common_noise = rng.normal(loc=0.0, scale=0.15, size=config.n_timesteps)
    latent = trend + daily + slow + common_noise

    features = np.empty((config.n_timesteps, config.n_features), dtype=np.float64)
    for feature_idx in range(config.n_features):
        phase = feature_idx * np.pi / max(config.n_features, 1)
        period = 24.0 + 6.0 * feature_idx
        feature_seasonal = np.sin(2.0 * np.pi * time / period + phase)
        lag = min(feature_idx * 2, config.n_timesteps - 1)
        lagged_latent = np.roll(latent, lag)
        if lag > 0:
            lagged_latent[:lag] = latent[0]
        feature_noise = rng.normal(
            loc=0.0,
            scale=0.08 + 0.02 * feature_idx,
            size=config.n_timesteps,
        )
        features[:, feature_idx] = (
            (1.0 - 0.06 * feature_idx) * latent
            + 0.15 * lagged_latent
            + 0.35 * feature_seasonal
            + 20.0
            + 0.2 * feature_idx
            + feature_noise
        )

    if config.mode == "nonlinear":
        features = _apply_nonlinear_effects(features, time, rng)

    return features


def make_synthetic_dataset(config: SyntheticDatasetConfig) -> TimeSeriesDatasetSplits:
    """Generate synthetic data and return train/val/test windows."""

    series = generate_synthetic_series(config)
    if config.split_strategy == "blocked_shuffle":
        return _make_blocked_shuffle_dataset(series, config)

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
        metadata=_metadata(config, row_count=series.shape[0]),
    )


def _validate_config(config: SyntheticDatasetConfig) -> None:
    if config.source != "synthetic":
        msg = f"source must be 'synthetic'; got {config.source!r}"
        raise ValueError(msg)
    if config.mode not in {"linear", "nonlinear"}:
        msg = f"mode must be 'linear' or 'nonlinear'; got {config.mode!r}"
        raise ValueError(msg)
    if config.split_strategy not in {"chronological", "blocked_shuffle"}:
        msg = (
            "split_strategy must be 'chronological' or 'blocked_shuffle'; "
            f"got {config.split_strategy!r}"
        )
        raise ValueError(msg)
    if config.n_timesteps <= 0:
        msg = f"n_timesteps must be positive; got {config.n_timesteps}"
        raise ValueError(msg)
    if config.n_features <= 0:
        msg = f"n_features must be positive; got {config.n_features}"
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


def _split_series(
    series: FloatArray,
    config: SyntheticDatasetConfig,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    train_end = int(config.n_timesteps * config.train_fraction)
    val_end = train_end + int(config.n_timesteps * config.val_fraction)
    split_lengths = {
        "train": train_end,
        "val": val_end - train_end,
        "test": config.n_timesteps - val_end,
    }

    min_required = config.input_length + config.forecast_horizon
    too_short = [
        f"{name}={length}" for name, length in split_lengths.items() if length < min_required
    ]
    if too_short:
        msg = (
            "Each split must contain at least input_length + forecast_horizon "
            f"timesteps ({min_required}); got {', '.join(too_short)}"
        )
        raise ValueError(msg)

    return series[:train_end], series[train_end:val_end], series[val_end:]


def _make_blocked_shuffle_dataset(
    series: FloatArray,
    config: SyntheticDatasetConfig,
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
            "blocked_shuffle produced an empty split; increase n_timesteps or adjust "
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
        metadata=_metadata(config, row_count=series.shape[0]),
    )


def _subset_windows(dataset: WindowedDataset, indices: NDArray[np.integer]) -> WindowedDataset:
    return WindowedDataset(X=dataset.X[indices], y=dataset.y[indices])


def _apply_nonlinear_effects(
    features: FloatArray,
    time: FloatArray,
    rng: np.random.Generator,
) -> FloatArray:
    nonlinear = features.copy()
    centered = features - features.mean(axis=0, keepdims=True)
    amplitude = 1.0 + 0.25 * np.sin(2.0 * np.pi * time / 180.0)
    regime_shift = np.where(time >= time[-1] * 0.55, 0.9, -0.2)

    nonlinear[:, 0] = (
        features[:, 0] * amplitude
        + 0.035 * centered[:, 0] ** 2
        + 0.35 * regime_shift
    )

    if features.shape[1] > 1:
        lagged_target = np.roll(centered[:, 0], 3)
        lagged_target[:3] = centered[0, 0]
        nonlinear[:, 1] = (
            features[:, 1]
            + 0.18 * lagged_target
            + 0.06 * centered[:, 0] * centered[:, 1]
        )

    for feature_idx in range(2, features.shape[1]):
        lag = min(feature_idx + 2, features.shape[0] - 1)
        lagged_previous = np.roll(centered[:, feature_idx - 1], lag)
        if lag > 0:
            lagged_previous[:lag] = centered[0, feature_idx - 1]
        nonlinear[:, feature_idx] = (
            features[:, feature_idx]
            + 0.12 * np.tanh(lagged_previous)
            + 0.15 * np.sin(centered[:, 0] / 3.0)
        )

    nonlinear += rng.normal(loc=0.0, scale=0.06, size=features.shape)
    return nonlinear


def _metadata(config: SyntheticDatasetConfig, *, row_count: int) -> dict[str, Any]:
    return {
        "source": "synthetic",
        "mode": config.mode,
        "split_strategy": config.split_strategy,
        "row_count": row_count,
        "n_timesteps": config.n_timesteps,
        "n_features": config.n_features,
        "input_length": config.input_length,
        "forecast_horizon": config.forecast_horizon,
        "target_column": "feature_0",
        "selected_feature_columns": [f"feature_{idx}" for idx in range(config.n_features)],
    }
