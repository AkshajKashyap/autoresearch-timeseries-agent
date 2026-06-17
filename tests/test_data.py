from __future__ import annotations

import numpy as np
import pytest

from autoresearch_timeseries_agent.data import (
    SyntheticDatasetConfig,
    create_windows,
    generate_synthetic_series,
    make_synthetic_dataset,
)


def test_synthetic_data_is_deterministic_with_same_seed() -> None:
    config = SyntheticDatasetConfig(
        n_timesteps=120,
        n_features=3,
        input_length=12,
        forecast_horizon=6,
        train_fraction=0.6,
        val_fraction=0.2,
        seed=123,
    )

    first = generate_synthetic_series(config)
    second = generate_synthetic_series(config)

    np.testing.assert_array_equal(first, second)


def test_nonlinear_synthetic_data_is_deterministic() -> None:
    config = SyntheticDatasetConfig(
        mode="nonlinear",
        n_timesteps=120,
        n_features=3,
        input_length=12,
        forecast_horizon=6,
        train_fraction=0.6,
        val_fraction=0.2,
        seed=123,
    )

    first = generate_synthetic_series(config)
    second = generate_synthetic_series(config)

    np.testing.assert_array_equal(first, second)


def test_nonlinear_mode_differs_from_linear_mode() -> None:
    linear_config = SyntheticDatasetConfig(
        mode="linear",
        n_timesteps=120,
        n_features=3,
        input_length=12,
        forecast_horizon=6,
        train_fraction=0.6,
        val_fraction=0.2,
        seed=123,
    )
    nonlinear_config = SyntheticDatasetConfig(
        mode="nonlinear",
        n_timesteps=120,
        n_features=3,
        input_length=12,
        forecast_horizon=6,
        train_fraction=0.6,
        val_fraction=0.2,
        seed=123,
    )

    linear = generate_synthetic_series(linear_config)
    nonlinear = generate_synthetic_series(nonlinear_config)

    assert nonlinear.shape == linear.shape
    assert not np.array_equal(nonlinear, linear)


def test_window_shapes() -> None:
    series = np.arange(40 * 3, dtype=np.float64).reshape(40, 3)

    dataset = create_windows(series, input_length=8, forecast_horizon=4)

    assert dataset.X.shape == (29, 8, 3)
    assert dataset.y.shape == (29, 4)
    np.testing.assert_array_equal(dataset.X[0], series[:8])
    np.testing.assert_array_equal(dataset.y[0], series[8:12, 0])


def test_invalid_window_errors() -> None:
    series = np.arange(10 * 2, dtype=np.float64).reshape(10, 2)

    with pytest.raises(ValueError, match="n_timesteps must be at least"):
        create_windows(series, input_length=8, forecast_horizon=4)
    with pytest.raises(ValueError, match="series must be a 2D array"):
        create_windows(np.arange(10, dtype=np.float64), input_length=3, forecast_horizon=2)
    with pytest.raises(ValueError, match="forecast_horizon must be positive"):
        create_windows(series, input_length=3, forecast_horizon=0)


def test_make_synthetic_dataset_returns_windowed_splits() -> None:
    config = SyntheticDatasetConfig(
        n_timesteps=150,
        n_features=4,
        input_length=10,
        forecast_horizon=5,
        train_fraction=0.6,
        val_fraction=0.2,
        seed=7,
    )

    splits = make_synthetic_dataset(config)

    assert splits.raw_series.shape == (150, 4)
    assert splits.train.X.shape[1:] == (10, 4)
    assert splits.val.y.shape[1] == 5
    assert splits.test.y.shape[1] == 5
