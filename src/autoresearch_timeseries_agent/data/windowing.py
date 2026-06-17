from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class WindowedDataset:
    """Supervised forecasting windows for one time-series split."""

    X: FloatArray
    y: FloatArray

    def __post_init__(self) -> None:
        if self.X.ndim != 3:
            msg = f"X must have shape (num_windows, input_length, n_features); got {self.X.shape}"
            raise ValueError(msg)
        if self.y.ndim != 2:
            msg = f"y must have shape (num_windows, forecast_horizon); got {self.y.shape}"
            raise ValueError(msg)
        if self.X.shape[0] != self.y.shape[0]:
            msg = (
                "X and y must contain the same number of windows; "
                f"got X={self.X.shape[0]} and y={self.y.shape[0]}"
            )
            raise ValueError(msg)


def create_windows(
    series: NDArray[np.floating],
    *,
    input_length: int,
    forecast_horizon: int,
    target_column: int = 0,
) -> WindowedDataset:
    """Convert a multivariate series into supervised forecasting windows."""

    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 2:
        msg = (
            "series must be a 2D array with shape (n_timesteps, n_features); "
            f"got shape {values.shape}"
        )
        raise ValueError(msg)
    if input_length <= 0:
        msg = f"input_length must be positive; got {input_length}"
        raise ValueError(msg)
    if forecast_horizon <= 0:
        msg = f"forecast_horizon must be positive; got {forecast_horizon}"
        raise ValueError(msg)

    n_timesteps, n_features = values.shape
    if target_column < 0 or target_column >= n_features:
        msg = f"target_column must be in [0, {n_features - 1}]; got {target_column}"
        raise ValueError(msg)

    min_required = input_length + forecast_horizon
    if n_timesteps < min_required:
        msg = (
            "Cannot create windows: n_timesteps must be at least "
            f"input_length + forecast_horizon ({min_required}); got {n_timesteps}"
        )
        raise ValueError(msg)

    num_windows = n_timesteps - input_length - forecast_horizon + 1
    X = np.empty((num_windows, input_length, n_features), dtype=np.float64)
    y = np.empty((num_windows, forecast_horizon), dtype=np.float64)

    for start in range(num_windows):
        input_end = start + input_length
        forecast_end = input_end + forecast_horizon
        X[start] = values[start:input_end]
        y[start] = values[input_end:forecast_end, target_column]

    return WindowedDataset(X=X, y=y)
