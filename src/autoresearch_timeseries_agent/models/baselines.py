from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge


FloatArray = NDArray[np.float64]


class PersistenceBaseline:
    """Predict the last observed target value across the full horizon."""

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> PersistenceBaseline:
        _, targets = _validate_training_arrays(X, y)
        self.forecast_horizon_ = targets.shape[1]
        return self

    def predict(self, X: NDArray[np.floating], forecast_horizon: int | None = None) -> FloatArray:
        values = _validate_input_array(X)
        if forecast_horizon is None:
            if not hasattr(self, "forecast_horizon_"):
                msg = "forecast_horizon is required before PersistenceBaseline.fit has been called"
                raise ValueError(msg)
            forecast_horizon = self.forecast_horizon_
        if forecast_horizon <= 0:
            msg = f"forecast_horizon must be positive; got {forecast_horizon}"
            raise ValueError(msg)

        last_target = values[:, -1, 0]
        return np.repeat(last_target[:, np.newaxis], forecast_horizon, axis=1)


class LinearBaseline:
    """Ridge regression over flattened input windows for direct multi-step forecasting."""

    def __init__(self, *, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> LinearBaseline:
        values, targets = _validate_training_arrays(X, y)
        self.input_shape_ = values.shape[1:]
        self.forecast_horizon_ = targets.shape[1]
        self.model.fit(_flatten_windows(values), targets)
        return self

    def predict(self, X: NDArray[np.floating]) -> FloatArray:
        values = _validate_input_array(X)
        if not hasattr(self, "input_shape_"):
            msg = "LinearBaseline must be fit before predict is called"
            raise ValueError(msg)
        if values.shape[1:] != self.input_shape_:
            msg = f"X window shape must be {self.input_shape_}; got {values.shape[1:]}"
            raise ValueError(msg)
        return np.asarray(self.model.predict(_flatten_windows(values)), dtype=np.float64)


def _flatten_windows(X: FloatArray) -> FloatArray:
    return X.reshape(X.shape[0], X.shape[1] * X.shape[2])


def _validate_input_array(X: NDArray[np.floating]) -> FloatArray:
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 3:
        msg = f"X must have shape (num_windows, input_length, n_features); got {values.shape}"
        raise ValueError(msg)
    if values.shape[0] == 0:
        msg = "X must contain at least one window"
        raise ValueError(msg)
    if values.shape[1] == 0 or values.shape[2] == 0:
        msg = f"X input_length and n_features must be positive; got {values.shape}"
        raise ValueError(msg)
    return values


def _validate_training_arrays(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
) -> tuple[FloatArray, FloatArray]:
    values = _validate_input_array(X)
    targets = np.asarray(y, dtype=np.float64)
    if targets.ndim != 2:
        msg = f"y must have shape (num_windows, forecast_horizon); got {targets.shape}"
        raise ValueError(msg)
    if values.shape[0] != targets.shape[0]:
        msg = (
            "X and y must contain the same number of windows; "
            f"got X={values.shape[0]} and y={targets.shape[0]}"
        )
        raise ValueError(msg)
    if targets.shape[1] == 0:
        msg = "y forecast_horizon must be positive"
        raise ValueError(msg)
    return values, targets
