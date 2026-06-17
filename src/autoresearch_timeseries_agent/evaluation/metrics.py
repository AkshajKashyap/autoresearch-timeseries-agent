from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def rmse(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> float:
    actual, predicted = _validate_metric_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def mae(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> float:
    actual, predicted = _validate_metric_arrays(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))


def mape(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    *,
    epsilon: float = 1e-8,
) -> float:
    actual, predicted = _validate_metric_arrays(y_true, y_pred)
    if epsilon <= 0:
        msg = f"epsilon must be positive; got {epsilon}"
        raise ValueError(msg)
    denominator = np.maximum(np.abs(actual), epsilon)
    return float(np.mean(np.abs((actual - predicted) / denominator)) * 100.0)


def per_horizon_rmse(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> FloatArray:
    actual, predicted = _validate_metric_arrays(y_true, y_pred)
    return np.sqrt(np.mean(np.square(actual - predicted), axis=0))


def evaluate_forecast(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> dict[str, float | list[float]]:
    horizon_rmse = per_horizon_rmse(y_true, y_pred)
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "per_horizon_rmse": horizon_rmse.tolist(),
    }


def _validate_metric_arrays(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> tuple[FloatArray, FloatArray]:
    actual = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    if actual.ndim != 2:
        msg = f"y_true must have shape (num_windows, forecast_horizon); got {actual.shape}"
        raise ValueError(msg)
    if predicted.ndim != 2:
        msg = f"y_pred must have shape (num_windows, forecast_horizon); got {predicted.shape}"
        raise ValueError(msg)
    if actual.shape != predicted.shape:
        msg = f"y_true and y_pred must have identical shapes; got {actual.shape} and {predicted.shape}"
        raise ValueError(msg)
    if actual.shape[0] == 0 or actual.shape[1] == 0:
        msg = f"metric arrays must be non-empty; got {actual.shape}"
        raise ValueError(msg)
    return actual, predicted
