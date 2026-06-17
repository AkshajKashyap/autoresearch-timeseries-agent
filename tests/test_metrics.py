from __future__ import annotations

import numpy as np
import pytest

from autoresearch_timeseries_agent.evaluation import mae, mape, per_horizon_rmse, rmse


def test_metric_functions() -> None:
    y_true = np.array([[1.0, 2.0], [3.0, 0.0]])
    y_pred = np.array([[1.0, 4.0], [1.0, 0.0]])

    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(2.0))
    assert mae(y_true, y_pred) == pytest.approx(1.0)
    assert mape(y_true, y_pred) == pytest.approx((0.0 + 1.0 + (2.0 / 3.0) + 0.0) / 4 * 100)
    np.testing.assert_allclose(per_horizon_rmse(y_true, y_pred), np.array([np.sqrt(2.0), np.sqrt(2.0)]))


def test_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        rmse(np.zeros((2, 2)), np.zeros((2, 3)))
