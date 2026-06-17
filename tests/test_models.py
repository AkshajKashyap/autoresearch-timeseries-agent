from __future__ import annotations

import numpy as np
import torch

from autoresearch_timeseries_agent.models import LSTMForecaster, LinearBaseline, PersistenceBaseline


def test_persistence_baseline_output_shape_and_values() -> None:
    X = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        ]
    )
    y = np.zeros((2, 4), dtype=np.float64)

    model = PersistenceBaseline().fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == (2, 4)
    np.testing.assert_array_equal(predictions[0], np.array([3.0, 3.0, 3.0, 3.0]))
    np.testing.assert_array_equal(predictions[1], np.array([6.0, 6.0, 6.0, 6.0]))


def test_linear_baseline_output_shape() -> None:
    rng = np.random.default_rng(12)
    X = rng.normal(size=(20, 5, 3))
    y = rng.normal(size=(20, 2))

    model = LinearBaseline(alpha=0.5).fit(X, y)
    predictions = model.predict(X[:4])

    assert predictions.shape == (4, 2)


def test_lstm_forecaster_forward_output_shape() -> None:
    model = LSTMForecaster(
        n_features=3,
        forecast_horizon=4,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
    )
    X = torch.randn(5, 12, 3)

    predictions = model(X)

    assert predictions.shape == (5, 4)
