from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """Direct multi-horizon forecaster using the final LSTM hidden state."""

    def __init__(
        self,
        *,
        n_features: int,
        forecast_horizon: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_features <= 0:
            msg = f"n_features must be positive; got {n_features}"
            raise ValueError(msg)
        if forecast_horizon <= 0:
            msg = f"forecast_horizon must be positive; got {forecast_horizon}"
            raise ValueError(msg)
        if hidden_size <= 0:
            msg = f"hidden_size must be positive; got {hidden_size}"
            raise ValueError(msg)
        if num_layers <= 0:
            msg = f"num_layers must be positive; got {num_layers}"
            raise ValueError(msg)

        self.n_features = n_features
        self.forecast_horizon = forecast_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 3:
            msg = f"X must have shape (batch, input_length, n_features); got {tuple(X.shape)}"
            raise ValueError(msg)
        if X.shape[2] != self.n_features:
            msg = f"Expected {self.n_features} features; got {X.shape[2]}"
            raise ValueError(msg)

        _, (hidden, _) = self.encoder(X.float())
        return self.head(hidden[-1])
