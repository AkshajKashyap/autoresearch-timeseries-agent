from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for fixed-length time-series windows."""

    def __init__(self, *, d_model: int, max_length: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X + self.encoding[:, : X.shape[1]]


class TransformerForecaster(nn.Module):
    """Small direct multi-horizon Transformer encoder forecaster."""

    def __init__(
        self,
        *,
        n_features: int,
        forecast_horizon: int,
        input_length: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_features <= 0:
            msg = f"n_features must be positive; got {n_features}"
            raise ValueError(msg)
        if forecast_horizon <= 0:
            msg = f"forecast_horizon must be positive; got {forecast_horizon}"
            raise ValueError(msg)
        if input_length <= 0:
            msg = f"input_length must be positive; got {input_length}"
            raise ValueError(msg)
        if d_model <= 0:
            msg = f"d_model must be positive; got {d_model}"
            raise ValueError(msg)
        if nhead <= 0 or d_model % nhead != 0:
            msg = f"nhead must be positive and divide d_model; got d_model={d_model}, nhead={nhead}"
            raise ValueError(msg)
        if num_layers <= 0:
            msg = f"num_layers must be positive; got {num_layers}"
            raise ValueError(msg)

        self.n_features = n_features
        self.forecast_horizon = forecast_horizon
        self.input_length = input_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.input_projection = nn.Linear(n_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_length=input_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, forecast_horizon),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 3:
            msg = f"X must have shape (batch, input_length, n_features); got {tuple(X.shape)}"
            raise ValueError(msg)
        if X.shape[1] != self.input_length:
            msg = f"Expected input_length {self.input_length}; got {X.shape[1]}"
            raise ValueError(msg)
        if X.shape[2] != self.n_features:
            msg = f"Expected {self.n_features} features; got {X.shape[2]}"
            raise ValueError(msg)

        encoded = self.input_projection(X.float())
        encoded = self.positional_encoding(encoded)
        encoded = self.encoder(encoded)
        return self.head(encoded[:, -1, :])
