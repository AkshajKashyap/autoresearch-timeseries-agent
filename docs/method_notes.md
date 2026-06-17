# Method Notes

## Baselines

`PersistenceBaseline` repeats the last observed target value across the full forecast
horizon. It is intentionally simple and catches whether a model beats a strong naive
forecast.

`LinearBaseline` flattens the input window and fits Ridge regression to predict the full
horizon directly. It is a strong baseline when the synthetic or CSV dynamics are mostly
linear and the feature set is informative.

`LSTMForecaster` uses a compact PyTorch LSTM and predicts the full horizon from the final
sequence representation.

`TransformerForecaster` uses a small PyTorch encoder with positional encoding and a
direct multi-horizon forecasting head.

## Splits

`chronological` is the realistic evaluation strategy. Train windows come first in time,
then validation, then test. Validation RMSE from chronological runs is the main model
selection signal.

`blocked_shuffle` is diagnostic-only. It creates windows and deterministically shuffles
them before splitting. It helps isolate model capacity from temporal distribution shift,
but it should not be used for deployment-style claims.

## Scaling and Normalization

Feature scaling and target normalization are fit from the train split only. Neural
models can use both options through config. Predictions are unnormalized before metrics
are computed.

## Model Selection

The comparison report ranks models by validation RMSE, not test RMSE. Test metrics are
reported as held-out evaluation context after selection.

## Why Linear Can Beat Neural Models

The current data is deterministic and relatively structured. Ridge regression can be
very competitive when lagged windows encode enough information and the dynamics are
mostly smooth. Small neural models can underperform when there is limited data, temporal
shift, short training, or scale sensitivity.
