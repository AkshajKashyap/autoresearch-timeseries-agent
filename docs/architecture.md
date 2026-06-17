# Architecture

The project is organized as a local forecasting benchmark pipeline:

1. Data source
2. Windowing
3. Model training
4. Evaluation metrics
5. Reports and plots
6. Deterministic experiment agent

## Data Sources

`src/autoresearch_timeseries_agent/data/` contains dataset loaders.

The synthetic loader creates deterministic multivariate series with trend, seasonality,
correlated features, nonlinear effects, and controlled distribution shift. The CSV
loader reads a local CSV file, optionally sorts by timestamp, validates numeric target
and feature columns, and orders the target as the first internal feature.

Both sources return the same `TimeSeriesDatasetSplits` shape: train, validation, test,
raw series, and metadata.

## Windowing

`create_windows` converts each multivariate sequence into supervised forecasting
examples:

- `X`: `(num_windows, input_length, n_features)`
- `y`: `(num_windows, forecast_horizon)`

The current target is always the first feature.

## Models

`src/autoresearch_timeseries_agent/models/` contains the forecasting models:

- `PersistenceBaseline`
- `LinearBaseline`
- `LSTMForecaster`
- `TransformerForecaster`

The experiment runner builds models from YAML configs and trains only the models that
need fitting.

## Metrics

`src/autoresearch_timeseries_agent/evaluation/` computes RMSE, MAE, MAPE, and
per-horizon RMSE. `run_experiment` evaluates train, validation, and test splits and
writes JSON and Markdown run reports.

## Reports

`src/autoresearch_timeseries_agent/reporting/` turns saved JSON reports into:

- comparison plots
- project summary reports
- repository health checks

Reports are intentionally file-based so future tools can inspect them without coupling
to internal Python objects.

## Agent Loop

The current agent is deterministic and local. It does not call LLMs or external APIs.

Its loop is:

Planner -> Config Writer -> Runner -> Evaluator -> Critic -> Report Writer

The agent writes generated configs only under `configs/agent_generated/`, calls the
existing experiment runner, refreshes comparisons, and writes reports under
`reports/agent/`.
