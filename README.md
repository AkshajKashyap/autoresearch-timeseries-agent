# Autoresearch Time-Series Agent

Stateful AI experiment agent for multivariate time-series forecasting.

The long-term goal is to compare persistence, linear, LSTM, and Transformer forecasting
models while using a controlled agent loop to plan experiments, run configs, evaluate
metrics, and write reproducible reports.

This implementation intentionally remains non-agentic. There is no LangChain,
LangGraph, OpenAI integration, web app, Docker setup, or Transformer code yet.

## Baseline Experiments

Run the persistence baseline with:

```bash
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/persistence.yaml
```

Run the Ridge linear baseline with:

```bash
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/linear.yaml
```

Run the LSTM baseline with:

```bash
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/lstm.yaml
```

Inspect the dataset used by an experiment config with:

```bash
python -m autoresearch_timeseries_agent.training.inspect_dataset --config configs/lstm.yaml
```

Run the small controlled LSTM tuning configs with:

```bash
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/lstm_small.yaml
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/lstm_medium.yaml
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/lstm_longer_train.yaml
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/lstm_normalized.yaml
```

Compare saved runs with:

```bash
python -m autoresearch_timeseries_agent.training.compare_runs
```

The runner:

- generates deterministic synthetic multivariate time-series data
- creates chronological train/validation/test forecasting windows
- trains the persistence, Ridge linear, or LSTM baseline
- evaluates RMSE, MAE, MAPE, and per-horizon RMSE
- saves prediction diagnostics, including small prediction samples and residual stats
- writes run reports under `reports/runs/{experiment_name}.json` and
  `reports/runs/{experiment_name}.md`

The dataset inspection command writes `reports/dataset_diagnostics.json` and
`reports/dataset_diagnostics.md` with split sizes, target and feature summary stats,
naive persistence RMSE, and basic train-to-validation/test range warnings.

The comparison command reads JSON files in `reports/runs/`, writes
`reports/model_comparison.json` and `reports/model_comparison.md`, and ranks models by
validation RMSE. Test metrics are reported for final held-out evaluation context only;
they are not used to choose the best model.

## Synthetic Modes

The synthetic dataset supports `dataset.mode: linear` and `dataset.mode: nonlinear`.
The linear mode preserves the original trend, seasonality, noise, and correlated feature
behavior. The nonlinear mode keeps the data deterministic and learnable, but adds feature
interactions, lagged feature dependencies, a mild regime shift, changing seasonal
amplitude, and slightly higher noise.

Linear baselines may win on simple synthetic data. The LSTM baseline is included to test
sequence modeling behavior under harder dynamics, especially when
`dataset.mode: nonlinear` is enabled.

The LSTM is currently being debugged and tuned before adding a Transformer. A large gap
between train and validation/test RMSE can indicate distribution shift, undertraining,
or scale issues; the tuning configs and `training.normalize_target` option are intended
to separate those cases with small CPU-friendly runs.

## Planned System

Planner -> Config Writer -> Runner -> Evaluator -> Critic -> Report Writer

## Core Models

- Persistence baseline
- Linear/Ridge baseline
- LSTM baseline
- Transformer from scratch in PyTorch

## Core Metrics

- RMSE
- MAE
- MAPE
- Per-horizon error
- Training/runtime cost

## Status

Forecasting foundation implemented:

- Synthetic deterministic multivariate dataset
- Supervised windowing for first-feature forecasting
- Persistence and Ridge linear baselines
- PyTorch LSTM baseline
- Evaluation metrics
- YAML-driven experiment runner
- Reproducible run reports and validation-ranked model comparison
- Dataset and prediction diagnostics
- Pytest coverage for data, models, metrics, runner, and comparison smoke tests

Agent orchestration is planned but not implemented.
