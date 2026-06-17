# Autoresearch Time-Series Agent

Stateful AI experiment agent for multivariate time-series forecasting.

The long-term goal is to compare persistence, linear, LSTM, and Transformer forecasting
models while using a controlled agent loop to plan experiments, run configs, evaluate
metrics, and write reproducible reports.

This first pass intentionally implements only the non-agent forecasting foundation.
There is no LangChain, LangGraph, OpenAI integration, web app, Docker setup, LSTM, or
Transformer code yet.

## Baseline Experiments

Run the persistence baseline with:

```bash
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/persistence.yaml
```

Run the Ridge linear baseline with:

```bash
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/linear.yaml
```

Compare saved runs with:

```bash
python -m autoresearch_timeseries_agent.training.compare_runs
```

The runner:

- generates deterministic synthetic multivariate time-series data
- creates chronological train/validation/test forecasting windows
- trains either the persistence or Ridge linear baseline
- evaluates RMSE, MAE, MAPE, and per-horizon RMSE
- writes run reports under `reports/runs/{experiment_name}.json` and
  `reports/runs/{experiment_name}.md`

The comparison command reads JSON files in `reports/runs/`, writes
`reports/model_comparison.json` and `reports/model_comparison.md`, and ranks models by
validation RMSE. Test metrics are reported for final held-out evaluation context only;
they are not used to choose the best model.

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
- Evaluation metrics
- YAML-driven experiment runner
- Reproducible run reports and validation-ranked model comparison
- Pytest coverage for data, models, metrics, runner, and comparison smoke tests

Agent orchestration is planned but not implemented.
