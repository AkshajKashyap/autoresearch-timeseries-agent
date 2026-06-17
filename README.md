# Autoresearch Time-Series Agent

Stateful AI experiment agent for multivariate time-series forecasting.

The long-term goal is to compare persistence, linear, LSTM, and Transformer forecasting
models while using a controlled agent loop to plan experiments, run configs, evaluate
metrics, and write reproducible reports.

This first pass intentionally implements only the non-agent forecasting foundation.
There is no LangChain, LangGraph, OpenAI integration, web app, Docker setup, LSTM, or
Transformer code yet.

## Baseline Command

Run the current synthetic-data baseline pipeline with:

```bash
python -m autoresearch_timeseries_agent.training.run_baseline --config configs/baseline.yaml
```

The runner:

- generates deterministic synthetic multivariate time-series data
- creates chronological train/validation/test forecasting windows
- trains either the persistence or Ridge linear baseline
- evaluates RMSE, MAE, MAPE, and per-horizon RMSE
- writes `reports/baseline_results.json` and `reports/baseline_report.md`

Use `model.name: persistence` or `model.name: linear` in `configs/baseline.yaml`.

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
- YAML-driven baseline runner
- Pytest coverage for data, models, metrics, and runner smoke test

Agent orchestration is planned but not implemented.
