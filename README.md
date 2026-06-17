# Autoresearch Time-Series Agent

Stateful AI experiment agent for multivariate time-series forecasting.

The goal is to compare persistence, linear, LSTM, and Transformer forecasting models while using a controlled agent loop to plan experiments, run configs, evaluate metrics, and write reproducible reports.

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

Initial scaffold.
