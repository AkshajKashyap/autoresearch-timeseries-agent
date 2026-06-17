# Autoresearch Time-Series Agent

Stateful AI experiment agent for multivariate time-series forecasting.

The long-term goal is to compare persistence, linear, LSTM, and Transformer forecasting
models while using a controlled agent loop to plan experiments, run configs, evaluate
metrics, and write reproducible reports.

This implementation intentionally remains non-agentic. There is no LangChain,
LangGraph, OpenAI integration, web app, or Docker setup yet.

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

Inspect and run the scaled LSTM diagnostics with:

```bash
python -m autoresearch_timeseries_agent.training.inspect_dataset --config configs/lstm_scaled.yaml
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/lstm_scaled.yaml
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/lstm_blocked_shuffle.yaml
```

Run the from-scratch PyTorch Transformer baseline with:

```bash
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/transformer_scaled.yaml
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/transformer_blocked_shuffle.yaml
```

Compare saved runs with:

```bash
python -m autoresearch_timeseries_agent.training.compare_runs
```

Run the deterministic local experiment agent with:

```bash
python -m autoresearch_timeseries_agent.agents.run_agent --objective "Improve chronological validation RMSE under a small CPU budget"
```

The runner:

- generates deterministic synthetic multivariate time-series data
- creates chronological train/validation/test forecasting windows
- trains the persistence, Ridge linear, LSTM, or Transformer baseline
- supports chronological and blocked-shuffle window splitting
- can scale features and normalize targets from train-split statistics only
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

## Deterministic Agent

The current agent is a controlled local experiment orchestrator, not an LLM agent. It
uses a fixed rule-based state machine:

Planner -> Config Writer -> Runner -> Evaluator -> Critic -> Report Writer

It reads existing comparison and dataset diagnostics reports when available, proposes at
most three safe experiments, writes generated configs only under
`configs/agent_generated/`, runs them through the existing experiment runner, refreshes
the comparison report, and writes `reports/agent/agent_plan.*` and
`reports/agent/agent_final_report.*`.

The agent does not edit source code, does not call OpenAI or external APIs, and does not
use LangChain, LangGraph, Streamlit, FastAPI, or Docker. It currently supports only
bounded config changes for `linear`, `lstm`, and `transformer` experiments.

## Synthetic Modes

The synthetic dataset supports `dataset.mode: linear` and `dataset.mode: nonlinear`.
The linear mode preserves the original trend, seasonality, noise, and correlated feature
behavior. The nonlinear mode keeps the data deterministic and learnable, but adds feature
interactions, lagged feature dependencies, a mild regime shift, changing seasonal
amplitude, and slightly higher noise.

It also supports `dataset.split_strategy: chronological` and
`dataset.split_strategy: blocked_shuffle`. Chronological splitting is the realistic
forecasting setup: train windows come first in time, then validation, then test. This can
expose distribution shift, which is often exactly what a deployment forecast must handle.
Blocked shuffle creates windows chronologically, then deterministically shuffles and
splits windows by seed. That is diagnostic only, not a deployment estimate, but it helps
separate model capacity from pure extrapolation difficulty.

Linear baselines may win on simple synthetic data. The LSTM baseline is included to test
sequence modeling behavior under harder dynamics, especially when
`dataset.mode: nonlinear` is enabled.

The Transformer baseline is a compact from-scratch PyTorch encoder model with input
projection, sinusoidal positional encoding, Transformer encoder blocks, last-token
pooling, and a direct multi-horizon forecasting head. It is intended as a CPU-practical
baseline, not a research-grade architecture.

A large gap between train and validation/test RMSE can indicate distribution shift,
undertraining, or scale issues; the tuning configs plus `training.scale_features` and
`training.normalize_target` are intended to separate those cases with small CPU-friendly
runs. Blocked-shuffle runs are diagnostic only. Realistic model claims should focus on
chronological results.

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
- PyTorch Transformer encoder baseline
- Evaluation metrics
- YAML-driven experiment runner
- Reproducible run reports and validation-ranked model comparison
- Dataset and prediction diagnostics
- Deterministic rule-based experiment agent
- Pytest coverage for data, models, metrics, runner, and comparison smoke tests

LLM-based agent orchestration is planned but not implemented.
