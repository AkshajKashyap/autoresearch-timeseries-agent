# Reproducibility

This project is designed to reproduce reports from local commands and checked-in
configs. No external APIs are required.

## Environment

```bash
pip install -e ".[dev]"
```

CI uses Python 3.11 and runs:

```bash
make check
```

## Core Checks

```bash
make check
```

## Generate Example CSV

The CSV file is ignored by git and can be regenerated deterministically:

```bash
python scripts/create_example_csv_dataset.py
```

## Run Core Experiments

```bash
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/persistence.yaml
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/linear.yaml
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/lstm.yaml
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/transformer_scaled.yaml
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/csv_linear.yaml
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/csv_transformer.yaml
```

## Diagnostics

```bash
python -m autoresearch_timeseries_agent.training.inspect_dataset --config configs/linear.yaml
python -m autoresearch_timeseries_agent.training.inspect_dataset --config configs/csv_linear.yaml
```

## Reports

```bash
make reports
```

This refreshes model comparison, figures, project summary, and repository health report
from existing run JSON files.

## Agent Run

```bash
python -m autoresearch_timeseries_agent.agents.run_agent --objective "Improve chronological validation RMSE under a small CPU budget"
python -m autoresearch_timeseries_agent.agents.run_agent --objective "Improve chronological validation RMSE on CSV data under a small CPU budget" --base-config configs/csv_linear.yaml
```

The agent writes generated configs under `configs/agent_generated/` and reports under
`reports/agent/`.
