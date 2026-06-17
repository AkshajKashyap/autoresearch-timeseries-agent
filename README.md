# Autoresearch Time-Series Agent

Reproducible local benchmark tooling for multivariate time-series forecasting, with a
deterministic experiment agent that plans and runs bounded config variants.

## Current Status

The repo currently supports synthetic and local CSV datasets, persistence/Ridge/LSTM/
Transformer baselines, chronological and blocked-shuffle splits, diagnostics, comparison
reports, plots, and a deterministic local experiment agent. It does not include OpenAI,
LangChain, LangGraph, external APIs, Streamlit, FastAPI, Docker, or code-writing agents.

## Quickstart

```bash
make check
python scripts/create_example_csv_dataset.py
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/linear.yaml
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/csv_linear.yaml
python -m autoresearch_timeseries_agent.training.compare_runs
python -m autoresearch_timeseries_agent.reporting.generate_plots
python -m autoresearch_timeseries_agent.reporting.build_project_summary
```

The generated reports live under `reports/`. Figures are saved under `reports/figures/`.

## Headline Results

Current saved reports show:

| Category | Experiment | Model | Val RMSE | Test RMSE | Note |
| --- | --- | --- | ---: | ---: | --- |
| Best realistic chronological | `linear` | Ridge linear | 0.5919 | 0.8092 | Use for model-selection claims |
| Best diagnostic blocked_shuffle | `lstm_blocked_shuffle` | LSTM | 0.5533 | 0.5875 | Diagnostic-only |
| Best CSV ingestion demo | `csv_linear` | Ridge linear | 0.6969 | 0.6506 | Local demo CSV only |
| CSV Transformer | `csv_transformer` | Transformer | 1.0149 | 2.3703 | Small CPU config |

Model selection should be based on validation RMSE. Test metrics are held out for final
evaluation context. `blocked_shuffle` is diagnostic-only and should not be treated as a
deployment estimate.

The included CSV is a deterministic local ingestion demo, not an external benchmark or a
claim about real-world forecasting performance.

## Key Commands

Run synthetic baselines:

```bash
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/persistence.yaml
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/linear.yaml
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/lstm.yaml
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/transformer_scaled.yaml
```

Generate and run CSV-backed experiments:

```bash
python scripts/create_example_csv_dataset.py
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/csv_linear.yaml
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/csv_transformer.yaml
```

Inspect datasets:

```bash
python -m autoresearch_timeseries_agent.training.inspect_dataset --config configs/lstm.yaml
python -m autoresearch_timeseries_agent.training.inspect_dataset --config configs/csv_linear.yaml
```

Compare runs:

```bash
python -m autoresearch_timeseries_agent.training.compare_runs
```

Generate plots:

```bash
python -m autoresearch_timeseries_agent.reporting.generate_plots
```

Build the GitHub-facing project summary:

```bash
python -m autoresearch_timeseries_agent.reporting.build_project_summary
```

Run the deterministic local experiment agent:

```bash
python -m autoresearch_timeseries_agent.agents.run_agent --objective "Improve chronological validation RMSE under a small CPU budget"
python -m autoresearch_timeseries_agent.agents.run_agent --objective "Improve chronological validation RMSE on CSV data under a small CPU budget" --base-config configs/csv_linear.yaml
```

Makefile shortcuts:

```bash
make check
make plots
make summary
make reports
```

`make reports` refreshes model comparison, plot PNGs, and the project summary from
existing run JSON files. It does not retrain models.

## Supported Data

Synthetic data supports:

- deterministic multivariate generation
- `dataset.mode: linear`
- `dataset.mode: nonlinear`
- trend, seasonality, correlated features, lagged dependencies, regime shift, and noise

CSV data supports:

- `dataset.source: csv`
- optional timestamp sorting
- configurable target column
- inferred numeric feature columns when `feature_columns: null`
- target-first internal feature ordering

Example CSV generation:

```bash
python scripts/create_example_csv_dataset.py
```

## Evaluation Design

`chronological` keeps train, validation, and test windows ordered in time. This is the
realistic forecasting setup and the right basis for model-selection claims.

`blocked_shuffle` creates windows, deterministically shuffles them by seed, and then
splits into train/validation/test. This is useful for diagnosing model capacity versus
temporal distribution shift, but it is not a deployment-style estimate.

## Deterministic Agent

The current agent is a controlled local experiment orchestrator, not an LLM agent. It
uses fixed planner/config-writer/runner/evaluator/critic/report-writer stages, writes
generated configs only under `configs/agent_generated/`, and calls the existing
experiment runner and comparison tools.

The agent does not edit source code, call external APIs, or use OpenAI/LangChain/
LangGraph. With `--base-config`, it preserves the base dataset block and generates
bounded model/config variants.

## Reports

Important outputs:

- `reports/runs/{experiment_name}.json`
- `reports/runs/{experiment_name}.md`
- `reports/model_comparison.json`
- `reports/model_comparison.md`
- `reports/figures/*.png`
- `reports/project_summary.json`
- `reports/project_summary.md`
- `reports/dataset_diagnostics.json`
- `reports/dataset_diagnostics.md`
- `reports/agent/agent_final_report.json`
- `reports/agent/agent_final_report.md`

## Limitations

- The included CSV is a local ingestion demo, not an external benchmark.
- `blocked_shuffle` is diagnostic-only.
- Neural models use small CPU-friendly configs, not exhaustive tuning.
- LSTM and Transformer results can lag linear models on simple or mostly linear data.
- The current agent is deterministic and config-bounded; LLM-based orchestration is not
  implemented.

## Planned Direction

The next useful layer is stronger reporting around real datasets and evaluation
assumptions. More model or agent complexity should come after the benchmark reports stay
clear, reproducible, and honest.
