# Autoresearch Time-Series Agent

![CI](https://img.shields.io/badge/CI-make%20check-blue)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)

Autoresearch Time-Series Agent is a reproducible local forecasting benchmark that turns
time-series configs into train/validation/test metrics, diagnostics, plots, project
summaries, and bounded deterministic agent experiments. It is designed to show the
engineering foundation for future autonomous research workflows without adding external
APIs, web apps, deployment, or LLM orchestration prematurely.

## Current Status

Implemented: synthetic and CSV data sources, supervised windowing, persistence/Ridge/
LSTM/Transformer models, chronological and blocked-shuffle splits, train-only scaling,
target normalization, diagnostics, comparison reports, plots, project summary, repo
health checks, GitHub Actions CI, and a deterministic local experiment agent.

Not implemented: OpenAI, LangChain, LangGraph, external APIs, Streamlit, FastAPI, Docker,
production deployment, new model families, or code-writing agents.

## Headline Results

| Category | Experiment | Model | Val RMSE | Test RMSE | Note |
| --- | --- | --- | ---: | ---: | --- |
| Best realistic chronological | `linear` | Ridge linear | 0.5919 | 0.8092 | Main model-selection signal |
| Best diagnostic blocked_shuffle | `lstm_blocked_shuffle` | LSTM | 0.5533 | 0.5875 | Diagnostic-only |
| Best CSV ingestion demo | `csv_linear` | Ridge linear | 0.6969 | 0.6506 | Local demo CSV only |

Validation RMSE drives model selection. Test metrics are held out for final context.
`blocked_shuffle` is diagnostic-only and is not a deployment estimate. The included CSV
is an ingestion demo, not an external benchmark claim.

## Quickstart

```bash
pip install -e ".[dev]"
make check
python scripts/create_example_csv_dataset.py
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/linear.yaml
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/csv_linear.yaml
make reports
```

## Core Commands

```bash
# Experiments
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/persistence.yaml
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/linear.yaml
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/lstm.yaml
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/transformer_scaled.yaml

# CSV demo
python scripts/create_example_csv_dataset.py
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/csv_linear.yaml
python -m autoresearch_timeseries_agent.training.run_experiment --config configs/csv_transformer.yaml

# Diagnostics and reports
python -m autoresearch_timeseries_agent.training.inspect_dataset --config configs/csv_linear.yaml
python -m autoresearch_timeseries_agent.training.compare_runs
python -m autoresearch_timeseries_agent.reporting.generate_plots
python -m autoresearch_timeseries_agent.reporting.build_project_summary
python -m autoresearch_timeseries_agent.reporting.check_repo_health

# Deterministic local agent
python -m autoresearch_timeseries_agent.agents.run_agent --objective "Improve chronological validation RMSE under a small CPU budget"
python -m autoresearch_timeseries_agent.agents.run_agent --objective "Improve chronological validation RMSE on CSV data under a small CPU budget" --base-config configs/csv_linear.yaml
```

Makefile shortcuts:

```bash
make check
make plots
make summary
make health
make reports
```

`make reports` refreshes comparison, figures, project summary, and repo health from
existing run JSON files. It does not retrain models.

## Repo Structure

```text
configs/                              Experiment configs
docs/                                 Architecture, methods, reproducibility notes
scripts/create_example_csv_dataset.py Local CSV demo generator
src/autoresearch_timeseries_agent/    Package source
tests/                                Fast pytest suite
reports/                              Generated metrics, figures, and summaries
```

## Deterministic Agent

The current agent is a fixed local loop:

Planner -> Config Writer -> Runner -> Evaluator -> Critic -> Report Writer

It writes generated configs under `configs/agent_generated/`, runs existing experiment
commands, refreshes comparisons, and writes `reports/agent/*`. It does not edit source
code, call external APIs, or use LLM orchestration.

## What This Proves

- A forecasting benchmark can stay reproducible, report-driven, and CI-checked.
- Strong linear baselines can outperform small neural models on structured time-series
  data.
- Chronological validation is the right default for realistic forecasting claims.
- A bounded local agent can orchestrate experiments safely before adding any LLM layer.

## Limitations

- Synthetic data is useful for controlled development, not real-world proof.
- The CSV file is a deterministic local ingestion demo, not an external benchmark.
- `blocked_shuffle` is diagnostic-only.
- Neural model configs are small CPU-friendly baselines, not exhaustive tuning.
- There is no production deployment or external data benchmark yet.

## Docs And Reports

Docs:

- [Architecture](docs/architecture.md)
- [Method Notes](docs/method_notes.md)
- [Reproducibility](docs/reproducibility.md)
- [Interview Notes](docs/interview_notes.md)
- [Limitations](docs/limitations.md)
- [Repository Description](docs/repo_description.md)

Generated reports:

- [Project Summary](reports/project_summary.md)
- [Model Comparison](reports/model_comparison.md)
- [Repo Health Check](reports/repo_health_check.md)
- [Final Portfolio Review](reports/final_portfolio_review.md)
- `reports/figures/*.png`
