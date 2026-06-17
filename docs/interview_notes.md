# Interview Notes

## What This Project Proves

The project shows a reproducible forecasting benchmark stack that can be controlled by
configuration files and inspected through reports. It covers data generation/loading,
windowing, model training, validation-ranked comparison, diagnostics, plots, and a
deterministic local experiment agent.

## Engineering Tradeoffs

The implementation favors small, readable modules over framework-heavy orchestration.
Reports are written to JSON and Markdown so humans and future tools can consume the same
artifacts. The agent is intentionally deterministic and bounded because the benchmark
foundation should be trustworthy before any LLM layer is introduced.

## Why Linear Can Win

Ridge regression is a strong baseline for this project because the input window already
contains lagged history and correlated features. When the dynamics are smooth or mostly
linear, a flattened window can expose enough signal for linear regression to generalize
well. Small LSTM and Transformer configs can lose when data is limited, distribution
shift is present, or training budgets are intentionally small.

## What To Emphasize

- Chronological validation RMSE is the main model-selection metric.
- Test metrics are held out for final context.
- Blocked shuffle is diagnostic-only.
- The CSV example proves ingestion, not external benchmark quality.
- The local agent demonstrates experiment orchestration without adding external
  dependencies or code-writing behavior.

## Good Next Questions

- How would this behave on a real public dataset with documented provenance?
- Which failure mode dominates neural models: undertraining, scale sensitivity, or
  temporal distribution shift?
- What additional diagnostics would be needed before making deployment claims?
