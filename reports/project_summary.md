# Project Summary

## Project Goal

Build a reproducible, local time-series forecasting benchmark foundation that a future agent layer can control through configs and report files.

## Current Capabilities

- Dataset sources: `synthetic, csv`
- Models: `persistence, linear, lstm, transformer`
- Metrics: RMSE, MAE, MAPE, per-horizon RMSE

## Evaluation Design

- Chronological: Train, validation, and test windows stay ordered in time. This is the realistic evaluation mode for forecasting claims.
- Blocked shuffle: Windows are created then deterministically shuffled into splits. This is diagnostic-only and helps separate model capacity from temporal shift.
- Caveat: Model selection should use validation RMSE from chronological runs. Test metrics are held out for final context, and blocked_shuffle runs are diagnostic-only.

## Headline Results

- Best realistic chronological: `linear` (linear, val RMSE=0.5919, test RMSE=0.8092).
- Best diagnostic blocked_shuffle: `lstm_blocked_shuffle` (lstm, val RMSE=0.5533, test RMSE=0.5875).

## CSV Results

| Experiment | Model | Split | Val RMSE | Test RMSE |
| --- | --- | --- | ---: | ---: |
| csv_linear | linear | chronological | 0.6969 | 0.6506 |
| agent_csv_linear_alpha_0_1_chronological | linear | chronological | 0.7668 | 0.6980 |
| csv_transformer | transformer | chronological | 1.0149 | 2.3703 |
| agent_csv_transformer_small_chronological | transformer | chronological | 1.0614 | 3.3126 |
| agent_csv_lstm_small_chronological | lstm | chronological | 1.3601 | 5.8869 |

The included CSV is a deterministic local ingestion demo, not an external benchmark.

## Agent Summary

Objective: Improve chronological validation RMSE on CSV data under a small CPU budget. Generated 3 config(s). Best generated run: `agent_csv_linear_alpha_0_1_chronological`. Critique: The objective was: Improve chronological validation RMSE on CSV data under a small CPU budget. The best generated run is agent_csv_linear_alpha_0_1_chronological with validation RMSE 0.7668; it is a realistic chronological result. The best generated run did not improve over the previous chronological best by 0.1750 validation RMSE. The overall comparison winner is lstm_blocked_shuffle (a diagnostic blocked-shuffle result).

## Dataset Diagnostics

Latest diagnostics source: `csv` with target `value` and no warnings.

## Limitations

- The included CSV is a deterministic local ingestion demo, not an external benchmark.
- Blocked-shuffle results are diagnostic-only and should not be treated as deployment estimates.
- Neural configs are intentionally small CPU-friendly baselines, not exhaustive tuning.
- The current local agent is deterministic and config-bounded; it is not an LLM agent.

## Next Steps

- Keep model selection focused on chronological validation RMSE.
- Add real external datasets only after documenting provenance and evaluation assumptions.
- Use the plotting and project summary outputs as the default GitHub-facing report layer.
- Consider future agent or benchmark expansion only after real-data evaluation assumptions are clear.
