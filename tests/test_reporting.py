from __future__ import annotations

import json
from pathlib import Path

from autoresearch_timeseries_agent.reporting.build_project_summary import build_project_summary
from autoresearch_timeseries_agent.reporting.plots import generate_plots


def test_generate_plots_creates_expected_files(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    figures_dir = tmp_path / "figures"
    runs_dir.mkdir()
    _write_run(
        runs_dir / "linear.json",
        experiment_name="linear",
        model_name="linear",
        split_strategy="chronological",
        dataset_source="synthetic",
        train_rmse=0.8,
        val_rmse=1.0,
        test_rmse=1.2,
    )
    _write_run(
        runs_dir / "csv_linear.json",
        experiment_name="csv_linear",
        model_name="linear",
        split_strategy="chronological",
        dataset_source="csv",
        train_rmse=0.7,
        val_rmse=0.9,
        test_rmse=1.1,
    )

    paths = generate_plots(runs_dir=runs_dir, output_dir=figures_dir)

    expected = {
        "validation_rmse_by_experiment.png",
        "test_rmse_by_experiment.png",
        "train_val_test_rmse_by_experiment.png",
        "per_horizon_rmse_selected_runs.png",
        "actual_vs_predicted_samples.png",
    }
    assert {path.name for path in paths} == expected
    for filename in expected:
        assert (figures_dir / filename).exists()


def test_build_project_summary_creates_markdown_and_json(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    runs_dir.mkdir()
    _write_run(
        runs_dir / "chronological_linear.json",
        experiment_name="chronological_linear",
        model_name="linear",
        split_strategy="chronological",
        dataset_source="synthetic",
        train_rmse=0.8,
        val_rmse=2.0,
        test_rmse=2.5,
    )
    _write_run(
        runs_dir / "blocked_lstm.json",
        experiment_name="blocked_lstm",
        model_name="lstm",
        split_strategy="blocked_shuffle",
        dataset_source="synthetic",
        train_rmse=0.4,
        val_rmse=1.0,
        test_rmse=1.1,
    )
    _write_run(
        runs_dir / "csv_linear.json",
        experiment_name="csv_linear",
        model_name="linear",
        split_strategy="chronological",
        dataset_source="csv",
        train_rmse=0.7,
        val_rmse=1.5,
        test_rmse=1.6,
    )

    summary = build_project_summary(
        runs_dir=runs_dir,
        comparison_path=tmp_path / "missing_comparison.json",
        diagnostics_path=tmp_path / "missing_diagnostics.json",
        agent_report_path=tmp_path / "missing_agent.json",
        output_dir=reports_dir,
    )

    assert (reports_dir / "project_summary.md").exists()
    assert (reports_dir / "project_summary.json").exists()
    assert summary["agent_summary"]["available"] is False
    assert summary["best_realistic_chronological"]["experiment_name"] == "csv_linear"
    assert summary["best_diagnostic_blocked_shuffle"]["experiment_name"] == "blocked_lstm"
    markdown = (reports_dir / "project_summary.md").read_text(encoding="utf-8")
    assert "Best realistic chronological" in markdown
    assert "Best diagnostic blocked_shuffle" in markdown
    assert "deterministic local ingestion demo" in markdown


def _write_run(
    path: Path,
    *,
    experiment_name: str,
    model_name: str,
    split_strategy: str,
    dataset_source: str,
    train_rmse: float,
    val_rmse: float,
    test_rmse: float,
) -> None:
    payload = {
        "experiment_name": experiment_name,
        "model": {"name": model_name, "params": {}},
        "dataset": {"source": dataset_source, "split_strategy": split_strategy},
        "dataset_source": dataset_source,
        "split_strategy": split_strategy,
        "scale_features": model_name in {"lstm", "transformer"},
        "normalize_target": model_name in {"lstm", "transformer"},
        "metrics": {
            "train": _metrics(train_rmse),
            "val": _metrics(val_rmse),
            "test": _metrics(test_rmse),
        },
        "prediction_diagnostics": {
            "test": {
                "y_true_sample": [[1.0, 2.0, 3.0]],
                "y_pred_sample": [[1.1, 1.9, 3.2]],
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(rmse: float) -> dict[str, float | list[float]]:
    return {
        "rmse": rmse,
        "mae": rmse * 0.8,
        "mape": rmse * 2.0,
        "per_horizon_rmse": [rmse, rmse + 0.1, rmse + 0.2],
    }
