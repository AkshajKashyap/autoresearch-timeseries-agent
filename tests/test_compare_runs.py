from __future__ import annotations

import json
from pathlib import Path

from autoresearch_timeseries_agent.training.compare_runs import compare_runs


def test_compare_runs_smoke_and_validation_ranking(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _write_run(
        runs_dir / "low_test_high_val.json",
        experiment_name="low_test_high_val",
        model_name="persistence",
        val_rmse=3.0,
        test_rmse=1.0,
    )
    _write_run(
        runs_dir / "high_test_low_val.json",
        experiment_name="high_test_low_val",
        model_name="linear",
        val_rmse=2.0,
        test_rmse=4.0,
    )

    comparison = compare_runs(runs_dir=runs_dir, output_dir=tmp_path)

    assert comparison["best_experiment"] == "high_test_low_val"
    assert comparison["runs"][0]["val_rmse"] == 2.0
    assert comparison["runs"][0]["test_rmse"] == 4.0
    assert (tmp_path / "model_comparison.json").exists()
    assert (tmp_path / "model_comparison.md").exists()


def _write_run(
    path: Path,
    *,
    experiment_name: str,
    model_name: str,
    val_rmse: float,
    test_rmse: float,
) -> None:
    payload = {
        "experiment_name": experiment_name,
        "model": {"name": model_name, "params": {}},
        "metrics": {
            "train": {"rmse": val_rmse, "mae": 1.0, "mape": 1.0, "per_horizon_rmse": [1.0]},
            "val": {"rmse": val_rmse, "mae": 1.0, "mape": 1.0, "per_horizon_rmse": [1.0]},
            "test": {"rmse": test_rmse, "mae": 1.0, "mape": 1.0, "per_horizon_rmse": [1.0]},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
