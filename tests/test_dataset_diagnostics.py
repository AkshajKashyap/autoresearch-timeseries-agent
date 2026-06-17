from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from autoresearch_timeseries_agent.training.inspect_dataset import inspect_dataset


def test_dataset_diagnostics_output_exists_and_has_expected_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "diagnostics.yaml"
    output_dir = tmp_path / "reports"
    config_path.write_text(
        f"""
experiment:
  name: diagnostics_test
dataset:
  name: synthetic
  mode: nonlinear
  split_strategy: chronological
  n_timesteps: 144
  n_features: 3
  input_length: 12
  forecast_horizon: 6
  train_fraction: 0.6
  val_fraction: 0.2
  seed: 9
model:
  name: lstm
  hidden_size: 8
  num_layers: 1
  dropout: 0.0
  batch_size: 16
  epochs: 2
  learning_rate: 0.01
  seed: 9
training:
  scale_features: true
  normalize_target: true
reporting:
  runs_dir: {tmp_path / "runs"}
""",
        encoding="utf-8",
    )

    diagnostics = inspect_dataset(config_path, output_dir=output_dir)

    assert (output_dir / "dataset_diagnostics.json").exists()
    assert (output_dir / "dataset_diagnostics.md").exists()
    assert set(diagnostics["split_sizes"]) == {"train", "val", "test"}
    assert diagnostics["split_strategy"] == "chronological"
    assert set(diagnostics["target"]) == {"train", "val", "test"}
    assert "target_range_overlap" in diagnostics
    assert "feature_shift" in diagnostics
    assert "naive_persistence_rmse" in diagnostics
    assert "warnings" in diagnostics


def test_dataset_diagnostics_csv_smoke(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path)
    config_path = tmp_path / "csv_diagnostics.yaml"
    output_dir = tmp_path / "reports"
    config_path.write_text(
        f"""
experiment:
  name: csv_diagnostics
dataset:
  source: csv
  path: {csv_path}
  timestamp_column: timestamp
  target_column: value
  feature_columns: null
  split_strategy: chronological
  input_length: 8
  forecast_horizon: 4
  train_fraction: 0.6
  val_fraction: 0.2
  seed: 9
model:
  name: linear
  alpha: 1.0
training:
  scale_features: false
  normalize_target: false
reporting:
  runs_dir: {tmp_path / "runs"}
""",
        encoding="utf-8",
    )

    diagnostics = inspect_dataset(config_path, output_dir=output_dir)

    assert diagnostics["source"] == "csv"
    assert diagnostics["row_count"] == 80
    assert diagnostics["target_column"] == "value"
    assert diagnostics["selected_feature_columns"] == ["feature_a", "feature_b"]
    assert diagnostics["split_strategy"] == "chronological"
    assert (output_dir / "dataset_diagnostics.json").exists()
    assert (output_dir / "dataset_diagnostics.md").exists()


def _write_csv(tmp_path: Path) -> Path:
    n_rows = 80
    time = np.arange(n_rows, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="h"),
            "value": 10.0 + 0.5 * time,
            "feature_a": time,
            "feature_b": np.sin(time / 6.0),
            "label": ["ignored"] * n_rows,
        }
    )
    path = tmp_path / "diagnostics.csv"
    frame.to_csv(path, index=False)
    return path
