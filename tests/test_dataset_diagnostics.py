from __future__ import annotations

from pathlib import Path

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
    assert set(diagnostics["target"]) == {"train", "val", "test"}
    assert "naive_persistence_rmse" in diagnostics
    assert "warnings" in diagnostics
