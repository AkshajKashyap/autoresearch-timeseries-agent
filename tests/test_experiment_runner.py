from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from autoresearch_timeseries_agent.data import SyntheticDatasetConfig
from autoresearch_timeseries_agent.models import (
    LSTMForecaster,
    LinearBaseline,
    PersistenceBaseline,
    TransformerForecaster,
)
from autoresearch_timeseries_agent.training import (
    ModelConfig,
    build_model,
    fit_feature_scaler,
    fit_target_scaler,
    load_experiment_config,
)


def test_load_experiment_config(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, experiment_name="linear_test", model_name="linear")

    config = load_experiment_config(config_path)

    assert config.experiment_name == "linear_test"
    assert config.dataset.mode == "nonlinear"
    assert config.dataset.n_timesteps == 144
    assert config.dataset.n_features == 3
    assert config.model.name == "linear"
    assert config.model.params["alpha"] == 0.25
    assert config.training.scale_features is True
    assert config.training.normalize_target is True
    assert config.runs_dir == tmp_path / "runs"


def test_model_factory_selects_models() -> None:
    dataset_config = SyntheticDatasetConfig(n_features=3, input_length=12, forecast_horizon=6)
    persistence = build_model(ModelConfig(name="persistence", params={}))
    linear = build_model(ModelConfig(name="linear", params={"alpha": 0.25}))
    lstm = build_model(
        ModelConfig(name="lstm", params={"hidden_size": 8, "num_layers": 1}),
        dataset_config,
    )
    transformer = build_model(
        ModelConfig(name="transformer", params={"d_model": 8, "nhead": 2}),
        dataset_config,
    )

    assert isinstance(persistence, PersistenceBaseline)
    assert isinstance(linear, LinearBaseline)
    assert isinstance(lstm, LSTMForecaster)
    assert isinstance(transformer, TransformerForecaster)
    assert linear.alpha == 0.25

    with pytest.raises(ValueError, match="dataset_config is required"):
        build_model(ModelConfig(name="lstm", params={}))
    with pytest.raises(ValueError, match="Unsupported baseline model"):
        build_model(ModelConfig(name="cnn", params={}))


def test_run_experiment_smoke(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        experiment_name="persistence_smoke",
        model_name="persistence",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = f"src{os.pathsep}{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoresearch_timeseries_agent.training.run_experiment",
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        cwd=Path.cwd(),
        env=env,
        text=True,
    )

    results_path = tmp_path / "runs" / "persistence_smoke.json"
    report_path = tmp_path / "runs" / "persistence_smoke.md"
    assert "persistence_smoke: val RMSE=" in result.stdout
    assert results_path.exists()
    assert report_path.exists()

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["experiment_name"] == "persistence_smoke"
    assert payload["metrics"]["test"]["rmse"] > 0
    assert len(payload["metrics"]["test"]["per_horizon_rmse"]) == 6
    assert "worst_horizon" in payload["per_horizon_analysis"]["test"]


def test_lstm_experiment_smoke(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        experiment_name="lstm_smoke",
        model_name="lstm",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = f"src{os.pathsep}{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoresearch_timeseries_agent.training.run_experiment",
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        cwd=Path.cwd(),
        env=env,
        text=True,
    )

    results_path = tmp_path / "runs" / "lstm_smoke.json"
    assert "lstm_smoke: val RMSE=" in result.stdout
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["model"]["name"] == "lstm"
    assert payload["split_strategy"] == "chronological"
    assert payload["scale_features"] is True
    assert payload["training"]["normalize_target"] is True
    assert payload["training"]["scalers"]["features"]["enabled"] is True
    assert len(payload["training"]["loss_history"]) == 2
    assert "prediction_diagnostics" in payload
    assert payload["metrics"]["test"]["rmse"] > 0


def test_transformer_experiment_smoke(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        experiment_name="transformer_smoke",
        model_name="transformer",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = f"src{os.pathsep}{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoresearch_timeseries_agent.training.run_experiment",
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        cwd=Path.cwd(),
        env=env,
        text=True,
    )

    results_path = tmp_path / "runs" / "transformer_smoke.json"
    assert "transformer_smoke: val RMSE=" in result.stdout
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["model"]["name"] == "transformer"
    assert payload["scale_features"] is True
    assert payload["normalize_target"] is True
    assert payload["training"]["scalers"]["features"]["enabled"] is True
    assert payload["training"]["scalers"]["target"]["enabled"] is True
    assert len(payload["training"]["loss_history"]) == 2
    assert payload["metrics"]["test"]["rmse"] > 0


def test_feature_scaler_uses_train_statistics_only() -> None:
    train_X = np.array(
        [
            [[1.0, 10.0], [3.0, 30.0]],
            [[5.0, 50.0], [7.0, 70.0]],
        ]
    )
    val_X = train_X + 100.0

    scaler = fit_feature_scaler(train_X, scale_features=True)

    np.testing.assert_allclose(scaler.mean.reshape(-1), np.array([4.0, 40.0]))
    np.testing.assert_allclose(scaler.transform(train_X).mean(axis=(0, 1)), np.zeros(2))
    assert scaler.transform(val_X).mean() > 1.0


def test_target_scaler_roundtrip() -> None:
    y = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    scaler = fit_target_scaler(y, normalize_target=True)

    restored = scaler.inverse_transform(scaler.transform(y))

    np.testing.assert_allclose(restored, y)

    disabled = fit_target_scaler(y, normalize_target=False)
    np.testing.assert_allclose(disabled.transform(y), y)
    np.testing.assert_allclose(disabled.inverse_transform(y), y)


def test_target_scaler_uses_train_statistics_only() -> None:
    train_y = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    val_y = train_y + 100.0

    scaler = fit_target_scaler(train_y, normalize_target=True)

    np.testing.assert_allclose(scaler.mean.reshape(-1), np.array([3.0, 6.0]))
    np.testing.assert_allclose(scaler.transform(train_y).mean(axis=0), np.zeros(2))
    assert scaler.transform(val_y).mean() > 1.0


def _write_config(tmp_path: Path, *, experiment_name: str, model_name: str) -> Path:
    alpha_line = "  alpha: 0.25\n" if model_name == "linear" else ""
    lstm_lines = (
        "  hidden_size: 8\n"
        "  num_layers: 1\n"
        "  dropout: 0.0\n"
        "  batch_size: 16\n"
        "  epochs: 2\n"
        "  learning_rate: 0.01\n"
        "  seed: 9\n"
        if model_name == "lstm"
        else ""
    )
    transformer_lines = (
        "  d_model: 8\n"
        "  nhead: 2\n"
        "  num_layers: 1\n"
        "  dim_feedforward: 16\n"
        "  dropout: 0.0\n"
        "  batch_size: 16\n"
        "  epochs: 2\n"
        "  learning_rate: 0.01\n"
        "  seed: 9\n"
        if model_name == "transformer"
        else ""
    )
    config_path = tmp_path / f"{experiment_name}.yaml"
    config_path.write_text(
        f"""
experiment:
  name: {experiment_name}
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
  name: {model_name}
{alpha_line}{lstm_lines}{transformer_lines}training:
  scale_features: true
  normalize_target: true
reporting:
  runs_dir: {tmp_path / "runs"}
""",
        encoding="utf-8",
    )
    return config_path
