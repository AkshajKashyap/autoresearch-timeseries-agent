from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from autoresearch_timeseries_agent.models import LinearBaseline, PersistenceBaseline
from autoresearch_timeseries_agent.training import ModelConfig, build_model, load_experiment_config


def test_load_experiment_config(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, experiment_name="linear_test", model_name="linear")

    config = load_experiment_config(config_path)

    assert config.experiment_name == "linear_test"
    assert config.dataset.n_timesteps == 144
    assert config.dataset.n_features == 3
    assert config.model.name == "linear"
    assert config.model.params["alpha"] == 0.25
    assert config.runs_dir == tmp_path / "runs"


def test_model_factory_selects_models() -> None:
    persistence = build_model(ModelConfig(name="persistence", params={}))
    linear = build_model(ModelConfig(name="linear", params={"alpha": 0.25}))

    assert isinstance(persistence, PersistenceBaseline)
    assert isinstance(linear, LinearBaseline)
    assert linear.alpha == 0.25

    with pytest.raises(ValueError, match="Unsupported baseline model"):
        build_model(ModelConfig(name="lstm", params={}))


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


def _write_config(tmp_path: Path, *, experiment_name: str, model_name: str) -> Path:
    alpha_line = "  alpha: 0.25\n" if model_name == "linear" else ""
    config_path = tmp_path / f"{experiment_name}.yaml"
    config_path.write_text(
        f"""
experiment:
  name: {experiment_name}
dataset:
  name: synthetic
  n_timesteps: 144
  n_features: 3
  input_length: 12
  forecast_horizon: 6
  train_fraction: 0.6
  val_fraction: 0.2
  seed: 9
model:
  name: {model_name}
{alpha_line}reporting:
  runs_dir: {tmp_path / "runs"}
""",
        encoding="utf-8",
    )
    return config_path
