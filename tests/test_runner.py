from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_baseline_runner_smoke(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        f"""
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
  name: persistence
reporting:
  output_dir: {report_dir}
""",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = f"src{os.pathsep}{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoresearch_timeseries_agent.training.run_baseline",
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        cwd=Path.cwd(),
        env=env,
        text=True,
    )

    results_path = report_dir / "baseline_results.json"
    report_path = report_dir / "baseline_report.md"
    assert "Baseline persistence" in result.stdout
    assert results_path.exists()
    assert report_path.exists()

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["model"] == "persistence"
    assert payload["metrics"]["test"]["rmse"] > 0
