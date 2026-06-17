from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import yaml

from autoresearch_timeseries_agent.training.compare_runs import compare_runs
from autoresearch_timeseries_agent.training.run_experiment import run_experiment


ExperimentRunner = Callable[[Path], dict[str, Any]]
ComparisonRunner = Callable[[], dict[str, Any]]


def load_comparison_report(path: Path = Path("reports/model_comparison.json")) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset_diagnostics(
    path: Path = Path("reports/dataset_diagnostics.json"),
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_experiment_config(
    config: dict[str, Any],
    *,
    config_name: str,
    generated_dir: Path = Path("configs/agent_generated"),
) -> Path:
    _validate_config_name(config_name)
    generated_dir.mkdir(parents=True, exist_ok=True)
    path = generated_dir / f"{config_name}.yaml"
    content = yaml.safe_dump(config, sort_keys=False)

    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            msg = f"Refusing to overwrite existing generated config with different content: {path}"
            raise FileExistsError(msg)
        return path

    path.write_text(content, encoding="utf-8")
    return path


def run_experiment_config(
    config_path: Path,
    *,
    runner: ExperimentRunner = run_experiment,
) -> dict[str, Any]:
    command = (
        "python -m autoresearch_timeseries_agent.training.run_experiment "
        f"--config {config_path.as_posix()}"
    )
    result = runner(config_path)
    return {"command": command, "result": result}


def run_comparison(
    *,
    runner: Callable[..., dict[str, Any]] = compare_runs,
) -> dict[str, Any]:
    command = "python -m autoresearch_timeseries_agent.training.compare_runs"
    result = runner()
    return {"command": command, "result": result}


def summarize_run_metrics(run_result: dict[str, Any]) -> dict[str, Any]:
    metrics = run_result["metrics"]
    model = run_result["model"]
    return {
        "experiment_name": run_result["experiment_name"],
        "model_name": model["name"],
        "split_strategy": run_result.get("split_strategy", "chronological"),
        "scale_features": run_result.get("scale_features", False),
        "normalize_target": run_result.get("normalize_target", False),
        "val_rmse": float(metrics["val"]["rmse"]),
        "test_rmse": float(metrics["test"]["rmse"]),
    }


def _validate_config_name(config_name: str) -> None:
    if not config_name or "/" in config_name or "\\" in config_name or config_name.endswith(".yaml"):
        msg = f"Invalid generated config name: {config_name!r}"
        raise ValueError(msg)
