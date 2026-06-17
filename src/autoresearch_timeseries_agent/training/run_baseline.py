from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from autoresearch_timeseries_agent.data import SyntheticDatasetConfig, make_synthetic_dataset
from autoresearch_timeseries_agent.evaluation import evaluate_forecast
from autoresearch_timeseries_agent.models import LinearBaseline, PersistenceBaseline


def run_baseline(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    dataset_config = _dataset_config_from_mapping(config)
    splits = make_synthetic_dataset(dataset_config)

    model_name = _model_name_from_mapping(config)
    model = _build_model(model_name, config)
    model.fit(splits.train.X, splits.train.y)

    metrics = {
        "train": _evaluate_split(model, splits.train.X, splits.train.y),
        "val": _evaluate_split(model, splits.val.X, splits.val.y),
        "test": _evaluate_split(model, splits.test.X, splits.test.y),
    }
    results = {
        "config_path": str(config_path),
        "model": model_name,
        "dataset": asdict(dataset_config),
        "metrics": metrics,
    }

    output_dir = _output_dir_from_mapping(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "baseline_results.json"
    report_path = output_dir / "baseline_report.md"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(_render_markdown_report(results), encoding="utf-8")

    test_metrics = metrics["test"]
    print(
        f"Baseline {model_name}: "
        f"test RMSE={test_metrics['rmse']:.4f}, "
        f"MAE={test_metrics['mae']:.4f}, "
        f"MAPE={test_metrics['mape']:.2f}%"
    )
    print(f"Wrote {results_path} and {report_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a baseline time-series forecast experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a baseline YAML config.")
    args = parser.parse_args()
    run_baseline(args.config)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        msg = f"Config file does not exist: {path}"
        raise FileNotFoundError(msg)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        msg = f"Config must be a YAML mapping; got {type(loaded).__name__}"
        raise ValueError(msg)
    return loaded


def _dataset_config_from_mapping(config: dict[str, Any]) -> SyntheticDatasetConfig:
    dataset = _mapping_section(config, "dataset")
    experiment = _mapping_section(config, "experiment")
    seed = dataset.get("seed", experiment.get("seed", SyntheticDatasetConfig.seed))

    name = dataset.get("name", "synthetic")
    if name != "synthetic":
        msg = f"Only the synthetic dataset is supported in this pass; got {name!r}"
        raise ValueError(msg)

    return SyntheticDatasetConfig(
        n_timesteps=int(dataset.get("n_timesteps", SyntheticDatasetConfig.n_timesteps)),
        n_features=int(dataset.get("n_features", SyntheticDatasetConfig.n_features)),
        input_length=int(dataset.get("input_length", SyntheticDatasetConfig.input_length)),
        forecast_horizon=int(
            dataset.get("forecast_horizon", SyntheticDatasetConfig.forecast_horizon)
        ),
        train_fraction=float(
            dataset.get("train_fraction", SyntheticDatasetConfig.train_fraction)
        ),
        val_fraction=float(dataset.get("val_fraction", SyntheticDatasetConfig.val_fraction)),
        seed=int(seed),
    )


def _model_name_from_mapping(config: dict[str, Any]) -> str:
    model = _mapping_section(config, "model")
    experiment = _mapping_section(config, "experiment")
    return str(model.get("name", experiment.get("model", "persistence"))).lower()


def _build_model(model_name: str, config: dict[str, Any]) -> PersistenceBaseline | LinearBaseline:
    model = _mapping_section(config, "model")
    if model_name == "persistence":
        return PersistenceBaseline()
    if model_name == "linear":
        return LinearBaseline(alpha=float(model.get("alpha", 1.0)))
    msg = f"Unsupported baseline model {model_name!r}; expected 'persistence' or 'linear'"
    raise ValueError(msg)


def _evaluate_split(
    model: PersistenceBaseline | LinearBaseline,
    X: Any,
    y: Any,
) -> dict[str, float | list[float]]:
    if isinstance(model, PersistenceBaseline):
        predictions = model.predict(X, forecast_horizon=y.shape[1])
    else:
        predictions = model.predict(X)
    return evaluate_forecast(y, predictions)


def _output_dir_from_mapping(config: dict[str, Any]) -> Path:
    reporting = _mapping_section(config, "reporting")
    return Path(reporting.get("output_dir", "reports"))


def _mapping_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        msg = f"Config section {key!r} must be a mapping; got {type(value).__name__}"
        raise ValueError(msg)
    return value


def _render_markdown_report(results: dict[str, Any]) -> str:
    dataset = results["dataset"]
    metrics = results["metrics"]
    lines = [
        "# Baseline Forecast Report",
        "",
        f"- Model: `{results['model']}`",
        f"- Timesteps: `{dataset['n_timesteps']}`",
        f"- Features: `{dataset['n_features']}`",
        f"- Input length: `{dataset['input_length']}`",
        f"- Forecast horizon: `{dataset['forecast_horizon']}`",
        "",
        "| Split | RMSE | MAE | MAPE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split_name in ("train", "val", "test"):
        split_metrics = metrics[split_name]
        lines.append(
            f"| {split_name} | {split_metrics['rmse']:.4f} | "
            f"{split_metrics['mae']:.4f} | {split_metrics['mape']:.2f}% |"
        )

    test_horizon = ", ".join(f"{value:.4f}" for value in metrics["test"]["per_horizon_rmse"])
    lines.extend(["", f"Test per-horizon RMSE: {test_horizon}", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
