from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from autoresearch_timeseries_agent.data import SyntheticDatasetConfig, make_synthetic_dataset
from autoresearch_timeseries_agent.evaluation import evaluate_forecast
from autoresearch_timeseries_agent.models import LinearBaseline, PersistenceBaseline


BaselineModel = PersistenceBaseline | LinearBaseline


@dataclass(frozen=True)
class ModelConfig:
    name: str
    params: dict[str, float | int | str | bool]


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    dataset: SyntheticDatasetConfig
    model: ModelConfig
    runs_dir: Path = Path("reports/runs")


def run_experiment(config_path: Path) -> dict[str, Any]:
    experiment_config = load_experiment_config(config_path)
    splits = make_synthetic_dataset(experiment_config.dataset)
    model = build_model(experiment_config.model)
    model.fit(splits.train.X, splits.train.y)

    metrics = {
        "train": _evaluate_split(model, splits.train.X, splits.train.y),
        "val": _evaluate_split(model, splits.val.X, splits.val.y),
        "test": _evaluate_split(model, splits.test.X, splits.test.y),
    }
    results = {
        "experiment_name": experiment_config.experiment_name,
        "config_path": str(config_path),
        "model": {
            "name": experiment_config.model.name,
            "params": experiment_config.model.params,
        },
        "dataset": asdict(experiment_config.dataset),
        "metrics": metrics,
        "per_horizon_analysis": {
            split_name: describe_per_horizon_rmse(split_metrics["per_horizon_rmse"])
            for split_name, split_metrics in metrics.items()
        },
    }

    results_path, report_path = _write_run_outputs(experiment_config, results)
    val_metrics = metrics["val"]
    test_metrics = metrics["test"]
    print(
        f"{experiment_config.experiment_name}: "
        f"val RMSE={val_metrics['rmse']:.4f}, "
        f"test RMSE={test_metrics['rmse']:.4f}"
    )
    print(f"Wrote {results_path} and {report_path}")
    return results


def load_experiment_config(config_path: Path) -> ExperimentConfig:
    config = _load_yaml(config_path)
    dataset_config = _dataset_config_from_mapping(config)
    model_config = _model_config_from_mapping(config)

    experiment = _mapping_section(config, "experiment")
    default_name = model_config.name
    experiment_name = str(experiment.get("name", default_name))
    if not experiment_name:
        msg = "experiment.name must not be empty"
        raise ValueError(msg)

    reporting = _mapping_section(config, "reporting")
    runs_dir = Path(reporting.get("runs_dir", "reports/runs"))
    return ExperimentConfig(
        experiment_name=experiment_name,
        dataset=dataset_config,
        model=model_config,
        runs_dir=runs_dir,
    )


def build_model(config: ModelConfig) -> BaselineModel:
    model_name = config.name.lower()
    if model_name == "persistence":
        return PersistenceBaseline()
    if model_name == "linear":
        return LinearBaseline(alpha=float(config.params.get("alpha", 1.0)))
    msg = f"Unsupported baseline model {config.name!r}; expected 'persistence' or 'linear'"
    raise ValueError(msg)


def describe_per_horizon_rmse(values: list[float]) -> dict[str, bool | int | float]:
    if not values:
        msg = "per-horizon RMSE values must not be empty"
        raise ValueError(msg)

    increases = sum(
        later >= earlier for earlier, later in zip(values, values[1:], strict=False)
    )
    steps = max(len(values) - 1, 1)
    worst_index = max(range(len(values)), key=values.__getitem__)
    best_index = min(range(len(values)), key=values.__getitem__)
    return {
        "generally_increases": bool(values[-1] > values[0] and increases >= steps / 2),
        "best_horizon": best_index + 1,
        "best_rmse": values[best_index],
        "worst_horizon": worst_index + 1,
        "worst_rmse": values[worst_index],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a time-series baseline experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to an experiment YAML config.")
    args = parser.parse_args()
    run_experiment(args.config)


def _write_run_outputs(
    experiment_config: ExperimentConfig,
    results: dict[str, Any],
) -> tuple[Path, Path]:
    experiment_config.runs_dir.mkdir(parents=True, exist_ok=True)
    results_path = experiment_config.runs_dir / f"{experiment_config.experiment_name}.json"
    report_path = experiment_config.runs_dir / f"{experiment_config.experiment_name}.md"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(_render_markdown_report(results), encoding="utf-8")
    return results_path, report_path


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


def _model_config_from_mapping(config: dict[str, Any]) -> ModelConfig:
    model = _mapping_section(config, "model")
    experiment = _mapping_section(config, "experiment")
    name = str(model.get("name", experiment.get("model", "persistence"))).lower()
    params = {key: value for key, value in model.items() if key != "name"}
    return ModelConfig(name=name, params=params)


def _evaluate_split(
    model: BaselineModel,
    X: Any,
    y: Any,
) -> dict[str, float | list[float]]:
    if isinstance(model, PersistenceBaseline):
        predictions = model.predict(X, forecast_horizon=y.shape[1])
    else:
        predictions = model.predict(X)
    return evaluate_forecast(y, predictions)


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
    model = results["model"]
    lines = [
        f"# {results['experiment_name']} Forecast Report",
        "",
        f"- Model: `{model['name']}`",
        f"- Model params: `{model['params']}`",
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
    lines.extend(
        [
            "",
            "## Per-Horizon Analysis",
            "",
            f"Test per-horizon RMSE: {test_horizon}",
            "",
            _render_horizon_interpretation(results["per_horizon_analysis"]["test"]),
            "",
        ]
    )
    return "\n".join(lines)


def _render_horizon_interpretation(analysis: dict[str, bool | int | float]) -> str:
    trend = "generally increases" if analysis["generally_increases"] else "does not generally increase"
    return (
        f"Error {trend} with horizon. "
        f"The best horizon is {analysis['best_horizon']} "
        f"(RMSE={analysis['best_rmse']:.4f}); "
        f"the worst horizon is {analysis['worst_horizon']} "
        f"(RMSE={analysis['worst_rmse']:.4f})."
    )


if __name__ == "__main__":
    main()
