from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from autoresearch_timeseries_agent.data import SyntheticDatasetConfig, make_synthetic_dataset
from autoresearch_timeseries_agent.data.windowing import WindowedDataset
from autoresearch_timeseries_agent.evaluation import evaluate_forecast
from autoresearch_timeseries_agent.models import LinearBaseline, LSTMForecaster, PersistenceBaseline


BaselineModel = PersistenceBaseline | LinearBaseline | LSTMForecaster
FloatArray = NDArray[np.float64]


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
    model = build_model(experiment_config.model, experiment_config.dataset)
    training_summary = _train_model(model, splits.train, experiment_config.model)

    metrics = {
        "train": _evaluate_split(model, splits.train),
        "val": _evaluate_split(model, splits.val),
        "test": _evaluate_split(model, splits.test),
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
        "training": training_summary,
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


def build_model(
    config: ModelConfig,
    dataset_config: SyntheticDatasetConfig | None = None,
) -> BaselineModel:
    model_name = config.name.lower()
    if model_name == "persistence":
        return PersistenceBaseline()
    if model_name == "linear":
        return LinearBaseline(alpha=float(config.params.get("alpha", 1.0)))
    if model_name == "lstm":
        if dataset_config is None:
            msg = "dataset_config is required to build the LSTM model"
            raise ValueError(msg)
        return LSTMForecaster(
            n_features=dataset_config.n_features,
            forecast_horizon=dataset_config.forecast_horizon,
            hidden_size=int(config.params.get("hidden_size", 32)),
            num_layers=int(config.params.get("num_layers", 1)),
            dropout=float(config.params.get("dropout", 0.0)),
        )
    msg = (
        f"Unsupported baseline model {config.name!r}; "
        "expected 'persistence', 'linear', or 'lstm'"
    )
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
        mode=str(dataset.get("mode", SyntheticDatasetConfig.mode)),
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


def _train_model(
    model: BaselineModel,
    train: WindowedDataset,
    model_config: ModelConfig,
) -> dict[str, Any]:
    if isinstance(model, LSTMForecaster):
        return _train_lstm(model, train, model_config)

    model.fit(train.X, train.y)
    return {"loss_history": []}


def _evaluate_split(
    model: BaselineModel,
    split: WindowedDataset,
) -> dict[str, float | list[float]]:
    if isinstance(model, PersistenceBaseline):
        predictions = model.predict(split.X, forecast_horizon=split.y.shape[1])
    elif isinstance(model, LSTMForecaster):
        predictions = _predict_lstm(model, split.X)
    else:
        predictions = model.predict(split.X)
    return evaluate_forecast(split.y, predictions)


def _train_lstm(
    model: LSTMForecaster,
    train: WindowedDataset,
    model_config: ModelConfig,
) -> dict[str, Any]:
    params = model_config.params
    seed = int(params.get("seed", 42))
    batch_size = int(params.get("batch_size", 32))
    epochs = int(params.get("epochs", 10))
    learning_rate = float(params.get("learning_rate", 0.001))
    if batch_size <= 0:
        msg = f"batch_size must be positive; got {batch_size}"
        raise ValueError(msg)
    if epochs <= 0:
        msg = f"epochs must be positive; got {epochs}"
        raise ValueError(msg)
    if learning_rate <= 0:
        msg = f"learning_rate must be positive; got {learning_rate}"
        raise ValueError(msg)

    _seed_torch(seed)
    X_train, y_train = _fit_lstm_scalers(model, train)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    loss_history: list[float] = []
    model.train()
    for _ in range(epochs):
        total_loss = 0.0
        total_examples = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            batch_size_actual = batch_X.shape[0]
            total_loss += float(loss.item()) * batch_size_actual
            total_examples += batch_size_actual
        loss_history.append(total_loss / total_examples)

    return {
        "loss_history": loss_history,
        "final_loss": loss_history[-1],
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
    }


def _fit_lstm_scalers(
    model: LSTMForecaster,
    train: WindowedDataset,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    feature_mean = train.X.mean(axis=(0, 1), keepdims=True)
    feature_std = train.X.std(axis=(0, 1), keepdims=True)
    target_mean = train.y.mean(axis=0, keepdims=True)
    target_std = train.y.std(axis=0, keepdims=True)

    model.feature_mean_ = feature_mean
    model.feature_std_ = np.maximum(feature_std, 1e-8)
    model.target_mean_ = target_mean
    model.target_std_ = np.maximum(target_std, 1e-8)
    return (
        _scale_lstm_X(model, train.X).astype(np.float32),
        ((train.y - model.target_mean_) / model.target_std_).astype(np.float32),
    )


def _predict_lstm(model: LSTMForecaster, X: FloatArray) -> FloatArray:
    if not hasattr(model, "feature_mean_"):
        msg = "LSTMForecaster must be trained before evaluation"
        raise ValueError(msg)
    model.eval()
    scaled_X = torch.from_numpy(_scale_lstm_X(model, X).astype(np.float32))
    with torch.no_grad():
        scaled_predictions = model(scaled_X).numpy()
    return scaled_predictions * model.target_std_ + model.target_mean_


def _scale_lstm_X(model: LSTMForecaster, X: FloatArray) -> FloatArray:
    return (X - model.feature_mean_) / model.feature_std_


def _seed_torch(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


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
        f"- Dataset mode: `{dataset['mode']}`",
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

    training = results.get("training", {})
    if training.get("loss_history"):
        lines.extend(
            [
                "",
                "## Training Summary",
                "",
                f"- Epochs: `{training['epochs']}`",
                f"- Final training loss: `{training['final_loss']:.6f}`",
                f"- Loss history: `{_format_loss_history(training['loss_history'])}`",
            ]
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


def _format_loss_history(loss_history: list[float]) -> str:
    return ", ".join(f"{value:.6f}" for value in loss_history)


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
