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

from autoresearch_timeseries_agent.data import (
    CsvDatasetConfig,
    DatasetConfig,
    SyntheticDatasetConfig,
    make_dataset,
)
from autoresearch_timeseries_agent.data.windowing import WindowedDataset
from autoresearch_timeseries_agent.evaluation import evaluate_forecast
from autoresearch_timeseries_agent.models import (
    LinearBaseline,
    LSTMForecaster,
    PersistenceBaseline,
    TransformerForecaster,
)


BaselineModel = PersistenceBaseline | LinearBaseline | LSTMForecaster | TransformerForecaster
NeuralModel = LSTMForecaster | TransformerForecaster
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    params: dict[str, float | int | str | bool]


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    runs_dir: Path = Path("reports/runs")


@dataclass(frozen=True)
class TrainingConfig:
    scale_features: bool = False
    normalize_target: bool = False


@dataclass(frozen=True)
class FeatureScaler:
    mean: FloatArray
    std: FloatArray
    enabled: bool

    def transform(self, X: FloatArray) -> FloatArray:
        return (X - self.mean) / self.std


@dataclass(frozen=True)
class TargetScaler:
    mean: FloatArray
    std: FloatArray
    enabled: bool

    def transform(self, y: FloatArray) -> FloatArray:
        return (y - self.mean) / self.std

    def inverse_transform(self, y: FloatArray) -> FloatArray:
        return y * self.std + self.mean


def run_experiment(config_path: Path) -> dict[str, Any]:
    experiment_config = load_experiment_config(config_path)
    splits = make_dataset(experiment_config.dataset)
    model = build_model(
        experiment_config.model,
        experiment_config.dataset,
        n_features=splits.train.X.shape[2],
    )
    training_summary = _train_model(
        model,
        splits.train,
        experiment_config.model,
        experiment_config.training,
    )
    metrics, prediction_diagnostics = _evaluate_splits(
        model,
        {"train": splits.train, "val": splits.val, "test": splits.test},
    )
    results = {
        "experiment_name": experiment_config.experiment_name,
        "config_path": str(config_path),
        "model": {
            "name": experiment_config.model.name,
            "params": experiment_config.model.params,
        },
        "dataset": asdict(experiment_config.dataset),
        "dataset_metadata": splits.metadata,
        "dataset_source": splits.metadata.get("source", "synthetic"),
        "split_strategy": experiment_config.dataset.split_strategy,
        "scale_features": experiment_config.training.scale_features,
        "normalize_target": experiment_config.training.normalize_target,
        "metrics": metrics,
        "training": training_summary,
        "prediction_diagnostics": prediction_diagnostics,
        "data_warnings": _target_range_warnings(
            {"train": splits.train, "val": splits.val, "test": splits.test},
            split_strategy=experiment_config.dataset.split_strategy,
        ),
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
    training_config = _training_config_from_mapping(config)

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
        training=training_config,
        runs_dir=runs_dir,
    )


def build_model(
    config: ModelConfig,
    dataset_config: DatasetConfig | None = None,
    *,
    n_features: int | None = None,
    input_length: int | None = None,
    forecast_horizon: int | None = None,
) -> BaselineModel:
    model_name = config.name.lower()
    if model_name == "persistence":
        return PersistenceBaseline()
    if model_name == "linear":
        return LinearBaseline(alpha=float(config.params.get("alpha", 1.0)))
    if model_name == "lstm":
        shape = _resolve_dataset_shape(dataset_config, n_features, input_length, forecast_horizon)
        return LSTMForecaster(
            n_features=shape["n_features"],
            forecast_horizon=shape["forecast_horizon"],
            hidden_size=int(config.params.get("hidden_size", 32)),
            num_layers=int(config.params.get("num_layers", 1)),
            dropout=float(config.params.get("dropout", 0.0)),
        )
    if model_name == "transformer":
        shape = _resolve_dataset_shape(dataset_config, n_features, input_length, forecast_horizon)
        return TransformerForecaster(
            n_features=shape["n_features"],
            forecast_horizon=shape["forecast_horizon"],
            input_length=shape["input_length"],
            d_model=int(config.params.get("d_model", 32)),
            nhead=int(config.params.get("nhead", 4)),
            num_layers=int(config.params.get("num_layers", 1)),
            dim_feedforward=int(config.params.get("dim_feedforward", 64)),
            dropout=float(config.params.get("dropout", 0.1)),
        )
    msg = (
        f"Unsupported baseline model {config.name!r}; "
        "expected 'persistence', 'linear', 'lstm', or 'transformer'"
    )
    raise ValueError(msg)


def _resolve_dataset_shape(
    dataset_config: DatasetConfig | None,
    n_features: int | None,
    input_length: int | None,
    forecast_horizon: int | None,
) -> dict[str, int]:
    if dataset_config is None and (
        n_features is None or input_length is None or forecast_horizon is None
    ):
        msg = "dataset_config is required to build sequence models"
        raise ValueError(msg)

    resolved_input_length = (
        input_length
        if input_length is not None
        else getattr(dataset_config, "input_length")
    )
    resolved_forecast_horizon = (
        forecast_horizon
        if forecast_horizon is not None
        else getattr(dataset_config, "forecast_horizon")
    )
    if n_features is not None:
        resolved_n_features = n_features
    elif isinstance(dataset_config, SyntheticDatasetConfig):
        resolved_n_features = dataset_config.n_features
    else:
        msg = "n_features is required when building sequence models for CSV datasets"
        raise ValueError(msg)

    return {
        "n_features": int(resolved_n_features),
        "input_length": int(resolved_input_length),
        "forecast_horizon": int(resolved_forecast_horizon),
    }


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


def fit_target_scaler(y: NDArray[np.floating], *, normalize_target: bool) -> TargetScaler:
    targets = np.asarray(y, dtype=np.float64)
    if targets.ndim != 2:
        msg = f"y must have shape (num_windows, forecast_horizon); got {targets.shape}"
        raise ValueError(msg)
    if normalize_target:
        mean = targets.mean(axis=0, keepdims=True)
        std = np.maximum(targets.std(axis=0, keepdims=True), 1e-8)
        return TargetScaler(mean=mean, std=std, enabled=True)

    return TargetScaler(
        mean=np.zeros((1, targets.shape[1]), dtype=np.float64),
        std=np.ones((1, targets.shape[1]), dtype=np.float64),
        enabled=False,
    )


def fit_feature_scaler(X: NDArray[np.floating], *, scale_features: bool) -> FeatureScaler:
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 3:
        msg = f"X must have shape (num_windows, input_length, n_features); got {values.shape}"
        raise ValueError(msg)
    if scale_features:
        mean = values.mean(axis=(0, 1), keepdims=True)
        std = np.maximum(values.std(axis=(0, 1), keepdims=True), 1e-8)
        return FeatureScaler(mean=mean, std=std, enabled=True)

    return FeatureScaler(
        mean=np.zeros((1, 1, values.shape[2]), dtype=np.float64),
        std=np.ones((1, 1, values.shape[2]), dtype=np.float64),
        enabled=False,
    )


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


def _dataset_config_from_mapping(config: dict[str, Any]) -> DatasetConfig:
    dataset = _mapping_section(config, "dataset")
    experiment = _mapping_section(config, "experiment")
    seed = dataset.get("seed", experiment.get("seed", SyntheticDatasetConfig.seed))

    source = str(dataset.get("source", dataset.get("name", "synthetic"))).lower()
    if source == "synthetic":
        return SyntheticDatasetConfig(
            source="synthetic",
            mode=str(dataset.get("mode", SyntheticDatasetConfig.mode)),
            split_strategy=str(
                dataset.get("split_strategy", SyntheticDatasetConfig.split_strategy)
            ),
            n_timesteps=int(dataset.get("n_timesteps", SyntheticDatasetConfig.n_timesteps)),
            n_features=int(dataset.get("n_features", SyntheticDatasetConfig.n_features)),
            input_length=int(
                dataset.get("input_length", SyntheticDatasetConfig.input_length)
            ),
            forecast_horizon=int(
                dataset.get("forecast_horizon", SyntheticDatasetConfig.forecast_horizon)
            ),
            train_fraction=float(
                dataset.get("train_fraction", SyntheticDatasetConfig.train_fraction)
            ),
            val_fraction=float(
                dataset.get("val_fraction", SyntheticDatasetConfig.val_fraction)
            ),
            seed=int(seed),
        )

    if source == "csv":
        return CsvDatasetConfig(
            source="csv",
            path=str(dataset.get("path", CsvDatasetConfig.path)),
            timestamp_column=_optional_string(dataset.get("timestamp_column")),
            target_column=str(dataset.get("target_column", CsvDatasetConfig.target_column)),
            feature_columns=_optional_string_list(dataset.get("feature_columns")),
            input_length=int(dataset.get("input_length", CsvDatasetConfig.input_length)),
            forecast_horizon=int(
                dataset.get("forecast_horizon", CsvDatasetConfig.forecast_horizon)
            ),
            train_fraction=float(
                dataset.get("train_fraction", CsvDatasetConfig.train_fraction)
            ),
            val_fraction=float(dataset.get("val_fraction", CsvDatasetConfig.val_fraction)),
            split_strategy=str(
                dataset.get("split_strategy", CsvDatasetConfig.split_strategy)
            ),
            seed=int(seed),
        )

    msg = f"Unsupported dataset source {source!r}; expected 'synthetic' or 'csv'"
    raise ValueError(msg)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        msg = f"feature_columns must be a list of strings or null; got {type(value).__name__}"
        raise ValueError(msg)
    return [str(item) for item in value]


def _model_config_from_mapping(config: dict[str, Any]) -> ModelConfig:
    model = _mapping_section(config, "model")
    experiment = _mapping_section(config, "experiment")
    name = str(model.get("name", experiment.get("model", "persistence"))).lower()
    params = {key: value for key, value in model.items() if key != "name"}
    return ModelConfig(name=name, params=params)


def _training_config_from_mapping(config: dict[str, Any]) -> TrainingConfig:
    training = _mapping_section(config, "training")
    return TrainingConfig(
        scale_features=_as_bool(training.get("scale_features", False)),
        normalize_target=_as_bool(training.get("normalize_target", False)),
    )


def _train_model(
    model: BaselineModel,
    train: WindowedDataset,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> dict[str, Any]:
    if isinstance(model, (LSTMForecaster, TransformerForecaster)):
        return _train_neural_model(model, train, model_config, training_config)

    model.fit(train.X, train.y)
    return {
        "loss_history": [],
        "scale_features": False,
        "normalize_target": False,
        "scalers": {},
    }


def _evaluate_splits(
    model: BaselineModel,
    splits: dict[str, WindowedDataset],
) -> tuple[dict[str, dict[str, float | list[float]]], dict[str, dict[str, Any]]]:
    metrics = {}
    diagnostics = {}
    for split_name, split in splits.items():
        predictions = _predict_split(model, split)
        split_metrics = evaluate_forecast(split.y, predictions)
        metrics[split_name] = split_metrics
        diagnostics[split_name] = build_prediction_diagnostics(
            split.y,
            predictions,
            split_metrics["per_horizon_rmse"],
        )
    return metrics, diagnostics


def build_prediction_diagnostics(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    per_horizon_rmse: list[float],
    *,
    sample_size: int = 2,
) -> dict[str, Any]:
    actual = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    residuals = predicted - actual
    horizon = describe_per_horizon_rmse(per_horizon_rmse)
    return {
        "y_true_sample": np.round(actual[:sample_size], 6).tolist(),
        "y_pred_sample": np.round(predicted[:sample_size], 6).tolist(),
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std()),
        "best_horizon": horizon["best_horizon"],
        "best_horizon_rmse": horizon["best_rmse"],
        "worst_horizon": horizon["worst_horizon"],
        "worst_horizon_rmse": horizon["worst_rmse"],
        "per_horizon_rmse_generally_increases": horizon["generally_increases"],
    }


def _predict_split(
    model: BaselineModel,
    split: WindowedDataset,
) -> FloatArray:
    if isinstance(model, PersistenceBaseline):
        return model.predict(split.X, forecast_horizon=split.y.shape[1])
    if isinstance(model, (LSTMForecaster, TransformerForecaster)):
        return _predict_neural_model(model, split.X)
    return model.predict(split.X)


def _train_neural_model(
    model: NeuralModel,
    train: WindowedDataset,
    model_config: ModelConfig,
    training_config: TrainingConfig,
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
    X_train, y_train = _fit_neural_scalers(
        model,
        train,
        scale_features=training_config.scale_features,
        normalize_target=training_config.normalize_target,
    )
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
        "scale_features": training_config.scale_features,
        "normalize_target": training_config.normalize_target,
        "scalers": _scaler_metadata(model),
    }


def _fit_neural_scalers(
    model: NeuralModel,
    train: WindowedDataset,
    *,
    scale_features: bool,
    normalize_target: bool,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    feature_scaler = fit_feature_scaler(train.X, scale_features=scale_features)
    target_scaler = fit_target_scaler(train.y, normalize_target=normalize_target)

    model.feature_scaler_ = feature_scaler
    model.target_scaler_ = target_scaler
    return (
        feature_scaler.transform(train.X).astype(np.float32),
        target_scaler.transform(train.y).astype(np.float32),
    )


def _predict_neural_model(model: NeuralModel, X: FloatArray) -> FloatArray:
    if not hasattr(model, "feature_scaler_"):
        msg = f"{model.__class__.__name__} must be trained before evaluation"
        raise ValueError(msg)
    model.eval()
    scaled_X = torch.from_numpy(model.feature_scaler_.transform(X).astype(np.float32))
    with torch.no_grad():
        scaled_predictions = model(scaled_X).numpy()
    return model.target_scaler_.inverse_transform(scaled_predictions)


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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    msg = f"Expected a boolean value; got {value!r}"
    raise ValueError(msg)


def _scaler_metadata(model: NeuralModel) -> dict[str, Any]:
    feature_scaler = model.feature_scaler_
    target_scaler = model.target_scaler_
    return {
        "features": {
            "enabled": feature_scaler.enabled,
            "mean": feature_scaler.mean.reshape(-1).tolist() if feature_scaler.enabled else [],
            "std": feature_scaler.std.reshape(-1).tolist() if feature_scaler.enabled else [],
        },
        "target": {
            "enabled": target_scaler.enabled,
            "mean": target_scaler.mean.reshape(-1).tolist() if target_scaler.enabled else [],
            "std": target_scaler.std.reshape(-1).tolist() if target_scaler.enabled else [],
        },
    }


def _target_range_warnings(
    splits: dict[str, WindowedDataset],
    *,
    split_strategy: str,
) -> list[str]:
    if split_strategy != "chronological":
        return []

    train_min = float(splits["train"].y.min())
    train_max = float(splits["train"].y.max())
    train_range = max(train_max - train_min, 1e-8)
    lower_bound = train_min - 0.25 * train_range
    upper_bound = train_max + 0.25 * train_range

    warnings = []
    for split_name in ("val", "test"):
        split_min = float(splits[split_name].y.min())
        split_max = float(splits[split_name].y.max())
        if split_min < lower_bound or split_max > upper_bound:
            warnings.append(
                f"{split_name} target range [{split_min:.4f}, {split_max:.4f}] "
                f"is far outside train range [{train_min:.4f}, {train_max:.4f}]"
            )
    return warnings


def _render_markdown_report(results: dict[str, Any]) -> str:
    dataset = results["dataset"]
    dataset_metadata = results.get("dataset_metadata", {})
    metrics = results["metrics"]
    model = results["model"]
    lines = [
        f"# {results['experiment_name']} Forecast Report",
        "",
        f"- Model: `{model['name']}`",
        f"- Model params: `{model['params']}`",
        f"- Dataset source: `{dataset_metadata.get('source', dataset.get('source', 'synthetic'))}`",
        f"- Split strategy: `{results['split_strategy']}`",
        f"- Feature scaling: `{results['scale_features']}`",
        f"- Target normalization: `{results['normalize_target']}`",
        f"- Rows: `{dataset_metadata.get('row_count', dataset.get('n_timesteps', 'unknown'))}`",
        f"- Features: `{dataset_metadata.get('n_features', dataset.get('n_features', 'unknown'))}`",
        f"- Input length: `{dataset['input_length']}`",
        f"- Forecast horizon: `{dataset['forecast_horizon']}`",
    ]
    if dataset_metadata.get("source") == "csv":
        lines.extend(
            [
                f"- CSV path: `{dataset_metadata['path']}`",
                f"- Target column: `{dataset_metadata['target_column']}`",
                f"- Selected feature columns: `{dataset_metadata['selected_feature_columns']}`",
            ]
        )
    else:
        lines.append(f"- Dataset mode: `{dataset.get('mode', 'linear')}`")

    lines.extend(
        [
            "",
            "| Split | RMSE | MAE | MAPE |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
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
                f"- Feature scaling: `{training['scale_features']}`",
                f"- Target normalization: `{training['normalize_target']}`",
                f"- Final training loss: `{training['final_loss']:.6f}`",
                f"- Loss history: `{_format_loss_history(training['loss_history'])}`",
            ]
        )

    warnings = results.get("data_warnings", [])
    if warnings:
        lines.extend(["", "## Data Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    diagnostics = results.get("prediction_diagnostics", {}).get("test")
    if diagnostics:
        lines.extend(
            [
                "",
                "## Prediction Diagnostics",
                "",
                f"- Test residual mean: `{diagnostics['residual_mean']:.6f}`",
                f"- Test residual std: `{diagnostics['residual_std']:.6f}`",
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
