from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from autoresearch_timeseries_agent.data import make_synthetic_dataset
from autoresearch_timeseries_agent.evaluation import rmse
from autoresearch_timeseries_agent.models import PersistenceBaseline
from autoresearch_timeseries_agent.training.run_experiment import load_experiment_config


def inspect_dataset(config_path: Path, *, output_dir: Path = Path("reports")) -> dict[str, Any]:
    experiment_config = load_experiment_config(config_path)
    splits = make_synthetic_dataset(experiment_config.dataset)
    raw_splits = _raw_split_series(splits.raw_series, experiment_config.dataset)

    diagnostics = {
        "config_path": str(config_path),
        "dataset": asdict(experiment_config.dataset),
        "split_sizes": {
            name: {"timesteps": raw.shape[0], "windows": getattr(splits, name).X.shape[0]}
            for name, raw in raw_splits.items()
        },
        "target": {
            name: _target_stats(getattr(splits, name).y)
            for name in ("train", "val", "test")
        },
        "features": {
            name: _feature_stats(raw)
            for name, raw in raw_splits.items()
        },
        "naive_persistence_rmse": {
            name: _persistence_rmse(getattr(splits, name).X, getattr(splits, name).y)
            for name in ("train", "val", "test")
        },
    }
    diagnostics["warnings"] = _range_warnings(diagnostics["target"])

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dataset_diagnostics.json"
    md_path = output_dir / "dataset_diagnostics.md"
    json_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(diagnostics), encoding="utf-8")

    print(f"Dataset diagnostics: {len(diagnostics['warnings'])} warning(s)")
    print(f"Wrote {json_path} and {md_path}")
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect synthetic dataset splits.")
    parser.add_argument("--config", type=Path, required=True, help="Path to an experiment YAML config.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    inspect_dataset(args.config, output_dir=args.output_dir)


def _raw_split_series(series: NDArray[np.float64], dataset_config: Any) -> dict[str, NDArray[np.float64]]:
    train_end = int(dataset_config.n_timesteps * dataset_config.train_fraction)
    val_end = train_end + int(dataset_config.n_timesteps * dataset_config.val_fraction)
    return {
        "train": series[:train_end],
        "val": series[train_end:val_end],
        "test": series[val_end:],
    }


def _target_stats(y: NDArray[np.float64]) -> dict[str, float]:
    target = y.reshape(-1)
    return {
        "mean": float(target.mean()),
        "std": float(target.std()),
        "min": float(target.min()),
        "max": float(target.max()),
    }


def _feature_stats(values: NDArray[np.float64]) -> dict[str, list[float]]:
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
    }


def _persistence_rmse(X: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    model = PersistenceBaseline().fit(X, y)
    predictions = model.predict(X)
    return rmse(y, predictions)


def _range_warnings(target_stats: dict[str, dict[str, float]]) -> list[str]:
    train = target_stats["train"]
    train_range = max(train["max"] - train["min"], 1e-8)
    lower_bound = train["min"] - 0.25 * train_range
    upper_bound = train["max"] + 0.25 * train_range

    warnings = []
    for split_name in ("val", "test"):
        split = target_stats[split_name]
        if split["min"] < lower_bound or split["max"] > upper_bound:
            warnings.append(
                f"{split_name} target range [{split['min']:.4f}, {split['max']:.4f}] "
                f"is far outside train range [{train['min']:.4f}, {train['max']:.4f}]"
            )
    return warnings


def _render_markdown(diagnostics: dict[str, Any]) -> str:
    lines = [
        "# Dataset Diagnostics",
        "",
        f"- Config: `{diagnostics['config_path']}`",
        f"- Dataset mode: `{diagnostics['dataset']['mode']}`",
        "",
        "## Split Sizes",
        "",
        "| Split | Timesteps | Windows | Persistence RMSE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split_name in ("train", "val", "test"):
        sizes = diagnostics["split_sizes"][split_name]
        persistence = diagnostics["naive_persistence_rmse"][split_name]
        lines.append(
            f"| {split_name} | {sizes['timesteps']} | {sizes['windows']} | {persistence:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Target Stats",
            "",
            "| Split | Mean | Std | Min | Max |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for split_name in ("train", "val", "test"):
        stats = diagnostics["target"][split_name]
        lines.append(
            f"| {split_name} | {stats['mean']:.4f} | {stats['std']:.4f} | "
            f"{stats['min']:.4f} | {stats['max']:.4f} |"
        )

    warnings = diagnostics["warnings"]
    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No major train-to-validation/test target range warning.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
