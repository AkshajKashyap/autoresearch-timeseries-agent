from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from autoresearch_timeseries_agent.data import make_dataset
from autoresearch_timeseries_agent.data.windowing import WindowedDataset
from autoresearch_timeseries_agent.evaluation import rmse
from autoresearch_timeseries_agent.models import PersistenceBaseline
from autoresearch_timeseries_agent.training.run_experiment import load_experiment_config


def inspect_dataset(config_path: Path, *, output_dir: Path = Path("reports")) -> dict[str, Any]:
    experiment_config = load_experiment_config(config_path)
    splits = make_dataset(experiment_config.dataset)
    split_map = {"train": splits.train, "val": splits.val, "test": splits.test}
    metadata = splits.metadata

    diagnostics = {
        "config_path": str(config_path),
        "dataset": asdict(experiment_config.dataset),
        "dataset_metadata": metadata,
        "source": metadata.get("source", "synthetic"),
        "path": metadata.get("path"),
        "row_count": metadata.get("row_count", splits.raw_series.shape[0]),
        "target_column": metadata.get("target_column", "feature_0"),
        "selected_feature_columns": metadata.get("selected_feature_columns", []),
        "split_strategy": experiment_config.dataset.split_strategy,
        "split_sizes": {
            name: _split_size(split)
            for name, split in split_map.items()
        },
        "target": {
            name: _target_stats(split.y)
            for name, split in split_map.items()
        },
        "target_range_overlap": {},
        "features": {
            name: _feature_stats(split.X)
            for name, split in split_map.items()
        },
        "feature_shift": {
            name: _feature_shift(split_map["train"].X, split_map[name].X)
            for name in split_map
        },
        "naive_persistence_rmse": {
            name: _persistence_rmse(split.X, split.y)
            for name, split in split_map.items()
        },
    }
    diagnostics["target_range_overlap"] = _target_range_overlap(diagnostics["target"])
    diagnostics["warnings"] = _range_warnings(
        diagnostics["target"],
        diagnostics["target_range_overlap"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dataset_diagnostics.json"
    md_path = output_dir / "dataset_diagnostics.md"
    json_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(diagnostics), encoding="utf-8")

    print(f"Dataset diagnostics: {len(diagnostics['warnings'])} warning(s)")
    print(f"Wrote {json_path} and {md_path}")
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dataset splits.")
    parser.add_argument("--config", type=Path, required=True, help="Path to an experiment YAML config.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    inspect_dataset(args.config, output_dir=args.output_dir)


def _split_size(split: WindowedDataset) -> dict[str, int]:
    return {
        "windows": split.X.shape[0],
        "input_length": split.X.shape[1],
        "forecast_horizon": split.y.shape[1],
    }


def _target_stats(y: Any) -> dict[str, float]:
    target = y.reshape(-1)
    return {
        "mean": float(target.mean()),
        "std": float(target.std()),
        "min": float(target.min()),
        "max": float(target.max()),
    }


def _feature_stats(values: Any) -> dict[str, list[float]]:
    flattened = values.reshape(-1, values.shape[-1])
    return {
        "mean": flattened.mean(axis=0).tolist(),
        "std": flattened.std(axis=0).tolist(),
    }


def _feature_shift(train_X: Any, split_X: Any) -> dict[str, float]:
    train = train_X.reshape(-1, train_X.shape[-1])
    split = split_X.reshape(-1, split_X.shape[-1])
    train_mean = train.mean(axis=0)
    train_std = train.std(axis=0)
    split_mean = split.mean(axis=0)
    split_std = split.std(axis=0)
    return {
        "mean_abs_mean_shift": float(abs(split_mean - train_mean).mean()),
        "mean_std_ratio": float((split_std / np.maximum(train_std, 1e-8)).mean()),
    }


def _persistence_rmse(X: Any, y: Any) -> float:
    model = PersistenceBaseline().fit(X, y)
    predictions = model.predict(X)
    return rmse(y, predictions)


def _target_range_overlap(
    target_stats: dict[str, dict[str, float]],
) -> dict[str, dict[str, float | bool]]:
    train = target_stats["train"]
    overlaps = {}
    for split_name in ("val", "test"):
        split = target_stats[split_name]
        overlap_min = max(train["min"], split["min"])
        overlap_max = min(train["max"], split["max"])
        overlap = max(overlap_max - overlap_min, 0.0)
        split_range = max(split["max"] - split["min"], 1e-8)
        overlaps[split_name] = {
            "overlaps_train": overlap > 0.0,
            "overlap_ratio": overlap / split_range,
        }
    return overlaps


def _range_warnings(
    target_stats: dict[str, dict[str, float]],
    target_overlap: dict[str, dict[str, float | bool]],
) -> list[str]:
    train = target_stats["train"]

    warnings = []
    for split_name in ("val", "test"):
        split = target_stats[split_name]
        overlap_ratio = float(target_overlap[split_name]["overlap_ratio"])
        if overlap_ratio < 0.8:
            warnings.append(
                f"{split_name} target range [{split['min']:.4f}, {split['max']:.4f}] "
                f"has only {overlap_ratio:.2f} overlap with train range "
                f"[{train['min']:.4f}, {train['max']:.4f}]"
            )
    return warnings


def _render_markdown(diagnostics: dict[str, Any]) -> str:
    source = diagnostics.get("source", diagnostics["dataset"].get("source", "synthetic"))
    lines = [
        "# Dataset Diagnostics",
        "",
        f"- Config: `{diagnostics['config_path']}`",
        f"- Dataset source: `{source}`",
        f"- Split strategy: `{diagnostics['split_strategy']}`",
        f"- Row count: `{diagnostics['row_count']}`",
        f"- Target column: `{diagnostics['target_column']}`",
        f"- Selected feature columns: `{diagnostics['selected_feature_columns']}`",
        "",
        "## Split Sizes",
        "",
        "| Split | Windows | Input Length | Forecast Horizon | Persistence RMSE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    if source == "csv":
        lines.insert(4, f"- CSV path: `{diagnostics['path']}`")
    else:
        lines.insert(4, f"- Dataset mode: `{diagnostics['dataset'].get('mode', 'linear')}`")
    for split_name in ("train", "val", "test"):
        sizes = diagnostics["split_sizes"][split_name]
        persistence = diagnostics["naive_persistence_rmse"][split_name]
        lines.append(
            f"| {split_name} | {sizes['windows']} | {sizes['input_length']} | "
            f"{sizes['forecast_horizon']} | {persistence:.4f} |"
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

    lines.extend(
        [
            "",
            "## Target Range Overlap",
            "",
            "| Split | Overlaps Train | Overlap Ratio |",
            "| --- | --- | ---: |",
        ]
    )
    for split_name in ("val", "test"):
        overlap = diagnostics["target_range_overlap"][split_name]
        lines.append(
            f"| {split_name} | {overlap['overlaps_train']} | "
            f"{overlap['overlap_ratio']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Feature Shift",
            "",
            "| Split | Mean Absolute Mean Shift | Mean Std Ratio |",
            "| --- | ---: | ---: |",
        ]
    )
    for split_name in ("train", "val", "test"):
        shift = diagnostics["feature_shift"][split_name]
        lines.append(
            f"| {split_name} | {shift['mean_abs_mean_shift']:.4f} | "
            f"{shift['mean_std_ratio']:.4f} |"
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
