from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def generate_plots(
    *,
    runs_dir: Path = Path("reports/runs"),
    output_dir: Path = Path("reports/figures"),
    max_selected_runs: int = 5,
) -> list[Path]:
    """Generate PNG figures from saved experiment run JSON files."""

    runs = _load_run_results(runs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = [
        _plot_single_metric(
            runs,
            metric_split="val",
            output_path=output_dir / "validation_rmse_by_experiment.png",
            title="Validation RMSE by Experiment",
        ),
        _plot_single_metric(
            runs,
            metric_split="test",
            output_path=output_dir / "test_rmse_by_experiment.png",
            title="Test RMSE by Experiment",
        ),
        _plot_split_rmse(
            runs,
            output_path=output_dir / "train_val_test_rmse_by_experiment.png",
        ),
    ]

    selected = _selected_runs(runs, max_selected_runs=max_selected_runs)
    horizon_plot = _plot_per_horizon_rmse(
        selected,
        output_path=output_dir / "per_horizon_rmse_selected_runs.png",
    )
    if horizon_plot is not None:
        generated.append(horizon_plot)

    sample_plot = _plot_actual_vs_predicted_samples(
        selected,
        output_path=output_dir / "actual_vs_predicted_samples.png",
    )
    if sample_plot is not None:
        generated.append(sample_plot)

    return generated


def _load_run_results(runs_dir: Path) -> list[dict[str, Any]]:
    if not runs_dir.exists():
        msg = f"Runs directory does not exist: {runs_dir}"
        raise FileNotFoundError(msg)

    paths = sorted(runs_dir.glob("*.json"))
    if not paths:
        msg = f"No run JSON files found in {runs_dir}"
        raise ValueError(msg)

    runs = []
    for path in paths:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            msg = f"Run JSON must contain an object: {path}"
            raise ValueError(msg)
        runs.append(loaded)
    return runs


def _plot_single_metric(
    runs: list[dict[str, Any]],
    *,
    metric_split: str,
    output_path: Path,
    title: str,
) -> Path:
    ordered = sorted(runs, key=lambda run: _rmse(run, metric_split))
    labels = [_experiment_name(run) for run in ordered]
    values = [_rmse(run, metric_split) for run in ordered]

    fig, ax = plt.subplots(figsize=_wide_figsize(labels))
    ax.bar(range(len(labels)), values, color="#3b82f6")
    ax.set_title(title)
    ax.set_ylabel("RMSE")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_split_rmse(runs: list[dict[str, Any]], *, output_path: Path) -> Path:
    ordered = sorted(runs, key=lambda run: _rmse(run, "val"))
    labels = [_experiment_name(run) for run in ordered]
    x_positions = list(range(len(labels)))
    width = 0.26
    colors = {"train": "#16a34a", "val": "#2563eb", "test": "#dc2626"}

    fig, ax = plt.subplots(figsize=_wide_figsize(labels))
    for offset, split in zip((-width, 0.0, width), ("train", "val", "test"), strict=True):
        values = [_rmse(run, split) for run in ordered]
        shifted = [position + offset for position in x_positions]
        ax.bar(shifted, values, width=width, label=split, color=colors[split])

    ax.set_title("Train/Validation/Test RMSE by Experiment")
    ax.set_ylabel("RMSE")
    ax.set_xticks(x_positions, labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_per_horizon_rmse(
    runs: list[dict[str, Any]],
    *,
    output_path: Path,
) -> Path | None:
    eligible = [
        run for run in runs if run.get("metrics", {}).get("test", {}).get("per_horizon_rmse")
    ]
    if not eligible:
        return None

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for run in eligible:
        values = run["metrics"]["test"]["per_horizon_rmse"]
        horizons = list(range(1, len(values) + 1))
        ax.plot(horizons, values, marker="o", linewidth=1.5, label=_experiment_name(run))

    ax.set_title("Test Per-Horizon RMSE for Selected Runs")
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("RMSE")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_actual_vs_predicted_samples(
    runs: list[dict[str, Any]],
    *,
    output_path: Path,
) -> Path | None:
    samples = [_prediction_sample(run) for run in runs]
    samples = [sample for sample in samples if sample is not None]
    if not samples:
        return None

    max_panels = min(len(samples), 4)
    fig, axes = plt.subplots(max_panels, 1, figsize=(9, 3.0 * max_panels), squeeze=False)
    for ax, sample in zip(axes.reshape(-1), samples[:max_panels], strict=False):
        horizons = list(range(1, len(sample["y_true"]) + 1))
        ax.plot(horizons, sample["y_true"], marker="o", label="actual", color="#111827")
        ax.plot(horizons, sample["y_pred"], marker="o", label="predicted", color="#dc2626")
        ax.set_title(sample["experiment_name"])
        ax.set_xlabel("Forecast Horizon")
        ax.set_ylabel("Target")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _prediction_sample(run: dict[str, Any]) -> dict[str, Any] | None:
    diagnostics = run.get("prediction_diagnostics", {}).get("test")
    if not diagnostics:
        return None

    y_true = diagnostics.get("y_true_sample") or []
    y_pred = diagnostics.get("y_pred_sample") or []
    if not y_true or not y_pred or not y_true[0] or not y_pred[0]:
        return None

    horizon = min(len(y_true[0]), len(y_pred[0]))
    if horizon <= 0:
        return None

    return {
        "experiment_name": _experiment_name(run),
        "y_true": y_true[0][:horizon],
        "y_pred": y_pred[0][:horizon],
    }


def _selected_runs(
    runs: list[dict[str, Any]],
    *,
    max_selected_runs: int,
) -> list[dict[str, Any]]:
    return sorted(runs, key=lambda run: (_rmse(run, "val"), _experiment_name(run)))[
        :max_selected_runs
    ]


def _rmse(run: dict[str, Any], split: str) -> float:
    return float(run["metrics"][split]["rmse"])


def _experiment_name(run: dict[str, Any]) -> str:
    return str(run["experiment_name"])


def _wide_figsize(labels: list[str]) -> tuple[float, float]:
    width = max(8.0, min(22.0, 0.65 * len(labels) + 4.0))
    return width, 5.5
