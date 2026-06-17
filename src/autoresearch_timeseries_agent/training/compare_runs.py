from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compare_runs(
    *,
    runs_dir: Path = Path("reports/runs"),
    output_dir: Path = Path("reports"),
) -> dict[str, Any]:
    run_results = _load_run_results(runs_dir)
    rows = [_comparison_row(result) for result in run_results]
    ranked_rows = sorted(rows, key=lambda row: (row["val_rmse"], row["experiment_name"]))

    comparison = {
        "ranking_metric": "val_rmse",
        "selection_note": "Models are ranked by validation RMSE. Test metrics are held out for final evaluation.",
        "best_experiment": ranked_rows[0]["experiment_name"],
        "runs": [
            {"rank": rank, **row}
            for rank, row in enumerate(ranked_rows, start=1)
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "model_comparison.json"
    md_path = output_dir / "model_comparison.md"
    json_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown_comparison(comparison), encoding="utf-8")

    best = comparison["runs"][0]
    print(
        f"Best by validation RMSE: {best['experiment_name']} "
        f"({best['model_name']}, val RMSE={best['val_rmse']:.4f})"
    )
    print(f"Wrote {json_path} and {md_path}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved baseline experiment runs.")
    parser.add_argument("--runs-dir", type=Path, default=Path("reports/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    compare_runs(runs_dir=args.runs_dir, output_dir=args.output_dir)


def _load_run_results(runs_dir: Path) -> list[dict[str, Any]]:
    if not runs_dir.exists():
        msg = f"Runs directory does not exist: {runs_dir}"
        raise FileNotFoundError(msg)

    paths = sorted(runs_dir.glob("*.json"))
    if not paths:
        msg = f"No run JSON files found in {runs_dir}"
        raise ValueError(msg)

    results = []
    for path in paths:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            msg = f"Run file must contain a JSON object: {path}"
            raise ValueError(msg)
        results.append(loaded)
    return results


def _comparison_row(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    model = result["model"]
    return {
        "experiment_name": result["experiment_name"],
        "model_name": model["name"],
        "val_rmse": float(metrics["val"]["rmse"]),
        "val_mae": float(metrics["val"]["mae"]),
        "val_mape": float(metrics["val"]["mape"]),
        "test_rmse": float(metrics["test"]["rmse"]),
        "test_mae": float(metrics["test"]["mae"]),
        "test_mape": float(metrics["test"]["mape"]),
    }


def _render_markdown_comparison(comparison: dict[str, Any]) -> str:
    lines = [
        "# Model Comparison",
        "",
        "Models are ranked by validation RMSE. Test metrics are included only as held-out "
        "evaluation context and are not used for model selection.",
        "",
        "| Rank | Experiment | Model | Val RMSE | Val MAE | Val MAPE | Test RMSE | Test MAE | Test MAPE |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison["runs"]:
        lines.append(
            f"| {row['rank']} | {row['experiment_name']} | {row['model_name']} | "
            f"{row['val_rmse']:.4f} | {row['val_mae']:.4f} | {row['val_mape']:.2f}% | "
            f"{row['test_rmse']:.4f} | {row['test_mae']:.4f} | {row['test_mape']:.2f}% |"
        )

    lines.extend(["", f"Best by validation RMSE: `{comparison['best_experiment']}`", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
