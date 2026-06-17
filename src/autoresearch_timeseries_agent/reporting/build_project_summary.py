from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_GOAL = (
    "Build a reproducible, local time-series forecasting benchmark foundation that a "
    "future agent layer can control through configs and report files."
)

SUPPORTED_DATASET_SOURCES = ["synthetic", "csv"]
SUPPORTED_MODELS = ["persistence", "linear", "lstm", "transformer"]
LIMITATIONS = [
    "The included CSV is a deterministic local ingestion demo, not an external benchmark.",
    "Blocked-shuffle results are diagnostic-only and should not be treated as deployment estimates.",
    "Neural configs are intentionally small CPU-friendly baselines, not exhaustive tuning.",
    "The current local agent is deterministic and config-bounded; it is not an LLM agent.",
]
NEXT_STEPS = [
    "Keep model selection focused on chronological validation RMSE.",
    "Add real external datasets only after documenting provenance and evaluation assumptions.",
    "Use the plotting and project summary outputs as the default GitHub-facing report layer.",
    "Consider Transformer or agent extensions only after the reporting baseline remains stable.",
]


def build_project_summary(
    *,
    runs_dir: Path = Path("reports/runs"),
    comparison_path: Path = Path("reports/model_comparison.json"),
    diagnostics_path: Path = Path("reports/dataset_diagnostics.json"),
    agent_report_path: Path = Path("reports/agent/agent_final_report.json"),
    output_dir: Path = Path("reports"),
) -> dict[str, Any]:
    runs = _load_run_results(runs_dir)
    comparison = _load_optional_json(comparison_path)
    diagnostics = _load_optional_json(diagnostics_path)
    agent_report = _load_optional_json(agent_report_path)

    summary = {
        "project_goal": PROJECT_GOAL,
        "supported_dataset_sources": SUPPORTED_DATASET_SOURCES,
        "supported_models": SUPPORTED_MODELS,
        "split_strategy_explanation": _split_strategy_explanation(),
        "evaluation_caveat": _evaluation_caveat(),
        "best_realistic_chronological": _best_by_split(runs, "chronological"),
        "best_diagnostic_blocked_shuffle": _best_by_split(runs, "blocked_shuffle"),
        "csv_results": _csv_results(runs),
        "agent_summary": _agent_summary(agent_report),
        "dataset_diagnostics": _diagnostics_summary(diagnostics),
        "comparison": _comparison_summary(comparison),
        "limitations": LIMITATIONS,
        "next_steps": NEXT_STEPS,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "project_summary.json"
    md_path = output_dir / "project_summary.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(summary), encoding="utf-8")

    print(f"Wrote {json_path} and {md_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the GitHub-facing project summary.")
    parser.add_argument("--runs-dir", type=Path, default=Path("reports/runs"))
    parser.add_argument(
        "--comparison-path",
        type=Path,
        default=Path("reports/model_comparison.json"),
    )
    parser.add_argument(
        "--diagnostics-path",
        type=Path,
        default=Path("reports/dataset_diagnostics.json"),
    )
    parser.add_argument(
        "--agent-report-path",
        type=Path,
        default=Path("reports/agent/agent_final_report.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    build_project_summary(
        runs_dir=args.runs_dir,
        comparison_path=args.comparison_path,
        diagnostics_path=args.diagnostics_path,
        agent_report_path=args.agent_report_path,
        output_dir=args.output_dir,
    )


def _load_run_results(runs_dir: Path) -> list[dict[str, Any]]:
    if not runs_dir.exists():
        msg = f"Runs directory does not exist: {runs_dir}"
        raise FileNotFoundError(msg)

    paths = sorted(runs_dir.glob("*.json"))
    if not paths:
        msg = f"No run JSON files found in {runs_dir}"
        raise ValueError(msg)

    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = f"JSON report must contain an object: {path}"
        raise ValueError(msg)
    return loaded


def _best_by_split(runs: list[dict[str, Any]], split_strategy: str) -> dict[str, Any] | None:
    candidates = [
        _run_summary(run)
        for run in runs
        if _split_strategy(run) == split_strategy
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row["val_rmse"], row["experiment_name"]))


def _csv_results(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    csv_runs = [
        _run_summary(run)
        for run in runs
        if _dataset_source(run) == "csv"
    ]
    return sorted(csv_runs, key=lambda row: (row["val_rmse"], row["experiment_name"]))


def _run_summary(run: dict[str, Any]) -> dict[str, Any]:
    metrics = run["metrics"]
    return {
        "experiment_name": str(run["experiment_name"]),
        "model_name": str(run["model"]["name"]),
        "dataset_source": _dataset_source(run),
        "split_strategy": _split_strategy(run),
        "val_rmse": float(metrics["val"]["rmse"]),
        "test_rmse": float(metrics["test"]["rmse"]),
        "scale_features": bool(run.get("scale_features", False)),
        "normalize_target": bool(run.get("normalize_target", False)),
    }


def _dataset_source(run: dict[str, Any]) -> str:
    dataset = run.get("dataset", {})
    metadata = run.get("dataset_metadata", {})
    return str(run.get("dataset_source", metadata.get("source", dataset.get("source", "synthetic"))))


def _split_strategy(run: dict[str, Any]) -> str:
    dataset = run.get("dataset", {})
    return str(run.get("split_strategy", dataset.get("split_strategy", "chronological")))


def _agent_summary(agent_report: dict[str, Any] | None) -> dict[str, Any]:
    if agent_report is None:
        return {
            "available": False,
            "best_generated": None,
            "generated_config_count": 0,
            "critique": "No agent final report was available.",
        }
    return {
        "available": True,
        "objective": agent_report.get("objective"),
        "generated_config_count": len(agent_report.get("generated_configs", [])),
        "best_generated": agent_report.get("best_generated"),
        "comparison_best": agent_report.get("comparison_best"),
        "critique": agent_report.get("critique"),
        "recommended_next_experiment": agent_report.get("recommended_next_experiment"),
    }


def _diagnostics_summary(diagnostics: dict[str, Any] | None) -> dict[str, Any]:
    if diagnostics is None:
        return {"available": False}
    return {
        "available": True,
        "source": diagnostics.get("source"),
        "row_count": diagnostics.get("row_count"),
        "target_column": diagnostics.get("target_column"),
        "selected_feature_columns": diagnostics.get("selected_feature_columns", []),
        "warnings": diagnostics.get("warnings", []),
    }


def _comparison_summary(comparison: dict[str, Any] | None) -> dict[str, Any]:
    if comparison is None:
        return {"available": False}
    return {
        "available": True,
        "ranking_metric": comparison.get("ranking_metric"),
        "best_experiment": comparison.get("best_experiment"),
        "selection_note": comparison.get("selection_note"),
    }


def _split_strategy_explanation() -> dict[str, str]:
    return {
        "chronological": (
            "Train, validation, and test windows stay ordered in time. This is the "
            "realistic evaluation mode for forecasting claims."
        ),
        "blocked_shuffle": (
            "Windows are created then deterministically shuffled into splits. This is "
            "diagnostic-only and helps separate model capacity from temporal shift."
        ),
    }


def _evaluation_caveat() -> str:
    return (
        "Model selection should use validation RMSE from chronological runs. Test metrics "
        "are held out for final context, and blocked_shuffle runs are diagnostic-only."
    )


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Project Summary",
        "",
        "## Project Goal",
        "",
        summary["project_goal"],
        "",
        "## Current Capabilities",
        "",
        f"- Dataset sources: `{', '.join(summary['supported_dataset_sources'])}`",
        f"- Models: `{', '.join(summary['supported_models'])}`",
        "- Metrics: RMSE, MAE, MAPE, per-horizon RMSE",
        "",
        "## Evaluation Design",
        "",
        f"- Chronological: {summary['split_strategy_explanation']['chronological']}",
        f"- Blocked shuffle: {summary['split_strategy_explanation']['blocked_shuffle']}",
        f"- Caveat: {summary['evaluation_caveat']}",
        "",
        "## Headline Results",
        "",
        _format_best("Best realistic chronological", summary["best_realistic_chronological"]),
        _format_best("Best diagnostic blocked_shuffle", summary["best_diagnostic_blocked_shuffle"]),
        "",
        "## CSV Results",
        "",
    ]
    lines.extend(_csv_table(summary["csv_results"]))
    lines.extend(
        [
            "",
            "The included CSV is a deterministic local ingestion demo, not an external benchmark.",
            "",
            "## Agent Summary",
            "",
            _format_agent(summary["agent_summary"]),
            "",
            "## Dataset Diagnostics",
            "",
            _format_diagnostics(summary["dataset_diagnostics"]),
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary["limitations"])
    lines.extend(["", "## Next Steps", ""])
    lines.extend(f"- {item}" for item in summary["next_steps"])
    lines.append("")
    return "\n".join(lines)


def _format_best(label: str, row: dict[str, Any] | None) -> str:
    if row is None:
        return f"- {label}: no matching runs found."
    return (
        f"- {label}: `{row['experiment_name']}` ({row['model_name']}, "
        f"val RMSE={row['val_rmse']:.4f}, test RMSE={row['test_rmse']:.4f})."
    )


def _csv_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No CSV-backed runs were found."]
    lines = [
        "| Experiment | Model | Split | Val RMSE | Test RMSE |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment_name']} | {row['model_name']} | {row['split_strategy']} | "
            f"{row['val_rmse']:.4f} | {row['test_rmse']:.4f} |"
        )
    return lines


def _format_agent(agent_summary: dict[str, Any]) -> str:
    if not agent_summary["available"]:
        return "No agent final report was available."
    best = agent_summary.get("best_generated") or {}
    best_name = best.get("experiment_name", "none")
    return (
        f"Objective: {agent_summary.get('objective')}. Generated "
        f"{agent_summary['generated_config_count']} config(s). Best generated run: "
        f"`{best_name}`. Critique: {agent_summary.get('critique')}"
    )


def _format_diagnostics(diagnostics: dict[str, Any]) -> str:
    if not diagnostics["available"]:
        return "No dataset diagnostics report was available."
    warnings = diagnostics.get("warnings", [])
    warning_text = f"{len(warnings)} warning(s)" if warnings else "no warnings"
    return (
        f"Latest diagnostics source: `{diagnostics.get('source')}` with target "
        f"`{diagnostics.get('target_column')}` and {warning_text}."
    )


if __name__ == "__main__":
    main()
