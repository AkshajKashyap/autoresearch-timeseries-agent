from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from autoresearch_timeseries_agent.tools import (
    load_comparison_report,
    load_dataset_diagnostics,
    run_comparison,
    run_experiment_config,
    summarize_run_metrics,
    write_experiment_config,
)


SUPPORTED_MODELS = {"linear", "lstm", "transformer"}
SUPPORTED_SPLITS = {"chronological", "blocked_shuffle"}


@dataclass(frozen=True)
class ExperimentProposal:
    name: str
    model_name: str
    split_strategy: str
    config: dict[str, Any]
    rationale: str


@dataclass(frozen=True)
class AgentRunResult:
    objective: str
    generated_configs: list[str]
    commands_run: list[str]
    metrics_observed: list[dict[str, Any]]
    best_generated: dict[str, Any] | None
    comparison_best: dict[str, Any] | None
    critique: str
    recommended_next_experiment: str


ExperimentTool = Callable[[Path], dict[str, Any]]
ComparisonTool = Callable[[], dict[str, Any]]


def run_agent(
    objective: str,
    *,
    project_root: Path = Path("."),
    base_config_path: Path | None = None,
    max_experiments: int = 3,
    experiment_tool: ExperimentTool = run_experiment_config,
    comparison_tool: ComparisonTool = run_comparison,
) -> AgentRunResult:
    if max_experiments > 3:
        msg = f"Refusing to run more than 3 generated experiments; got {max_experiments}"
        raise ValueError(msg)

    comparison = load_comparison_report(project_root / "reports/model_comparison.json")
    diagnostics = load_dataset_diagnostics(project_root / "reports/dataset_diagnostics.json")
    base_config = _load_base_config(project_root, base_config_path, objective)
    proposals = plan_experiments(
        objective,
        comparison_report=comparison,
        dataset_diagnostics=diagnostics,
        base_config=base_config,
        max_experiments=max_experiments,
    )

    agent_dir = project_root / "reports/agent"
    generated_dir = project_root / "configs/agent_generated"
    _write_plan(objective, proposals, agent_dir)

    generated_configs = []
    commands_run = []
    metrics_observed = []
    for proposal in proposals:
        config_path = write_experiment_config(
            proposal.config,
            config_name=proposal.name,
            generated_dir=generated_dir,
        )
        generated_configs.append(config_path.as_posix())
        run_output = experiment_tool(config_path)
        commands_run.append(run_output["command"])
        metrics_observed.append(summarize_run_metrics(run_output["result"]))

    comparison_output = comparison_tool()
    commands_run.append(comparison_output["command"])
    final_comparison = comparison_output["result"]
    best_generated = _best_generated(metrics_observed)
    comparison_best = final_comparison["runs"][0] if final_comparison.get("runs") else None
    critique = _critique_result(
        objective=objective,
        best_generated=best_generated,
        comparison_best=comparison_best,
        previous_comparison=comparison,
    )
    recommended_next = _recommended_next_experiment(best_generated, objective)
    result = AgentRunResult(
        objective=objective,
        generated_configs=generated_configs,
        commands_run=commands_run,
        metrics_observed=metrics_observed,
        best_generated=best_generated,
        comparison_best=comparison_best,
        critique=critique,
        recommended_next_experiment=recommended_next,
    )
    _write_final_report(result, agent_dir)
    return result


def plan_experiments(
    objective: str,
    *,
    comparison_report: dict[str, Any] | None,
    dataset_diagnostics: dict[str, Any] | None,
    base_config: dict[str, Any] | None = None,
    max_experiments: int = 3,
) -> list[ExperimentProposal]:
    if max_experiments > 3:
        msg = f"Refusing to plan more than 3 experiments; got {max_experiments}"
        raise ValueError(msg)

    if base_config is not None:
        proposals = _template_probes(base_config, objective)
        split_strategy = proposals[0].split_strategy if proposals else "chronological"
    else:
        split_strategy = _choose_split_strategy(objective)
        proposals = [
            _linear_probe(split_strategy),
            _transformer_probe(split_strategy),
            _lstm_probe(split_strategy),
        ]
    if (
        base_config is None
        and _has_chronological_shift(dataset_diagnostics)
        and split_strategy == "chronological"
    ):
        proposals[1] = _transformer_shift_probe()

    _ = comparison_report
    return proposals[:max_experiments]


def validate_experiment_proposal(proposal: ExperimentProposal) -> None:
    if proposal.model_name not in SUPPORTED_MODELS:
        msg = f"Unsupported model name: {proposal.model_name!r}"
        raise ValueError(msg)
    if proposal.split_strategy not in SUPPORTED_SPLITS:
        msg = f"Unsupported split strategy: {proposal.split_strategy!r}"
        raise ValueError(msg)


def _load_base_config(
    project_root: Path,
    base_config_path: Path | None,
    objective: str,
) -> dict[str, Any] | None:
    candidate = base_config_path
    if candidate is None and "csv" in objective.lower():
        default_csv = project_root / "configs" / "csv_linear.yaml"
        candidate = default_csv if default_csv.exists() else None
    if candidate is None:
        return None

    path = candidate if candidate.is_absolute() else project_root / candidate
    if not path.exists():
        msg = f"Base config does not exist: {path}"
        raise FileNotFoundError(msg)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = f"Base config must be a YAML mapping: {path}"
        raise ValueError(msg)
    return loaded


def _choose_split_strategy(objective: str) -> str:
    lowered = objective.lower()
    if "blocked" in lowered or "diagnostic" in lowered:
        return "blocked_shuffle"
    return "chronological"


def _template_probes(base_config: dict[str, Any], objective: str) -> list[ExperimentProposal]:
    dataset = dict(base_config.get("dataset", {}))
    split_strategy = str(dataset.get("split_strategy", _choose_split_strategy(objective)))
    source = str(dataset.get("source", dataset.get("name", "synthetic"))).lower()
    prefix = "agent_csv" if source == "csv" else "agent_template"
    probes = [
        _template_probe(
            base_config,
            name=f"{prefix}_linear_alpha_0_1_{split_strategy}",
            model={"name": "linear", "alpha": 0.1},
            split_strategy=split_strategy,
            rationale="Probe a lower Ridge alpha while preserving the base dataset config.",
        ),
        _template_probe(
            base_config,
            name=f"{prefix}_transformer_small_{split_strategy}",
            model={
                "name": "transformer",
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "dim_feedforward": 64,
                "dropout": 0.1,
                "batch_size": 32,
                "epochs": 8,
                "learning_rate": 0.003,
                "seed": 42,
            },
            split_strategy=split_strategy,
            rationale="Run a small scaled Transformer variant on the base dataset config.",
        ),
        _template_probe(
            base_config,
            name=f"{prefix}_lstm_small_{split_strategy}",
            model={
                "name": "lstm",
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.0,
                "batch_size": 32,
                "epochs": 8,
                "learning_rate": 0.003,
                "seed": 42,
            },
            split_strategy=split_strategy,
            rationale="Run a small scaled LSTM variant on the base dataset config.",
        ),
    ]
    return probes


def _template_probe(
    base_config: dict[str, Any],
    *,
    name: str,
    model: dict[str, Any],
    split_strategy: str,
    rationale: str,
) -> ExperimentProposal:
    config = json.loads(json.dumps(base_config))
    config["experiment"] = {**config.get("experiment", {}), "name": name}
    config["dataset"] = {**config.get("dataset", {}), "split_strategy": split_strategy}
    config["model"] = model
    config["training"] = {
        **config.get("training", {}),
        "scale_features": model["name"] in {"lstm", "transformer"},
        "normalize_target": model["name"] in {"lstm", "transformer"},
    }
    config["reporting"] = {**config.get("reporting", {}), "runs_dir": "reports/runs"}
    proposal = ExperimentProposal(
        name=name,
        model_name=str(model["name"]),
        split_strategy=split_strategy,
        config=config,
        rationale=rationale,
    )
    validate_experiment_proposal(proposal)
    return proposal


def _base_config(name: str, model: dict[str, Any], split_strategy: str) -> dict[str, Any]:
    return {
        "experiment": {"name": name},
        "dataset": {
            "name": "synthetic",
            "mode": "nonlinear",
            "split_strategy": split_strategy,
            "n_timesteps": 600,
            "n_features": 4,
            "input_length": 48,
            "forecast_horizon": 24,
            "train_fraction": 0.7,
            "val_fraction": 0.15,
            "seed": 42,
        },
        "model": model,
        "training": {
            "scale_features": model["name"] in {"lstm", "transformer"},
            "normalize_target": model["name"] in {"lstm", "transformer"},
        },
        "reporting": {"runs_dir": "reports/runs"},
    }


def _linear_probe(split_strategy: str) -> ExperimentProposal:
    name = f"agent_linear_alpha_0_1_{split_strategy}"
    proposal = ExperimentProposal(
        name=name,
        model_name="linear",
        split_strategy=split_strategy,
        config=_base_config(name, {"name": "linear", "alpha": 0.1}, split_strategy),
        rationale="Probe the current strongest chronological baseline with a slightly lower Ridge alpha.",
    )
    validate_experiment_proposal(proposal)
    return proposal


def _transformer_probe(split_strategy: str) -> ExperimentProposal:
    name = f"agent_transformer_small_{split_strategy}"
    proposal = ExperimentProposal(
        name=name,
        model_name="transformer",
        split_strategy=split_strategy,
        config=_base_config(
            name,
            {
                "name": "transformer",
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "dim_feedforward": 64,
                "dropout": 0.1,
                "batch_size": 32,
                "epochs": 12,
                "learning_rate": 0.002,
                "seed": 42,
            },
            split_strategy,
        ),
        rationale="Run a small scaled Transformer candidate under the requested split.",
    )
    validate_experiment_proposal(proposal)
    return proposal


def _transformer_shift_probe() -> ExperimentProposal:
    proposal = _transformer_probe("chronological")
    config = json.loads(json.dumps(proposal.config))
    config["experiment"]["name"] = "agent_transformer_shift_probe_chronological"
    config["model"]["learning_rate"] = 0.001
    config["model"]["epochs"] = 16
    return ExperimentProposal(
        name="agent_transformer_shift_probe_chronological",
        model_name="transformer",
        split_strategy="chronological",
        config=config,
        rationale="Use a gentler Transformer learning rate because diagnostics indicate target shift.",
    )


def _lstm_probe(split_strategy: str) -> ExperimentProposal:
    name = f"agent_lstm_small_{split_strategy}"
    proposal = ExperimentProposal(
        name=name,
        model_name="lstm",
        split_strategy=split_strategy,
        config=_base_config(
            name,
            {
                "name": "lstm",
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.0,
                "batch_size": 32,
                "epochs": 12,
                "learning_rate": 0.003,
                "seed": 42,
            },
            split_strategy,
        ),
        rationale="Run a small scaled LSTM candidate for a controlled neural comparison.",
    )
    validate_experiment_proposal(proposal)
    return proposal


def _has_chronological_shift(dataset_diagnostics: dict[str, Any] | None) -> bool:
    if not dataset_diagnostics:
        return False
    return bool(dataset_diagnostics.get("warnings"))


def _write_plan(objective: str, proposals: list[ExperimentProposal], agent_dir: Path) -> None:
    agent_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "objective": objective,
        "stages": ["Planner", "Config Writer", "Runner", "Evaluator", "Critic", "Report Writer"],
        "experiments": [asdict(proposal) for proposal in proposals],
    }
    (agent_dir / "agent_plan.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Agent Plan",
        "",
        f"Objective: {objective}",
        "",
        "| Experiment | Model | Split | Rationale |",
        "| --- | --- | --- | --- |",
    ]
    for proposal in proposals:
        lines.append(
            f"| {proposal.name} | {proposal.model_name} | "
            f"{proposal.split_strategy} | {proposal.rationale} |"
        )
    lines.append("")
    (agent_dir / "agent_plan.md").write_text("\n".join(lines), encoding="utf-8")


def _write_final_report(result: AgentRunResult, agent_dir: Path) -> None:
    agent_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    (agent_dir / "agent_final_report.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Agent Final Report",
        "",
        f"Objective: {result.objective}",
        "",
        "## Generated Configs",
        "",
    ]
    lines.extend(f"- `{path}`" for path in result.generated_configs)
    lines.extend(["", "## Commands Run", ""])
    lines.extend(f"- `{command}`" for command in result.commands_run)
    lines.extend(
        [
            "",
            "## Metrics Observed",
            "",
            "| Experiment | Model | Split | Val RMSE | Test RMSE |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for metric in result.metrics_observed:
        lines.append(
            f"| {metric['experiment_name']} | {metric['model_name']} | "
            f"{metric['split_strategy']} | {metric['val_rmse']:.4f} | "
            f"{metric['test_rmse']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Best generated experiment: `{_name_or_none(result.best_generated)}`",
            f"Best comparison experiment: `{_name_or_none(result.comparison_best)}`",
            "",
            "## Critique",
            "",
            result.critique,
            "",
            "## Recommended Next Experiment",
            "",
            result.recommended_next_experiment,
            "",
        ]
    )
    (agent_dir / "agent_final_report.md").write_text("\n".join(lines), encoding="utf-8")


def _best_generated(metrics: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not metrics:
        return None
    return min(metrics, key=lambda row: (row["val_rmse"], row["experiment_name"]))


def _critique_result(
    *,
    objective: str,
    best_generated: dict[str, Any] | None,
    comparison_best: dict[str, Any] | None,
    previous_comparison: dict[str, Any] | None,
) -> str:
    if best_generated is None:
        return "No generated experiments were run."

    realism = _realism_label(best_generated)
    previous_best = _best_previous_chronological(previous_comparison)
    improvement = ""
    if previous_best:
        delta = previous_best["val_rmse"] - best_generated["val_rmse"]
        direction = "improved" if delta > 0 else "did not improve"
        improvement = (
            f" The best generated run {direction} over the previous chronological best "
            f"by {abs(delta):.4f} validation RMSE."
        )

    comparison_note = ""
    if comparison_best:
        comparison_note = (
            f" The overall comparison winner is {comparison_best['experiment_name']} "
            f"({ _realism_label(comparison_best) })."
        )
    return (
        f"The objective was: {objective}. The best generated run is "
        f"{best_generated['experiment_name']} with validation RMSE "
        f"{best_generated['val_rmse']:.4f}; it is {realism}.{improvement}{comparison_note}"
    )


def _recommended_next_experiment(best_generated: dict[str, Any] | None, objective: str) -> str:
    if best_generated is None:
        return "Re-run the agent with at least one generated experiment enabled."
    if best_generated["model_name"] == "linear":
        return (
            "Try one more chronological linear regularization probe around the winning alpha, "
            "then compare against the scaled Transformer with the same validation protocol."
        )
    if "chronological" in objective.lower():
        return (
            "Keep the chronological split and run a smaller learning-rate neural probe focused "
            "on extrapolation robustness rather than blocked-shuffle capacity."
        )
    return "Run the best diagnostic model under chronological splitting before making realistic claims."


def _best_previous_chronological(comparison_report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not comparison_report:
        return None
    chronological = [
        row for row in comparison_report.get("runs", []) if row.get("split_strategy") == "chronological"
    ]
    if not chronological:
        return None
    return min(chronological, key=lambda row: (row["val_rmse"], row["experiment_name"]))


def _realism_label(row: dict[str, Any]) -> str:
    return (
        "a realistic chronological result"
        if row.get("split_strategy") == "chronological"
        else "a diagnostic blocked-shuffle result"
    )


def _name_or_none(row: dict[str, Any] | None) -> str:
    if row is None:
        return "none"
    return str(row["experiment_name"])
