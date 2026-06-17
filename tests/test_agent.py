from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch_timeseries_agent.agents import (
    ExperimentProposal,
    plan_experiments,
    run_agent,
    validate_experiment_proposal,
)
from autoresearch_timeseries_agent.tools import write_experiment_config


def test_agent_planner_produces_at_most_three_experiments() -> None:
    proposals = plan_experiments(
        "Improve chronological validation RMSE under a small CPU budget",
        comparison_report=None,
        dataset_diagnostics=None,
        max_experiments=3,
    )

    assert len(proposals) <= 3
    assert all(proposal.split_strategy == "chronological" for proposal in proposals)


def test_agent_planner_uses_blocked_shuffle_only_when_requested() -> None:
    proposals = plan_experiments(
        "Run a diagnostic blocked shuffle capacity check",
        comparison_report=None,
        dataset_diagnostics=None,
        max_experiments=2,
    )

    assert len(proposals) == 2
    assert all(proposal.split_strategy == "blocked_shuffle" for proposal in proposals)


def test_generated_configs_are_written_under_agent_generated(tmp_path: Path) -> None:
    config = {"experiment": {"name": "agent_test"}}

    path = write_experiment_config(
        config,
        config_name="agent_test",
        generated_dir=tmp_path / "configs" / "agent_generated",
    )

    assert path.parent == tmp_path / "configs" / "agent_generated"
    assert path.name == "agent_test.yaml"


def test_unsupported_model_names_are_rejected() -> None:
    proposal = ExperimentProposal(
        name="bad",
        model_name="cnn",
        split_strategy="chronological",
        config={},
        rationale="invalid",
    )

    with pytest.raises(ValueError, match="Unsupported model name"):
        validate_experiment_proposal(proposal)


def test_unsupported_split_strategies_are_rejected() -> None:
    proposal = ExperimentProposal(
        name="bad",
        model_name="linear",
        split_strategy="random",
        config={},
        rationale="invalid",
    )

    with pytest.raises(ValueError, match="Unsupported split strategy"):
        validate_experiment_proposal(proposal)


def test_run_agent_smoke_with_mocked_execution(tmp_path: Path) -> None:
    def fake_experiment_tool(config_path: Path) -> dict[str, object]:
        name = config_path.stem
        result = {
            "experiment_name": name,
            "model": {"name": "linear" if "linear" in name else "transformer"},
            "split_strategy": "chronological",
            "scale_features": "linear" not in name,
            "normalize_target": "linear" not in name,
            "metrics": {
                "val": {"rmse": 1.0 if "linear" in name else 2.0},
                "test": {"rmse": 1.5 if "linear" in name else 2.5},
            },
        }
        return {
            "command": f"python -m autoresearch_timeseries_agent.training.run_experiment --config {config_path}",
            "result": result,
        }

    def fake_comparison_tool() -> dict[str, object]:
        return {
            "command": "python -m autoresearch_timeseries_agent.training.compare_runs",
            "result": {
                "runs": [
                    {
                        "experiment_name": "agent_linear_alpha_0_1_chronological",
                        "model_name": "linear",
                        "split_strategy": "chronological",
                        "val_rmse": 1.0,
                        "test_rmse": 1.5,
                    }
                ]
            },
        }

    result = run_agent(
        "Improve chronological validation RMSE under a small CPU budget",
        project_root=tmp_path,
        max_experiments=2,
        experiment_tool=fake_experiment_tool,
        comparison_tool=fake_comparison_tool,
    )

    assert len(result.generated_configs) == 2
    assert (tmp_path / "reports" / "agent" / "agent_plan.json").exists()
    assert (tmp_path / "reports" / "agent" / "agent_plan.md").exists()
    assert (tmp_path / "reports" / "agent" / "agent_final_report.json").exists()
    assert (tmp_path / "reports" / "agent" / "agent_final_report.md").exists()
    assert result.best_generated is not None
    assert result.best_generated["experiment_name"] == "agent_linear_alpha_0_1_chronological"


def test_agent_does_not_overwrite_existing_handwritten_configs(tmp_path: Path) -> None:
    handwritten = tmp_path / "configs" / "linear.yaml"
    handwritten.parent.mkdir(parents=True)
    handwritten.write_text("handwritten: true\n", encoding="utf-8")

    write_experiment_config(
        {"experiment": {"name": "agent_linear"}},
        config_name="agent_linear",
        generated_dir=tmp_path / "configs" / "agent_generated",
    )

    assert handwritten.read_text(encoding="utf-8") == "handwritten: true\n"
