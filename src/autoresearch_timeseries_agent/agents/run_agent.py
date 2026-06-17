from __future__ import annotations

import argparse

from autoresearch_timeseries_agent.agents.experiment_agent import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic experiment agent.")
    parser.add_argument("--objective", required=True, help="Experiment objective for the local agent.")
    args = parser.parse_args()
    result = run_agent(args.objective)
    best = result.best_generated
    if best is None:
        print("Agent completed without generated experiment metrics.")
        return
    print(
        f"Agent best generated run: {best['experiment_name']} "
        f"val RMSE={best['val_rmse']:.4f}, test RMSE={best['test_rmse']:.4f}"
    )
    print("Wrote reports/agent/agent_plan.* and reports/agent/agent_final_report.*")


if __name__ == "__main__":
    main()
