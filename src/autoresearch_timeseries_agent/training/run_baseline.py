import argparse
from pathlib import Path
from typing import Any

from autoresearch_timeseries_agent.training.run_experiment import run_experiment


def run_baseline(config_path: Path) -> dict[str, Any]:
    return run_experiment(config_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a baseline experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a baseline YAML config.")
    args = parser.parse_args()
    run_baseline(args.config)


if __name__ == "__main__":
    main()
