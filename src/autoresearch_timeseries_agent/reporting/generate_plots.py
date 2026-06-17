from __future__ import annotations

import argparse
from pathlib import Path

from autoresearch_timeseries_agent.reporting.plots import generate_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report figures from run JSON files.")
    parser.add_argument("--runs-dir", type=Path, default=Path("reports/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--max-selected-runs", type=int, default=5)
    args = parser.parse_args()

    paths = generate_plots(
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        max_selected_runs=args.max_selected_runs,
    )
    print(f"Generated {len(paths)} figure(s) under {args.output_dir}")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
