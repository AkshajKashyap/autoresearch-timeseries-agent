.PHONY: install test lint check plots summary health reports all

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

check:
	ruff check .
	pytest -q

plots:
	python -m autoresearch_timeseries_agent.reporting.generate_plots

summary:
	python -m autoresearch_timeseries_agent.reporting.build_project_summary

health:
	python -m autoresearch_timeseries_agent.reporting.check_repo_health

reports:
	python -m autoresearch_timeseries_agent.training.compare_runs
	python -m autoresearch_timeseries_agent.reporting.generate_plots
	python -m autoresearch_timeseries_agent.reporting.build_project_summary
	python -m autoresearch_timeseries_agent.reporting.check_repo_health

all: check reports
