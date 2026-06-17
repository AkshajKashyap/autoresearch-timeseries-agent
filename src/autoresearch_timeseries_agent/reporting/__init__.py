from importlib import import_module
from typing import Any

__all__ = [
    "build_project_summary",
    "generate_plots",
]


def __getattr__(name: str) -> Any:
    if name == "build_project_summary":
        module = import_module("autoresearch_timeseries_agent.reporting.build_project_summary")
        return module.build_project_summary
    if name == "generate_plots":
        module = import_module("autoresearch_timeseries_agent.reporting.plots")
        return module.generate_plots
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
