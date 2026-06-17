"""Training entry points and experiment runners."""

from importlib import import_module
from typing import Any

__all__ = [
    "ExperimentConfig",
    "ModelConfig",
    "build_model",
    "load_experiment_config",
    "run_experiment",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module("autoresearch_timeseries_agent.training.run_experiment")
        return getattr(module, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
