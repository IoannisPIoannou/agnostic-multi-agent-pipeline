"""
experiments/configs/schema.py — Pydantic configuration schemas for all experiment types.

Load a config with:
    from experiments.configs.schema import load_config
    cfg = load_config("experiments/configs/convergence_example.yaml")
"""

from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ConvergenceConfig(BaseModel):
    """Experiment 1: run the same goal N times and measure convergence stability."""
    type: Literal["convergence"] = "convergence"
    name: str
    description: str = ""
    goal: str
    num_runs: int = Field(default=3, ge=1, le=10)
    pipeline_config_path: str | None = None  # override the default config.yaml
    seed: int | None = Field(default=None)  # set for reproducible Ollama runs


class AblationConfig(BaseModel):
    """Experiment 2: disable one evaluator at a time to measure each agent's contribution."""
    type: Literal["ablation"] = "ablation"
    name: str
    description: str = ""
    goal: str
    # Each listed agent produces one ablated run; the full run is always included.
    ablate_agents: list[Literal["end_user", "policy", "software"]] = [
        "end_user", "policy", "software"
    ]


class BaselineConfig(BaseModel):
    """Experiment 3: compare pipeline output against a single-LLM baseline via a judge."""
    type: Literal["baseline"] = "baseline"
    name: str
    description: str = ""
    goal: str
    baseline_model: str = "gemini-2.5-flash"
    judge_model: str = "gemini-2.5-flash"
    judge_criteria: list[str] = Field(default_factory=lambda: [
        "usability",
        "clarity",
        "feasibility",
        "stakeholder_balance",
        "overall_quality",
    ])


class RuntimeConfig(BaseModel):
    """Experiment 4: measure wall time, API call counts, and estimated cost per run."""
    type: Literal["runtime"] = "runtime"
    name: str
    description: str = ""
    goal: str
    num_runs: int = Field(default=1, ge=1, le=5)
    # Approximate Gemini 2.5 Flash pricing (USD per 1M tokens, as of 2025).
    # Update these if pricing changes.
    gemini_input_cost_per_mtok: float = 0.075
    gemini_output_cost_per_mtok: float = 0.300
    # Rough average token estimates per Gemini call type (adjust after profiling).
    avg_orchestrator_input_tokens: int = 800
    avg_orchestrator_output_tokens: int = 400
    avg_coding_input_tokens: int = 1200
    avg_coding_output_tokens: int = 800
    avg_synthesis_input_tokens: int = 1500
    avg_synthesis_output_tokens: int = 1000


def load_config(
    path: str,
) -> ConvergenceConfig | AblationConfig | BaselineConfig | RuntimeConfig:
    """Load and validate an experiment config from a YAML file."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    dispatch = {
        "convergence": ConvergenceConfig,
        "ablation":    AblationConfig,
        "baseline":    BaselineConfig,
        "runtime":     RuntimeConfig,
    }
    exp_type = raw.get("type")
    cls = dispatch.get(exp_type)
    if cls is None:
        raise ValueError(
            f"Unknown experiment type {exp_type!r}. "
            f"Must be one of: {list(dispatch)}"
        )
    return cls(**raw)
