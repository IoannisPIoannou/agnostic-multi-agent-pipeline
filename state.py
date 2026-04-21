"""
state.py — Shared pipeline state.

PipelineState is the single source of truth passed through every LangGraph node.
Design rules:
  - Nodes return dicts updating ONLY the keys they own.
  - Parallel nodes (driver / policy / software) each write to a unique key → no merge conflict.
  - List fields that multiple nodes append to (log_entries, metrics_history) use
    Annotated[list, operator.add] so LangGraph merges them automatically.
"""

import operator
from typing import Annotated, Any, TypedDict


class PipelineState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    user_goal: str          # Original problem statement
    run_id: str             # UUID for this run, set once in main.py

    # ── Control flow ──────────────────────────────────────────────────────────
    iteration: int          # Current iteration (incremented by orchestrator)
    max_iterations: int     # Hard ceiling from config
    score_threshold: float  # Pipeline stops when evaluation_score >= this

    # ── Orchestrator outputs ───────────────────────────────────────────────────
    task_decomposition: dict | None       # subproblems, constraints, criteria
    coordination_instructions: dict       # {agent_name: specific_instruction}
    revision_strategy: str | None         # Populated on iteration > 1
    priority_focus: list                  # Ordered list of improvement areas

    # ── Parallel evaluator outputs (each agent owns ONE unique key) ────────────
    # These are set independently during the fan-out phase and never conflict.
    driver_feedback: dict | None
    policy_feedback: dict | None
    software_feedback: dict | None

    # ── Aggregated feedback (set by aggregator_node after fan-in) ─────────────
    aggregated_feedback: dict | None

    # ── Solution produced by the coding agent ─────────────────────────────────
    solution_artifact: dict | None

    # ── Evaluation results (set by evaluator_node) ────────────────────────────
    evaluation_score: float
    evaluation_details: dict

    # ── Final synthesis report (set by synthesis_node at the very end) ────────
    final_report: str | None

    # ── Append-only list fields — Annotated reducer merges parallel writes ─────
    # Multiple nodes may append entries; operator.add concatenates the lists.
    metrics_history: Annotated[list, operator.add]  # one IterationMetrics per cycle
    log_entries: Annotated[list, operator.add]       # structured event log

    # ── Stop signal (set by evaluator_node) ──────────────────────────────────
    should_stop: bool
    stop_reason: str  # "threshold_reached" | "converged" | "max_iterations" | ""


def initial_state(user_goal: str, run_id: str, cfg: dict) -> PipelineState:
    """Build a fresh PipelineState for a new run."""
    return PipelineState(
        user_goal=user_goal,
        run_id=run_id,
        iteration=0,
        max_iterations=cfg["pipeline"]["max_iterations"],
        score_threshold=cfg["pipeline"]["score_threshold"],
        task_decomposition=None,
        coordination_instructions={},
        revision_strategy=None,
        priority_focus=[],
        driver_feedback=None,
        policy_feedback=None,
        software_feedback=None,
        aggregated_feedback=None,
        solution_artifact=None,
        evaluation_score=0.0,
        evaluation_details={},
        final_report=None,
        metrics_history=[],
        log_entries=[],
        should_stop=False,
        stop_reason="",
    )
