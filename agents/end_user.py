"""
agents/end_user.py — End-user perspective evaluation agent (Ollama).

Evaluates the current solution from the perspective of the primary end user,
as identified and described by the orchestrator for each specific goal:
  - Usability: how easy is the system to interact with day-to-day?
  - Clarity: how clear and understandable is the output?
  - Cost: how affordable and cost-effective for the end user?
  - Practicality: how well does it fit the end user's real-world workflows?

Returns: DriverFeedback (scores 0-10, confidence 0-1, concerns, recommendations)
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agents._ollama_base import _call_ollama, build_ollama_llm
from agents.schemas import DriverFeedback
from state import PipelineState
from utils.logging_config import get_logger, make_log_entry

logger = get_logger(__name__)

_SYSTEM = """\
You are the End User evaluation agent in a multi-agent pipeline.
Your job is to evaluate proposed solutions from the perspective of the primary end user \
described in the evaluation task below.

Score each criterion from 0 (terrible) to 10 (excellent):
  - usability_score:    How intuitive and easy is the system to use day-to-day?
  - clarity_score:      How clear, readable, and actionable is the information presented?
  - cost_score:         How affordable and cost-effective is this for the end user?
  - practicality_score: How well does it fit the end user's real-world workflows and constraints?

Also report your confidence (0-1) in your assessment.
List the top concerns and specific, actionable recommendations.

Stay strictly within the end-user perspective — do not assess policy/compliance or system architecture.
Respond with a valid JSON object matching this exact schema — nothing else.
"""

_USER_TEMPLATE = """\
## Goal
{user_goal}

## Evaluation Task
{instruction}

## Solution to Evaluate
{solution_summary}

## Components
{components}

## Decision Logic
{decision_logic}

## What to Focus On
{priority_focus}
"""


def _build_messages(state: PipelineState) -> list:
    cfg_instructions = state["coordination_instructions"] or {}
    instruction = cfg_instructions.get("driver", "Evaluate from the primary end-user perspective.")
    artifact = state["solution_artifact"] or {}

    components = "\n".join(
        f"  - {c.get('name', '?')}: {c.get('description', '')}"
        for c in artifact.get("components", [])
    ) or "  (no components yet)"

    priority = ", ".join(state.get("priority_focus", [])) or "general evaluation"

    user_content = _USER_TEMPLATE.format(
        user_goal=state["user_goal"],
        instruction=instruction,
        solution_summary=artifact.get("design_summary", "No solution yet — evaluate the feasibility of the goal."),
        components=components,
        decision_logic=artifact.get("decision_logic", "N/A"),
        priority_focus=priority,
    )
    return [SystemMessage(content=_SYSTEM), HumanMessage(content=user_content)]


def end_user_node(state: PipelineState) -> dict:
    """LangGraph node: evaluate solution from end-user perspective."""
    run_id = state["run_id"]
    iteration = state["iteration"]
    logger.info(
        "End User agent evaluating",
        extra={"run_id": run_id, "iteration": iteration, "node": "end_user"},
    )

    llm = build_ollama_llm("driver_agent")
    messages = _build_messages(state)

    result, fallback_logs = _call_ollama(
        llm=llm,
        messages=messages,
        schema=DriverFeedback,
        agent_name="end_user",
        state_run_id=run_id,
        state_iteration=iteration,
    )

    avg_score = (
        result.usability_score
        + result.clarity_score
        + result.cost_score
        + result.practicality_score
    ) / 4

    log_entry = make_log_entry(
        event="end_user_complete",
        node="end_user",
        run_id=run_id,
        iteration=iteration,
        message=f"End User avg={avg_score:.2f}, confidence={result.confidence:.2f}",
        avg_score=avg_score,
        confidence=result.confidence,
    )
    logger.info(log_entry["message"], extra=log_entry)

    return {
        "driver_feedback": result.model_dump(),
        "log_entries": fallback_logs + [log_entry],
    }
