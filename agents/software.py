"""
agents/software.py — Software feasibility evaluation agent (Ollama).

Evaluates the solution from a technical / implementation perspective:
  - Complexity: how difficult is it to build? (10 = simple)
  - Scalability: how well will it handle growth in users / data?
  - Maintainability: how easy will it be to extend, debug, and operate?

Returns: SoftwareFeedback (scores 0-10, confidence 0-1, concerns, recommendations)
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agents._ollama_base import _call_ollama, build_ollama_llm
from agents.schemas import SoftwareFeedback
from state import PipelineState
from utils.logging_config import get_logger, make_log_entry

logger = get_logger(__name__)

_SYSTEM = """\
You are the Software Feasibility evaluation agent in a multi-agent pipeline.
Your job is to assess proposed solutions from a software engineering perspective.

Score each criterion from 0 (very poor) to 10 (excellent):
  - complexity_score:      Simplicity of implementation (10 = very easy to build).
  - scalability_score:     Ability to handle growth in load and data volume.
  - maintainability_score: How easy it is to extend, debug, and operate over time.

Also report your confidence (0-1) in your assessment.
Identify top concerns and concrete, actionable recommendations.

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

## Implementation Notes
{impl_notes}

## What to Focus On
{priority_focus}
"""


def _build_messages(state: PipelineState) -> list:
    cfg_instructions = state["coordination_instructions"] or {}
    instruction = cfg_instructions.get(
        "software", "Evaluate from a software engineering perspective."
    )
    artifact = state["solution_artifact"] or {}

    components = "\n".join(
        f"  - {c.get('name', '?')}: {c.get('responsibility', '')}"
        for c in artifact.get("components", [])
    ) or "  (no components yet)"

    priority = ", ".join(state.get("priority_focus", [])) or "general evaluation"

    user_content = _USER_TEMPLATE.format(
        user_goal=state["user_goal"],
        instruction=instruction,
        solution_summary=artifact.get("design_summary", "No solution yet — evaluate the feasibility of the goal."),
        components=components,
        decision_logic=artifact.get("decision_logic", "N/A"),
        impl_notes=artifact.get("implementation_notes", "N/A"),
        priority_focus=priority,
    )
    return [SystemMessage(content=_SYSTEM), HumanMessage(content=user_content)]


def software_node(state: PipelineState) -> dict:
    """LangGraph node: evaluate solution from software feasibility perspective."""
    run_id = state["run_id"]
    iteration = state["iteration"]
    logger.info(
        "Software agent evaluating",
        extra={"run_id": run_id, "iteration": iteration, "node": "software"},
    )

    llm = build_ollama_llm("software_agent")
    messages = _build_messages(state)

    result, fallback_logs = _call_ollama(
        llm=llm,
        messages=messages,
        schema=SoftwareFeedback,
        agent_name="software",
        state_run_id=run_id,
        state_iteration=iteration,
    )

    avg_score = (
        result.complexity_score
        + result.scalability_score
        + result.maintainability_score
    ) / 3

    log_entry = make_log_entry(
        event="software_complete",
        node="software",
        run_id=run_id,
        iteration=iteration,
        message=f"Software avg={avg_score:.2f}, confidence={result.confidence:.2f}",
        avg_score=avg_score,
        confidence=result.confidence,
        fallback_used=bool(fallback_logs),
    )
    logger.info(log_entry["message"], extra=log_entry)

    return {
        "software_feedback": result.model_dump(),
        "log_entries": fallback_logs + [log_entry],
    }
