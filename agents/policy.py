"""
agents/policy.py — Policy / stakeholder / systemic evaluation agent (Ollama).

Evaluates the solution from a regulatory and systemic perspective:
  - Safety: are there safety risks for drivers or the public?
  - Compliance: does it meet transport regulations and data-privacy rules?
  - System impact: does it improve or worsen overall network efficiency / fairness?

Returns: PolicyFeedback (scores 0-10, confidence 0-1, concerns, recommendations)
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agents._ollama_base import _call_ollama, build_ollama_llm
from agents.schemas import PolicyFeedback
from state import PipelineState
from utils.logging_config import get_logger, make_log_entry

logger = get_logger(__name__)

_SYSTEM = """\
You are the Policy/Stakeholder evaluation agent in a multi-agent pipeline.
Your job is to assess proposed solutions from a regulatory, safety, and systemic perspective.

Score each criterion from 0 (very poor) to 10 (excellent):
  - safety_score:        How safe is the system for drivers, other road users, and the public?
  - compliance_score:    How well does it satisfy transport regulations and privacy laws?
  - system_impact_score: How positive is the impact on overall transport network efficiency?

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

## What to Focus On
{priority_focus}
"""


def _build_messages(state: PipelineState) -> list:
    cfg_instructions = state["coordination_instructions"] or {}
    instruction = cfg_instructions.get(
        "policy", "Evaluate from the policy and safety perspective."
    )
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

    # Inject revision request when the audit agent flagged the previous feedback.
    revision_req = (state.get("policy_audit") or {}).get("revision_request")
    if revision_req and state.get("policy_feedback") is not None:
        user_content += (
            f"\n\n## REVISION REQUEST (from audit agent)\n{revision_req}\n\n"
            "Your previous feedback was flagged. Revise it addressing the issues above."
        )

    return [SystemMessage(content=_SYSTEM), HumanMessage(content=user_content)]


def policy_node(state: PipelineState) -> dict:
    """LangGraph node: evaluate solution from policy/stakeholder perspective."""
    run_id = state["run_id"]
    iteration = state["iteration"]
    logger.info(
        "Policy agent evaluating",
        extra={"run_id": run_id, "iteration": iteration, "node": "policy"},
    )

    llm = build_ollama_llm("policy_agent")
    messages = _build_messages(state)

    result, fallback_logs = _call_ollama(
        llm=llm,
        messages=messages,
        schema=PolicyFeedback,
        agent_name="policy",
        state_run_id=run_id,
        state_iteration=iteration,
    )

    avg_score = (
        result.safety_score
        + result.compliance_score
        + result.system_impact_score
    ) / 3

    log_entry = make_log_entry(
        event="policy_complete",
        node="policy",
        run_id=run_id,
        iteration=iteration,
        message=f"Policy avg={avg_score:.2f}, confidence={result.confidence:.2f}",
        avg_score=avg_score,
        confidence=result.confidence,
        fallback_used=bool(fallback_logs),
    )
    logger.info(log_entry["message"], extra=log_entry)

    return {
        "policy_feedback": result.model_dump(),
        "log_entries": fallback_logs + [log_entry],
    }
