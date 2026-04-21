"""
agents/orchestrator.py — Orchestrator agent (Gemini).

Responsibilities:
  - Iteration 1: Decompose the user goal into structured subproblems and provide
    targeted evaluation instructions for each specialist agent.
  - Iteration > 1: Receive aggregated feedback (including conflicts + confidence
    signals) and produce a revision strategy that directs the coding agent toward
    the highest-impact improvements.

LLM: Gemini (via langchain-google-genai) with structured output.
"""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.schemas import OrchestratorOutput
from state import PipelineState
from utils.config_loader import get_config
from utils.logging_config import get_logger, make_log_entry

logger = get_logger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM = """\
You are the master orchestrator of a multi-agent decision-support pipeline.

Your role depends on the iteration:
  - Iteration 1: Decompose the problem, set evaluation focus for each agent.
  - Iteration > 1: Analyse feedback, detect what is blocking a higher score,
    and produce a targeted revision strategy.

Three specialist agents will evaluate the solution:
  1. End User Agent   — usability, cost, practicality, clarity (primary end-user view)
  2. Policy Agent     — safety, compliance, systemic impact (stakeholder view)
  3. Software Agent   — complexity, scalability, maintainability (tech view)

For the End User Agent: identify the primary end user of this system (e.g. fraud analyst,
warehouse operator, clinician, customer, driver) and frame its mandate from that persona's
perspective, describing who they are and what their day-to-day tasks involve.

Be analytical, concise, and specific.  Avoid vague instructions.
"""

_USER_FIRST = """\
## User Goal
{user_goal}

## Task
This is iteration 1.  Decompose the problem and give each agent a specific,
actionable evaluation mandate.  Identify hard constraints and success criteria.

For the End User Agent (driver_instruction): first identify who the primary end user
of this system is and what their day-to-day tasks involve, then frame the evaluation
mandate from that persona's perspective.  Begin the instruction with a sentence like:
"Evaluate from the perspective of a <persona> who <daily context>..."
"""

_USER_REVISION = """\
## User Goal
{user_goal}

## Current Iteration: {iteration}
## Current Score: {score:.2f} / {threshold}

## Aggregated Feedback from Previous Iteration

Agent averages:
  - Driver:   {driver_avg:.2f}
  - Policy:   {policy_avg:.2f}
  - Software: {software_avg:.2f}

Agent confidences:
  - Driver:   {driver_conf:.2f}
  - Policy:   {policy_conf:.2f}
  - Software: {software_conf:.2f}

Unresolved conflicts (agents that diverge from consensus): {unresolved}
Resolved since last iteration: {resolved}

Top concerns raised:
{concerns}

Top recommendations:
{recommendations}

## Current Solution
### Design Summary
{solution_summary}

### Components
{solution_components}

## Task
Produce a revision strategy.  Focus the agents on the highest-impact unresolved
issues.  Update coordination instructions and priority_focus accordingly.

For the End User Agent (driver_instruction): retain the same end-user persona identified
in iteration 1 and frame any revised mandate from that same persona's perspective.
"""

# Caps applied before sending concerns/recommendations to Gemini.
_CONCERN_CAP = 8
_REC_CAP = 8


def _build_prompt(state: PipelineState, iteration: int) -> str:
    """Build the user-turn prompt. `iteration` is the already-incremented value."""
    cfg = get_config()

    if state["iteration"] == 0:
        # state["iteration"] is the pre-increment value; 0 means first pass
        return _USER_FIRST.format(user_goal=state["user_goal"])

    agg = state["aggregated_feedback"] or {}
    agent_avgs = agg.get("agent_averages", {})
    # Confidences from aggregated_feedback — single source of truth after aggregator refactor
    agent_confs = agg.get("agent_confidences", {})

    all_concerns_full = agg.get("all_concerns", [])
    all_recs_full     = agg.get("all_recommendations", [])
    shown_concerns    = all_concerns_full[:_CONCERN_CAP]
    shown_recs        = all_recs_full[:_REC_CAP]

    if len(all_concerns_full) > _CONCERN_CAP:
        logger.warning(
            f"[orchestrator] Truncating concerns: showing {_CONCERN_CAP}/{len(all_concerns_full)}",
            extra={
                "run_id": state["run_id"],
                "iteration": iteration,
                "node": "orchestrator",
                "total_concerns": len(all_concerns_full),
                "shown_concerns": _CONCERN_CAP,
            },
        )
    if len(all_recs_full) > _REC_CAP:
        logger.warning(
            f"[orchestrator] Truncating recommendations: showing {_REC_CAP}/{len(all_recs_full)}",
            extra={
                "run_id": state["run_id"],
                "iteration": iteration,
                "node": "orchestrator",
                "total_recommendations": len(all_recs_full),
                "shown_recommendations": _REC_CAP,
            },
        )

    concerns        = "\n".join(f"  - {c}" for c in shown_concerns) or "  (none reported)"
    recommendations = "\n".join(f"  - {r}" for r in shown_recs)     or "  (none reported)"

    # Brief component list to give the orchestrator structural context for its mandate
    artifact = state["solution_artifact"] or {}
    solution_components = "\n".join(
        f"  - {c.get('name', '?')}: {c.get('description', '')}"
        for c in artifact.get("components", [])
    ) or "  (none yet)"

    return _USER_REVISION.format(
        user_goal=state["user_goal"],
        iteration=iteration,
        score=state["evaluation_score"],
        threshold=cfg["pipeline"]["score_threshold"],
        driver_avg=agent_avgs.get("driver",   0.0),
        policy_avg=agent_avgs.get("policy",   0.0),
        software_avg=agent_avgs.get("software", 0.0),
        driver_conf=agent_confs.get("driver",   0.0),
        policy_conf=agent_confs.get("policy",   0.0),
        software_conf=agent_confs.get("software", 0.0),
        unresolved=", ".join(agg.get("unresolved_conflicts", [])) or "none",
        resolved=", ".join(agg.get("resolved_conflicts",   [])) or "none",
        concerns=concerns,
        recommendations=recommendations,
        solution_summary=artifact.get("design_summary", "N/A"),
        solution_components=solution_components,
    )


# ── LLM client — built once per process; ChatGoogleGenerativeAI is stateless ──

_llm: ChatGoogleGenerativeAI | None = None


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        cfg = get_config()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is not set.\n"
                "Set it in your shell or create a .env file (see .env.example)."
            )
        orch_cfg = cfg["models"]["orchestrator"]
        _llm = ChatGoogleGenerativeAI(
            model=orch_cfg["model"],
            temperature=orch_cfg["temperature"],
            google_api_key=api_key,
        )
    return _llm


# ── Node function ─────────────────────────────────────────────────────────────

def orchestrator_node(state: PipelineState) -> dict:
    """
    LangGraph node: orchestrates task decomposition and revision strategy.
    Increments the iteration counter before doing any work.
    """
    new_iteration = state["iteration"] + 1
    run_id = state["run_id"]
    mode = "initial" if state["iteration"] == 0 else "revision"

    logger.info(
        f"Orchestrator starting ({mode})",
        extra={"run_id": run_id, "iteration": new_iteration, "node": "orchestrator", "mode": mode},
    )

    structured_llm = _get_llm().with_structured_output(OrchestratorOutput)
    user_prompt = _build_prompt(state, new_iteration)
    messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=user_prompt)]
    log_entries: list[dict] = []

    try:
        result: OrchestratorOutput = structured_llm.invoke(messages)
    except Exception as exc:
        error_msg = f"[orchestrator] LLM call failed (iteration={new_iteration}, mode={mode}): {exc}"
        logger.error(
            error_msg,
            extra={"run_id": run_id, "iteration": new_iteration, "node": "orchestrator"},
        )
        log_entries.append(make_log_entry(
            event="orchestrator_error",
            node="orchestrator",
            run_id=run_id,
            iteration=new_iteration,
            message=error_msg,
            error=str(exc),
            mode=mode,
        ))
        raise

    log_entry = make_log_entry(
        event="orchestrator_complete",
        node="orchestrator",
        run_id=run_id,
        iteration=new_iteration,
        message=f"Decomposed into {len(result.subproblems)} subproblems ({mode})",
        revision_strategy=result.revision_strategy,
        priority_focus=result.priority_focus,
        mode=mode,
    )
    logger.info(log_entry["message"], extra=log_entry)
    log_entries.append(log_entry)

    return {
        "iteration": new_iteration,
        "task_decomposition": result.model_dump(
            exclude={"driver_instruction", "policy_instruction", "software_instruction"}
        ),
        "coordination_instructions": {
            "driver":   result.driver_instruction,
            "policy":   result.policy_instruction,
            "software": result.software_instruction,
        },
        "revision_strategy": result.revision_strategy,
        "priority_focus":    result.priority_focus,
        "log_entries":       log_entries,
    }
