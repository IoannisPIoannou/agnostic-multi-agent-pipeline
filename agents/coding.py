"""
agents/coding.py — Solution design / refinement agent (Gemini).

Produces or refines a concrete solution artifact based on:
  - The orchestrator's task decomposition and coordination instructions
  - The aggregated feedback from all evaluator agents
  - Any revision strategy and priority focus from the orchestrator

On iteration 1 it generates an initial design.
On subsequent iterations it refines the previous design, explicitly addressing
the unresolved conflicts and highest-priority concerns while preserving design
decisions that are not under criticism.

LLM: Gemini with structured output.
"""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.schemas import CodingOutput
from state import PipelineState
from utils.config_loader import get_config
from utils.logging_config import get_logger, make_log_entry

logger = get_logger(__name__)

# Caps applied before sending concerns/recommendations to Gemini.
_CONCERN_CAP = 10
_REC_CAP = 8

_SYSTEM = """\
You are the Solution Architecture agent in a multi-agent evaluation pipeline.

Your job is to produce or refine a concrete, implementable solution design.

Guidelines:
  - Be specific: name real components, data structures, and decision algorithms.
  - Be practical: the solution should be buildable by a small team.
  - Address concerns: explicitly state which agent concerns you are resolving.
  - Focus: if a revision_strategy is provided, prioritise those areas above all else.

Output a complete solution design as structured JSON.
"""

_USER_FIRST = """\
## User Goal
{user_goal}

## Task Decomposition
Subproblems: {subproblems}
Constraints: {constraints}
Success criteria: {success_criteria}

## Coordination notes
{coordination_notes}

Design the initial solution.
"""

_USER_REVISION = """\
## User Goal
{user_goal}

## Iteration {iteration} — Revision Mandate
{revision_strategy}

## Priority Focus
{priority_focus}

## Unresolved Conflicts (must address)
{unresolved_conflicts}

## All Agent Concerns
{all_concerns}

## Top Recommendations
{top_recommendations}

## Previous Solution (from Iteration {prev_iteration})

### Design Summary
{prev_summary}

### Components
{prev_components}

### Decision Logic
{prev_decision_logic}

### Evaluation Criteria
{prev_criteria}

### Implementation Notes
{prev_implementation_notes}

### Concerns Already Addressed
{prev_addressed_concerns}

## Refinement Rules
- Preserve all design decisions not implicated by the concerns or conflicts above.
- Only change components, logic, or decisions that are directly targeted by the listed concerns/conflicts.
- Build on the previous solution; do not regenerate it from scratch.
- List every concern you address in the addressed_concerns field.

Refine the solution based on the above. Keep what works; fix what is criticised.
"""


def _build_messages(state: PipelineState, run_id: str) -> list:
    td = state["task_decomposition"] or {}
    agg = state["aggregated_feedback"] or {}

    if state["iteration"] == 1:
        subproblems = "\n".join(
            f"  [{sp.get('priority', '?')}] {sp.get('title', '')}: {sp.get('description', '')}"
            for sp in td.get("subproblems", [])
        ) or "  (see user goal)"

        user_content = _USER_FIRST.format(
            user_goal=state["user_goal"],
            subproblems=subproblems,
            constraints="\n".join(f"  - {c}" for c in td.get("constraints", [])) or "  none",
            success_criteria="\n".join(f"  - {c}" for c in td.get("success_criteria", [])) or "  none",
            coordination_notes=td.get("coordination_notes", ""),
        )
    else:
        # ── Truncation with explicit visibility ───────────────────────────────
        all_concerns_full = agg.get("all_concerns", [])
        all_recs_full = agg.get("all_recommendations", [])
        shown_concerns = all_concerns_full[:_CONCERN_CAP]
        shown_recs = all_recs_full[:_REC_CAP]

        if len(all_concerns_full) > _CONCERN_CAP:
            logger.warning(
                f"[coding] Truncating concerns: showing {_CONCERN_CAP}/{len(all_concerns_full)}",
                extra={
                    "run_id": run_id,
                    "iteration": state["iteration"],
                    "node": "coding",
                    "total_concerns": len(all_concerns_full),
                    "shown_concerns": _CONCERN_CAP,
                },
            )
        if len(all_recs_full) > _REC_CAP:
            logger.warning(
                f"[coding] Truncating recommendations: showing {_REC_CAP}/{len(all_recs_full)}",
                extra={
                    "run_id": run_id,
                    "iteration": state["iteration"],
                    "node": "coding",
                    "total_recommendations": len(all_recs_full),
                    "shown_recommendations": _REC_CAP,
                },
            )

        concerns = "\n".join(f"  - {c}" for c in shown_concerns) or "  (none)"
        recommendations = "\n".join(f"  - {r}" for r in shown_recs) or "  (none)"
        unresolved = ", ".join(agg.get("unresolved_conflicts", [])) or "none"
        priority = "\n".join(f"  - {p}" for p in state.get("priority_focus", [])) or "  (general)"

        # ── Full previous solution for true refinement, not blind regeneration ─
        prev = state["solution_artifact"] or {}
        prev_components = "\n".join(
            f"  - {c.get('name', '?')}: {c.get('description', '')} "
            f"[responsibility: {c.get('responsibility', '')}]"
            for c in prev.get("components", [])
        ) or "  (none)"
        prev_criteria = "\n".join(
            f"  - {ec.get('name', '?')}: {ec.get('description', '')} "
            f"[metric: {ec.get('metric', '')}]"
            for ec in prev.get("evaluation_criteria", [])
        ) or "  (none)"

        user_content = _USER_REVISION.format(
            user_goal=state["user_goal"],
            iteration=state["iteration"],
            revision_strategy=state.get("revision_strategy") or "Improve overall quality.",
            priority_focus=priority,
            unresolved_conflicts=unresolved,
            all_concerns=concerns,
            top_recommendations=recommendations,
            prev_iteration=prev.get("iteration", "?"),
            prev_summary=prev.get("design_summary", "N/A"),
            prev_components=prev_components,
            prev_decision_logic=prev.get("decision_logic", "N/A"),
            prev_criteria=prev_criteria,
            prev_implementation_notes=prev.get("implementation_notes", "N/A"),
            prev_addressed_concerns=", ".join(prev.get("addressed_concerns", [])) or "none",
        )

    return [SystemMessage(content=_SYSTEM), HumanMessage(content=user_content)]


# ── LLM client — built once per process; ChatGoogleGenerativeAI is stateless ──

_llm: ChatGoogleGenerativeAI | None = None


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        cfg = get_config()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set.")
        coding_cfg = cfg["models"]["coding_agent"]
        _llm = ChatGoogleGenerativeAI(
            model=coding_cfg["model"],
            temperature=coding_cfg["temperature"],
            google_api_key=api_key,
        )
    return _llm


def coding_node(state: PipelineState) -> dict:
    """LangGraph node: generate or refine the solution artifact."""
    run_id = state["run_id"]
    iteration = state["iteration"]
    mode = "initial" if iteration == 1 else "refinement"

    logger.info(
        f"Coding agent running ({mode})",
        extra={"run_id": run_id, "iteration": iteration, "node": "coding", "mode": mode},
    )

    structured_llm = _get_llm().with_structured_output(CodingOutput)
    messages = _build_messages(state, run_id)
    log_entries: list[dict] = []

    try:
        result: CodingOutput = structured_llm.invoke(messages)
    except Exception as exc:
        error_msg = f"[coding] LLM call failed (iteration={iteration}, mode={mode}): {exc}"
        logger.error(
            error_msg,
            extra={"run_id": run_id, "iteration": iteration, "node": "coding"},
        )
        log_entries.append(make_log_entry(
            event="coding_error",
            node="coding",
            run_id=run_id,
            iteration=iteration,
            message=error_msg,
            error=str(exc),
            mode=mode,
        ))
        raise

    log_entries.append(make_log_entry(
        event="coding_complete",
        node="coding",
        run_id=run_id,
        iteration=iteration,
        message=f"Solution {mode}: {len(result.components)} components",
        num_components=len(result.components),
        addressed_concerns=result.addressed_concerns,
        mode=mode,
    ))
    logger.info(log_entries[-1]["message"], extra=log_entries[-1])

    artifact = result.model_dump()
    artifact["iteration"] = iteration

    return {
        "solution_artifact": artifact,
        "log_entries": log_entries,
    }
