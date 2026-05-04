"""
agents/independent_evaluator.py — Independent quality evaluator (Gemini).

Acts as an unbiased judge of the solution artifact against the original goal.
It deliberately sees ONLY the goal, success criteria, and solution — never the
parallel agents' scores or concerns — so its score is not contaminated by the
same feedback loop it is measuring.

Its overall_score becomes the pipeline's stopping criterion, replacing the
confidence-weighted average of the parallel agents.  The parallel agents
continue to provide improvement feedback to the coding agent unchanged.
"""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.schemas import IndependentEvaluation
from state import PipelineState
from utils.config_loader import get_config
from utils.logging_config import get_logger, make_log_entry

logger = get_logger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM = """\
You are an independent quality auditor for AI-generated system designs.

Your role is to evaluate a proposed solution objectively and rigorously against
the original goal.  You have no knowledge of how other agents scored this
solution — you must form your own independent judgement.

Score each criterion on a 1–10 scale (1 = very poor, 10 = excellent):
  - goal_alignment_score : Does the solution directly and fully address the stated goal?
  - completeness_score   : Are all required aspects, edge cases, and stakeholders covered?
  - feasibility_score    : Is this practically implementable given real-world constraints?
  - clarity_score        : Is the design clear, well-structured, and unambiguous?
  - innovation_score     : Does the solution show strong design quality and sound decisions?
  - overall_score        : Your holistic, independent overall quality judgement (not a simple average).

Set confidence (0–1) to reflect how certain you are in your assessment given
the information available.  Be critical but fair.  Identify real gaps and
provide actionable recommendations.
"""

_USER_TEMPLATE = """\
## Original Goal
{user_goal}

## Success Criteria
{success_criteria}

## Constraints
{constraints}

## Proposed Solution (Iteration {iteration})

### Design Summary
{design_summary}

### Components
{components}

### Decision Logic
{decision_logic}

### Implementation Notes
{implementation_notes}

## Task
Evaluate this solution independently against the goal and success criteria above.
Do not assume any prior feedback was incorporated correctly — judge only what is
written here.  Provide honest scores, your key concerns, and concrete recommendations.
"""

_CONCERN_CAP = 8
_REC_CAP = 8


def _build_prompt(state: PipelineState) -> str:
    td       = state.get("task_decomposition") or {}
    artifact = state.get("solution_artifact") or {}

    success_criteria = "\n".join(
        f"  - {c}" for c in td.get("success_criteria", [])
    ) or "  (not specified)"
    constraints = "\n".join(
        f"  - {c}" for c in td.get("constraints", [])
    ) or "  (not specified)"
    components = "\n".join(
        f"  - {c.get('name', '?')}: {c.get('description', '')}"
        for c in artifact.get("components", [])
    ) or "  (none)"

    return _USER_TEMPLATE.format(
        user_goal=state["user_goal"],
        success_criteria=success_criteria,
        constraints=constraints,
        iteration=state["iteration"],
        design_summary=artifact.get("design_summary", "N/A"),
        components=components,
        decision_logic=artifact.get("decision_logic", "N/A"),
        implementation_notes=artifact.get("implementation_notes", "N/A"),
    )


# ── LLM client singleton ──────────────────────────────────────────────────────

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
        model_cfg = cfg["models"]["independent_evaluator"]
        _llm = ChatGoogleGenerativeAI(
            model=model_cfg["model"],
            temperature=model_cfg["temperature"],
            google_api_key=api_key,
        )
    return _llm


# ── Node function ─────────────────────────────────────────────────────────────

def independent_evaluator_node(state: PipelineState) -> dict:
    """
    LangGraph node: independently score the current solution artifact.

    Reads only user_goal, task_decomposition, and solution_artifact.
    Writes independent_evaluation to state.
    """
    run_id    = state["run_id"]
    iteration = state["iteration"]

    logger.info(
        "Independent evaluator running",
        extra={"run_id": run_id, "iteration": iteration, "node": "independent_evaluator"},
    )

    structured_llm = _get_llm().with_structured_output(IndependentEvaluation)
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=_build_prompt(state)),
    ]

    try:
        result: IndependentEvaluation = structured_llm.invoke(messages)
    except Exception as exc:
        error_msg = (
            f"[independent_evaluator] LLM call failed "
            f"(iteration={iteration}): {exc}"
        )
        logger.error(
            error_msg,
            extra={"run_id": run_id, "iteration": iteration, "node": "independent_evaluator"},
        )
        raise

    log_entry = make_log_entry(
        event="independent_evaluator_complete",
        node="independent_evaluator",
        run_id=run_id,
        iteration=iteration,
        message=(
            f"overall={result.overall_score:.2f} | "
            f"goal={result.goal_alignment_score:.1f} "
            f"complete={result.completeness_score:.1f} "
            f"feasible={result.feasibility_score:.1f} "
            f"clarity={result.clarity_score:.1f} "
            f"innov={result.innovation_score:.1f} | "
            f"conf={result.confidence:.2f}"
        ),
        overall_score=result.overall_score,
        confidence=result.confidence,
    )
    logger.info(log_entry["message"], extra=log_entry)

    return {
        "independent_evaluation": result.model_dump(),
        "log_entries":            [log_entry],
    }
