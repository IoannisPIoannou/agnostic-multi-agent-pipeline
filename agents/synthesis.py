"""
agents/synthesis.py — Final synthesis / report generation agent (Gemini).

Runs exactly once, after the pipeline stops (score threshold reached,
convergence detected, or max iterations exceeded).

Produces:
  - A structured Markdown final report saved to outputs/{run_id}/final_report.md
  - The report text is also stored in state["final_report"] for programmatic access

The report includes:
  1. Executive summary
  2. Final solution design
  3. Score progression and convergence analysis
  4. Key trade-offs and remaining concerns
  5. Recommended next steps
"""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from state import PipelineState
from utils.config_loader import get_config
from utils.logging_config import get_logger, make_log_entry

logger = get_logger(__name__)

# Prefix written by _safe_fallback in _ollama_base.py when parsing fails
_FALLBACK_MARKER = "[Parse failure"
_FALLBACK_REPLACEMENT = "(evaluation unavailable due to parse failure)"

_SYSTEM = """\
You are producing the final deliverable of a multi-agent evaluation pipeline.
Write a professional, structured Markdown report.
Be concise but complete.  Use headers, bullet points, and tables where appropriate.
Only use information present in the provided data sections.  Do not invent unsupported claims or details.
"""

_USER_TEMPLATE = """\
# Pipeline Run: {run_id}

## Original Goal
{user_goal}

## Run Summary
- Iterations completed: {iterations}
- Stopping reason: {stop_reason}
- Final score: {final_score:.2f} / {threshold}

## Score Progression
{metrics_table}

## Final Solution
### Design Summary
{design_summary}

### Key Components
{components}

### Decision Logic
{decision_logic}

### Implementation Notes
{impl_notes}

### Evaluation Criteria
{evaluation_criteria}

### Concerns Addressed in Final Iteration
{addressed_concerns}

## Final Agent Feedback
### Driver Perspective
{driver_summary}

### Policy Perspective
{policy_summary}

### Software Perspective
{software_summary}

## Unresolved Agent Conflicts (at pipeline stop)
{unresolved_conflicts}

## All Concerns Raised During Evaluation
{all_concerns}

---

Now produce the professional final report with these sections:
1. Executive Summary (3 sentences max)
2. Final Solution Design (components + decision logic)
3. Evaluation Summary (score table + what improved each iteration)
4. Key Trade-offs and Remaining Concerns
5. Recommended Next Steps
"""


def _metrics_table(metrics_history: list) -> str:
    if not metrics_history:
        return "No iterations recorded."
    rows = [
        "| Iter | Driver | Policy | Software | Overall | Converged |",
        "|------|--------|--------|----------|---------|-----------|",
    ]
    for m in metrics_history:
        rows.append(
            f"| {m.get('iteration', '?')} "
            f"| {m.get('driver_avg', 0):.2f} "
            f"| {m.get('policy_avg', 0):.2f} "
            f"| {m.get('software_avg', 0):.2f} "
            f"| {m.get('overall_score', 0):.2f} "
            f"| {'yes' if m.get('converged') else 'no'} |"
        )
    return "\n".join(rows)


def _safe_summary(summary: str) -> str:
    """Replace _safe_fallback placeholder text with a neutral indicator."""
    if summary.startswith(_FALLBACK_MARKER):
        return _FALLBACK_REPLACEMENT
    return summary


def _build_messages(state: PipelineState) -> list:
    cfg = get_config()
    artifact    = state["solution_artifact"]  or {}
    driver_fb   = state["driver_feedback"]    or {}
    policy_fb   = state["policy_feedback"]    or {}
    software_fb = state["software_feedback"]  or {}
    agg         = state["aggregated_feedback"] or {}

    components = "\n".join(
        f"- **{c.get('name')}**: {c.get('description')} ({c.get('responsibility')})"
        for c in artifact.get("components", [])
    ) or "- (none)"

    evaluation_criteria = "\n".join(
        f"- **{ec.get('name', '?')}**: {ec.get('description', '')} — metric: {ec.get('metric', '')}"
        for ec in artifact.get("evaluation_criteria", [])
    ) or "- (none)"

    addressed_concerns = "\n".join(
        f"- {c}" for c in artifact.get("addressed_concerns", [])
    ) or "- (none)"

    # Correctly distinguish unresolved conflicts from the full concern list
    unresolved_conflicts = "\n".join(
        f"- {c}" for c in agg.get("unresolved_conflicts", [])
    ) or "- None"

    all_concerns = "\n".join(
        f"- {c}" for c in agg.get("all_concerns", [])
    ) or "- None"

    user_content = _USER_TEMPLATE.format(
        run_id=state["run_id"],
        user_goal=state["user_goal"],
        iterations=state["iteration"],
        stop_reason=state.get("stop_reason", "unknown"),
        final_score=state["evaluation_score"],
        threshold=cfg["pipeline"]["score_threshold"],
        metrics_table=_metrics_table(state.get("metrics_history", [])),
        design_summary=artifact.get("design_summary", "N/A"),
        components=components,
        decision_logic=artifact.get("decision_logic", "N/A"),
        impl_notes=artifact.get("implementation_notes", "N/A"),
        evaluation_criteria=evaluation_criteria,
        addressed_concerns=addressed_concerns,
        driver_summary=_safe_summary(driver_fb.get("overall_summary", "N/A")),
        policy_summary=_safe_summary(policy_fb.get("overall_summary", "N/A")),
        software_summary=_safe_summary(software_fb.get("overall_summary", "N/A")),
        unresolved_conflicts=unresolved_conflicts,
        all_concerns=all_concerns,
    )
    return [SystemMessage(content=_SYSTEM), HumanMessage(content=user_content)]


def _build_llm() -> ChatGoogleGenerativeAI:
    cfg = get_config()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")
    syn_cfg = cfg["models"]["synthesis"]
    return ChatGoogleGenerativeAI(
        model=syn_cfg["model"],
        temperature=syn_cfg["temperature"],
        google_api_key=api_key,
    )


def synthesis_node(state: PipelineState) -> dict:
    """LangGraph node: generate the final structured report."""
    run_id = state["run_id"]
    iteration = state["iteration"]
    logger.info(
        "Synthesis agent generating final report",
        extra={"run_id": run_id, "iteration": iteration, "node": "synthesis"},
    )

    llm = _build_llm()
    messages = _build_messages(state)
    log_entries: list[dict] = []

    try:
        response = llm.invoke(messages)
        report_text = response.content
    except Exception as exc:
        error_msg = f"[synthesis] LLM call failed (iteration={iteration}): {exc}"
        logger.error(
            error_msg,
            extra={"run_id": run_id, "iteration": iteration, "node": "synthesis"},
        )
        log_entries.append(make_log_entry(
            event="synthesis_error",
            node="synthesis",
            run_id=run_id,
            iteration=iteration,
            message=error_msg,
            error=str(exc),
        ))
        raise

    log_entries.append(make_log_entry(
        event="synthesis_complete",
        node="synthesis",
        run_id=run_id,
        iteration=iteration,
        message=f"Final report generated ({len(report_text)} chars)",
        report_length=len(report_text),
        stop_reason=state.get("stop_reason"),
    ))
    logger.info(log_entries[-1]["message"], extra=log_entries[-1])

    return {
        "final_report": report_text,
        "log_entries": log_entries,
    }
