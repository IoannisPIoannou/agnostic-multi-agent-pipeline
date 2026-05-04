"""
agents/end_user_audit.py — Audit node for end-user evaluator feedback.

Reads driver_feedback and user_goal, scores the feedback quality using a small
Ollama LLM (llama3.2:3b), and returns AuditOutput written to state["end_user_audit"].

Routing function route_end_user_audit() is imported by graph/builder.py.

Pass rule: overall_audit_score >= cfg["audit_layer"]["audit_approval_threshold"]
           AND confidence >= cfg["audit_layer"]["min_audit_confidence"]

Revision boundary: attempts < max_audit_revisions  → "revise"
                   attempts >= max_audit_revisions  → "done" (failed_after_max_revisions)
where `attempts` is the post-increment counter written by this node each run.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agents._ollama_base import _call_ollama, build_ollama_llm
from agents.schemas import AuditOutput
from state import PipelineState
from utils.config_loader import get_config
from utils.logging_config import get_logger, make_log_entry

logger = get_logger(__name__)

_SYSTEM = """\
You are a quality auditor for driver/end-user evaluation feedback in a multi-agent AI pipeline.
Your role is to assess whether the provided feedback correctly represents a truck driver or \
primary end-user perspective.

Score each criterion 0–10. Set confidence (0–1) based on the clarity of the feedback you are auditing.

Audit criteria:
  role_adherence_score      : Does the feedback consistently adopt the end-user/driver persona?
                              No engineering jargon or policy/compliance language.
  grounding_score           : Are scores grounded in the actual solution content, not generic statements?
  specificity_score         : Are concerns and recommendations specific and actionable, not vague?
  score_text_consistency_score : Do numeric scores align with the severity of concerns and recommendations?
  overall_audit_score       : Holistic quality of the feedback as useful driver-perspective input.

Set approved=true if overall_audit_score >= threshold AND confidence >= min_confidence.
If approved=false, set revision_request to a clear, specific instruction for how to revise.
Respond with a valid JSON object matching the required schema — nothing else.
"""

_USER_TEMPLATE = """\
## User Goal
{user_goal}

## Feedback Being Audited (End-User / Driver Perspective)
{feedback_text}

## Approval Criteria (for reference — do not include in output)
overall_audit_score >= {threshold} AND confidence >= {min_confidence}
"""


def _feedback_to_text(feedback: dict) -> str:
    lines = [
        f"Usability score    : {feedback.get('usability_score', 'N/A')}",
        f"Clarity score      : {feedback.get('clarity_score', 'N/A')}",
        f"Cost score         : {feedback.get('cost_score', 'N/A')}",
        f"Practicality score : {feedback.get('practicality_score', 'N/A')}",
        f"Confidence         : {feedback.get('confidence', 'N/A')}",
        f"Overall summary    : {feedback.get('overall_summary', 'N/A')}",
        "Key concerns:",
    ]
    for c in feedback.get("key_concerns", []):
        lines.append(f"  - {c}")
    lines.append("Recommendations:")
    for r in feedback.get("recommendations", []):
        lines.append(f"  - {r}")
    return "\n".join(lines)


def _audit_fallback() -> dict:
    """
    Hardcoded fallback when the audit LLM call fails completely.
    Does NOT use _safe_fallback() — AuditOutput has a bool field which
    _safe_fallback raises TypeError on.  approved=False ensures the pipeline
    does not silently pass a failed audit.
    """
    return {
        "role_adherence_score": 0.0,
        "grounding_score": 0.0,
        "specificity_score": 0.0,
        "score_text_consistency_score": 0.0,
        "overall_audit_score": 0.0,
        "approved": False,
        "issues": ["[end_user_audit LLM failure — using hardcoded fallback]"],
        "revision_request": None,
        "confidence": 0.0,
    }


def end_user_audit_node(state: PipelineState) -> dict:
    """
    LangGraph node: audit the quality of end_user agent feedback.

    Increments end_user_audit_attempts by 1 on each run.
    Sets end_user_audit_status:
      "approved"                  — audit passed
      ""                          — failed but revisions remain
      "failed_after_max_revisions"— failed and quota exhausted
      "disabled"                  — audit_layer.enabled=False
    """
    cfg = get_config()
    audit_cfg = cfg.get("audit_layer", {})
    run_id = state["run_id"]
    iteration = state["iteration"]
    new_attempts = state.get("end_user_audit_attempts", 0) + 1
    max_revisions = audit_cfg.get("max_audit_revisions", 2)

    logger.info(
        f"EndUser audit running (attempt {new_attempts})",
        extra={"run_id": run_id, "iteration": iteration, "node": "end_user_audit"},
    )

    # ── Disabled fast-path ─────────────────────────────────────────────────────
    if not audit_cfg.get("enabled", True):
        log_entry = make_log_entry(
            event="end_user_audit_skipped",
            node="end_user_audit",
            run_id=run_id,
            iteration=iteration,
            message="Audit layer disabled — end_user audit skipped",
            attempt=new_attempts,
        )
        logger.info(log_entry["message"], extra=log_entry)
        return {
            "end_user_audit": {
                "role_adherence_score": 10.0,
                "grounding_score": 10.0,
                "specificity_score": 10.0,
                "score_text_consistency_score": 10.0,
                "overall_audit_score": 10.0,
                "approved": True,
                "issues": [],
                "revision_request": None,
                "confidence": 1.0,
            },
            "end_user_audit_attempts": new_attempts,
            "end_user_audit_status": "disabled",
            "log_entries": [log_entry],
        }

    # ── Active audit ───────────────────────────────────────────────────────────
    threshold = audit_cfg.get("audit_approval_threshold", 8.5)
    min_conf = audit_cfg.get("min_audit_confidence", 0.5)

    feedback = state.get("driver_feedback") or {}
    feedback_text = _feedback_to_text(feedback) if feedback else "(no feedback available)"

    user_content = _USER_TEMPLATE.format(
        user_goal=state["user_goal"],
        feedback_text=feedback_text,
        threshold=threshold,
        min_confidence=min_conf,
    )
    messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=user_content)]

    llm = build_ollama_llm("end_user_audit_agent")
    log_entries_out: list[dict] = []

    try:
        result, fallback_logs = _call_ollama(
            llm=llm,
            messages=messages,
            schema=AuditOutput,
            agent_name="end_user_audit",
            state_run_id=run_id,
            state_iteration=iteration,
        )
        log_entries_out.extend(fallback_logs)
        audit_dict = result.model_dump()
    except Exception as exc:
        # _call_ollama's internal _safe_fallback raises TypeError on bool fields.
        # Catch it here and use our own hardcoded fallback instead.
        logger.error(
            f"[end_user_audit] Using hardcoded fallback: {exc}",
            extra={"run_id": run_id, "iteration": iteration, "node": "end_user_audit"},
        )
        log_entries_out.append(make_log_entry(
            event="end_user_audit_fallback",
            node="end_user_audit",
            run_id=run_id,
            iteration=iteration,
            message=f"[end_user_audit] Hardcoded fallback after LLM failure: {exc}",
            error=str(exc),
            attempt=new_attempts,
        ))
        audit_dict = _audit_fallback()

    # Re-evaluate approval from scores (overrides model's own approved field to
    # ensure the threshold logic is always applied consistently).
    actually_approved = (
        audit_dict.get("overall_audit_score", 0.0) >= threshold
        and audit_dict.get("confidence", 0.0) >= min_conf
    )
    audit_dict["approved"] = actually_approved

    # Determine status
    if actually_approved:
        status = "approved"
    elif new_attempts < max_revisions:
        status = ""  # revisions still available; routing will send "revise"
    else:
        status = "failed_after_max_revisions"

    log_entry = make_log_entry(
        event="end_user_audit_complete",
        node="end_user_audit",
        run_id=run_id,
        iteration=iteration,
        message=(
            f"EndUser audit attempt={new_attempts} | "
            f"overall={audit_dict['overall_audit_score']:.2f} | "
            f"approved={actually_approved} | "
            f"max_revisions_reached={new_attempts >= max_revisions}"
        ),
        attempt=new_attempts,
        overall_audit_score=audit_dict["overall_audit_score"],
        approved=actually_approved,
        confidence=audit_dict.get("confidence", 0.0),
        max_revisions_reached=new_attempts >= max_revisions,
        status=status,
    )
    logger.info(log_entry["message"], extra=log_entry)
    log_entries_out.append(log_entry)

    return {
        "end_user_audit": audit_dict,
        "end_user_audit_attempts": new_attempts,
        "end_user_audit_status": status,
        "log_entries": log_entries_out,
    }


def route_end_user_audit(state: PipelineState) -> str:
    """
    Routing function for end_user_audit conditional edge.
    Reads the post-increment attempt count (already written by end_user_audit_node).

    Returns:
        "done"   — audit approved, or revision quota exhausted
        "revise" — audit failed and revisions remain
    """
    cfg = get_config()
    audit_cfg = cfg.get("audit_layer", {})
    max_revisions = audit_cfg.get("max_audit_revisions", 2)

    audit_out = state.get("end_user_audit") or {}
    approved = audit_out.get("approved", False)
    attempts = state.get("end_user_audit_attempts", 0)  # post-increment

    if approved:
        return "done"
    if attempts < max_revisions:
        return "revise"
    return "done"
