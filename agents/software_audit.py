"""
agents/software_audit.py — Audit node for software evaluator feedback.

Reads software_feedback and user_goal, scores the feedback quality using
qwen2.5-coder:7b-instruct via Ollama, and returns AuditOutput written to
state["software_audit"].

Routing function route_software_audit() is imported by graph/builder.py.
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
You are a quality auditor for software/engineering evaluation feedback in a multi-agent AI pipeline.
Your role is to assess whether the provided feedback correctly represents a software engineering \
perspective.

Score each criterion 0–10. Set confidence (0–1) based on the clarity of the feedback you are auditing.

Audit criteria:
  role_adherence_score      : Does the feedback focus on technical architecture, scalability,
                              maintainability? No user experience or policy/compliance concerns.
  grounding_score           : Are technical concerns grounded in the solution's actual components
                              and design, not generic platitudes?
  specificity_score         : Are technical recommendations concrete and implementable
                              (not just "improve scalability" or "add caching")?
  score_text_consistency_score : Do scores accurately reflect the technical severity of concerns?
  overall_audit_score       : Holistic quality of the feedback as useful engineering-perspective input.

Set approved=true if overall_audit_score >= threshold AND confidence >= min_confidence.
If approved=false, set revision_request to a clear, specific instruction for how to revise.
Respond with a valid JSON object matching the required schema — nothing else.
"""

_USER_TEMPLATE = """\
## User Goal
{user_goal}

## Feedback Being Audited (Software / Engineering Perspective)
{feedback_text}

## Approval Criteria (for reference — do not include in output)
overall_audit_score >= {threshold} AND confidence >= {min_confidence}
"""


def _feedback_to_text(feedback: dict) -> str:
    lines = [
        f"Complexity score     : {feedback.get('complexity_score', 'N/A')}",
        f"Scalability score    : {feedback.get('scalability_score', 'N/A')}",
        f"Maintainability score: {feedback.get('maintainability_score', 'N/A')}",
        f"Confidence           : {feedback.get('confidence', 'N/A')}",
        f"Overall summary      : {feedback.get('overall_summary', 'N/A')}",
        "Key concerns:",
    ]
    for c in feedback.get("key_concerns", []):
        lines.append(f"  - {c}")
    lines.append("Recommendations:")
    for r in feedback.get("recommendations", []):
        lines.append(f"  - {r}")
    return "\n".join(lines)


def _audit_fallback() -> dict:
    return {
        "role_adherence_score": 0.0,
        "grounding_score": 0.0,
        "specificity_score": 0.0,
        "score_text_consistency_score": 0.0,
        "overall_audit_score": 0.0,
        "approved": False,
        "issues": ["[software_audit LLM failure — using hardcoded fallback]"],
        "revision_request": None,
        "confidence": 0.0,
    }


def software_audit_node(state: PipelineState) -> dict:
    """
    LangGraph node: audit the quality of software agent feedback.

    Increments software_audit_attempts by 1 on each run.
    Sets software_audit_status: "approved" | "" | "failed_after_max_revisions" | "disabled"
    """
    cfg = get_config()
    audit_cfg = cfg.get("audit_layer", {})
    run_id = state["run_id"]
    iteration = state["iteration"]
    new_attempts = state.get("software_audit_attempts", 0) + 1
    max_revisions = audit_cfg.get("max_audit_revisions", 2)

    logger.info(
        f"Software audit running (attempt {new_attempts})",
        extra={"run_id": run_id, "iteration": iteration, "node": "software_audit"},
    )

    # ── Disabled fast-path ─────────────────────────────────────────────────────
    if not audit_cfg.get("enabled", True):
        log_entry = make_log_entry(
            event="software_audit_skipped",
            node="software_audit",
            run_id=run_id,
            iteration=iteration,
            message="Audit layer disabled — software audit skipped",
            attempt=new_attempts,
        )
        logger.info(log_entry["message"], extra=log_entry)
        return {
            "software_audit": {
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
            "software_audit_attempts": new_attempts,
            "software_audit_status": "disabled",
            "log_entries": [log_entry],
        }

    # ── Active audit ───────────────────────────────────────────────────────────
    threshold = audit_cfg.get("audit_approval_threshold", 8.5)
    min_conf = audit_cfg.get("min_audit_confidence", 0.5)

    feedback = state.get("software_feedback") or {}
    feedback_text = _feedback_to_text(feedback) if feedback else "(no feedback available)"

    user_content = _USER_TEMPLATE.format(
        user_goal=state["user_goal"],
        feedback_text=feedback_text,
        threshold=threshold,
        min_confidence=min_conf,
    )
    messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=user_content)]

    llm = build_ollama_llm("software_audit_agent")
    log_entries_out: list[dict] = []

    try:
        result, fallback_logs = _call_ollama(
            llm=llm,
            messages=messages,
            schema=AuditOutput,
            agent_name="software_audit",
            state_run_id=run_id,
            state_iteration=iteration,
        )
        log_entries_out.extend(fallback_logs)
        audit_dict = result.model_dump()
    except Exception as exc:
        logger.error(
            f"[software_audit] Using hardcoded fallback: {exc}",
            extra={"run_id": run_id, "iteration": iteration, "node": "software_audit"},
        )
        log_entries_out.append(make_log_entry(
            event="software_audit_fallback",
            node="software_audit",
            run_id=run_id,
            iteration=iteration,
            message=f"[software_audit] Hardcoded fallback after LLM failure: {exc}",
            error=str(exc),
            attempt=new_attempts,
        ))
        audit_dict = _audit_fallback()

    actually_approved = (
        audit_dict.get("overall_audit_score", 0.0) >= threshold
        and audit_dict.get("confidence", 0.0) >= min_conf
    )
    audit_dict["approved"] = actually_approved

    if actually_approved:
        status = "approved"
    elif new_attempts < max_revisions:
        status = ""
    else:
        status = "failed_after_max_revisions"

    log_entry = make_log_entry(
        event="software_audit_complete",
        node="software_audit",
        run_id=run_id,
        iteration=iteration,
        message=(
            f"Software audit attempt={new_attempts} | "
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
        "software_audit": audit_dict,
        "software_audit_attempts": new_attempts,
        "software_audit_status": status,
        "log_entries": log_entries_out,
    }


def route_software_audit(state: PipelineState) -> str:
    """Routing function for software_audit conditional edge."""
    cfg = get_config()
    audit_cfg = cfg.get("audit_layer", {})
    max_revisions = audit_cfg.get("max_audit_revisions", 2)

    audit_out = state.get("software_audit") or {}
    approved = audit_out.get("approved", False)
    attempts = state.get("software_audit_attempts", 0)

    if approved:
        return "done"
    if attempts < max_revisions:
        return "revise"
    return "done"
