"""
agents/aggregator.py — Feedback aggregation with conflict detection (pure Python).

Runs after the three parallel evaluator agents complete (fan-in).
Responsibilities:
  1. Compute per-agent score averages
  2. Apply confidence-adjusted weights (with fallback if total weight is too low)
  3. Detect inter-agent conflicts and track which are resolved vs. new
  4. Merge all concerns and recommendations for the orchestrator's next prompt

No LLM is used here — this is deterministic and fast.

aggregated_feedback provides conflict detection and improvement context.

evaluation/evaluator.py uses unresolved_conflicts for convergence checks.
The stopping score is provided by independent_evaluation.overall_score.

weighted_score is NOT used for stopping decisions.
It is passed to the coding agent as a diagnostic signal only.
"""

from __future__ import annotations

from statistics import mean, median

from state import PipelineState
from utils.config_loader import get_config
from utils.logging_config import get_logger, make_log_entry

logger = get_logger(__name__)


# ── Score helpers ─────────────────────────────────────────────────────────────

def _driver_avg(fb: dict) -> float:
    return mean([
        float(fb.get("usability_score", 5.0)),
        float(fb.get("clarity_score", 5.0)),
        float(fb.get("cost_score", 5.0)),
        float(fb.get("practicality_score", 5.0)),
    ])


def _policy_avg(fb: dict) -> float:
    return mean([
        float(fb.get("safety_score", 5.0)),
        float(fb.get("compliance_score", 5.0)),
        float(fb.get("system_impact_score", 5.0)),
    ])


def _software_avg(fb: dict) -> float:
    return mean([
        float(fb.get("complexity_score", 5.0)),
        float(fb.get("scalability_score", 5.0)),
        float(fb.get("maintainability_score", 5.0)),
    ])


# ── Conflict detection ────────────────────────────────────────────────────────

def _detect_conflicts(
    agent_avgs: dict[str, float],
    prev_aggregated: dict | None,
    threshold: float,
) -> tuple[list[str], list[str]]:
    """
    Return (unresolved_conflicts, resolved_conflicts).

    Uses the median rather than the mean as the reference point. With three
    agents the median is the middle value, so two agents in agreement are never
    flagged as conflicts just because a single outlier shifts the mean toward
    the middle. Only true outliers (score deviating from the median by more
    than threshold) are detected.
    """
    ref = median(agent_avgs.values())
    current_conflicts = [
        agent
        for agent, score in agent_avgs.items()
        if abs(score - ref) > threshold
    ]

    prev_unresolved: list[str] = (prev_aggregated or {}).get("unresolved_conflicts", [])
    resolved_conflicts = [c for c in prev_unresolved if c not in current_conflicts]
    # All currently detected conflicts are considered "unresolved" until they disappear
    unresolved_conflicts = current_conflicts

    return unresolved_conflicts, resolved_conflicts


# ── Node function ─────────────────────────────────────────────────────────────

def aggregator_node(state: PipelineState) -> dict:
    """
    LangGraph node: merge parallel evaluator outputs into aggregated_feedback.
    """
    cfg = get_config()
    run_id = state["run_id"]
    iteration = state["iteration"]
    pipeline_cfg = cfg["pipeline"]

    logger.info(
        "Aggregator running",
        extra={"run_id": run_id, "iteration": iteration, "node": "aggregator"},
    )

    driver_fb   = state["driver_feedback"]   or {}
    policy_fb   = state["policy_feedback"]   or {}
    software_fb = state["software_feedback"] or {}

    # ── Per-agent score averages ──────────────────────────────────────────────
    d_avg = _driver_avg(driver_fb)
    p_avg = _policy_avg(policy_fb)
    s_avg = _software_avg(software_fb)

    # Default 0.0 — absent/failed agents must be down-weighted, matching the
    # explicit confidence=0.0 set by _safe_fallback in _ollama_base.py.
    d_conf = float(driver_fb.get("confidence", 0.0))
    p_conf = float(policy_fb.get("confidence", 0.0))
    s_conf = float(software_fb.get("confidence", 0.0))

    base_weights = cfg["agent_weights"]

    # ── Confidence-adjusted effective weights ─────────────────────────────────
    effective = {
        "driver":   base_weights["driver"]   * d_conf,
        "policy":   base_weights["policy"]   * p_conf,
        "software": base_weights["software"] * s_conf,
    }
    total_effective = sum(effective.values())

    # Guard: if all agents have near-zero confidence, fall back to raw weights
    min_weight = pipeline_cfg["min_total_effective_weight"]
    used_weight_fallback = total_effective < min_weight
    if used_weight_fallback:
        logger.warning(
            f"Total effective weight {total_effective:.3f} < {min_weight}. "
            "Falling back to raw config weights.",
            extra={"run_id": run_id, "iteration": iteration, "node": "aggregator"},
        )
        effective = dict(base_weights)
        total_effective = sum(effective.values())

    # Normalise
    normalised = {k: v / total_effective for k, v in effective.items()}

    weighted_score = (
        normalised["driver"]   * d_avg
        + normalised["policy"]   * p_avg
        + normalised["software"] * s_avg
    )

    # ── Conflict detection ────────────────────────────────────────────────────
    agent_avgs = {"driver": d_avg, "policy": p_avg, "software": s_avg}
    conflict_threshold = pipeline_cfg["conflict_threshold"]
    prev_agg = state.get("aggregated_feedback")
    unresolved, resolved = _detect_conflicts(agent_avgs, prev_agg, conflict_threshold)

    # ── Merge all concerns and recommendations ────────────────────────────────
    all_concerns = (
        [f"[driver] {c}" for c in driver_fb.get("key_concerns", [])]
        + [f"[policy] {c}" for c in policy_fb.get("key_concerns", [])]
        + [f"[software] {c}" for c in software_fb.get("key_concerns", [])]
    )
    all_recommendations = (
        [f"[driver] {r}" for r in driver_fb.get("recommendations", [])]
        + [f"[policy] {r}" for r in policy_fb.get("recommendations", [])]
        + [f"[software] {r}" for r in software_fb.get("recommendations", [])]
    )

    log_entry = make_log_entry(
        event="aggregator_complete",
        node="aggregator",
        run_id=run_id,
        iteration=iteration,
        message=(
            f"Weighted score={weighted_score:.2f} | "
            f"conflicts={unresolved} | fallback={used_weight_fallback}"
        ),
        weighted_score=weighted_score,
        unresolved_conflicts=unresolved,
        resolved_conflicts=resolved,
        used_weight_fallback=used_weight_fallback,
    )
    logger.info(log_entry["message"], extra=log_entry)

    return {
        "aggregated_feedback": {
            "weighted_score":       weighted_score,
            "agent_averages":       agent_avgs,
            # Raw confidences stored here so evaluator.py can read them without
            # re-reading individual feedback dicts or recomputing the formula.
            "agent_confidences":    {"driver": d_conf, "policy": p_conf, "software": s_conf},
            "normalised_weights":   normalised,
            "used_weight_fallback": used_weight_fallback,
            "unresolved_conflicts": unresolved,
            "resolved_conflicts":   resolved,
            "all_concerns":         all_concerns,
            "all_recommendations":  all_recommendations,
            # Audit diagnostics — captured after all branches have completed.
            "audit_diagnostics": {
                "end_user": {
                    "status":             state.get("end_user_audit_status", ""),
                    "attempts":           state.get("end_user_audit_attempts", 0),
                    "overall_audit_score": (state.get("end_user_audit") or {}).get("overall_audit_score"),
                    "issues":             (state.get("end_user_audit") or {}).get("issues", []),
                },
                "policy": {
                    "status":             state.get("policy_audit_status", ""),
                    "attempts":           state.get("policy_audit_attempts", 0),
                    "overall_audit_score": (state.get("policy_audit") or {}).get("overall_audit_score"),
                    "issues":             (state.get("policy_audit") or {}).get("issues", []),
                },
                "software": {
                    "status":             state.get("software_audit_status", ""),
                    "attempts":           state.get("software_audit_attempts", 0),
                    "overall_audit_score": (state.get("software_audit") or {}).get("overall_audit_score"),
                    "issues":             (state.get("software_audit") or {}).get("issues", []),
                },
            },
        },
        "log_entries": [log_entry],
    }
