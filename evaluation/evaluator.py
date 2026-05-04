"""
evaluation/evaluator.py — Convergence detection and pipeline control (pure Python).

Reads the pre-computed weighted score from aggregated_feedback (the aggregator's
single source of truth) and determines whether the pipeline should stop.

Stopping conditions (ANY one is sufficient):
  - score >= score_threshold  → "threshold_reached"
  - convergence (ALL three):
      (a) |score_n - score_(n-1)| < convergence_delta
      (b) no unresolved conflicts
      (c) all agent confidences >= min_confidence_threshold
    → "converged"
  - iteration >= max_iterations → "max_iterations"

Also appends an IterationMetrics record to state["metrics_history"].
"""

from __future__ import annotations

from state import PipelineState
from utils.config_loader import get_config
from utils.logging_config import get_logger, make_log_entry

logger = get_logger(__name__)


# ── Score computation ─────────────────────────────────────────────────────────

def _compute_overall_score(state: PipelineState, cfg: dict) -> tuple[float, dict]:
    """
    Read the independent evaluator's score as the stopping criterion.

    The independent_evaluator node scores the solution against the original
    goal without seeing parallel-agent feedback, removing self-evaluation bias.
    Unresolved conflicts are still read from aggregated_feedback for the
    convergence check — the parallel agents remain the conflict signal source.
    """
    ind = state.get("independent_evaluation") or {}
    agg = state.get("aggregated_feedback")    or {}

    return ind.get("overall_score", 0.0), {
        "goal_alignment":       ind.get("goal_alignment_score", 5.0),
        "completeness":         ind.get("completeness_score",   5.0),
        "feasibility":          ind.get("feasibility_score",    5.0),
        "clarity":              ind.get("clarity_score",        5.0),
        "innovation":           ind.get("innovation_score",     5.0),
        "confidence":           ind.get("confidence",           0.0),
        "evaluation_summary":   ind.get("evaluation_summary",  ""),
        "unresolved_conflicts": agg.get("unresolved_conflicts", []),
    }


# ── Convergence check ─────────────────────────────────────────────────────────

def _is_converged(
    current_score: float,
    state: PipelineState,
    cfg: dict,
    detail: dict,
) -> bool:
    """
    True iff ALL three convergence conditions are met simultaneously:
      (a) Score change below convergence_delta
      (b) No unresolved agent conflicts
      (c) All agent confidences >= min_confidence_threshold
    """
    pipeline_cfg = cfg["pipeline"]
    history = state.get("metrics_history", [])

    # (a) Score stability
    if not history:
        return False
    prev_score = history[-1]["overall_score"]
    score_stable = abs(current_score - prev_score) < pipeline_cfg["convergence_delta"]

    # (b) No unresolved conflicts
    unresolved = detail.get("unresolved_conflicts", [])
    no_conflicts = len(unresolved) <= pipeline_cfg["max_unresolved_conflicts"]

    # (c) Independent evaluator is sufficiently confident in its assessment.
    min_conf = pipeline_cfg["min_confidence_threshold"]
    confident_enough = detail["confidence"] >= min_conf

    return score_stable and no_conflicts and confident_enough


# ── Node function ─────────────────────────────────────────────────────────────

def evaluator_node(state: PipelineState) -> dict:
    """
    LangGraph node: compute score, check stopping conditions, record metrics.
    """
    cfg = get_config()
    run_id = state["run_id"]
    iteration = state["iteration"]
    pipeline_cfg = cfg["pipeline"]

    logger.info(
        "Evaluator running",
        extra={"run_id": run_id, "iteration": iteration, "node": "evaluator"},
    )

    overall_score, detail = _compute_overall_score(state, cfg)
    converged = _is_converged(overall_score, state, cfg, detail)

    threshold = state["score_threshold"]
    max_iter  = state["max_iterations"]

    stop_threshold = overall_score >= threshold
    stop_max_iter  = iteration >= max_iter
    should_stop    = stop_threshold or converged or stop_max_iter

    if stop_threshold:
        stop_reason = "threshold_reached"
    elif converged:
        stop_reason = "converged"
    elif stop_max_iter:
        stop_reason = "max_iterations"
    else:
        stop_reason = ""

    # ── Append metrics history entry ─────────────────────────────────────────
    metrics_entry = {
        "iteration":            iteration,
        "overall_score":        overall_score,
        "goal_alignment":       detail["goal_alignment"],
        "completeness":         detail["completeness"],
        "feasibility":          detail["feasibility"],
        "clarity":              detail["clarity"],
        "innovation":           detail["innovation"],
        "confidence":           detail["confidence"],
        "unresolved_conflicts": len(detail["unresolved_conflicts"]),
        "converged":            converged,
        "stop_reason":          stop_reason,
        "used_weight_fallback": (state.get("aggregated_feedback") or {}).get("used_weight_fallback", False),
    }

    log_entry = make_log_entry(
        event="evaluator_complete",
        node="evaluator",
        run_id=run_id,
        iteration=iteration,
        message=(
            f"Score={overall_score:.2f}/{threshold} | "
            f"stop={should_stop} ({stop_reason or 'continue'})"
        ),
        overall_score=overall_score,
        converged=converged,
        should_stop=should_stop,
        stop_reason=stop_reason,
    )
    logger.info(log_entry["message"], extra=log_entry)

    return {
        "evaluation_score":   overall_score,
        "evaluation_details": detail,
        "should_stop":        should_stop,
        "stop_reason":        stop_reason,
        "metrics_history":    [metrics_entry],   # Annotated reducer appends this
        "log_entries":        [log_entry],
    }


# ── Routing function (used by graph/builder.py) ───────────────────────────────

def routing_function(state: PipelineState) -> str:
    """Return 'stop' or 'continue' based on should_stop flag."""
    return "stop" if state["should_stop"] else "continue"
