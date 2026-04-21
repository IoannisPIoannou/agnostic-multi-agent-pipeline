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
    Read the pre-computed weighted score and agent details from aggregated_feedback.

    aggregator_node is the single source of truth for scoring; reading from its
    output eliminates the risk of the two implementations diverging silently.
    """
    agg  = state.get("aggregated_feedback") or {}
    avgs  = agg.get("agent_averages", {})
    confs = agg.get("agent_confidences", {})

    return agg.get("weighted_score", 0.0), {
        "driver_avg":           avgs.get("driver",   5.0),
        "policy_avg":           avgs.get("policy",   5.0),
        "software_avg":         avgs.get("software", 5.0),
        "driver_conf":          confs.get("driver",   0.0),
        "policy_conf":          confs.get("policy",   0.0),
        "software_conf":        confs.get("software", 0.0),
        "normalised_weights":   agg.get("normalised_weights", {}),
        "used_weight_fallback": agg.get("used_weight_fallback", False),
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

    # (c) All active agents confident enough.
    # Agents with confidence=0.0 are treated as absent (stub or parse-failure
    # fallback) and excluded from this gate so they do not permanently block
    # convergence when one evaluator is disabled during ablation.
    min_conf = pipeline_cfg["min_confidence_threshold"]
    confidences = [detail["driver_conf"], detail["policy_conf"], detail["software_conf"]]
    confident_enough = all(c >= min_conf for c in confidences if c > 0.0)

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

    threshold = pipeline_cfg["score_threshold"]
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
        "iteration":          iteration,
        "overall_score":      overall_score,
        "driver_avg":         detail["driver_avg"],
        "policy_avg":         detail["policy_avg"],
        "software_avg":       detail["software_avg"],
        "driver_conf":        detail["driver_conf"],
        "policy_conf":        detail["policy_conf"],
        "software_conf":      detail["software_conf"],
        "used_weight_fallback":  detail["used_weight_fallback"],
        "unresolved_conflicts":  len(detail["unresolved_conflicts"]),
        "converged":             converged,
        "stop_reason":           stop_reason,
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
