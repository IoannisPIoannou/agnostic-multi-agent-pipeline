"""
graph/runner.py — Pipeline execution wrapper.

Handles:
  - Running the compiled graph with an initial state
  - Saving intermediate snapshots after each iteration (if configured)
  - Final persistence after the graph completes
  - Printing a metrics summary table to stdout

The intermediate-save hook works by using LangGraph's on_step callback
(stream mode) to intercept state after the evaluator node completes each cycle.
"""

from __future__ import annotations

from pathlib import Path

from graph.builder import build_graph
from state import PipelineState
from utils.config_loader import get_config
from utils.logging_config import get_logger
from utils.persistence import save_final_outputs, save_intermediate

logger = get_logger(__name__)


def _print_metrics_table(metrics_history: list) -> None:
    """Print a formatted metrics table to stdout after the run."""
    if not metrics_history:
        print("\nNo metrics recorded.")
        return

    header = f"{'Iter':>4}  {'Driver':>6}  {'Policy':>6}  {'Software':>8}  {'Overall':>7}  {'Conf_D':>6}  {'Conf_P':>6}  {'Conf_S':>6}  Stop"
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print("  PIPELINE METRICS SUMMARY")
    print(sep)
    print(header)
    print(sep)
    for m in metrics_history:
        print(
            f"{m.get('iteration', '?'):>4}  "
            f"{m.get('driver_avg', 0):>6.2f}  "
            f"{m.get('policy_avg', 0):>6.2f}  "
            f"{m.get('software_avg', 0):>8.2f}  "
            f"{m.get('overall_score', 0):>7.2f}  "
            f"{m.get('driver_conf', 0):>6.2f}  "
            f"{m.get('policy_conf', 0):>6.2f}  "
            f"{m.get('software_conf', 0):>6.2f}  "
            f"{m.get('stop_reason', '') or 'continue'}"
        )
    print(sep)
    final = metrics_history[-1]
    print(
        f"  Final score: {final.get('overall_score', 0):.2f}  |  "
        f"Stop reason: {final.get('stop_reason', 'unknown')}"
    )
    print("=" * len(header))


def run_pipeline(initial_state: PipelineState) -> PipelineState:
    """
    Execute the multi-agent pipeline and return the final state.

    Args:
        initial_state: Starting state (built by state.initial_state()).

    Returns:
        The final PipelineState after the graph terminates.
    """
    cfg = get_config()
    persist_cfg = cfg.get("persistence", {})
    save_intermediate_flag = persist_cfg.get("save_intermediate", True)
    output_dir = Path(persist_cfg.get("output_dir", "outputs"))

    graph = build_graph()
    run_id = initial_state["run_id"]

    logger.info(
        f"Pipeline starting | run_id={run_id} | "
        f"max_iter={initial_state['max_iterations']} | "
        f"threshold={initial_state['score_threshold']}",
        extra={"run_id": run_id, "node": "runner"},
    )

    final_state: PipelineState | None = None
    seen_iterations: set[int] = set()

    logger.info(
        "Graph execution started",
        extra={"run_id": run_id, "node": "runner", "event": "run_started"},
    )

    try:
        # Stream the graph so we can hook into intermediate states
        for chunk in graph.stream(initial_state, stream_mode="values"):
            final_state = chunk

            # Save a snapshot only after the evaluator has completed a full cycle:
            # evaluation_details must be present, metrics_history must be non-empty,
            # and the iteration counter must match the last metrics entry — ensuring
            # we never snapshot mid-cycle state from an earlier or later node.
            if save_intermediate_flag:
                iteration = chunk.get("iteration", 0)
                score = chunk.get("evaluation_score", 0.0)
                metrics = chunk.get("metrics_history", [])
                if (
                    chunk.get("evaluation_details")
                    and metrics
                    and iteration == metrics[-1].get("iteration")
                    and iteration not in seen_iterations
                ):
                    seen_iterations.add(iteration)
                    save_intermediate(chunk, output_dir)
                    logger.info(
                        f"Intermediate snapshot saved | iter={iteration} | score={score:.2f}",
                        extra={"run_id": run_id, "iteration": iteration, "node": "runner"},
                    )
    except Exception as exc:
        last_iteration = (final_state or {}).get("iteration", "unknown")
        logger.error(
            f"Pipeline execution failed | run_id={run_id} | last_iter={last_iteration} | error={exc}",
            extra={"run_id": run_id, "iteration": last_iteration, "node": "runner", "error": str(exc)},
        )
        raise

    if final_state is None:
        raise RuntimeError("Pipeline produced no output state.")

    # ── Persist final outputs ─────────────────────────────────────────────────
    try:
        report_path, json_path = save_final_outputs(final_state, output_dir)
    except Exception as exc:
        logger.error(
            f"Failed to persist final outputs | run_id={run_id} | error={exc}",
            extra={"run_id": run_id, "node": "runner", "error": str(exc)},
        )
        raise
    logger.info(
        f"Run complete | stop_reason={final_state.get('stop_reason')} | "
        f"final_score={final_state.get('evaluation_score', 0):.2f}",
        extra={"run_id": run_id, "node": "runner"},
    )

    # ── Print metrics table ───────────────────────────────────────────────────
    _print_metrics_table(final_state.get("metrics_history", []))

    print(f"\n  Outputs saved to:")
    print(f"    {report_path}")
    print(f"    {json_path}\n")

    return final_state
