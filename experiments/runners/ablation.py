"""
experiments/runners/ablation.py — Experiment 2: per-agent ablation study.

Runs the pipeline once with all agents, then once per disabled agent.
Disabled agents are replaced with stub nodes that return neutral scores
(confidence=0.0, all scores=5.0) so the aggregator's weight fallback kicks in
and the remaining two agents drive the final score.

Result shows what each evaluator perspective actually contributes.

Validity note:
    Each variant runs exactly once with a fresh run ID and no seed control.
    The orchestrator, coding, and synthesis agents are LLM-backed with
    temperature > 0, so run-to-run variability is present. Score differences
    between variants reflect the combined effect of the disabled agent AND
    normal LLM randomness. Treat results as indicative trends, not statistically
    robust measurements. For higher confidence, increase num_runs_per_variant
    (not yet implemented) and average scores across replicas.

Usage:
    from experiments.runners.ablation import run_ablation_experiment
    from experiments.configs.schema import AblationConfig
    cfg = AblationConfig(name="test", goal="...")
    result = run_ablation_experiment(cfg, Path("experiments/results"))
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from experiments.configs.schema import AblationConfig
from state import PipelineState, initial_state
from utils.config_loader import get_config
from utils.logging_config import get_logger, make_log_entry
from utils.persistence import save_final_outputs

logger = get_logger(__name__)

# ── Neutral stub feedback values ──────────────────────────────────────────────
# Scores are neutral (5.0/10), confidence is 0.0 so the aggregator treats this
# agent as absent and applies the weight-fallback rule for the remaining agents.

_STUB_FEEDBACK: dict[str, dict] = {
    "end_user": {
        "usability_score":    5.0,
        "clarity_score":      5.0,
        "cost_score":         5.0,
        "practicality_score": 5.0,
        "confidence":         0.0,
        "key_concerns":       [],
        "recommendations":    [],
        "overall_summary":    "(end_user agent disabled for ablation study)",
    },
    "policy": {
        "safety_score":       5.0,
        "compliance_score":   5.0,
        "system_impact_score":5.0,
        "confidence":         0.0,
        "key_concerns":       [],
        "recommendations":    [],
        "overall_summary":    "(policy agent disabled for ablation study)",
    },
    "software": {
        "complexity_score":     5.0,
        "scalability_score":    5.0,
        "maintainability_score":5.0,
        "confidence":           0.0,
        "key_concerns":         [],
        "recommendations":      [],
        "overall_summary":      "(software agent disabled for ablation study)",
    },
}

# Maps agent name → the state key it writes to
_FEEDBACK_KEY: dict[str, str] = {
    "end_user": "driver_feedback",
    "policy":   "policy_feedback",
    "software": "software_feedback",
}


def _make_stub_node(agent_name: str):
    """Return a LangGraph-compatible stub node for a disabled agent."""
    feedback_key  = _FEEDBACK_KEY[agent_name]
    stub_feedback = _STUB_FEEDBACK[agent_name]

    def stub_node(state: PipelineState) -> dict:
        log_entry = make_log_entry(
            event=f"{agent_name}_ablated",
            node=agent_name,
            run_id=state["run_id"],
            iteration=state["iteration"],
            message=f"{agent_name} agent disabled for ablation — returning neutral stub",
        )
        return {
            feedback_key: stub_feedback,
            "log_entries": [log_entry],
        }

    stub_node.__name__ = f"{agent_name}_stub"
    return stub_node


def _build_ablated_graph(disabled_agents: list[str]):
    """
    Build a fresh (non-cached) LangGraph graph with stub nodes for each
    disabled agent. Mirrors the topology in graph/builder.py exactly.
    """
    from langgraph.graph import END, START, StateGraph

    from agents.aggregator import aggregator_node
    from agents.coding import coding_node
    from agents.end_user import end_user_node
    from agents.orchestrator import orchestrator_node
    from agents.policy import policy_node
    from agents.software import software_node
    from agents.synthesis import synthesis_node
    from evaluation.evaluator import evaluator_node, routing_function

    node_funcs = {
        "end_user": end_user_node,
        "policy":   policy_node,
        "software": software_node,
    }
    for agent in disabled_agents:
        node_funcs[agent] = _make_stub_node(agent)

    builder = StateGraph(PipelineState)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("end_user",     node_funcs["end_user"])
    builder.add_node("policy",       node_funcs["policy"])
    builder.add_node("software",     node_funcs["software"])
    builder.add_node("aggregator",   aggregator_node)
    builder.add_node("coding",       coding_node)
    builder.add_node("evaluator",    evaluator_node)
    builder.add_node("synthesis",    synthesis_node)

    builder.add_edge(START, "orchestrator")
    builder.add_edge("orchestrator", "end_user")
    builder.add_edge("orchestrator", "policy")
    builder.add_edge("orchestrator", "software")
    builder.add_edge("end_user",  "aggregator")
    builder.add_edge("policy",    "aggregator")
    builder.add_edge("software",  "aggregator")
    builder.add_edge("aggregator", "coding")
    builder.add_edge("coding",     "evaluator")
    builder.add_conditional_edges(
        "evaluator",
        routing_function,
        {"continue": "orchestrator", "stop": "synthesis"},
    )
    builder.add_edge("synthesis", END)

    return builder.compile()


def _run_graph(graph, state: PipelineState, output_dir: Path) -> PipelineState:
    """Execute a compiled graph via stream and return the final state."""
    from utils.persistence import save_intermediate

    pipeline_cfg = get_config().get("persistence", {})
    save_intermediate_flag = pipeline_cfg.get("save_intermediate", True)

    final_state = None
    seen_iterations: set[int] = set()

    for chunk in graph.stream(state, stream_mode="values"):
        final_state = chunk
        if save_intermediate_flag:
            iteration = chunk.get("iteration", 0)
            metrics   = chunk.get("metrics_history", [])
            if (
                chunk.get("evaluation_details")
                and metrics
                and iteration == metrics[-1].get("iteration")
                and iteration not in seen_iterations
            ):
                seen_iterations.add(iteration)
                save_intermediate(chunk, output_dir)

    if final_state is None:
        raise RuntimeError("Graph produced no output state.")
    return final_state


def _extract_variant_metrics(final_state: dict, disabled: list[str]) -> dict:
    history = final_state.get("metrics_history", [])
    artifact = final_state.get("solution_artifact") or {}
    return {
        "disabled_agents":   disabled,
        "label":             "full" if not disabled else f"no_{'+'.join(disabled)}",
        "iterations_run":    final_state.get("iteration", 0),
        "final_score":       final_state.get("evaluation_score", 0.0),
        "stop_reason":       final_state.get("stop_reason", "unknown"),
        "score_progression": [m.get("overall_score", 0.0) for m in history],
        "unresolved_conflicts_final": (
            history[-1].get("unresolved_conflicts", 0) if history else 0
        ),
        "design_summary":    artifact.get("design_summary", ""),
        "num_components":    len(artifact.get("components", [])),
        "final_report_path": str(
            Path("outputs") / final_state.get("run_id", "") / "final_report.md"
        ),
    }


def run_ablation_experiment(cfg: AblationConfig, output_dir: Path) -> dict:
    """
    Run the full pipeline, then one run per disabled agent.
    Returns the full result dict and saves it to output_dir.
    """
    pipeline_cfg = get_config()
    experiment_id = (
        f"ablation_{cfg.name.lower().replace(' ', '_')}_"
        f"{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    exp_dir = output_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Variants: full run first, then one per ablated agent
    variants_to_run: list[list[str]] = [[]] + [[a] for a in cfg.ablate_agents]
    variant_results = []

    for disabled in variants_to_run:
        label  = "full" if not disabled else f"no_{disabled[0]}"
        run_id = str(uuid.uuid4())[:8]
        logger.info(
            f"[ablation] Variant={label} | run_id={run_id}",
            extra={"run_id": run_id, "node": "experiment"},
        )

        state  = initial_state(user_goal=cfg.goal, run_id=run_id, cfg=pipeline_cfg)
        graph  = _build_ablated_graph(disabled)
        run_output_dir = Path("outputs")

        try:
            final_state = _run_graph(graph, state, run_output_dir)
            save_final_outputs(final_state, run_output_dir)
        except Exception as exc:
            logger.error(
                f"[ablation] Variant={label} failed: {exc}",
                extra={"run_id": run_id, "node": "experiment"},
            )
            variant_results.append({
                "label": label,
                "disabled_agents": disabled,
                "run_id": run_id,
                "error": str(exc),
            })
            continue

        metrics = _extract_variant_metrics(final_state, disabled)
        metrics["run_id"] = run_id
        variant_results.append(metrics)
        logger.info(
            f"[ablation] Variant={label} done | "
            f"score={metrics['final_score']:.2f} | "
            f"iters={metrics['iterations_run']}",
            extra={"run_id": run_id, "node": "experiment"},
        )

    result = {
        "experiment_id":   experiment_id,
        "experiment_type": "ablation",
        "name":            cfg.name,
        "description":     cfg.description,
        "goal":            cfg.goal,
        "ablated_agents":  cfg.ablate_agents,
        "timestamp_utc":   datetime.now(tz=timezone.utc).isoformat(),
        "variants":        variant_results,
    }

    out_path = exp_dir / "results.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(f"[ablation] Results saved to {out_path}")
    return result
