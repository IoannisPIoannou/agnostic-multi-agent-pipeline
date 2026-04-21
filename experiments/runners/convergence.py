"""
experiments/runners/convergence.py — Experiment 1: score convergence stability.

Runs the same goal N times with independent run IDs and collects:
  - final score and stop reason
  - iterations needed
  - score progression per iteration
  - unresolved conflicts per iteration
  - per-agent confidence per iteration

Usage:
    from experiments.runners.convergence import run_convergence_experiment
    from experiments.configs.schema import ConvergenceConfig
    cfg = ConvergenceConfig(name="test", goal="...", num_runs=3)
    result = run_convergence_experiment(cfg, Path("experiments/results"))
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from experiments.configs.schema import ConvergenceConfig
from graph.builder import reset_graph
from graph.runner import run_pipeline
from state import initial_state
from utils.config_loader import get_config, reload_config
from utils.logging_config import get_logger

logger = get_logger(__name__)


def _extract_run_metrics(final_state: dict) -> dict:
    """Pull the fields needed for convergence analysis from a completed pipeline state."""
    history = final_state.get("metrics_history", [])
    return {
        "iterations_run":               final_state.get("iteration", 0),
        "final_score":                  final_state.get("evaluation_score", 0.0),
        "stop_reason":                  final_state.get("stop_reason", "unknown"),
        "converged_internally":         any(m.get("converged", False) for m in history),
        "score_progression":            [m.get("overall_score", 0.0) for m in history],
        "driver_avg_per_iter":          [m.get("driver_avg", 0.0) for m in history],
        "policy_avg_per_iter":          [m.get("policy_avg", 0.0) for m in history],
        "software_avg_per_iter":        [m.get("software_avg", 0.0) for m in history],
        "driver_conf_per_iter":         [m.get("driver_conf", 0.0) for m in history],
        "policy_conf_per_iter":         [m.get("policy_conf", 0.0) for m in history],
        "software_conf_per_iter":       [m.get("software_conf", 0.0) for m in history],
        "unresolved_conflicts_per_iter":[m.get("unresolved_conflicts", 0) for m in history],
        "used_weight_fallback_any_iter": any(
            m.get("used_weight_fallback", False) for m in history
        ),
        "fallback_events": sum(
            1 for e in final_state.get("log_entries", [])
            if e.get("event") == "fallback_activated"
        ),
    }


def run_convergence_experiment(cfg: ConvergenceConfig, output_dir: Path) -> dict:
    """
    Run the pipeline cfg.num_runs times with the same goal.
    Returns the full result dict and saves it to output_dir.
    """
    pipeline_cfg = (
        reload_config(cfg.pipeline_config_path)
        if cfg.pipeline_config_path
        else get_config()
    )

    experiment_id = (
        f"convergence_{cfg.name.lower().replace(' ', '_')}_"
        f"{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    exp_dir = output_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    reset_graph()

    runs = []
    for i in range(cfg.num_runs):
        if cfg.seed is not None:
            os.environ["OLLAMA_SEED"] = str(cfg.seed + i)
        else:
            os.environ.pop("OLLAMA_SEED", None)

        run_id = str(uuid.uuid4())[:8]
        logger.info(
            f"[convergence] Run {i + 1}/{cfg.num_runs} | run_id={run_id}",
            extra={"run_id": run_id, "node": "experiment"},
        )

        state = initial_state(user_goal=cfg.goal, run_id=run_id, cfg=pipeline_cfg)

        try:
            final_state = run_pipeline(state)
        except Exception as exc:
            logger.error(
                f"[convergence] Run {i + 1} failed: {exc}",
                extra={"run_id": run_id, "node": "experiment"},
            )
            runs.append({
                "run_index": i + 1,
                "run_id": run_id,
                "error": str(exc),
            })
            continue

        run_result = {
            "run_index": i + 1,
            "run_id":    run_id,
            **_extract_run_metrics(final_state),
        }
        runs.append(run_result)
        logger.info(
            f"[convergence] Run {i + 1} done | "
            f"score={run_result['final_score']:.2f} | "
            f"iters={run_result['iterations_run']} | "
            f"stop={run_result['stop_reason']}",
            extra={"run_id": run_id, "node": "experiment"},
        )

    result = {
        "experiment_id":   experiment_id,
        "experiment_type": "convergence",
        "name":            cfg.name,
        "description":     cfg.description,
        "goal":            cfg.goal,
        "num_runs":        cfg.num_runs,
        "timestamp_utc":   datetime.now(tz=timezone.utc).isoformat(),
        "gemini_cache_note": (
            "Gemini server-side prompt caching may inflate consistency. "
            "Runs with identical goals and early context may share cached responses."
        ),
        "runs":            runs,
    }

    out_path = exp_dir / "results.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(f"[convergence] Results saved to {out_path}")
    return result
