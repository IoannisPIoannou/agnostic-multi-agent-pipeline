"""
experiments/runners/baseline.py — Experiment 3: pipeline vs single-LLM baseline.

Steps:
  1. Generate a baseline solution using a single Gemini call (no iteration loop).
  2. Run the full multi-agent pipeline for the same goal.
  3. Use a judge LLM to compare both solutions on configurable criteria.
  4. Save the full comparison result.

Fairness design:
  - The pipeline is represented by its synthesis final_report (the polished prose
    deliverable) rather than the raw coding-agent artifact. This avoids implicit
    identity leakage from agent-referencing language in implementation_notes, and
    presents the pipeline's best output to the judge.
  - The baseline artifact is converted to a comparable prose block for the judge.
  - Both sides are shown without labels indicating their origin.

Validity note:
  - Each experiment is a single paired comparison (one pipeline run, one baseline
    call). Results are indicative trends, not statistically robust measurements.
    LLM variability affects both sides; re-run multiple times and compare the
    distribution for higher confidence.
  - The judge and both solutions use the same model family (gemini-2.5-flash),
    which may introduce same-model stylistic preference. Consider using a
    different judge model for independent validation.

Usage:
    from experiments.runners.baseline import run_baseline_experiment
    from experiments.configs.schema import BaselineConfig
    cfg = BaselineConfig(name="test", goal="...")
    result = run_baseline_experiment(cfg, Path("experiments/results"))
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from experiments.baselines.simple_llm import generate_baseline
from experiments.configs.schema import BaselineConfig
from experiments.judges.llm_judge import judge_solutions
from graph.runner import run_pipeline
from state import initial_state
from utils.config_loader import get_config
from utils.logging_config import get_logger

logger = get_logger(__name__)


def run_baseline_experiment(cfg: BaselineConfig, output_dir: Path) -> dict:
    """
    Compare the pipeline's output against a single-LLM baseline solution.
    Returns the full comparison result dict and saves it to output_dir.
    """
    pipeline_cfg = get_config()
    experiment_id = (
        f"baseline_{cfg.name.lower().replace(' ', '_')}_"
        f"{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    exp_dir = output_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Baseline solution ─────────────────────────────────────────────
    logger.info("[baseline] Generating baseline solution...")
    baseline_artifact = generate_baseline(goal=cfg.goal, model=cfg.baseline_model)

    # ── Step 2: Pipeline solution ─────────────────────────────────────────────
    run_id = str(uuid.uuid4())[:8]
    logger.info(f"[baseline] Running pipeline | run_id={run_id}")
    state = initial_state(user_goal=cfg.goal, run_id=run_id, cfg=pipeline_cfg)

    try:
        final_state = run_pipeline(state)
    except Exception as exc:
        logger.error(f"[baseline] Pipeline run failed: {exc}")
        result = {
            "experiment_id":   experiment_id,
            "experiment_type": "baseline",
            "name":            cfg.name,
            "goal":            cfg.goal,
            "timestamp_utc":   datetime.now(tz=timezone.utc).isoformat(),
            "error":           str(exc),
        }
        (exp_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    pipeline_artifact = final_state.get("solution_artifact") or {}
    pipeline_score    = final_state.get("evaluation_score", 0.0)
    pipeline_iters    = final_state.get("iteration", 0)
    pipeline_stop     = final_state.get("stop_reason", "unknown")

    # Prefer the synthesis final_report for the judge — it is the pipeline's
    # polished deliverable and avoids agent-referencing language that could
    # signal to the judge which solution came from the multi-agent system.
    pipeline_text = final_state.get("final_report") or None

    # ── Step 3: Judge comparison ──────────────────────────────────────────────
    logger.info(
        f"[baseline] Running judge comparison | "
        f"pipeline_text={'yes' if pipeline_text else 'no (falling back to artifact)'}..."
    )
    try:
        judgement = judge_solutions(
            goal=cfg.goal,
            pipeline_artifact=pipeline_artifact,
            baseline_artifact=baseline_artifact,
            criteria=cfg.judge_criteria,
            model=cfg.judge_model,
            pipeline_text=pipeline_text,
        )
    except Exception as exc:
        logger.error(f"[baseline] Judge call failed: {exc}")
        judgement = {"judgement_error": str(exc)}

    result = {
        "experiment_id":   experiment_id,
        "experiment_type": "baseline",
        "name":            cfg.name,
        "description":     cfg.description,
        "goal":            cfg.goal,
        "timestamp_utc":   datetime.now(tz=timezone.utc).isoformat(),
        "pipeline": {
            "run_id":            run_id,
            "final_score":       pipeline_score,
            "stop_reason":       pipeline_stop,
            "iterations":        pipeline_iters,
            "num_components":    len(pipeline_artifact.get("components", [])),
            "design_summary":    pipeline_artifact.get("design_summary", ""),
            "used_final_report": pipeline_text is not None,
        },
        "baseline": {
            "model":          cfg.baseline_model,
            "num_components": len(baseline_artifact.get("components", [])),
            "design_summary": baseline_artifact.get("design_summary", ""),
        },
        "judgement": judgement,
    }

    out_path = exp_dir / "results.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Save raw artifacts for manual inspection
    (exp_dir / "pipeline_artifact.json").write_text(
        json.dumps(pipeline_artifact, indent=2), encoding="utf-8"
    )
    (exp_dir / "baseline_artifact.json").write_text(
        json.dumps(baseline_artifact, indent=2), encoding="utf-8"
    )

    if "judgement_error" not in judgement:
        logger.info(
            f"[baseline] Done | pipeline={pipeline_score:.2f} | "
            f"judge_pipeline={judgement['overall_score_pipeline']:.1f} | "
            f"judge_baseline={judgement['overall_score_baseline']:.1f} | "
            f"preferred={'pipeline' if judgement['pipeline_preferred'] else 'baseline'}"
        )
    else:
        logger.warning(
            f"[baseline] Done (judge failed) | pipeline={pipeline_score:.2f} | "
            f"judge_error={judgement['judgement_error']}"
        )
    return result
