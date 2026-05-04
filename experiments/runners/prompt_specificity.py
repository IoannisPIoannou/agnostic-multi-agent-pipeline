"""
experiments/runners/prompt_specificity.py — Experiment 5: prompt specificity sensitivity.

Runs 4 sub-experiments (convergence, ablation, baseline, threshold sweep) across
3 specificity levels (vague / moderate / highly_specific) for the same domain goal.

Key feature: a JSON-backed RunRegistry caches every executed pipeline run.
Cache entries are reused when prompt text, threshold, max_iterations, variant,
repetition, AND model configuration all match.  The registry lives at a stable
path scoped to cfg.name, so a second invocation of the same config reuses all
compatible runs across timestamped experiment directories.

Registry location (persists across runs):
    output_dir / "prompt_specificity" / {cfg.name} / run_registry.json

Results location (per invocation):
    output_dir / "prompt_specificity" / {cfg.name} / {experiment_id} / *

run_registry.json — stores only actually-executed runs (keyed by run parameters).
results.json       — stores all logical experiment entries, including reused records.

Run count (defaults: 3 reps, thresholds=[7.0, 8.0, 9.0], convergence_threshold=8.0):
  Phase 1 convergence : 9  new runs
  Phase 2 ablation    : 27 new + 9 reused  = 36 variant-runs
  Phase 3 baseline    : 0  new pipeline    + 3 LLM calls + 3 judge calls
  Phase 4 sweep       : 18 new + 9 reused  = 27 sweep-runs
  Total new pipeline runs : 54 (27 reused)

Usage:
    from experiments.runners.prompt_specificity import run_prompt_specificity_experiment
    from experiments.configs.schema import PromptSpecificityConfig
    cfg = PromptSpecificityConfig(name="test", domain="logistics", prompts={...})
    result = run_prompt_specificity_experiment(cfg, Path("experiments/results"))
"""

from __future__ import annotations

import copy
import csv
import json
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from experiments.configs.schema import PromptSpecificityConfig
from experiments.runners.ablation import _build_ablated_graph, _run_graph
from experiments.baselines.simple_llm import generate_baseline
from experiments.judges.llm_judge import judge_solutions
from graph.runner import run_pipeline
from state import initial_state
from utils.config_loader import get_config
from utils.logging_config import get_logger
from utils.persistence import save_final_outputs

logger = get_logger(__name__)

_SPECIFICITIES = ["vague", "moderate", "highly_specific"]


# ── Model fingerprint ─────────────────────────────────────────────────────────

def _model_fingerprint(pipeline_cfg: dict) -> dict:
    """
    Extract a dict of {agent: model_id} for the Gemini-backed agents.
    Used to detect config changes that would invalidate cached runs.
    """
    models = pipeline_cfg.get("models", {})
    return {
        agent: models.get(agent, {}).get("model", "")
        for agent in ["orchestrator", "independent_evaluator", "coding", "synthesis"]
    }


def _make_meta(goal: str, threshold: float, max_iterations: int, pipeline_cfg: dict) -> dict:
    """Build the metadata fingerprint stored with every registry entry."""
    return {
        "prompt_text":        goal,
        "threshold":          threshold,
        "max_iterations":     max_iterations,
        "model_fingerprint":  _model_fingerprint(pipeline_cfg),
    }


# ── RunRegistry ───────────────────────────────────────────────────────────────

class RunRegistry:
    """
    JSON-backed cache for completed pipeline runs, keyed by run parameters.

    The registry persists at a stable path across multiple experiment invocations.
    Each entry includes a `_meta` fingerprint (prompt text, threshold, max_iterations,
    model config).  A cache hit is only valid when the fingerprint matches the
    current request — any config change forces a fresh run.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        self._path.write_text(
            json.dumps(self._data, indent=2, default=str), encoding="utf-8"
        )

    def get(self, key: str, expected_meta: dict | None = None) -> dict | None:
        """
        Return the cached entry for key, or None on miss.
        If expected_meta is provided, returns None when the stored metadata
        does not match (prompt changed, threshold changed, model changed, etc.).
        """
        entry = self._data.get(key)
        if entry is None:
            return None
        if expected_meta is not None and entry.get("_meta") != expected_meta:
            logger.info(
                f"[registry] Cache miss (meta mismatch) for key={key}",
                extra={"run_id": "", "node": "prompt_specificity"},
            )
            return None
        return entry

    def put(self, key: str, data: dict) -> None:
        """Persist a newly executed run (data must include _meta)."""
        self._data[key] = data
        self._save()

    @staticmethod
    def make_key(specificity: str, threshold: float, variant: str, rep: int) -> str:
        """
        Build a registry key.
        variant: "full" | "no_end_user" | "no_policy" | "no_software"
        rep: 1-indexed
        """
        return f"{specificity}__{threshold:.1f}__{variant}__{rep}"


# ── Low-level execution helpers ───────────────────────────────────────────────

def _extract_run_result(final_state: dict, disabled: list[str] | None = None) -> dict:
    """Extract key metrics and artifacts from a completed pipeline state."""
    history  = final_state.get("metrics_history", [])
    artifact = final_state.get("solution_artifact") or {}
    return {
        "run_id":            final_state.get("run_id", ""),
        "disabled_agents":   list(disabled or []),
        "final_score":       final_state.get("evaluation_score", 0.0),
        "stop_reason":       final_state.get("stop_reason", "unknown"),
        "iterations_run":    final_state.get("iteration", 0),
        "score_progression": [m.get("overall_score", 0.0) for m in history],
        "num_components":    len(artifact.get("components", [])),
        "solution_artifact": artifact,
        "final_report":      final_state.get("final_report"),
    }


def _run_full_pipeline(goal: str, run_id: str, threshold: float, max_iterations: int) -> dict:
    """Execute a full (non-ablated) pipeline run and return the extracted result."""
    cfg = copy.deepcopy(get_config())
    cfg["pipeline"]["score_threshold"] = threshold
    cfg["pipeline"]["max_iterations"]  = max_iterations
    state       = initial_state(user_goal=goal, run_id=run_id, cfg=cfg)
    final_state = run_pipeline(state)
    return _extract_run_result(final_state)


def _run_ablated_pipeline(
    goal: str,
    run_id: str,
    disabled: list[str],
    threshold: float,
    max_iterations: int,
) -> dict:
    """Execute an ablated pipeline run (one agent stubbed out) and return the extracted result."""
    cfg = copy.deepcopy(get_config())
    cfg["pipeline"]["score_threshold"] = threshold
    cfg["pipeline"]["max_iterations"]  = max_iterations
    state       = initial_state(user_goal=goal, run_id=run_id, cfg=cfg)
    graph       = _build_ablated_graph(disabled)
    output_dir  = Path("outputs")
    final_state = _run_graph(graph, state, output_dir)
    save_final_outputs(final_state, output_dir)
    return _extract_run_result(final_state, disabled)


# ── Registry-backed run helpers ───────────────────────────────────────────────

def _get_or_run_full(
    registry: RunRegistry,
    spec: str,
    threshold: float,
    max_iters: int,
    rep: int,
    goal: str,
) -> dict:
    """Return a cached full-pipeline result, or execute and cache a new one."""
    pipeline_cfg  = get_config()
    expected_meta = _make_meta(goal, threshold, max_iters, pipeline_cfg)
    key           = RunRegistry.make_key(spec, threshold, "full", rep)

    cached = registry.get(key, expected_meta)
    if cached:
        logger.info(
            f"[prompt_specificity] REUSE spec={spec} thr={threshold:.1f} full rep={rep} "
            f"id={cached.get('run_id','')}",
            extra={"run_id": cached.get("run_id", ""), "node": "prompt_specificity"},
        )
        return {**cached, "status": "reused_existing"}

    run_id = str(uuid.uuid4())[:8]
    logger.info(
        f"[prompt_specificity] RUN spec={spec} thr={threshold:.1f} full rep={rep} id={run_id}",
        extra={"run_id": run_id, "node": "prompt_specificity"},
    )
    try:
        result = _run_full_pipeline(goal, run_id, threshold, max_iters)
    except Exception as exc:
        logger.error(
            f"[prompt_specificity] FAILED spec={spec} thr={threshold:.1f} full rep={rep}: {exc}",
            extra={"run_id": run_id, "node": "prompt_specificity"},
        )
        result = {
            "run_id": run_id, "error": str(exc), "final_score": 0.0,
            "stop_reason": "error", "iterations_run": 0, "num_components": 0,
            "score_progression": [], "solution_artifact": {}, "final_report": None,
        }

    result["_meta"]  = expected_meta
    result["status"] = "executed_new"
    registry.put(key, result)
    return result


def _get_or_run_ablated(
    registry: RunRegistry,
    spec: str,
    threshold: float,
    max_iters: int,
    rep: int,
    goal: str,
    agent: str,
) -> dict:
    """Return a cached ablated-pipeline result, or execute and cache a new one."""
    pipeline_cfg  = get_config()
    expected_meta = _make_meta(goal, threshold, max_iters, pipeline_cfg)
    variant       = f"no_{agent}"
    key           = RunRegistry.make_key(spec, threshold, variant, rep)

    cached = registry.get(key, expected_meta)
    if cached:
        logger.info(
            f"[prompt_specificity] REUSE spec={spec} thr={threshold:.1f} {variant} rep={rep} "
            f"id={cached.get('run_id','')}",
            extra={"run_id": cached.get("run_id", ""), "node": "prompt_specificity"},
        )
        return {**cached, "status": "reused_existing"}

    run_id = str(uuid.uuid4())[:8]
    logger.info(
        f"[prompt_specificity] RUN spec={spec} thr={threshold:.1f} {variant} rep={rep} id={run_id}",
        extra={"run_id": run_id, "node": "prompt_specificity"},
    )
    try:
        result = _run_ablated_pipeline(goal, run_id, [agent], threshold, max_iters)
    except Exception as exc:
        logger.error(
            f"[prompt_specificity] FAILED spec={spec} {variant} rep={rep}: {exc}",
            extra={"run_id": run_id, "node": "prompt_specificity"},
        )
        result = {
            "run_id": run_id, "error": str(exc), "final_score": 0.0,
            "stop_reason": "error", "iterations_run": 0, "num_components": 0,
            "score_progression": [], "solution_artifact": {}, "final_report": None,
        }

    result["_meta"]  = expected_meta
    result["status"] = "executed_new"
    registry.put(key, result)
    return result


# ── Aggregation helper ────────────────────────────────────────────────────────

def _agg(runs: list[dict]) -> dict:
    good   = [r for r in runs if "error" not in r]
    scores = [r["final_score"] for r in good]
    iters  = [r["iterations_run"] for r in good]
    return {
        "mean_score": round(statistics.mean(scores), 4) if scores else 0.0,
        "score_std":  round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
        "score_min":  round(min(scores), 2) if scores else 0.0,
        "score_max":  round(max(scores), 2) if scores else 0.0,
        "mean_iters": round(statistics.mean(iters), 2) if iters else 0.0,
    }


# ── Phase runners ─────────────────────────────────────────────────────────────

def _run_convergence_phase(cfg: PromptSpecificityConfig, registry: RunRegistry) -> dict:
    print(f"\n  [Phase 1 — Convergence] threshold={cfg.convergence_threshold}  "
          f"reps={cfg.convergence_num_runs}  total={3 * cfg.convergence_num_runs} runs")

    results = []
    for spec in _SPECIFICITIES:
        goal = cfg.prompts[spec]
        runs = []
        for rep in range(1, cfg.convergence_num_runs + 1):
            r = _get_or_run_full(
                registry, spec, cfg.convergence_threshold,
                cfg.convergence_max_iterations, rep, goal,
            )
            r["rep"] = rep
            runs.append(r)
            tag = "REUSE" if r["status"] == "reused_existing" else "NEW  "
            print(f"    [{tag}] spec={spec:<18} rep={rep}  "
                  f"score={r.get('final_score', 0):.2f}  "
                  f"iters={r.get('iterations_run', 0)}  "
                  f"stop={r.get('stop_reason','?')}")

        agg = _agg(runs)
        results.append({"specificity": spec, "runs": runs, **agg})

    return {
        "threshold":             cfg.convergence_threshold,
        "max_iterations":        cfg.convergence_max_iterations,
        "runs_per_specificity":  cfg.convergence_num_runs,
        "results":               results,
    }


def _run_ablation_phase(cfg: PromptSpecificityConfig, registry: RunRegistry) -> dict:
    variants = ["full"] + [f"no_{a}" for a in cfg.ablation_agents]
    print(f"\n  [Phase 2 — Ablation] variants={variants}  "
          f"reps={cfg.ablation_num_runs_per_variant}  "
          f"total={3 * len(variants) * cfg.ablation_num_runs_per_variant} variant-runs")

    results = []
    for spec in _SPECIFICITIES:
        goal            = cfg.prompts[spec]
        variant_results = []

        for variant in variants:
            runs = []
            for rep in range(1, cfg.ablation_num_runs_per_variant + 1):
                if variant == "full":
                    r = _get_or_run_full(
                        registry, spec, cfg.convergence_threshold,
                        cfg.convergence_max_iterations, rep, goal,
                    )
                else:
                    agent = variant.removeprefix("no_")
                    r = _get_or_run_ablated(
                        registry, spec, cfg.convergence_threshold,
                        cfg.convergence_max_iterations, rep, goal, agent,
                    )
                r["rep"] = rep
                runs.append(r)
                tag = "REUSE" if r["status"] == "reused_existing" else "NEW  "
                print(f"    [{tag}] spec={spec:<18} variant={variant:<14} rep={rep}  "
                      f"score={r.get('final_score', 0):.2f}")

            agg = _agg(runs)
            variant_results.append({"variant": variant, "runs": runs, **agg})

        full_mean = next(
            (v["mean_score"] for v in variant_results if v["variant"] == "full"), 0.0
        )
        for v in variant_results:
            v["delta_vs_full"] = (
                None if v["variant"] == "full"
                else round(v["mean_score"] - full_mean, 4)
            )

        results.append({"specificity": spec, "variants": variant_results})

    return {"ablation_agents": cfg.ablation_agents, "results": results}


def _run_baseline_phase(cfg: PromptSpecificityConfig, registry: RunRegistry) -> dict:
    print(f"\n  [Phase 3 — Baseline] "
          f"baseline_model={cfg.baseline_model}  judge_model={cfg.judge_model}")

    results = []
    for spec in _SPECIFICITIES:
        goal = cfg.prompts[spec]

        # Reuse (or execute) pipeline runs up to baseline_num_runs
        pipeline_runs = []
        for rep in range(1, cfg.baseline_num_runs + 1):
            r = _get_or_run_full(
                registry, spec, cfg.convergence_threshold,
                cfg.convergence_max_iterations, rep, goal,
            )
            r["rep"] = rep
            pipeline_runs.append(r)
            tag = "REUSE" if r["status"] == "reused_existing" else "NEW  "
            print(f"    [{tag}] pipeline spec={spec:<18} rep={rep}  "
                  f"score={r.get('final_score', 0):.2f}")

        good_runs  = [r for r in pipeline_runs if "error" not in r]
        mean_score = (
            round(statistics.mean([r["final_score"] for r in good_runs]), 4)
            if good_runs else 0.0
        )
        best_run = max(good_runs, key=lambda r: r.get("final_score", 0)) if good_runs else {}

        # Generate single-LLM baseline
        print(f"    [NEW  ] baseline LLM spec={spec}")
        try:
            baseline_artifact = generate_baseline(goal=goal, model=cfg.baseline_model)
        except Exception as exc:
            logger.error(
                f"[prompt_specificity] Baseline generation failed for {spec}: {exc}",
                extra={"run_id": "", "node": "prompt_specificity"},
            )
            results.append({
                "specificity": spec, "pipeline_mean_score": mean_score,
                "pipeline_runs": pipeline_runs, "error": str(exc),
            })
            continue

        # Judge best pipeline vs baseline
        print(f"    [NEW  ] judge spec={spec}")
        try:
            judgement = judge_solutions(
                goal=goal,
                pipeline_artifact=best_run.get("solution_artifact", {}),
                baseline_artifact=baseline_artifact,
                criteria=cfg.judge_criteria,
                model=cfg.judge_model,
                pipeline_text=best_run.get("final_report"),
            )
        except Exception as exc:
            logger.error(
                f"[prompt_specificity] Judge failed for {spec}: {exc}",
                extra={"run_id": "", "node": "prompt_specificity"},
            )
            judgement = {"judgement_error": str(exc)}

        pref = "pipeline" if judgement.get("pipeline_preferred") else "baseline"
        print(f"    spec={spec:<18} pipeline_mean={mean_score:.2f}  "
              f"judge_p={judgement.get('overall_score_pipeline', 0):.1f}  "
              f"judge_b={judgement.get('overall_score_baseline', 0):.1f}  "
              f"preferred={pref}")

        results.append({
            "specificity":         spec,
            "pipeline_mean_score": mean_score,
            "pipeline_runs":       pipeline_runs,
            "baseline_model":      cfg.baseline_model,
            "judgement":           judgement,
        })

    return {
        "baseline_num_runs": cfg.baseline_num_runs,
        "baseline_model":    cfg.baseline_model,
        "judge_model":       cfg.judge_model,
        "judge_criteria":    cfg.judge_criteria,
        "results":           results,
    }


def _run_threshold_sweep_phase(cfg: PromptSpecificityConfig, registry: RunRegistry) -> dict:
    print(f"\n  [Phase 4 — Threshold Sweep] thresholds={cfg.sweep_thresholds}  "
          f"reps={cfg.sweep_num_runs}  max_iters={cfg.sweep_max_iterations}")

    results = []
    for spec in _SPECIFICITIES:
        goal              = cfg.prompts[spec]
        threshold_results = []

        for thr in cfg.sweep_thresholds:
            runs = []
            for rep in range(1, cfg.sweep_num_runs + 1):
                r = _get_or_run_full(
                    registry, spec, thr, cfg.sweep_max_iterations, rep, goal,
                )
                r["rep"] = rep
                runs.append(r)
                tag = "REUSE" if r["status"] == "reused_existing" else "NEW  "
                print(f"    [{tag}] spec={spec:<18} thr={thr:.1f} rep={rep}  "
                      f"score={r.get('final_score', 0):.2f}  "
                      f"iters={r.get('iterations_run', 0)}  "
                      f"stop={r.get('stop_reason','?')}")

            agg   = _agg(runs)
            stops = [r.get("stop_reason", "unknown") for r in runs if "error" not in r]
            threshold_results.append({
                "threshold":    thr,
                "runs":         runs,
                "stop_reasons": stops,
                **agg,
            })

        results.append({"specificity": spec, "threshold_results": threshold_results})

    return {
        "thresholds":     cfg.sweep_thresholds,
        "max_iterations": cfg.sweep_max_iterations,
        "sweep_num_runs": cfg.sweep_num_runs,
        "results":        results,
    }


# ── CSV writers ───────────────────────────────────────────────────────────────

def _write_convergence_csv(phase: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "specificity", "rep", "final_score", "iterations", "stop_reason", "status", "run_id",
        ])
        w.writeheader()
        for spec_r in phase["results"]:
            for r in spec_r["runs"]:
                w.writerow({
                    "specificity": spec_r["specificity"],
                    "rep":         r.get("rep", ""),
                    "final_score": r.get("final_score", 0),
                    "iterations":  r.get("iterations_run", 0),
                    "stop_reason": r.get("stop_reason", ""),
                    "status":      r.get("status", ""),
                    "run_id":      r.get("run_id", ""),
                })


def _write_ablation_csv(phase: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "specificity", "variant", "rep", "final_score", "status", "run_id",
        ])
        w.writeheader()
        for spec_r in phase["results"]:
            for v in spec_r["variants"]:
                for r in v["runs"]:
                    w.writerow({
                        "specificity": spec_r["specificity"],
                        "variant":     v["variant"],
                        "rep":         r.get("rep", ""),
                        "final_score": r.get("final_score", 0),
                        "status":      r.get("status", ""),
                        "run_id":      r.get("run_id", ""),
                    })


def _write_baseline_csv(phase: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "specificity", "pipeline_score", "baseline_score", "delta", "pipeline_preferred",
        ])
        w.writeheader()
        for r in phase["results"]:
            jdg     = r.get("judgement", {})
            p_score = jdg.get("overall_score_pipeline", 0.0)
            b_score = jdg.get("overall_score_baseline", 0.0)
            w.writerow({
                "specificity":        r["specificity"],
                "pipeline_score":     p_score,
                "baseline_score":     b_score,
                "delta":              round(p_score - b_score, 2),
                "pipeline_preferred": jdg.get("pipeline_preferred", False),
            })


def _write_sweep_csv(phase: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "specificity", "threshold", "rep", "final_score",
            "iterations", "stop_reason", "status", "run_id",
        ])
        w.writeheader()
        for spec_r in phase["results"]:
            for thr_r in spec_r["threshold_results"]:
                for r in thr_r["runs"]:
                    w.writerow({
                        "specificity": spec_r["specificity"],
                        "threshold":   thr_r["threshold"],
                        "rep":         r.get("rep", ""),
                        "final_score": r.get("final_score", 0),
                        "iterations":  r.get("iterations_run", 0),
                        "stop_reason": r.get("stop_reason", ""),
                        "status":      r.get("status", ""),
                        "run_id":      r.get("run_id", ""),
                    })


# ── Main entry point ──────────────────────────────────────────────────────────

def run_prompt_specificity_experiment(
    cfg: PromptSpecificityConfig,
    output_dir: Path,
) -> dict:
    """
    Run all 4 prompt-specificity sub-experiments and return the full result dict.

    Directory layout:
      output_dir/prompt_specificity/{cfg.name}/run_registry.json   ← shared cache
      output_dir/prompt_specificity/{cfg.name}/{experiment_id}/    ← per-run results
        results.json, convergence_summary.csv, ablation_summary.csv,
        baseline_summary.csv, sweep_summary.csv
    """
    experiment_id = (
        f"prompt_specificity_{cfg.name.lower().replace(' ', '_')}_"
        f"{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    # Stable parent dir for this experiment name (registry shared across invocations)
    exp_name_dir = output_dir / "prompt_specificity" / cfg.name
    exp_name_dir.mkdir(parents=True, exist_ok=True)

    # Per-invocation results dir
    exp_dir = exp_name_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Registry lives at the stable name-level path
    registry = RunRegistry(exp_name_dir / "run_registry.json")

    # Validate prompt keys
    missing = [k for k in _SPECIFICITIES if k not in cfg.prompts]
    if missing:
        raise ValueError(
            f"PromptSpecificityConfig.prompts is missing keys: {missing}. "
            f"Required: {_SPECIFICITIES}"
        )

    print(f"\n{'='*72}")
    print(f"  PROMPT SPECIFICITY EXPERIMENT: {cfg.name}")
    print(f"  Domain   : {cfg.domain}")
    print(f"  Exp ID   : {experiment_id}")
    print(f"  Registry : {exp_name_dir / 'run_registry.json'}")
    print(f"  Results  : {exp_dir}")
    print(f"{'='*72}")

    start = datetime.now(tz=timezone.utc)

    conv_phase  = _run_convergence_phase(cfg, registry)
    abl_phase   = _run_ablation_phase(cfg, registry)
    base_phase  = _run_baseline_phase(cfg, registry)
    sweep_phase = _run_threshold_sweep_phase(cfg, registry)

    elapsed = (datetime.now(tz=timezone.utc) - start).total_seconds() / 60

    result = {
        "experiment_id":   experiment_id,
        "experiment_type": "prompt_specificity",
        "name":            cfg.name,
        "description":     cfg.description,
        "domain":          cfg.domain,
        "timestamp_utc":   datetime.now(tz=timezone.utc).isoformat(),
        "elapsed_minutes": round(elapsed, 2),
        "prompts":         cfg.prompts,
        "convergence":     conv_phase,
        "ablation":        abl_phase,
        "baseline":        base_phase,
        "threshold_sweep": sweep_phase,
    }

    (exp_dir / "results.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    _write_convergence_csv(conv_phase, exp_dir / "convergence_summary.csv")
    _write_ablation_csv(abl_phase,     exp_dir / "ablation_summary.csv")
    _write_baseline_csv(base_phase,    exp_dir / "baseline_summary.csv")
    _write_sweep_csv(sweep_phase,      exp_dir / "sweep_summary.csv")

    logger.info(
        f"[prompt_specificity] Done. Results saved to {exp_dir}",
        extra={"run_id": "", "node": "prompt_specificity"},
    )
    print(f"\n  Done in {elapsed:.1f} min.")
    print(f"  Registry : {exp_name_dir / 'run_registry.json'}")
    print(f"  Results  : {exp_dir}")
    return result
