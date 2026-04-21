"""
experiments/analysis/summarize.py — Post-experiment analysis utilities.

Each summarize_* function accepts a result dict (as returned by the runners
or loaded from results.json) and returns a formatted string table.

Usage:
    from experiments.analysis.summarize import load_result, summarize_convergence
    result = load_result("experiments/results/convergence_.../results.json")
    print(summarize_convergence(result))
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_result(path: str | Path) -> dict:
    """Load a JSON result file produced by any experiment runner."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_latest_result(results_dir: str | Path, experiment_type: str) -> Path:
    """Return the path to the most recently created result for an experiment type."""
    results_dir = Path(results_dir)
    matches = sorted(
        results_dir.glob(f"{experiment_type}_*/results.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"No {experiment_type} results found in {results_dir}"
        )
    return matches[0]


# ── Convergence ───────────────────────────────────────────────────────────────

def summarize_convergence(result: dict) -> str:
    all_runs = result.get("runs", [])
    runs = [r for r in all_runs if "error" not in r]
    failed = len(all_runs) - len(runs)
    if not runs:
        return "No successful runs to summarize."

    runs_label = f"{len(runs)} successful / {len(all_runs)} total"
    if failed:
        runs_label += f"  ({failed} failed)"

    lines = [
        f"Experiment : {result.get('name')}",
        f"Goal       : {result.get('goal', '')[:80]}...",
        f"Runs       : {runs_label}",
        "",
        f"{'Run':>3}  {'Score':>7}  {'Iters':>5}  {'Stop Reason':<20}  {'Fallbacks':>9}",
        "-" * 58,
    ]
    if len(runs) < 2:
        lines.append(
            "  WARNING: fewer than 2 successful runs — "
            "std dev and consistency metrics are unreliable."
        )
    scores = []
    iters  = []
    for r in runs:
        score = r.get("final_score", 0.0)
        it    = r.get("iterations_run", 0)
        scores.append(score)
        iters.append(it)
        lines.append(
            f"{r['run_index']:>3}  {score:>7.2f}  {it:>5}  "
            f"{r.get('stop_reason', '?'):<20}  "
            f"{r.get('fallback_events', 0):>9}"
        )

    lines += [
        "-" * 58,
        f"  Avg score : {statistics.mean(scores):.2f}",
        f"  Std score : {statistics.stdev(scores):.2f}" if len(scores) > 1 else "  Std score : N/A",
        f"  Avg iters : {statistics.mean(iters):.1f}",
        f"  Min/Max   : {min(iters)} / {max(iters)} iterations",
    ]

    # Score progression table if all runs have the same iteration count
    first_prog = runs[0].get("score_progression", [])
    if first_prog and all(len(r.get("score_progression", [])) == len(first_prog) for r in runs):
        lines += ["", "Score progression per iteration:", f"{'Iter':>4}", ""]
        header = "Iter  " + "  ".join(f"Run{r['run_index']:>1}" for r in runs)
        lines.append(header)
        for i, _ in enumerate(first_prog):
            row = f"{i + 1:>4}  " + "  ".join(
                f"{r['score_progression'][i]:>4.2f}" for r in runs
            )
            lines.append(row)

    return "\n".join(lines)


# ── Ablation ──────────────────────────────────────────────────────────────────

def summarize_ablation(result: dict) -> str:
    variants = [v for v in result.get("variants", []) if "error" not in v]
    if not variants:
        return "No successful variants to summarize."

    full    = next((v for v in variants if not v.get("disabled_agents")), None)
    ablated = [v for v in variants if v.get("disabled_agents")]

    lines = [
        f"Experiment : {result.get('name')}",
        f"Goal       : {result.get('goal', '')[:80]}...",
        "",
        f"{'Variant':<18}  {'Score':>7}  {'Iters':>5}  {'Conflicts':>9}  {'Stop Reason':<20}",
        "-" * 70,
    ]

    def _row(v: dict) -> str:
        label     = v.get("label", "?")
        score     = v.get("final_score", 0.0)
        iters     = v.get("iterations_run", 0)
        conflicts = v.get("unresolved_conflicts_final", 0)
        stop      = v.get("stop_reason", "?")
        return f"{label:<18}  {score:>7.2f}  {iters:>5}  {conflicts:>9}  {stop:<20}"

    if full:
        lines.append(_row(full))
    for v in ablated:
        lines.append(_row(v))

    lines.append("-" * 70)

    if full:
        full_score = full.get("final_score", 0.0)
        lines.append("")
        lines.append("Score delta vs full pipeline:")
        for v in ablated:
            delta = v.get("final_score", 0.0) - full_score
            lines.append(
                f"  {v.get('label', '?'):<18}  {delta:+.2f}  "
                f"({'higher' if delta > 0 else 'lower'})"
            )

    return "\n".join(lines)


# ── Baseline comparison ───────────────────────────────────────────────────────

def summarize_baseline(result: dict) -> str:
    pipeline  = result.get("pipeline", {})
    baseline  = result.get("baseline", {})
    judgement = result.get("judgement", {})

    lines = [
        f"Experiment        : {result.get('name')}",
        f"Goal              : {result.get('goal', '')[:80]}...",
        "",
        f"Pipeline score    : {pipeline.get('final_score', 0):.2f} "
        f"(internal | {pipeline.get('iterations', 0)} iters)",
        f"Pipeline components: {pipeline.get('num_components', 0)}",
        "",
        f"Baseline model    : {baseline.get('model', '?')}",
        f"Baseline components: {baseline.get('num_components', 0)}",
        "",
        f"Judge model       : {judgement.get('judge_model', '?')}",
        f"Judge score — pipeline : {judgement.get('overall_score_pipeline', 0):.1f} / 10",
        f"Judge score — baseline : {judgement.get('overall_score_baseline', 0):.1f} / 10",
        f"Preferred         : {'pipeline' if judgement.get('pipeline_preferred') else 'baseline'}",
        f"Reason            : {judgement.get('preference_reason', '')}",
        "",
    ]

    scores = judgement.get("criterion_scores", [])
    if scores:
        lines += [
            f"{'Criterion':<25}  {'Pipeline':>8}  {'Baseline':>8}",
            "-" * 48,
        ]
        for s in scores:
            lines.append(
                f"{s.get('criterion', '?'):<25}  "
                f"{s.get('score_pipeline', 0):>8.1f}  "
                f"{s.get('score_baseline', 0):>8.1f}"
            )
        lines.append("-" * 48)

    return "\n".join(lines)


# ── Runtime / cost ────────────────────────────────────────────────────────────

def summarize_runtime(result: dict) -> str:
    runs = [r for r in result.get("runs", []) if "error" not in r]
    if not runs:
        return "No successful runs to summarize."

    lines = [
        f"Experiment : {result.get('name')}",
        f"Goal       : {result.get('goal', '')[:80]}...",
        f"Runs       : {len(runs)}",
        "",
        f"{'Run':>3}  {'Wall(s)':>7}  {'Iters':>5}  {'Score':>6}  "
        f"{'Gemini':>6}  {'Ollama':>6}  {'Retry':>5}  {'Fallb':>5}  {'Cost($)':>8}",
        "-" * 72,
    ]

    total_walls   = []
    total_costs   = []
    total_gemini  = []
    total_ollama  = []
    total_retries = []

    for r in runs:
        wall    = r.get("total_wall_s", 0.0)
        iters   = r.get("iterations_run", 0)
        score   = r.get("final_score", 0.0)
        api     = r.get("api_calls", {})
        cost    = r.get("cost_estimate", {})
        gem     = api.get("total_gemini", 0)
        oll     = api.get("total_ollama", 0)
        retry   = api.get("retry_attempts", 0)
        fallb   = api.get("fallbacks", 0)
        usd     = cost.get("estimated_total_cost_usd", 0.0)

        total_walls.append(wall)
        total_costs.append(usd)
        total_gemini.append(gem)
        total_ollama.append(oll)
        total_retries.append(retry)

        lines.append(
            f"{r['run_index']:>3}  {wall:>7.1f}  {iters:>5}  {score:>6.2f}  "
            f"{gem:>6}  {oll:>6}  {retry:>5}  {fallb:>5}  {usd:>8.4f}"
        )

    lines += [
        "-" * 72,
        f"  Avg wall time  : {statistics.mean(total_walls):.1f}s  (includes persistence I/O)",
        f"  Avg Gemini calls: {statistics.mean(total_gemini):.1f}",
        f"  Avg Ollama calls: {statistics.mean(total_ollama):.1f}",
        f"  Avg retries    : {statistics.mean(total_retries):.1f}  (failed LLM attempts)",
        f"  Avg est. cost  : ${statistics.mean(total_costs):.4f} per run",
        "",
        "  Note: cost uses flat avg_*_tokens estimates — multi-iteration runs",
        "        will exceed estimates as context grows. Retry calls included.",
        "        Ollama runs locally (no API cost).",
    ]

    # Per-iteration timing from first run if available
    first_timing = runs[0].get("timing", {}).get("per_iteration", [])
    if first_timing:
        lines += ["", "Per-iteration timing (run 1):", f"  {'Iter':>4}  {'Duration(s)':>11}"]
        for it in first_timing:
            approx = "  *" if it.get("note") else ""
            lines.append(f"  {it['iteration']:>4}  {it['duration_s']:>11.2f}{approx}")
        if any(it.get("note") for it in first_timing):
            lines.append("  * last iteration duration is approximate")

    return "\n".join(lines)


# ── Cross-experiment dispatcher ───────────────────────────────────────────────

def summarize(result: dict) -> str:
    """Dispatch to the correct summarizer based on experiment_type."""
    dispatch = {
        "convergence": summarize_convergence,
        "ablation":    summarize_ablation,
        "baseline":    summarize_baseline,
        "runtime":     summarize_runtime,
    }
    fn = dispatch.get(result.get("experiment_type"))
    if fn is None:
        return f"Unknown experiment type: {result.get('experiment_type')}"
    return fn(result)
