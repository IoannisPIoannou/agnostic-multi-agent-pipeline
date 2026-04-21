"""
experiments/runners/runtime.py — Experiment 4: runtime and cost benchmarking.

Measures per-run:
  - Total wall time and per-iteration wall time (boundary-based from log timestamps)
  - API call counts broken down by component (Gemini vs Ollama), including retries
  - Ollama fallback frequency
  - Estimated cost per run based on configurable token + pricing assumptions

Wall-time scope note:
  total_wall_s is end-to-end runtime including persistence I/O (save_intermediate
  and save_final_outputs file writes inside run_pipeline). It represents the full
  wall time experienced by the caller, not pure model compute time. Disable
  save_intermediate in config to reduce I/O overhead in the measurement.

Usage:
    from experiments.runners.runtime import run_runtime_experiment
    from experiments.configs.schema import RuntimeConfig
    cfg = RuntimeConfig(name="test", goal="...")
    result = run_runtime_experiment(cfg, Path("experiments/results"))
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from experiments.configs.schema import RuntimeConfig
from graph.runner import run_pipeline
from state import initial_state
from utils.config_loader import get_config
from utils.logging_config import get_logger

logger = get_logger(__name__)

_FALLBACK_EVENT = "fallback_activated"
_RETRY_EVENTS   = {"parse_error", "connection_error"}


def _count_api_calls(log_entries: list[dict]) -> dict:
    """
    Tally API calls by component from the structured log.

    Completion events (orchestrator_complete etc.) count successful calls.
    parse_error and connection_error events each represent one real LLM call
    that failed before producing a usable result — they are counted separately
    as retries so the total reflects actual calls made, not just successes.
    """
    counts: dict[str, int] = {
        "orchestrator":        0,
        "coding":              0,
        "synthesis":           0,
        "end_user":            0,
        "policy":              0,
        "software":            0,
        "fallbacks":           0,
        "retry_attempts":      0,
        "total_gemini":        0,
        "total_ollama":        0,
    }
    _gemini_agents = {"orchestrator", "coding", "synthesis"}
    _ollama_agents  = {"end_user", "policy", "software"}
    event_map = {
        "orchestrator_complete": "orchestrator",
        "coding_complete":       "coding",
        "synthesis_complete":    "synthesis",
        "end_user_complete":     "end_user",
        "policy_complete":       "policy",
        "software_complete":     "software",
    }
    for entry in log_entries:
        event = entry.get("event", "")
        key   = event_map.get(event)
        if key:
            counts[key] += 1
        elif event == _FALLBACK_EVENT:
            counts["fallbacks"] += 1
        elif event in _RETRY_EVENTS:
            # Each retry event is one real LLM invocation that did not produce
            # a valid structured output. Route to the correct per-agent counter
            # so totals include both successful and failed calls.
            node = entry.get("node", "")
            if node in counts:
                counts[node] += 1
            counts["retry_attempts"] += 1

    counts["total_gemini"] = sum(counts[a] for a in _gemini_agents)
    counts["total_ollama"] = sum(counts[a] for a in _ollama_agents)
    return counts


def _estimate_cost(api_counts: dict, cfg: RuntimeConfig) -> dict:
    """
    Estimate Gemini API cost from call counts and average token assumptions.
    Ollama is local (zero marginal cost).

    Cost caveats:
      - Token counts use flat per-call averages from config. Orchestrator and
        coding inputs grow with iteration count (accumulated context); synthesis
        input includes the full solution history. Multi-iteration runs will
        exceed these estimates — adjust avg_*_tokens after profiling.
      - api_counts includes retry calls, so cost reflects actual calls made,
        not just successful completions.
      - Enable Gemini token usage logging for exact per-call token figures.
    """
    input_tok = (
        api_counts["orchestrator"] * cfg.avg_orchestrator_input_tokens
        + api_counts["coding"]       * cfg.avg_coding_input_tokens
        + api_counts["synthesis"]    * cfg.avg_synthesis_input_tokens
    )
    output_tok = (
        api_counts["orchestrator"] * cfg.avg_orchestrator_output_tokens
        + api_counts["coding"]       * cfg.avg_coding_output_tokens
        + api_counts["synthesis"]    * cfg.avg_synthesis_output_tokens
    )
    input_cost  = (input_tok  / 1_000_000) * cfg.gemini_input_cost_per_mtok
    output_cost = (output_tok / 1_000_000) * cfg.gemini_output_cost_per_mtok
    total_cost  = input_cost + output_cost

    return {
        "estimated_input_tokens":    input_tok,
        "estimated_output_tokens":   output_tok,
        "estimated_input_cost_usd":  round(input_cost,  6),
        "estimated_output_cost_usd": round(output_cost, 6),
        "estimated_total_cost_usd":  round(total_cost,  6),
        "cost_note": (
            "Estimates use flat avg_*_tokens config values. "
            "Synthesis and orchestrator inputs grow with iteration count — "
            "actual cost on multi-iteration runs will exceed this estimate. "
            "Retry calls are included in api_counts but use the same per-call "
            "token average. Enable Gemini usage logging for exact figures."
        ),
    }


def _estimate_per_iter_timing(
    log_entries: list[dict], total_wall_s: float, iterations: int
) -> dict:
    """
    Estimate per-iteration timing from log timestamps.

    Iteration boundaries are defined as start-of-iter-N to start-of-iter-(N+1),
    so the inter-iteration gap (orchestrator startup, state transitions between
    iterations) is included in the preceding iteration rather than silently
    dropped. The previous approach (max_ts - min_ts within each iteration) lost
    all inter-iteration time, causing sum(per_iter) < total_wall_s.

    The last iteration has no following iteration to anchor against, so its end
    is approximated as first_iter_start + total_wall_s. This is slightly
    generous (total_wall_s starts before the first log entry) but far more
    accurate than the intra-iteration span.

    Falls back to equal division when fewer than 2 iterations have timestamps.
    """
    timestamps_by_iter: dict[int, list[float]] = {}
    for entry in log_entries:
        ts_str = entry.get("timestamp")
        itr    = entry.get("iteration")
        if ts_str and itr is not None:
            try:
                ts = datetime.fromisoformat(ts_str).timestamp()
                timestamps_by_iter.setdefault(itr, []).append(ts)
            except ValueError:
                pass

    if len(timestamps_by_iter) >= 2:
        iter_nums  = sorted(timestamps_by_iter)
        iter_first = {i: min(timestamps_by_iter[i]) for i in iter_nums}

        per_iter: list[dict] = []
        for j, i in enumerate(iter_nums[:-1]):
            next_i   = iter_nums[j + 1]
            duration = round(iter_first[next_i] - iter_first[i], 2)
            per_iter.append({"iteration": i, "duration_s": duration})

        # Last iteration: no following start to anchor against — approximate
        # end as first-iteration start + total wall time.
        run_end_approx = iter_first[iter_nums[0]] + total_wall_s
        last_duration  = round(run_end_approx - iter_first[iter_nums[-1]], 2)
        per_iter.append({
            "iteration":  iter_nums[-1],
            "duration_s": max(last_duration, 0.0),
            "note":       "approximate — derived from total wall time",
        })
    else:
        avg = round(total_wall_s / max(iterations, 1), 2)
        per_iter = [{"iteration": i + 1, "duration_s": avg} for i in range(iterations)]

    return {
        "per_iteration": per_iter,
        "avg_iter_s": round(
            sum(x["duration_s"] for x in per_iter) / max(len(per_iter), 1), 2
        ),
    }


def run_runtime_experiment(cfg: RuntimeConfig, output_dir: Path) -> dict:
    """
    Run the pipeline cfg.num_runs times with timing and cost instrumentation.
    Returns the full result dict and saves it to output_dir.
    """
    pipeline_cfg = get_config()
    experiment_id = (
        f"runtime_{cfg.name.lower().replace(' ', '_')}_"
        f"{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    exp_dir = output_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for i in range(cfg.num_runs):
        run_id = str(uuid.uuid4())[:8]
        logger.info(
            f"[runtime] Run {i + 1}/{cfg.num_runs} | run_id={run_id}",
            extra={"run_id": run_id, "node": "experiment"},
        )

        state = initial_state(user_goal=cfg.goal, run_id=run_id, cfg=pipeline_cfg)

        # total_wall_s is end-to-end including persistence I/O inside run_pipeline.
        t_start = time.perf_counter()
        try:
            final_state = run_pipeline(state)
        except Exception as exc:
            t_end = time.perf_counter()
            logger.error(f"[runtime] Run {i + 1} failed: {exc}")
            runs.append({
                "run_index":    i + 1,
                "run_id":       run_id,
                "total_wall_s": round(t_end - t_start, 2),
                "error":        str(exc),
            })
            continue
        t_end = time.perf_counter()

        total_wall_s = round(t_end - t_start, 2)
        iterations   = final_state.get("iteration", 0)
        log_entries  = final_state.get("log_entries", [])

        api_counts = _count_api_calls(log_entries)
        cost       = _estimate_cost(api_counts, cfg)
        timing     = _estimate_per_iter_timing(log_entries, total_wall_s, iterations)

        run_result = {
            "run_index":      i + 1,
            "run_id":         run_id,
            "iterations_run": iterations,
            "final_score":    final_state.get("evaluation_score", 0.0),
            "stop_reason":    final_state.get("stop_reason", "unknown"),
            "total_wall_s":   total_wall_s,
            "timing":         timing,
            "api_calls":      api_counts,
            "cost_estimate":  cost,
        }
        runs.append(run_result)
        logger.info(
            f"[runtime] Run {i + 1} done | "
            f"wall={total_wall_s:.1f}s | "
            f"gemini_calls={api_counts['total_gemini']} | "
            f"ollama_calls={api_counts['total_ollama']} | "
            f"retries={api_counts['retry_attempts']} | "
            f"est_cost=${cost['estimated_total_cost_usd']:.4f}",
            extra={"run_id": run_id, "node": "experiment"},
        )

    result = {
        "experiment_id":   experiment_id,
        "experiment_type": "runtime",
        "name":            cfg.name,
        "description":     cfg.description,
        "goal":            cfg.goal,
        "num_runs":        cfg.num_runs,
        "timestamp_utc":   datetime.now(tz=timezone.utc).isoformat(),
        "pricing_config": {
            "gemini_input_cost_per_mtok":  cfg.gemini_input_cost_per_mtok,
            "gemini_output_cost_per_mtok": cfg.gemini_output_cost_per_mtok,
        },
        "runs": runs,
    }

    out_path = exp_dir / "results.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(f"[runtime] Results saved to {out_path}")
    return result
