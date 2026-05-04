"""
experiments/threshold_sweep.py — Threshold sweep experiment.

Tests 5 domain-diverse prompts at 3 score thresholds (7.0, 8.0, 9.0).
Produces 15 pipeline runs total (1 run per prompt × threshold combination).

Results are saved to:
  experiments/results/threshold_sweep_<timestamp>/results.json
  experiments/results/threshold_sweep_<timestamp>/summary.csv

Metrics captured per run:
  domain, prompt, threshold, final_score, stop_reason, iterations_used,
  runtime_s, unresolved_conflicts_final, converged, gemini_calls, ollama_calls

Usage:
    python experiments/threshold_sweep.py
    python experiments/threshold_sweep.py --output-dir experiments/results
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv()

from graph.builder import build_graph, reset_graph
from state import initial_state
from utils.config_loader import get_config
from utils.logging_config import get_logger

logger = get_logger(__name__)

# ── Experiment definition ─────────────────────────────────────────────────────

PROMPTS = [
    {
        "domain": "logistics",
        "label": "Truck Parking",
        "goal": (
            "Design a decision-support system that recommends truck parking "
            "locations based on distance, availability, and driver preferences."
        ),
    },
    {
        "domain": "healthcare",
        "label": "Patient Scheduling",
        "goal": (
            "Design an AI-powered patient scheduling system for a multi-specialty "
            "outpatient clinic that minimizes wait times and maximizes doctor utilization."
        ),
    },
    {
        "domain": "smart_city",
        "label": "Traffic Management",
        "goal": (
            "Design a smart city traffic management system that dynamically adjusts "
            "signal timings based on real-time congestion data and incident reports."
        ),
    },
    {
        "domain": "e-commerce",
        "label": "Product Recommendations",
        "goal": (
            "Design a personalized product recommendation engine for an e-commerce "
            "platform that balances user preferences, inventory levels, and profit margins."
        ),
    },
    {
        "domain": "education",
        "label": "Course Planning",
        "goal": (
            "Design an AI-assisted university course planning system that helps students "
            "select courses based on prerequisites, career goals, and schedule constraints."
        ),
    },
]

THRESHOLDS = [7.0, 8.0, 9.0]
MAX_ITERATIONS = 5


# ── API call counting ─────────────────────────────────────────────────────────

_GEMINI_COMPLETIONS = {"orchestrator_complete", "independent_evaluator_complete", "coding_complete", "synthesis_complete"}
_OLLAMA_COMPLETIONS = {"end_user_complete", "policy_complete", "software_complete"}


def _count_api_calls(log_entries: list[dict]) -> dict[str, int]:
    gemini = sum(1 for e in log_entries if e.get("event") in _GEMINI_COMPLETIONS)
    ollama = sum(1 for e in log_entries if e.get("event") in _OLLAMA_COMPLETIONS)
    return {"gemini_calls": gemini, "ollama_calls": ollama}


# ── Single run ────────────────────────────────────────────────────────────────

def _run_one(prompt: dict, threshold: float, base_cfg: dict) -> dict:
    """Run the pipeline for one prompt/threshold combination."""
    run_id = str(uuid.uuid4())[:8]
    label = prompt["label"]

    logger.info(
        f"[sweep] domain={prompt['domain']} threshold={threshold} run_id={run_id}",
        extra={"run_id": run_id, "node": "threshold_sweep"},
    )

    cfg = copy.deepcopy(base_cfg)
    cfg["pipeline"]["score_threshold"] = threshold
    cfg["pipeline"]["max_iterations"] = MAX_ITERATIONS
    cfg["persistence"]["save_intermediate"] = False

    state = initial_state(user_goal=prompt["goal"], run_id=run_id, cfg=cfg)

    reset_graph()
    graph = build_graph()

    t_start = time.perf_counter()
    try:
        final_state = None
        for chunk in graph.stream(state, stream_mode="values"):
            final_state = chunk
    except Exception as exc:
        t_end = time.perf_counter()
        logger.error(f"[sweep] Run failed: {exc}", extra={"run_id": run_id, "node": "threshold_sweep"})
        return {
            "domain": prompt["domain"],
            "label": label,
            "prompt": prompt["goal"],
            "threshold": threshold,
            "run_id": run_id,
            "error": str(exc),
            "runtime_s": round(time.perf_counter() - t_start, 2),
        }
    t_end = time.perf_counter()

    if final_state is None:
        return {
            "domain": prompt["domain"],
            "label": label,
            "prompt": prompt["goal"],
            "threshold": threshold,
            "run_id": run_id,
            "error": "no output state",
            "runtime_s": round(t_end - t_start, 2),
        }

    metrics = final_state.get("metrics_history", [])
    last_metrics = metrics[-1] if metrics else {}
    api = _count_api_calls(final_state.get("log_entries", []))
    stop_reason = final_state.get("stop_reason", "unknown")

    result = {
        "domain": prompt["domain"],
        "label": label,
        "prompt": prompt["goal"],
        "threshold": threshold,
        "run_id": run_id,
        "final_score": round(final_state.get("evaluation_score", 0.0), 4),
        "stop_reason": stop_reason,
        "iterations_used": final_state.get("iteration", 0),
        "runtime_s": round(t_end - t_start, 2),
        "unresolved_conflicts_final": last_metrics.get("unresolved_conflicts", 0),
        "converged": stop_reason == "converged",
        "gemini_calls": api["gemini_calls"],
        "ollama_calls": api["ollama_calls"],
    }

    logger.info(
        f"[sweep] done | domain={prompt['domain']} threshold={threshold} "
        f"score={result['final_score']:.2f} stop={stop_reason} "
        f"iters={result['iterations_used']} wall={result['runtime_s']:.1f}s",
        extra={"run_id": run_id, "node": "threshold_sweep"},
    )
    return result


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(output_dir: Path) -> None:
    base_cfg = get_config()
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    sweep_dir = output_dir / f"threshold_sweep_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    total = len(PROMPTS) * len(THRESHOLDS)
    runs: list[dict] = []
    run_num = 0

    for threshold in THRESHOLDS:
        for prompt in PROMPTS:
            run_num += 1
            print(
                f"\n[{run_num}/{total}] domain={prompt['domain']}  "
                f"threshold={threshold}  label={prompt['label']}"
            )
            result = _run_one(prompt, threshold, base_cfg)
            runs.append(result)

    # ── JSON ──────────────────────────────────────────────────────────────────
    full_result = {
        "experiment": "threshold_sweep",
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "thresholds": THRESHOLDS,
        "prompts": [{"domain": p["domain"], "label": p["label"], "goal": p["goal"]} for p in PROMPTS],
        "max_iterations": MAX_ITERATIONS,
        "total_runs": total,
        "runs": runs,
    }
    json_path = sweep_dir / "results.json"
    json_path.write_text(json.dumps(full_result, indent=2), encoding="utf-8")
    print(f"\nJSON saved -> {json_path}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_cols = [
        "domain", "label", "threshold", "final_score", "stop_reason",
        "iterations_used", "runtime_s", "unresolved_conflicts_final",
        "converged", "gemini_calls", "ollama_calls", "run_id",
    ]
    csv_path = sweep_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        writer.writeheader()
        for r in runs:
            if "error" not in r:
                writer.writerow(r)
            else:
                writer.writerow({
                    "domain": r.get("domain", ""),
                    "label": r.get("label", ""),
                    "threshold": r.get("threshold", ""),
                    "final_score": "ERROR",
                    "stop_reason": r.get("error", ""),
                    "iterations_used": "",
                    "runtime_s": r.get("runtime_s", ""),
                    "unresolved_conflicts_final": "",
                    "converged": "",
                    "gemini_calls": "",
                    "ollama_calls": "",
                    "run_id": r.get("run_id", ""),
                })
    print(f"CSV saved  -> {csv_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  THRESHOLD SWEEP SUMMARY")
    print("=" * 72)
    print(f"  {'Domain':<22}  {'Thr':>5}  {'Score':>6}  {'Stop Reason':<18}  {'Iters':>5}  {'Wall(s)':>7}")
    print("-" * 72)
    for r in runs:
        if "error" in r:
            print(f"  {r.get('domain',''):<22}  {r.get('threshold',''):>5}  {'ERROR':>6}  {r.get('error','')[:18]:<18}")
        else:
            print(
                f"  {r['domain']:<22}  {r['threshold']:>5.1f}  {r['final_score']:>6.2f}  "
                f"{r['stop_reason']:<18}  {r['iterations_used']:>5}  {r['runtime_s']:>7.1f}"
            )
    print("=" * 72)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threshold sweep experiment")
    parser.add_argument(
        "--output-dir",
        default="experiments/results",
        help="Directory to write results (default: experiments/results)",
    )
    args = parser.parse_args()
    run_sweep(Path(args.output_dir))
