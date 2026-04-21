"""
utils/persistence.py — Run artifact persistence.

Saves two types of output:
  1. Per-iteration snapshots  → outputs/{run_id}/iter_{n}.json
  2. Final run record         → outputs/{run_id}/final.json   (machine-readable)
                              → outputs/{run_id}/final_report.md (human-readable)

The final.json includes tracing diagnostics:
  - parse_fallbacks: list of parse-failure events extracted from log_entries
  - weight_fallback_iterations: which iterations triggered the weight fallback
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state import PipelineState


def _run_dir(state: "PipelineState", output_dir: Path) -> Path:
    run_dir = output_dir / state["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_intermediate(state: "PipelineState", output_dir: Path) -> None:
    """
    Write a lightweight snapshot after each iteration.
    Called by graph/runner.py if persistence.save_intermediate == true.
    """
    run_dir = _run_dir(state, output_dir)
    snap = {
        # Identity
        "run_id": state["run_id"],
        "iteration": state["iteration"],
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        # Score
        "evaluation_score": state["evaluation_score"],
        # Agent diagnostics
        "agent_confidences": {
            "driver": (state["driver_feedback"] or {}).get("confidence"),
            "policy": (state["policy_feedback"] or {}).get("confidence"),
            "software": (state["software_feedback"] or {}).get("confidence"),
        },
        "used_weight_fallback": (state["aggregated_feedback"] or {}).get("used_weight_fallback"),
        "unresolved_conflicts": (state["aggregated_feedback"] or {}).get("unresolved_conflicts", []),
        "resolved_conflicts": (state["aggregated_feedback"] or {}).get("resolved_conflicts", []),
        # Solution snapshot
        "solution_summary": (state["solution_artifact"] or {}).get("design_summary"),
        "stop_reason": state.get("stop_reason", ""),
    }
    path = run_dir / f"iter_{state['iteration']}.json"
    path.write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")


def save_final_outputs(state: "PipelineState", output_dir: Path) -> tuple[Path, Path]:
    """
    Write the complete final.json and final_report.md after the pipeline stops.
    """
    run_dir = _run_dir(state, output_dir)

    # ── final_report.md ──────────────────────────────────────────────────────
    report_path = run_dir / "final_report.md"
    report_path.write_text(state.get("final_report") or "", encoding="utf-8")

    # ── final.json ───────────────────────────────────────────────────────────
    log_entries = state.get("log_entries", [])
    metrics_history = state.get("metrics_history", [])

    final_json = {
        # Identity
        "run_id": state["run_id"],
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "user_goal": state["user_goal"],
        # Outcome
        "final_score": state["evaluation_score"],
        "stop_reason": state["stop_reason"],
        "iterations_run": state["iteration"],
        # Result artefacts
        "solution_artifact": state["solution_artifact"],
        "aggregated_feedback": state["aggregated_feedback"],
        "evaluation_details": state["evaluation_details"],
        # Full trace
        "metrics_history": metrics_history,
        "log_entries": log_entries,
        # Diagnostics — extracted from the trace for quick inspection
        "parse_fallbacks": [
            e for e in log_entries if e.get("event") == "parse_fallback"
        ],
        "weight_fallback_iterations": [
            m["iteration"]
            for m in metrics_history
            if m.get("used_weight_fallback")
        ],
    }
    final_path = run_dir / "final.json"
    final_path.write_text(json.dumps(final_json, indent=2, default=str), encoding="utf-8")

    return report_path, final_path
