"""
experiments/run_audit_on_experiments.py -- Audit ON experimental batch.

Runs all four experiment types with audit_layer.enabled=true and saves
results to experiments/results/audit_on/. Generates comparison reports
against existing Audit OFF results.

Run matrix:
  Phase 1 -- Convergence     : 5 domains x 3 runs          = 15 pipeline runs
  Phase 2 -- Ablation        : 5 domains x 4 variants x 3  = 60 pipeline runs
  Phase 3 -- Threshold sweep : 5 domains x 3 thresholds    = 15 pipeline runs
  Phase 4 -- Prompt spec     : 54 new + 27 reused          = 54 new runs
  TOTAL NEW: 144 pipeline runs

Audit OFF baseline files used for comparison are hardcoded below.
Do NOT rerun those experiments -- use them as-is.

Usage:
    python experiments/run_audit_on_experiments.py
"""

from __future__ import annotations

import copy
import csv
import json
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AUDIT_ON_DIR = Path("experiments/results/audit_on")
RESULTS_DIR  = Path("experiments/results")

DOMAINS = [
    {
        "domain":   "logistics",
        "label":    "Truck Parking",
        "name_key": "truck_parking",
        "goal": (
            "Design a decision-support system that recommends truck parking "
            "locations based on distance, availability, and driver preferences."
        ),
    },
    {
        "domain":   "healthcare",
        "label":    "Patient Scheduling",
        "name_key": "patient_scheduling",
        "goal": (
            "Design an AI-powered patient scheduling system for a multi-specialty "
            "outpatient clinic that minimizes wait times and maximizes doctor utilization."
        ),
    },
    {
        "domain":   "smart_city",
        "label":    "Traffic Management",
        "name_key": "traffic_management",
        "goal": (
            "Design a smart city traffic management system that dynamically adjusts "
            "signal timings based on real-time congestion data and incident reports."
        ),
    },
    {
        "domain":   "e-commerce",
        "label":    "Product Recommendations",
        "name_key": "product_recommendations",
        "goal": (
            "Design a personalized product recommendation engine for an e-commerce "
            "platform that balances user preferences, inventory levels, and profit margins."
        ),
    },
    {
        "domain":   "education",
        "label":    "Course Planning",
        "name_key": "course_planning",
        "goal": (
            "Design an AI-assisted university course planning system that helps students "
            "select courses based on prerequisites, career goals, and schedule constraints."
        ),
    },
]

# Prompt specificity prompts -- same as existing Audit OFF run
PS_PROMPTS = {
    "vague": "Design a truck parking system.",
    "moderate": (
        "Design a decision-support system that recommends truck parking locations "
        "based on distance, availability, and driver preferences."
    ),
    "highly_specific": (
        "Design a decision-support system for long-haul truck drivers that recommends "
        "parking locations along their route. The system must: (1) ingest real-time "
        "occupancy feeds from IoT-equipped truck stops and public rest areas, "
        "(2) rank locations using a weighted score of distance-to-destination, "
        "estimated occupancy at ETA, amenity ratings, driver preference history, "
        "and Hours-of-Service (HOS) compliance windows, (3) support convoy bookings "
        "for 2-5 trucks, (4) send push notifications 30 minutes before HOS limits, "
        "and (5) fall back to offline cached rankings when connectivity is lost. "
        "Design for 10,000 concurrent drivers with sub-2-second response time."
    ),
}

# Hardcoded Audit OFF result paths (most recent valid set per experiment type)
AUDIT_OFF_FILES = {
    "convergence": {
        "truck_parking":          RESULTS_DIR / "convergence_truck_parking_convergence_20260501_162014/results.json",
        "patient_scheduling":     RESULTS_DIR / "convergence_patient_scheduling_convergence_20260501_162422/results.json",
        "traffic_management":     RESULTS_DIR / "convergence_traffic_management_convergence_20260501_162825/results.json",
        "product_recommendations":RESULTS_DIR / "convergence_product_recommendations_convergence_20260501_163238/results.json",
        "course_planning":        RESULTS_DIR / "convergence_course_planning_convergence_20260501_163649/results.json",
    },
    "ablation": {
        "truck_parking":          RESULTS_DIR / "ablation_truck_parking_ablation_20260502_033200/results.json",
        "patient_scheduling":     RESULTS_DIR / "ablation_patient_scheduling_ablation_20260502_034709/results.json",
        "traffic_management":     RESULTS_DIR / "ablation_traffic_management_ablation_20260502_040205/results.json",
        "product_recommendations":RESULTS_DIR / "ablation_product_recommendations_ablation_20260502_041754/results.json",
        "course_planning":        RESULTS_DIR / "ablation_course_planning_ablation_20260502_043305/results.json",
    },
    "threshold_sweep":     RESULTS_DIR / "threshold_sweep_20260502_023429/results.json",
    "prompt_specificity":  RESULTS_DIR / "prompt_specificity/logistics_truck_parking/prompt_specificity_logistics_truck_parking_20260502_155441/results.json",
}

# ---------------------------------------------------------------------------
# Audit diagnostics extraction
# ---------------------------------------------------------------------------

def _audit_diag_from_file(run_id: str) -> dict:
    """Read outputs/{run_id}/final.json and return flat audit diagnostic fields."""
    path = Path("outputs") / run_id / "final.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return _flatten_audit_diag(
            (data.get("aggregated_feedback") or {}).get("audit_diagnostics", {})
        )
    except Exception:
        return {}


def _audit_diag_from_state(state: dict) -> dict:
    """Extract audit diagnostics directly from a final pipeline state dict."""
    return _flatten_audit_diag(
        (state.get("aggregated_feedback") or {}).get("audit_diagnostics", {})
    )


def _flatten_audit_diag(diag: dict) -> dict:
    if not diag:
        return {}
    result: dict = {}
    for branch in ("end_user", "policy", "software"):
        bd = diag.get(branch, {})
        result[f"{branch}_audit_status"]   = bd.get("status", "")
        result[f"{branch}_audit_attempts"] = bd.get("attempts", 0)
        result[f"{branch}_audit_score"]    = bd.get("overall_audit_score")
    statuses = [diag.get(b, {}).get("status", "") for b in ("end_user", "policy", "software")]
    attempts = [diag.get(b, {}).get("attempts", 0) for b in ("end_user", "policy", "software")]
    result["total_audit_attempts"]       = sum(attempts)
    result["approved_count"]             = statuses.count("approved")
    result["failed_max_revisions_count"] = statuses.count("failed_after_max_revisions")
    result["disabled_count"]             = statuses.count("disabled")
    return result

# ---------------------------------------------------------------------------
# Phase 1: Convergence
# ---------------------------------------------------------------------------

def run_convergence_phase() -> list[dict]:
    from experiments.configs.schema import ConvergenceConfig
    from experiments.runners.convergence import run_convergence_experiment
    from graph.builder import reset_graph

    print("\n" + "=" * 72)
    print("  PHASE 1: CONVERGENCE AUDIT ON (5 domains x 3 runs = 15 runs)")
    print("=" * 72)

    all_results: list[dict] = []
    for i, p in enumerate(DOMAINS, 1):
        print(f"\n[{i}/{len(DOMAINS)}] Convergence (Audit ON) -- {p['label']}")
        cfg = ConvergenceConfig(
            name=f"{p['name_key']}_convergence",
            description=f"Convergence (Audit ON) for {p['label']}.",
            goal=p["goal"],
            num_runs=3,
        )
        reset_graph()
        result = run_convergence_experiment(cfg, AUDIT_ON_DIR)

        # Enrich each run with audit diagnostics from outputs/
        for run in result.get("runs", []):
            if "error" not in run:
                run.update(_audit_diag_from_file(run["run_id"]))

        # Re-save enriched result
        exp_dir = AUDIT_ON_DIR / result["experiment_id"]
        (exp_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

        all_results.append({
            "domain":   p["domain"],
            "label":    p["label"],
            "name_key": p["name_key"],
            "result":   result,
        })
    return all_results

# ---------------------------------------------------------------------------
# Phase 2: Ablation
# ---------------------------------------------------------------------------

def run_ablation_phase() -> list[dict]:
    from experiments.configs.schema import AblationConfig
    from experiments.runners.ablation import run_ablation_experiment

    print("\n" + "=" * 72)
    print("  PHASE 2: ABLATION AUDIT ON (5 domains x 4 variants x 3 reps = 60 runs)")
    print("=" * 72)

    all_results: list[dict] = []
    for i, p in enumerate(DOMAINS, 1):
        print(f"\n[{i}/{len(DOMAINS)}] Ablation (Audit ON) -- {p['label']}")
        cfg = AblationConfig(
            name=f"{p['name_key']}_ablation",
            description=f"Ablation (Audit ON) for {p['label']}.",
            goal=p["goal"],
            ablate_agents=["end_user", "policy", "software"],
            num_runs_per_variant=3,
        )
        result = run_ablation_experiment(cfg, AUDIT_ON_DIR)

        # Enrich each run with audit diagnostics from outputs/
        for variant in result.get("variants", []):
            for run in variant.get("runs", []):
                if "error" not in run:
                    run.update(_audit_diag_from_file(run["run_id"]))

        # Re-save enriched result
        exp_dir = AUDIT_ON_DIR / result["experiment_id"]
        (exp_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

        all_results.append({
            "domain":   p["domain"],
            "label":    p["label"],
            "name_key": p["name_key"],
            "result":   result,
        })
    return all_results

# ---------------------------------------------------------------------------
# Phase 3: Threshold sweep (custom -- captures audit diagnostics inline)
# ---------------------------------------------------------------------------

_SWEEP_THRESHOLDS  = [7.0, 8.0, 9.0]
_SWEEP_MAX_ITERS   = 5
_GEMINI_EVENTS     = {"orchestrator_complete", "independent_evaluator_complete",
                      "coding_complete", "synthesis_complete"}
_OLLAMA_EVENTS     = {"end_user_complete", "policy_complete", "software_complete"}


def run_threshold_sweep_phase() -> dict:
    from graph.builder import build_graph, reset_graph
    from state import initial_state
    from utils.config_loader import get_config
    from utils.logging_config import get_logger

    logger = get_logger(__name__)
    base_cfg = get_config()

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    sweep_dir = AUDIT_ON_DIR / f"threshold_sweep_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    total = len(DOMAINS) * len(_SWEEP_THRESHOLDS)
    runs: list[dict] = []
    run_num = 0

    print("\n" + "=" * 72)
    print(f"  PHASE 3: THRESHOLD SWEEP AUDIT ON (5 domains x 3 thresholds = 15 runs)")
    print("=" * 72)

    for threshold in _SWEEP_THRESHOLDS:
        for prompt in DOMAINS:
            run_num += 1
            run_id = str(uuid.uuid4())[:8]
            print(
                f"\n[{run_num}/{total}] domain={prompt['domain']}  "
                f"threshold={threshold}  label={prompt['label']}"
            )
            cfg = copy.deepcopy(base_cfg)
            cfg["pipeline"]["score_threshold"] = threshold
            cfg["pipeline"]["max_iterations"]  = _SWEEP_MAX_ITERS
            cfg["persistence"]["save_intermediate"] = False

            state = initial_state(user_goal=prompt["goal"], run_id=run_id, cfg=cfg)
            reset_graph()
            graph = build_graph()

            t0 = time.perf_counter()
            try:
                final_state = None
                for chunk in graph.stream(state, stream_mode="values"):
                    final_state = chunk
            except Exception as exc:
                logger.error(f"[audit_on_sweep] failed: {exc}", extra={"run_id": run_id, "node": "sweep"})
                runs.append({
                    "domain": prompt["domain"], "label": prompt["label"],
                    "prompt": prompt["goal"],   "threshold": threshold,
                    "run_id": run_id,           "error": str(exc),
                    "runtime_s": round(time.perf_counter() - t0, 2),
                })
                continue
            runtime_s = round(time.perf_counter() - t0, 2)

            if final_state is None:
                runs.append({
                    "domain": prompt["domain"], "label": prompt["label"],
                    "prompt": prompt["goal"],   "threshold": threshold,
                    "run_id": run_id,           "error": "no output state",
                    "runtime_s": runtime_s,
                })
                continue

            metrics = final_state.get("metrics_history", [])
            last_m  = metrics[-1] if metrics else {}
            stop    = final_state.get("stop_reason", "unknown")
            gemini  = sum(1 for e in final_state.get("log_entries", []) if e.get("event") in _GEMINI_EVENTS)
            ollama  = sum(1 for e in final_state.get("log_entries", []) if e.get("event") in _OLLAMA_EVENTS)

            run_result: dict = {
                "domain":                    prompt["domain"],
                "label":                     prompt["label"],
                "prompt":                    prompt["goal"],
                "threshold":                 threshold,
                "run_id":                    run_id,
                "final_score":               round(final_state.get("evaluation_score", 0.0), 4),
                "stop_reason":               stop,
                "iterations_used":           final_state.get("iteration", 0),
                "runtime_s":                 runtime_s,
                "unresolved_conflicts_final": last_m.get("unresolved_conflicts", 0),
                "converged":                 stop == "converged",
                "gemini_calls":              gemini,
                "ollama_calls":              ollama,
            }
            run_result.update(_audit_diag_from_state(final_state))
            runs.append(run_result)

            print(
                f"  score={run_result['final_score']:.2f}  "
                f"stop={stop}  iters={run_result['iterations_used']}  "
                f"wall={runtime_s:.1f}s  "
                f"audit_attempts={run_result.get('total_audit_attempts', '?')}"
            )

    full_result = {
        "experiment":   "threshold_sweep_audit_on",
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "thresholds":   _SWEEP_THRESHOLDS,
        "max_iterations": _SWEEP_MAX_ITERS,
        "total_runs":   total,
        "runs":         runs,
    }

    (sweep_dir / "results.json").write_text(json.dumps(full_result, indent=2), encoding="utf-8")

    # Summary CSV
    csv_fields = [
        "domain", "label", "threshold", "final_score", "stop_reason",
        "iterations_used", "runtime_s", "unresolved_conflicts_final",
        "total_audit_attempts", "approved_count", "failed_max_revisions_count",
        "gemini_calls", "ollama_calls", "run_id",
    ]
    with (sweep_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            if "error" not in r:
                w.writerow(r)

    print(f"\n  Sweep saved to {sweep_dir}")
    return full_result

# ---------------------------------------------------------------------------
# Phase 4: Prompt specificity (fresh registry)
# ---------------------------------------------------------------------------

def run_prompt_specificity_phase() -> dict:
    from experiments.configs.schema import PromptSpecificityConfig
    from experiments.runners.prompt_specificity import run_prompt_specificity_experiment

    print("\n" + "=" * 72)
    print("  PHASE 4: PROMPT SPECIFICITY AUDIT ON (logistics, fresh registry)")
    print("=" * 72)

    cfg = PromptSpecificityConfig(
        name="logistics_truck_parking_audit_on",
        description="Prompt specificity sensitivity with Audit ON (logistics).",
        domain="logistics",
        prompts=PS_PROMPTS,
        convergence_threshold=8.0,
        convergence_max_iterations=5,
        convergence_num_runs=3,
        ablation_agents=["end_user", "policy", "software"],
        ablation_num_runs_per_variant=3,
        sweep_thresholds=[7.0, 8.0, 9.0],
        sweep_num_runs=3,
        sweep_max_iterations=5,
    )
    return run_prompt_specificity_experiment(cfg, AUDIT_ON_DIR)

# ---------------------------------------------------------------------------
# Comparison report generation
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_mean(vals: list) -> float | None:
    nums = [v for v in vals if isinstance(v, (int, float))]
    return round(statistics.mean(nums), 4) if nums else None


def _safe_pct(count: int, total: int) -> str:
    return f"{100*count/total:.1f}%" if total > 0 else "N/A"


def generate_comparison_reports(
    on_convergence: list[dict],
    on_ablation:    list[dict],
    on_sweep:       dict,
    on_ps:          dict,
) -> None:
    AUDIT_ON_DIR.mkdir(parents=True, exist_ok=True)

    # Load Audit OFF results
    off_conv  = {dk: _load_json(p) for dk, p in AUDIT_OFF_FILES["convergence"].items()}
    off_abl   = {dk: _load_json(p) for dk, p in AUDIT_OFF_FILES["ablation"].items()}
    off_sweep = _load_json(AUDIT_OFF_FILES["threshold_sweep"])
    off_ps    = _load_json(AUDIT_OFF_FILES["prompt_specificity"])

    # ── audit_on_results.json ─────────────────────────────────────────────────
    combined = {
        "generated_at":              datetime.now(tz=timezone.utc).isoformat(),
        "audit_on_convergence":      [{"domain": e["name_key"], "label": e["label"], "result": e["result"]} for e in on_convergence],
        "audit_on_ablation":         [{"domain": e["name_key"], "label": e["label"], "result": e["result"]} for e in on_ablation],
        "audit_on_sweep":            on_sweep,
        "audit_on_prompt_specificity": on_ps,
    }
    (AUDIT_ON_DIR / "audit_on_results.json").write_text(
        json.dumps(combined, indent=2, default=str), encoding="utf-8"
    )

    # ── audit_on_summary.csv (one row per pipeline run) ───────────────────────
    csv_fields = [
        "experiment_type", "domain", "variant", "rep", "run_id",
        "final_score", "iterations_run", "stop_reason", "runtime_s",
        "unresolved_conflicts",
        "total_audit_attempts", "approved_count", "failed_max_revisions_count",
        "end_user_audit_status", "end_user_audit_attempts", "end_user_audit_score",
        "policy_audit_status",   "policy_audit_attempts",   "policy_audit_score",
        "software_audit_status", "software_audit_attempts", "software_audit_score",
    ]
    summary_rows: list[dict] = []

    for entry in on_convergence:
        for r in entry["result"].get("runs", []):
            if "error" in r:
                continue
            urc = r.get("unresolved_conflicts_per_iter", [])
            summary_rows.append({
                "experiment_type": "convergence",
                "domain":          entry["name_key"],
                "variant":         "full",
                "rep":             r.get("run_index", ""),
                "run_id":          r.get("run_id", ""),
                "final_score":     r.get("final_score", ""),
                "iterations_run":  r.get("iterations_run", ""),
                "stop_reason":     r.get("stop_reason", ""),
                "runtime_s":       "",
                "unresolved_conflicts": urc[-1] if urc else 0,
                **{k: r.get(k, "") for k in [
                    "total_audit_attempts", "approved_count", "failed_max_revisions_count",
                    "end_user_audit_status", "end_user_audit_attempts", "end_user_audit_score",
                    "policy_audit_status",   "policy_audit_attempts",   "policy_audit_score",
                    "software_audit_status", "software_audit_attempts", "software_audit_score",
                ]},
            })

    for entry in on_ablation:
        for v in entry["result"].get("variants", []):
            for r in v.get("runs", []):
                if "error" in r:
                    continue
                summary_rows.append({
                    "experiment_type": "ablation",
                    "domain":          entry["name_key"],
                    "variant":         v["label"],
                    "rep":             r.get("run_index", ""),
                    "run_id":          r.get("run_id", ""),
                    "final_score":     r.get("final_score", ""),
                    "iterations_run":  r.get("iterations_run", ""),
                    "stop_reason":     r.get("stop_reason", ""),
                    "runtime_s":       "",
                    "unresolved_conflicts": r.get("unresolved_conflicts_final", 0),
                    **{k: r.get(k, "") for k in [
                        "total_audit_attempts", "approved_count", "failed_max_revisions_count",
                        "end_user_audit_status", "end_user_audit_attempts", "end_user_audit_score",
                        "policy_audit_status",   "policy_audit_attempts",   "policy_audit_score",
                        "software_audit_status", "software_audit_attempts", "software_audit_score",
                    ]},
                })

    for r in on_sweep.get("runs", []):
        if "error" in r:
            continue
        summary_rows.append({
            "experiment_type": "threshold_sweep",
            "domain":          r.get("domain", ""),
            "variant":         f"thr={r.get('threshold')}",
            "rep":             1,
            "run_id":          r.get("run_id", ""),
            "final_score":     r.get("final_score", ""),
            "iterations_run":  r.get("iterations_used", ""),
            "stop_reason":     r.get("stop_reason", ""),
            "runtime_s":       r.get("runtime_s", ""),
            "unresolved_conflicts": r.get("unresolved_conflicts_final", 0),
            **{k: r.get(k, "") for k in [
                "total_audit_attempts", "approved_count", "failed_max_revisions_count",
                "end_user_audit_status", "end_user_audit_attempts", "end_user_audit_score",
                "policy_audit_status",   "policy_audit_attempts",   "policy_audit_score",
                "software_audit_status", "software_audit_attempts", "software_audit_score",
            ]},
        })

    with (AUDIT_ON_DIR / "audit_on_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary_rows)

    # ── audit_on_vs_off_comparison.csv ────────────────────────────────────────
    comp_fields = [
        "experiment_type", "domain", "variant",
        "audit_off_mean_score", "audit_on_mean_score", "score_delta",
        "audit_off_mean_iters", "audit_on_mean_iters", "iter_delta",
        "audit_off_total_conflicts", "audit_on_total_conflicts",
        "audit_on_mean_total_attempts", "audit_on_failed_max_revisions",
        "audit_off_runtime_s", "audit_on_runtime_s", "runtime_delta_s",
    ]
    comp_rows: list[dict] = []

    # Convergence
    for entry in on_convergence:
        dk       = entry["name_key"]
        on_runs  = [r for r in entry["result"].get("runs", []) if "error" not in r]
        off_data = off_conv.get(dk, {})
        off_runs = [r for r in off_data.get("runs", []) if "error" not in r]
        if not on_runs or not off_runs:
            continue
        on_scores  = [r["final_score"] for r in on_runs]
        off_scores = [r["final_score"] for r in off_runs]
        on_iters   = [r["iterations_run"] for r in on_runs]
        off_iters  = [r["iterations_run"] for r in off_runs]
        on_conf = [sum(r.get("unresolved_conflicts_per_iter", [])) for r in on_runs]
        off_conf= [sum(r.get("unresolved_conflicts_per_iter", [])) for r in off_runs]
        atts    = [r.get("total_audit_attempts", 0) for r in on_runs]
        fails   = [r.get("failed_max_revisions_count", 0) for r in on_runs]
        comp_rows.append({
            "experiment_type":             "convergence",
            "domain":                      dk,
            "variant":                     "full",
            "audit_off_mean_score":        round(statistics.mean(off_scores), 4),
            "audit_on_mean_score":         round(statistics.mean(on_scores), 4),
            "score_delta":                 round(statistics.mean(on_scores) - statistics.mean(off_scores), 4),
            "audit_off_mean_iters":        round(statistics.mean(off_iters), 2),
            "audit_on_mean_iters":         round(statistics.mean(on_iters), 2),
            "iter_delta":                  round(statistics.mean(on_iters) - statistics.mean(off_iters), 2),
            "audit_off_total_conflicts":   sum(off_conf),
            "audit_on_total_conflicts":    sum(on_conf),
            "audit_on_mean_total_attempts":round(statistics.mean(atts), 2),
            "audit_on_failed_max_revisions":sum(fails),
            "audit_off_runtime_s":         "",
            "audit_on_runtime_s":          "",
            "runtime_delta_s":             "",
        })

    # Ablation
    for entry in on_ablation:
        dk       = entry["name_key"]
        off_data = off_abl.get(dk, {})
        off_vs   = {v["label"]: v for v in off_data.get("variants", [])}
        for v in entry["result"].get("variants", []):
            on_runs  = [r for r in v.get("runs", []) if "error" not in r]
            off_v    = off_vs.get(v["label"], {})
            off_runs = [r for r in off_v.get("runs", []) if "error" not in r]
            if not on_runs or not off_runs:
                continue
            on_scores  = [r["final_score"] for r in on_runs]
            off_scores = [r["final_score"] for r in off_runs]
            on_iters   = [r["iterations_run"] for r in on_runs]
            off_iters  = [r["iterations_run"] for r in off_runs]
            atts  = [r.get("total_audit_attempts", 0) for r in on_runs]
            fails = [r.get("failed_max_revisions_count", 0) for r in on_runs]
            comp_rows.append({
                "experiment_type":             "ablation",
                "domain":                      dk,
                "variant":                     v["label"],
                "audit_off_mean_score":        round(statistics.mean(off_scores), 4),
                "audit_on_mean_score":         round(statistics.mean(on_scores), 4),
                "score_delta":                 round(statistics.mean(on_scores) - statistics.mean(off_scores), 4),
                "audit_off_mean_iters":        round(statistics.mean(off_iters), 2),
                "audit_on_mean_iters":         round(statistics.mean(on_iters), 2),
                "iter_delta":                  round(statistics.mean(on_iters) - statistics.mean(off_iters), 2),
                "audit_off_total_conflicts":   off_v.get("unresolved_conflicts_final", ""),
                "audit_on_total_conflicts":    v.get("unresolved_conflicts_final", ""),
                "audit_on_mean_total_attempts":round(statistics.mean(atts), 2),
                "audit_on_failed_max_revisions":sum(fails),
                "audit_off_runtime_s":         "",
                "audit_on_runtime_s":          "",
                "runtime_delta_s":             "",
            })

    # Threshold sweep
    off_by_key = {(r["domain"], r["threshold"]): r for r in off_sweep.get("runs", []) if "error" not in r}
    for r in on_sweep.get("runs", []):
        if "error" in r:
            continue
        off_r = off_by_key.get((r["domain"], r["threshold"]), {})
        if not off_r:
            continue
        on_rt  = r.get("runtime_s")
        off_rt = off_r.get("runtime_s")
        rt_delta = round(on_rt - off_rt, 2) if isinstance(on_rt, (int, float)) and isinstance(off_rt, (int, float)) else ""
        score_d = round(r.get("final_score", 0) - off_r.get("final_score", 0), 4) if isinstance(off_r.get("final_score"), (int, float)) else ""
        iter_d  = (r.get("iterations_used", 0) - off_r.get("iterations_used", 0)) if isinstance(off_r.get("iterations_used"), int) else ""
        comp_rows.append({
            "experiment_type":             "threshold_sweep",
            "domain":                      r["domain"],
            "variant":                     f"thr={r['threshold']}",
            "audit_off_mean_score":        off_r.get("final_score", ""),
            "audit_on_mean_score":         r.get("final_score", ""),
            "score_delta":                 score_d,
            "audit_off_mean_iters":        off_r.get("iterations_used", ""),
            "audit_on_mean_iters":         r.get("iterations_used", ""),
            "iter_delta":                  iter_d,
            "audit_off_total_conflicts":   off_r.get("unresolved_conflicts_final", ""),
            "audit_on_total_conflicts":    r.get("unresolved_conflicts_final", ""),
            "audit_on_mean_total_attempts":r.get("total_audit_attempts", ""),
            "audit_on_failed_max_revisions":r.get("failed_max_revisions_count", ""),
            "audit_off_runtime_s":         off_rt,
            "audit_on_runtime_s":          on_rt,
            "runtime_delta_s":             rt_delta,
        })

    with (AUDIT_ON_DIR / "audit_on_vs_off_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=comp_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(comp_rows)

    # ── audit_on_vs_off_summary.md ────────────────────────────────────────────
    _write_summary_md(comp_rows, on_convergence, on_ps, off_ps)

    print(f"\n  Comparison reports written to {AUDIT_ON_DIR}/")
    for f in ("audit_on_results.json", "audit_on_summary.csv",
              "audit_on_vs_off_comparison.csv", "audit_on_vs_off_summary.md"):
        print(f"    {f}")


def _write_summary_md(
    comp_rows:      list[dict],
    on_convergence: list[dict],
    on_ps:          dict,
    off_ps:         dict,
) -> None:
    conv_rows  = [r for r in comp_rows if r["experiment_type"] == "convergence"]
    abl_full   = [r for r in comp_rows if r["experiment_type"] == "ablation" and r["variant"] == "full"]
    sweep_rows = [r for r in comp_rows if r["experiment_type"] == "threshold_sweep"]

    # Score deltas
    conv_score_d  = [r["score_delta"] for r in conv_rows if isinstance(r.get("score_delta"), (int, float))]
    abl_score_d   = [r["score_delta"] for r in abl_full   if isinstance(r.get("score_delta"), (int, float))]
    sweep_score_d = [r["score_delta"] for r in sweep_rows if isinstance(r.get("score_delta"), (int, float))]
    conv_iter_d   = [r["iter_delta"]  for r in conv_rows  if isinstance(r.get("iter_delta"),  (int, float))]
    rt_deltas     = [r["runtime_delta_s"] for r in sweep_rows if isinstance(r.get("runtime_delta_s"), (int, float))]

    # Audit attempt stats from convergence runs
    on_conv_runs  = [r for e in on_convergence for r in e["result"].get("runs", []) if "error" not in r]
    all_attempts  = [r.get("total_audit_attempts", 0) for r in on_conv_runs if r.get("total_audit_attempts") is not None]
    all_failed    = sum(r.get("failed_max_revisions_count", 0) for r in on_conv_runs)
    all_approved  = sum(r.get("approved_count", 0) for r in on_conv_runs)
    total_branches = len(on_conv_runs) * 3  # 3 audit branches per run

    # Conflict delta
    conf_deltas = [
        r["audit_on_total_conflicts"] - r["audit_off_total_conflicts"]
        for r in conv_rows
        if isinstance(r.get("audit_on_total_conflicts"), (int, float))
        and isinstance(r.get("audit_off_total_conflicts"), (int, float))
    ]

    mcd  = _safe_mean(conv_score_d)
    mad  = _safe_mean(abl_score_d)
    msd  = _safe_mean(sweep_score_d)
    mcid = _safe_mean(conv_iter_d)
    mrt  = _safe_mean(rt_deltas)
    mconf= _safe_mean(conf_deltas)
    mats = _safe_mean(all_attempts)

    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def _dir(v: float | None) -> str:
        if v is None: return "N/A"
        if v > 0.01:  return "improved"
        if v < -0.01: return "degraded"
        return "within noise"

    lines = [
        "# Audit ON vs Audit OFF -- Comparison Summary",
        "",
        f"Generated: {now}",
        "Audit OFF: experiments run before audit layer was implemented.",
        "Audit ON : experiments run with audit_layer.enabled=true (threshold=8.5, max_revisions=2).",
        "",
        "---",
        "",
        "## 1. Did Audit ON improve final scores?",
        "",
    ]

    if mcd is not None:
        lines.append(f"**Convergence (5 domains x 3 runs):** mean score delta = {mcd:+.4f} ({_dir(mcd)}).")
    if mad is not None:
        lines.append(f"**Ablation full-pipeline (5 domains x 3 runs):** mean score delta = {mad:+.4f} ({_dir(mad)}).")
    if msd is not None:
        lines.append(f"**Threshold sweep (15 runs):** mean score delta = {msd:+.4f} ({_dir(msd)}).")
    lines += [
        "",
        "_Note: No fixed seed -- small deltas should be treated as within LLM run-to-run noise._",
        "",
        "---",
        "",
        "## 2. Did Audit ON increase runtime/cost?",
        "",
    ]
    if rt_deltas:
        lines.append(f"**Threshold sweep runtime delta:** mean = {mrt:+.1f}s per run." if mrt else "")
        lines.append("Audit adds 3 extra Ollama calls per iteration as baseline (6 per revision loop).")
    else:
        lines.append("Runtime not captured for convergence/ablation. Only threshold sweep tracks runtime_s.")
    lines += [
        "",
        "---",
        "",
        "## 3. Did Audit ON reduce unresolved conflicts?",
        "",
    ]
    if conf_deltas and mconf is not None:
        lines.append(f"**Convergence conflict delta:** mean = {mconf:+.2f} total unresolved conflicts per 3-run batch ({_dir(-mconf)} = fewer is better).")
    else:
        lines.append("Conflict delta not available (insufficient data).")
    lines += [
        "",
        "---",
        "",
        "## 4. How often did audit branches require revision?",
        "",
    ]
    if all_attempts:
        revision_runs = sum(1 for a in all_attempts if a > 3)  # >3 means at least 1 revision
        lines += [
            f"**Convergence runs with at least one audit revision:** {revision_runs}/{len(all_attempts)} runs ({_safe_pct(revision_runs, len(all_attempts))}).",
            f"**Mean total audit attempts per run:** {mats:.2f} (baseline=3 when all branches pass on first try).",
        ]
    lines += [
        "",
        "---",
        "",
        "## 5. How often did audit branches fail after max revisions?",
        "",
        f"**Convergence (15 runs x 3 branches = {total_branches} audit branch evaluations):**",
        f"  - failed_after_max_revisions: {all_failed} ({_safe_pct(all_failed, total_branches)})",
        f"  - approved: {all_approved} ({_safe_pct(all_approved, total_branches)})",
        f"  - remaining: disabled or status unknown",
        "",
        "Branches that fail after max revisions still contribute their final feedback to the aggregator.",
        "Failed status means the audit could not obtain quality feedback via revision alone.",
        "",
        "---",
        "",
        "## 6. Were benefits stronger for vague, moderate, or highly specific prompts?",
        "",
    ]
    try:
        off_conv_res = off_ps.get("convergence", {}).get("results", [])
        on_conv_res  = on_ps.get("convergence", {}).get("results", [])
        for spec in ("vague", "moderate", "highly_specific"):
            off_spec = next((s for s in off_conv_res if s["specificity"] == spec), {})
            on_spec  = next((s for s in on_conv_res  if s["specificity"] == spec), {})
            off_m = off_spec.get("mean_score")
            on_m  = on_spec.get("mean_score")
            delta = round(on_m - off_m, 4) if isinstance(off_m, (int, float)) and isinstance(on_m, (int, float)) else "N/A"
            delta_str = f"{delta:+.4f}" if isinstance(delta, float) else str(delta)
            lines.append(f"  - **{spec}**: OFF mean={off_m}, ON mean={on_m}, delta={delta_str}")
        lines.append("")
    except Exception as ex:
        lines.append(f"_Could not generate prompt specificity comparison: {ex}_")
        lines.append("")

    # Convergence table
    lines += [
        "---",
        "",
        "## Convergence Comparison Table",
        "",
        "| Domain | OFF Score | ON Score | Delta | OFF Iters | ON Iters | Iter Delta | ON Failed Rev |",
        "|--------|-----------|----------|-------|-----------|----------|------------|---------------|",
    ]
    for r in conv_rows:
        sd = r.get("score_delta")
        id_ = r.get("iter_delta")
        lines.append(
            f"| {r['domain']} "
            f"| {r.get('audit_off_mean_score', 'N/A')} "
            f"| {r.get('audit_on_mean_score', 'N/A')} "
            f"| {sd:+.4f} " if isinstance(sd, float) else "| N/A "
            f"| {r.get('audit_off_mean_iters', 'N/A')} "
            f"| {r.get('audit_on_mean_iters', 'N/A')} "
            f"| {id_:+.2f} " if isinstance(id_, float) else "| N/A "
            f"| {r.get('audit_on_failed_max_revisions', 'N/A')} |"
        )

    # Threshold sweep table
    lines += [
        "",
        "---",
        "",
        "## Threshold Sweep Comparison Table",
        "",
        "| Domain | Thr | OFF Score | ON Score | Delta | OFF iters | ON iters | OFF rt(s) | ON rt(s) | RT delta |",
        "|--------|-----|-----------|----------|-------|-----------|----------|-----------|----------|----------|",
    ]
    for r in sweep_rows:
        sd  = r.get("score_delta")
        id_ = r.get("iter_delta")
        rtd = r.get("runtime_delta_s")
        lines.append(
            f"| {r['domain']} | {r['variant']} "
            f"| {r.get('audit_off_mean_score', '')} | {r.get('audit_on_mean_score', '')} "
            f"| {sd:+.4f} | {r.get('audit_off_mean_iters','')} | {r.get('audit_on_mean_iters','')} "
            f"| {r.get('audit_off_runtime_s','')} | {r.get('audit_on_runtime_s','')} "
            f"| {rtd:+.1f} |"
            if all(isinstance(x, (int, float)) for x in [sd, rtd]) else
            f"| {r.get('domain','')} | {r.get('variant','')} | (incomplete data) |"
        )

    lines += ["", "---", f"_Generated by run_audit_on_experiments.py -- {now}_", ""]

    (AUDIT_ON_DIR / "audit_on_vs_off_summary.md").write_text("\n".join(lines), encoding="utf-8")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def verify_audit_enabled() -> None:
    from utils.config_loader import get_config, reload_config
    cfg = reload_config()
    if not cfg.get("audit_layer", {}).get("enabled", False):
        print("\n[ERROR] audit_layer.enabled is False in config.yaml -- aborting.")
        sys.exit(1)
    alth = cfg["audit_layer"]["audit_approval_threshold"]
    maxr = cfg["audit_layer"]["max_audit_revisions"]
    print(f"\n  audit_layer.enabled=True  threshold={alth}  max_revisions={maxr}")


def main() -> None:
    AUDIT_ON_DIR.mkdir(parents=True, exist_ok=True)
    verify_audit_enabled()

    start = datetime.now(tz=timezone.utc)
    print(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Output : {AUDIT_ON_DIR.resolve()}")

    on_convergence = run_convergence_phase()
    on_ablation    = run_ablation_phase()
    on_sweep       = run_threshold_sweep_phase()
    on_ps          = run_prompt_specificity_phase()

    elapsed = (datetime.now(tz=timezone.utc) - start).total_seconds() / 60
    print(f"\n  All experiments completed in {elapsed:.1f} minutes.")
    print("\n  Generating comparison reports...")
    generate_comparison_reports(on_convergence, on_ablation, on_sweep, on_ps)
    print(f"\n  DONE. Total elapsed: {elapsed:.1f} minutes.")


if __name__ == "__main__":
    main()
