"""
experiments/run_all_experiments.py — Full re-run of all four experiments
across all five domains under the updated independent-evaluator architecture.

Schedule:
  Phase 1 — Convergence : 5 domains × 3 runs  = 15 pipeline runs
  Phase 2 — Ablation    : 5 domains × 4 variants = 20 pipeline runs
  Phase 3 — Baseline    : 5 domains × 1 run    =  5 pipeline runs
  Phase 4 — Threshold   : 5 domains × 3 thresholds = 15 pipeline runs
  Total: 55 pipeline runs

Usage:
    python experiments/run_all_experiments.py
"""

from __future__ import annotations

import sys
import statistics
from pathlib import Path
from datetime import datetime, timezone

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv()

from experiments.configs.schema import ConvergenceConfig, AblationConfig, BaselineConfig
from experiments.runners.convergence import run_convergence_experiment
from experiments.runners.ablation import run_ablation_experiment
from experiments.runners.baseline import run_baseline_experiment
from experiments.threshold_sweep import run_sweep

OUTPUT_DIR = Path("experiments/results")

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


# ── Phase 1: Convergence ──────────────────────────────────────────────────────

def run_all_convergence() -> list[dict]:
    print("\n" + "=" * 70)
    print("  PHASE 1: CONVERGENCE (5 domains x 3 runs = 15 pipeline runs)")
    print("=" * 70)
    results = []
    for i, p in enumerate(DOMAINS, 1):
        print(f"\n[{i}/{len(DOMAINS)}] Convergence -- {p['label']}")
        cfg = ConvergenceConfig(
            name=f"{p['name_key']}_convergence",
            description=f"Convergence stability for the {p['label']} goal.",
            goal=p["goal"],
            num_runs=3,
        )
        result = run_convergence_experiment(cfg, OUTPUT_DIR)
        results.append({"domain": p["domain"], "label": p["label"], "result": result})
    return results


# ── Phase 2: Ablation ─────────────────────────────────────────────────────────

def run_all_ablation() -> list[dict]:
    print("\n" + "=" * 70)
    print("  PHASE 2: ABLATION (5 domains x 4 variants = 20 pipeline runs)")
    print("=" * 70)
    results = []
    for i, p in enumerate(DOMAINS, 1):
        print(f"\n[{i}/{len(DOMAINS)}] Ablation -- {p['label']}")
        cfg = AblationConfig(
            name=f"{p['name_key']}_ablation",
            description=f"Per-agent ablation for the {p['label']} goal.",
            goal=p["goal"],
            ablate_agents=["end_user", "policy", "software"],
        )
        result = run_ablation_experiment(cfg, OUTPUT_DIR)
        results.append({"domain": p["domain"], "label": p["label"], "result": result})
    return results


# ── Phase 3: Baseline ─────────────────────────────────────────────────────────

def run_all_baseline() -> list[dict]:
    print("\n" + "=" * 70)
    print("  PHASE 3: BASELINE (5 domains x 1 run = 5 pipeline runs)")
    print("=" * 70)
    results = []
    for i, p in enumerate(DOMAINS, 1):
        print(f"\n[{i}/{len(DOMAINS)}] Baseline -- {p['label']}")
        cfg = BaselineConfig(
            name=f"{p['name_key']}_baseline",
            description=f"Pipeline vs single-LLM baseline for {p['label']}.",
            goal=p["goal"],
        )
        result = run_baseline_experiment(cfg, OUTPUT_DIR)
        results.append({"domain": p["domain"], "label": p["label"], "result": result})
    return results


# ── Phase 4: Threshold sweep ──────────────────────────────────────────────────

def run_threshold_sweep() -> None:
    print("\n" + "=" * 70)
    print("  PHASE 4: THRESHOLD SWEEP (5 domains x 3 thresholds = 15 runs)")
    print("=" * 70)
    run_sweep(OUTPUT_DIR)


# ── Summary printers ──────────────────────────────────────────────────────────

def print_convergence_summary(all_results: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("  CONVERGENCE SUMMARY")
    print("=" * 78)
    print(f"  {'Domain':<24}  {'Avg Score':>9}  {'Std Dev':>7}  {'Avg Iters':>9}  {'Conflicts':>9}")
    print("  " + "-" * 64)
    for r in all_results:
        runs = [x for x in r["result"].get("runs", []) if "error" not in x]
        if not runs:
            print(f"  {r['label']:<24}  NO SUCCESSFUL RUNS")
            continue
        scores = [x["final_score"] for x in runs]
        iters  = [x["iterations_run"] for x in runs]
        confs  = [sum(x["unresolved_conflicts_per_iter"]) for x in runs]
        std    = statistics.stdev(scores) if len(scores) > 1 else 0.0
        print(
            f"  {r['label']:<24}  {statistics.mean(scores):>9.2f}  {std:>7.2f}  "
            f"{statistics.mean(iters):>9.1f}  {sum(confs):>9}"
        )
    print("=" * 78)


def print_ablation_summary(all_results: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("  ABLATION SUMMARY")
    print("=" * 90)
    print(f"  {'Domain':<24}  {'Variant':<16}  {'Score':>6}  {'Delta':>6}  {'Iters':>5}  {'Conflicts':>9}  {'Components':>10}")
    print("  " + "-" * 82)
    for r in all_results:
        variants = [v for v in r["result"].get("variants", []) if "error" not in v]
        full = next((v for v in variants if not v.get("disabled_agents")), None)
        full_score = full["final_score"] if full else 0.0
        for v in variants:
            delta = f"{v['final_score'] - full_score:+.2f}" if v.get("disabled_agents") else "---"
            print(
                f"  {r['label']:<24}  {v['label']:<16}  {v['final_score']:>6.2f}  "
                f"{delta:>6}  {v['iterations_run']:>5}  {v['unresolved_conflicts_final']:>9}  "
                f"{v['num_components']:>10}"
            )
        print("  " + "-" * 82)
    print("=" * 90)


def print_baseline_summary(all_results: list[dict]) -> None:
    DOMAIN_LABELS = {
        "logistics":  "Truck Parking",
        "healthcare": "Patient Scheduling",
        "smart_city": "Traffic Mgmt",
        "e-commerce": "Product Recs",
        "education":  "Course Planning",
    }
    print("\n" + "=" * 88)
    print("  BASELINE SUMMARY")
    print("=" * 88)
    print(f"  {'Domain':<22}  {'Pipeline':>8}  {'Baseline':>8}  {'Delta':>6}  {'Verdict':<10}  {'Iters':>5}  {'Comp P/B':>10}")
    print("  " + "-" * 78)
    for r in all_results:
        res  = r["result"]
        pip  = res.get("pipeline", {})
        jdg  = res.get("judgement", {})
        label = DOMAIN_LABELS.get(r["domain"], r["domain"])
        p_score = jdg.get("overall_score_pipeline", 0.0)
        b_score = jdg.get("overall_score_baseline", 0.0)
        verdict = "pipeline" if jdg.get("pipeline_preferred") else "baseline"
        comp = f"{pip.get('num_components',0)} / {res.get('baseline',{}).get('num_components',0)}"
        print(
            f"  {label:<22}  {p_score:>8.1f}  {b_score:>8.1f}  {p_score-b_score:>+6.1f}  "
            f"{verdict:<10}  {pip.get('iterations',0):>5}  {comp:>10}"
        )
    print("=" * 88)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    start = datetime.now(tz=timezone.utc)
    print(f"\nStarted at {start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("Total planned: 55 pipeline runs across 4 experiment types, 5 domains")

    conv_results = run_all_convergence()
    abl_results  = run_all_ablation()
    base_results = run_all_baseline()
    run_threshold_sweep()

    end     = datetime.now(tz=timezone.utc)
    elapsed = (end - start).total_seconds() / 60

    print_convergence_summary(conv_results)
    print_ablation_summary(abl_results)
    print_baseline_summary(base_results)

    print(f"\nAll done. Total elapsed: {elapsed:.1f} minutes")
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
