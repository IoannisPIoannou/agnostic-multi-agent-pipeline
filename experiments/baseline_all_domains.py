"""
experiments/baseline_all_domains.py — Baseline comparison for all 5 domain prompts.

Runs the pipeline vs single-LLM baseline experiment for 4 remaining domains
(healthcare, smart_city, e-commerce, education) and loads the existing truck
parking result, then prints a combined summary table.

Usage:
    python experiments/baseline_all_domains.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv()

from experiments.configs.schema import BaselineConfig
from experiments.runners.baseline import run_baseline_experiment

OUTPUT_DIR = Path("experiments/results")

EXISTING_RESULT = Path(
    "experiments/results/baseline_truck_parking_baseline_20260429_012335/results.json"
)

NEW_PROMPTS = [
    {
        "domain": "healthcare",
        "name": "patient_scheduling_baseline",
        "goal": (
            "Design an AI-powered patient scheduling system for a multi-specialty "
            "outpatient clinic that minimizes wait times and maximizes doctor utilization."
        ),
    },
    {
        "domain": "smart_city",
        "name": "traffic_management_baseline",
        "goal": (
            "Design a smart city traffic management system that dynamically adjusts "
            "signal timings based on real-time congestion data and incident reports."
        ),
    },
    {
        "domain": "e-commerce",
        "name": "product_recommendations_baseline",
        "goal": (
            "Design a personalized product recommendation engine for an e-commerce "
            "platform that balances user preferences, inventory levels, and profit margins."
        ),
    },
    {
        "domain": "education",
        "name": "course_planning_baseline",
        "goal": (
            "Design an AI-assisted university course planning system that helps students "
            "select courses based on prerequisites, career goals, and schedule constraints."
        ),
    },
]


def _extract_row(result: dict, domain: str) -> dict:
    pipeline   = result.get("pipeline", {})
    baseline   = result.get("baseline", {})
    judgement  = result.get("judgement", {})
    scores     = {s["criterion"]: (s["score_pipeline"], s["score_baseline"])
                  for s in judgement.get("criterion_scores", [])}
    return {
        "domain":             domain,
        "pipeline_score":     pipeline.get("final_score", 0.0),
        "pipeline_iters":     pipeline.get("iterations", 0),
        "pipeline_components": pipeline.get("num_components", 0),
        "baseline_components": baseline.get("num_components", 0),
        "judge_pipeline":     judgement.get("overall_score_pipeline", 0.0),
        "judge_baseline":     judgement.get("overall_score_baseline", 0.0),
        "preferred":          "pipeline" if judgement.get("pipeline_preferred") else "baseline",
        "usability":          scores.get("usability", (0, 0)),
        "clarity":            scores.get("clarity", (0, 0)),
        "feasibility":        scores.get("feasibility", (0, 0)),
        "stakeholder_balance": scores.get("stakeholder_balance", (0, 0)),
        "overall_quality":    scores.get("overall_quality", (0, 0)),
        "reason":             judgement.get("preference_reason", ""),
    }


def print_summary(rows: list[dict]) -> None:
    DOMAINS = {
        "logistics":  "Truck Parking",
        "healthcare": "Patient Scheduling",
        "smart_city": "Traffic Mgmt",
        "e-commerce": "Product Recs",
        "education":  "Course Planning",
    }

    print("\n" + "=" * 90)
    print("  BASELINE COMPARISON — ALL 5 DOMAINS")
    print("=" * 90)

    # Table 1: head-to-head scores
    print("\nTable 1: Head-to-Head Judge Scores (out of 10)")
    print(f"  {'Domain':<22}  {'Pipeline':>8}  {'Baseline':>8}  {'Delta':>6}  {'Verdict':<10}  {'Iters':>5}  {'Components (P/B)':>18}")
    print("  " + "-" * 82)
    for r in rows:
        label = DOMAINS.get(r["domain"], r["domain"])
        delta = r["judge_pipeline"] - r["judge_baseline"]
        comp  = f"{r['pipeline_components']} / {r['baseline_components']}"
        print(
            f"  {label:<22}  {r['judge_pipeline']:>8.1f}  {r['judge_baseline']:>8.1f}  "
            f"{delta:>+6.1f}  {r['preferred']:<10}  {r['pipeline_iters']:>5}  {comp:>18}"
        )
    avg_p = sum(r["judge_pipeline"] for r in rows) / len(rows)
    avg_b = sum(r["judge_baseline"] for r in rows) / len(rows)
    print("  " + "-" * 82)
    print(f"  {'AVERAGE':<22}  {avg_p:>8.1f}  {avg_b:>8.1f}  {avg_p-avg_b:>+6.1f}  {'pipeline' if avg_p > avg_b else 'baseline':<10}")

    # Table 2: per-criterion breakdown
    print("\nTable 2: Per-Criterion Judge Scores — Pipeline (P) vs Baseline (B)")
    criteria = ["usability", "clarity", "feasibility", "stakeholder_balance", "overall_quality"]
    header = f"  {'Domain':<22}" + "".join(f"  {c[:8]:>10}" for c in criteria)
    print(header)
    print("  " + "-" * (22 + len(criteria) * 12))
    for r in rows:
        label = DOMAINS.get(r["domain"], r["domain"])
        cells = ""
        for c in criteria:
            p, b = r[c]
            cells += f"  {p:.0f}v{b:.0f}({p-b:+.0f})"
        print(f"  {label:<22}{cells}")

    # Table 3: preference count
    n_pipeline = sum(1 for r in rows if r["preferred"] == "pipeline")
    n_baseline = sum(1 for r in rows if r["preferred"] == "baseline")
    print(f"\nTable 3: Verdict Tally")
    print(f"  Pipeline preferred : {n_pipeline} / {len(rows)}")
    print(f"  Baseline preferred : {n_baseline} / {len(rows)}")

    # Table 4: judge reasons
    print("\nTable 4: Judge Preference Reasons")
    for r in rows:
        label = DOMAINS.get(r["domain"], r["domain"])
        reason = r["reason"][:120] + ("..." if len(r["reason"]) > 120 else "")
        print(f"  [{label}] {reason}")

    print("=" * 90)


def main() -> None:
    all_rows: list[dict] = []

    # Load existing truck parking result
    if EXISTING_RESULT.exists():
        print(f"Loading existing truck parking result from {EXISTING_RESULT}")
        data = json.loads(EXISTING_RESULT.read_text(encoding="utf-8"))
        all_rows.append(_extract_row(data, "logistics"))
    else:
        print(f"WARNING: existing result not found at {EXISTING_RESULT}, skipping.")

    # Run the 4 remaining domains
    total = len(NEW_PROMPTS)
    for i, p in enumerate(NEW_PROMPTS, 1):
        print(f"\n[{i}/{total}] Running baseline for domain={p['domain']} ...")
        cfg = BaselineConfig(name=p["name"], goal=p["goal"])
        result = run_baseline_experiment(cfg, OUTPUT_DIR)
        all_rows.append(_extract_row(result, p["domain"]))

    print_summary(all_rows)


if __name__ == "__main__":
    main()
