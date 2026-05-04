"""
experiments/run_ablation_3x.py — Re-run ablation with 3 runs per variant.

5 domains x 4 variants x 3 runs = 60 pipeline runs total.
Results include mean, std dev, min, max per variant for statistical reliability.

Usage:
    python experiments/run_ablation_3x.py
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

from experiments.configs.schema import AblationConfig
from experiments.runners.ablation import run_ablation_experiment

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


def print_summary(all_results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("  ABLATION SUMMARY — 3 RUNS PER VARIANT")
    print("=" * 100)
    print(f"  {'Domain':<24}  {'Variant':<16}  {'Mean':>6}  {'Std':>5}  {'Min':>5}  {'Max':>5}  {'Delta':>6}  {'Iters':>5}  {'Components':>10}")
    print("  " + "-" * 92)
    for r in all_results:
        variants = [v for v in r["result"].get("variants", []) if v.get("runs")]
        full = next((v for v in variants if not v.get("disabled_agents")), None)
        full_mean = full["final_score"] if full else 0.0
        for v in variants:
            is_full = not v.get("disabled_agents")
            delta = "---" if is_full else f"{v['final_score'] - full_mean:+.2f}"
            print(
                f"  {r['label'] if is_full else '':<24}  {v['label']:<16}  "
                f"{v['final_score']:>6.2f}  {v.get('score_std', 0):>5.2f}  "
                f"{v.get('score_min', 0):>5.1f}  {v.get('score_max', 0):>5.1f}  "
                f"{delta:>6}  {v['iterations_run']:>5.1f}  {v['num_components']:>10}"
            )
        print("  " + "-" * 92)
    print("=" * 100)


def main() -> None:
    start = datetime.now(tz=timezone.utc)
    total_runs = len(DOMAINS) * 4 * 3
    print(f"\nStarted at {start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Ablation: 5 domains x 4 variants x 3 runs = {total_runs} pipeline runs")

    all_results = []
    for i, p in enumerate(DOMAINS, 1):
        print(f"\n[{i}/{len(DOMAINS)}] Ablation (3x) -- {p['label']}")
        cfg = AblationConfig(
            name=f"{p['name_key']}_ablation",
            description=f"Per-agent ablation (3 runs/variant) for the {p['label']} goal.",
            goal=p["goal"],
            ablate_agents=["end_user", "policy", "software"],
            num_runs_per_variant=3,
        )
        result = run_ablation_experiment(cfg, OUTPUT_DIR)
        all_results.append({"domain": p["domain"], "label": p["label"], "result": result})

    end     = datetime.now(tz=timezone.utc)
    elapsed = (end - start).total_seconds() / 60
    print_summary(all_results)
    print(f"\nAll done. Total elapsed: {elapsed:.1f} minutes")
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
