"""
experiments/run_prompt_specificity.py — CLI entry point for Experiment 5.

Loads a prompt-specificity config YAML, runs all 4 sub-experiments
(convergence, ablation, baseline, threshold sweep) with run reuse, then
prints a per-specificity summary table.

Usage:
    python experiments/run_prompt_specificity.py
    python experiments/run_prompt_specificity.py path/to/my_config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv()

from experiments.configs.schema import load_config, PromptSpecificityConfig
from experiments.runners.prompt_specificity import run_prompt_specificity_experiment

OUTPUT_DIR    = Path("experiments/results")
DEFAULT_CONFIG = Path("experiments/configs/prompt_specificity_example.yaml")

_SPECIFICITIES = ["vague", "moderate", "highly_specific"]


def _print_summary(result: dict) -> None:
    conv  = result.get("convergence",     {}).get("results", [])
    abl   = result.get("ablation",        {}).get("results", [])
    base  = result.get("baseline",        {}).get("results", [])
    sweep = result.get("threshold_sweep", {}).get("results", [])

    # ── Convergence summary ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  CONVERGENCE  (threshold={})".format(result["convergence"]["threshold"]))
    print("=" * 72)
    print(f"  {'Specificity':<20}  {'Mean':>6}  {'Std':>5}  {'Min':>5}  {'Max':>5}  {'Iters':>6}")
    print("  " + "-" * 60)
    for r in conv:
        print(f"  {r['specificity']:<20}  {r['mean_score']:>6.2f}  {r['score_std']:>5.2f}  "
              f"{r['score_min']:>5.1f}  {r['score_max']:>5.1f}  {r['mean_iters']:>6.1f}")

    # ── Ablation summary ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  ABLATION")
    print("=" * 80)
    print(f"  {'Specificity':<20}  {'Variant':<16}  {'Mean':>6}  {'Std':>5}  {'Delta':>7}")
    print("  " + "-" * 68)
    for r in abl:
        for v in r.get("variants", []):
            delta = "---" if v.get("delta_vs_full") is None else f"{v['delta_vs_full']:+.2f}"
            print(f"  {r['specificity'] if v['variant']=='full' else '':<20}  "
                  f"{v['variant']:<16}  {v['mean_score']:>6.2f}  "
                  f"{v['score_std']:>5.2f}  {delta:>7}")
        print("  " + "-" * 68)

    # ── Baseline summary ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  BASELINE  (judge comparison)")
    print("=" * 72)
    print(f"  {'Specificity':<20}  {'Pipeline':>8}  {'Baseline':>8}  {'Delta':>6}  {'Preferred':<10}")
    print("  " + "-" * 60)
    for r in base:
        jdg     = r.get("judgement", {})
        p_score = jdg.get("overall_score_pipeline", 0.0)
        b_score = jdg.get("overall_score_baseline", 0.0)
        pref    = "pipeline" if jdg.get("pipeline_preferred") else "baseline"
        print(f"  {r['specificity']:<20}  {p_score:>8.1f}  {b_score:>8.1f}  "
              f"{p_score - b_score:>+6.1f}  {pref:<10}")

    # ── Threshold sweep summary ────────────────────────────────────────────
    thresholds = result.get("threshold_sweep", {}).get("thresholds", [])
    print("\n" + "=" * 80)
    print("  THRESHOLD SWEEP")
    print("=" * 80)
    header_thr = "  ".join(f"thr={t:.1f}(mean/iters)" for t in thresholds)
    print(f"  {'Specificity':<20}  {header_thr}")
    print("  " + "-" * 76)
    for r in sweep:
        parts = []
        for thr_r in r.get("threshold_results", []):
            parts.append(f"{thr_r['mean_score']:>5.2f} / {thr_r['mean_iters']:>4.1f}")
        print(f"  {r['specificity']:<20}  {'   '.join(parts)}")

    print("\n" + "=" * 72)
    print(f"  Elapsed : {result.get('elapsed_minutes', 0):.1f} min")
    print(f"  Results : {OUTPUT_DIR.resolve()}/prompt_specificity/{result['experiment_id']}")
    print("=" * 72)


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    cfg = load_config(str(config_path))
    if not isinstance(cfg, PromptSpecificityConfig):
        print(f"Config type must be 'prompt_specificity', got {cfg.type!r}")
        sys.exit(1)

    start = datetime.now(tz=timezone.utc)
    print(f"\nStarted at {start.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    result = run_prompt_specificity_experiment(cfg, OUTPUT_DIR)
    _print_summary(result)


if __name__ == "__main__":
    main()
