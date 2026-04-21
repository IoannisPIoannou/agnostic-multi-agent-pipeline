"""
experiments/run_experiment.py — CLI entry point for the experimental framework.

Usage:
    python experiments/run_experiment.py --config experiments/configs/convergence_example.yaml
    python experiments/run_experiment.py --config experiments/configs/ablation_example.yaml
    python experiments/run_experiment.py --config experiments/configs/baseline_example.yaml
    python experiments/run_experiment.py --config experiments/configs/runtime_example.yaml

    # Summarize an existing result without re-running
    python experiments/run_experiment.py --summarize experiments/results/.../results.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path when run as a script directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv()

# Pipeline logging must be set up before any pipeline imports
from utils.config_loader import get_config
from utils.logging_config import setup_logging

setup_logging(get_config())

from experiments.analysis.summarize import load_result, summarize
from experiments.configs.schema import (
    AblationConfig,
    BaselineConfig,
    ConvergenceConfig,
    RuntimeConfig,
    load_config,
)

RESULTS_DIR = Path("experiments/results")


def _run(cfg_path: str) -> None:
    cfg = load_config(cfg_path)

    if isinstance(cfg, ConvergenceConfig):
        from experiments.runners.convergence import run_convergence_experiment
        result = run_convergence_experiment(cfg, RESULTS_DIR)

    elif isinstance(cfg, AblationConfig):
        from experiments.runners.ablation import run_ablation_experiment
        result = run_ablation_experiment(cfg, RESULTS_DIR)

    elif isinstance(cfg, BaselineConfig):
        from experiments.runners.baseline import run_baseline_experiment
        result = run_baseline_experiment(cfg, RESULTS_DIR)

    elif isinstance(cfg, RuntimeConfig):
        from experiments.runners.runtime import run_runtime_experiment
        result = run_runtime_experiment(cfg, RESULTS_DIR)

    else:
        print(f"[ERROR] Unknown config type: {type(cfg)}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY")
    print("=" * 70)
    print(summarize(result))
    print("=" * 70 + "\n")


def _summarize(result_path: str) -> None:
    result = load_result(result_path)
    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY")
    print("=" * 70)
    print(summarize(result))
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or summarize a pipeline experiment."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config", metavar="PATH",
        help="Path to an experiment YAML config file.",
    )
    group.add_argument(
        "--summarize", metavar="PATH",
        help="Path to an existing results.json file to summarize without re-running.",
    )
    args = parser.parse_args()

    if args.config:
        _run(args.config)
    else:
        _summarize(args.summarize)


if __name__ == "__main__":
    main()
