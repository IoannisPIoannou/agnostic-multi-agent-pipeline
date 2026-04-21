"""
main.py — Entry point for the multi-agent pipeline.

Usage:
    python main.py                          # uses built-in truck-parking example
    python main.py --goal "your goal here" # custom goal
    python main.py --config path/to/config.yaml

Environment:
    GEMINI_API_KEY must be set (or present in a .env file in this directory).
    Ollama must be running locally (default: http://localhost:11434).
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Load .env before anything imports os.environ
load_dotenv(Path(__file__).parent / ".env")

from state import initial_state
from utils.config_loader import get_config
from utils.logging_config import get_logger, setup_logging

# ── Sample problem ────────────────────────────────────────────────────────────
# Used when no --goal is provided.  Domain: truck parking recommendation.

SAMPLE_GOAL = """\
Design a simple decision-support system that recommends truck parking locations
based on distance, estimated availability, and driver preferences.

The system should balance:
  - Driver convenience   (short detour, low cost, familiar locations)
  - System efficiency    (avoid over-utilised locations, distribute load)
  - Implementation feasibility (simple, scalable, easy to maintain)

Provide a structured solution including:
  - System design with key components
  - Core decision logic for ranking locations
  - Evaluation criteria to validate the solution
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-agent LangGraph pipeline for goal-driven decision support."
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="High-level problem statement.  Defaults to the truck-parking example.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a config.yaml file.  Defaults to config.yaml in this directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load config first (needed for logging setup) ──────────────────────────
    from utils.config_loader import get_config, reload_config
    cfg = reload_config(args.config) if args.config else get_config()

    # ── Set up structured logging ─────────────────────────────────────────────
    setup_logging(cfg)
    logger = get_logger("main")

    # ── Resolve goal ──────────────────────────────────────────────────────────
    user_goal = args.goal or SAMPLE_GOAL

    # ── Generate a unique run ID ──────────────────────────────────────────────
    run_id = str(uuid.uuid4())[:8]   # short UUID for readable output paths

    print(f"\n{'='*60}")
    print(f"  Multi-Agent Pipeline")
    print(f"  Run ID : {run_id}")
    print(f"  Max iterations : {cfg['pipeline']['max_iterations']}")
    print(f"  Score threshold: {cfg['pipeline']['score_threshold']}")
    print(f"{'='*60}\n")
    print("Goal:\n" + user_goal.strip())
    print(f"\n{'='*60}\n")

    logger.info(
        f"Pipeline initialised | run_id={run_id}",
        extra={"run_id": run_id, "node": "main"},
    )

    # ── Build initial state ───────────────────────────────────────────────────
    state = initial_state(user_goal=user_goal, run_id=run_id, cfg=cfg)

    # ── Run the pipeline ──────────────────────────────────────────────────────
    from graph.runner import run_pipeline

    try:
        final_state = run_pipeline(state)
    except EnvironmentError as exc:
        print(f"\n[ERROR] {exc}\n", file=sys.stderr)
        sys.exit(1)
    except ConnectionError as exc:
        print(
            f"\n[ERROR] Could not connect to Ollama: {exc}\n"
            f"Is Ollama running?  Check: {cfg['pipeline']['ollama_base_url']}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Print final solution to stdout ────────────────────────────────────────
    artifact = final_state.get("solution_artifact") or {}
    print("\n" + "="*60)
    print("  FINAL SOLUTION DESIGN")
    print("="*60)
    print(artifact.get("design_summary", "(no design summary)"))
    print()

    components = artifact.get("components", [])
    if components:
        print("Components:")
        for c in components:
            print(f"  • {c.get('name')}: {c.get('description')}")
    print()
    print(f"Stop reason   : {final_state.get('stop_reason')}")
    print(f"Final score   : {final_state.get('evaluation_score', 0):.2f}")
    print(f"Iterations run: {final_state.get('iteration')}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
