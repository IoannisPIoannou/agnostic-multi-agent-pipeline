# Agnostic Multi-Agent Pipeline

A domain-agnostic multi-agent system built with [LangGraph](https://github.com/langchain-ai/langgraph) that iteratively evaluates and refines solutions to any goal through structured multi-perspective feedback.

## How it works

The pipeline accepts a free-text goal and runs an iterative refinement loop until the solution converges or a quality threshold is reached.

```
START
  └─► orchestrator          (Gemini) — decomposes the goal, coordinates agents
        ├─► end_user   ─┐
        ├─► policy     ─┤  (Ollama, parallel) — evaluate from three perspectives
        └─► software   ─┘
              └─► aggregator        — confidence-weighted score, conflict detection
                    └─► coding      (Gemini) — refines the solution artifact
                          └─► evaluator      — checks convergence / stopping
                                ├─[continue]─► orchestrator   (loop)
                                └─[stop]─────► synthesis
                                                  └─► END     (Gemini) — final report
```

**Evaluator agents run in parallel on Ollama (local):**
- `end_user` — scores usability, clarity, cost, practicality from the primary user's perspective
- `policy` — scores safety, compliance, and systemic impact
- `software` — scores complexity, scalability, and maintainability

**Stopping conditions** (any one is sufficient):
- Score ≥ threshold (`score_threshold` in config)
- All active agents converge (score stable + no conflicts + all confident)
- Maximum iterations reached

---

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) running locally with `llama3.2:3b` pulled
- A [Gemini API key](https://aistudio.google.com/app/apikey)

### Install

```bash
git clone https://github.com/IoannisPIoannou/agnostic-multi-agent-pipeline.git
cd agnostic-multi-agent-pipeline
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# edit .env and add your Gemini API key:
# GEMINI_API_KEY=your-key-here
```

Pull the Ollama model:

```bash
ollama pull llama3.2:3b
```

### Run

```bash
python main.py
```

You will be prompted to enter a goal. Example:

```
Enter your goal: Design a decision-support system that recommends truck parking
locations based on distance, availability, and driver preferences.
```

Outputs are saved to `outputs/<run_id>/` as a markdown report and a JSON artifact.

---

## Configuration

All pipeline settings live in `config.yaml`:

| Key | Description |
|-----|-------------|
| `pipeline.max_iterations` | Hard cap on refinement cycles |
| `pipeline.score_threshold` | Stop when weighted score reaches this |
| `pipeline.convergence_delta` | Min score change to consider stable |
| `pipeline.conflict_threshold` | Score deviation that counts as an agent conflict |
| `agent_weights` | Base weights for driver / policy / software agents |
| `models.*` | Model names and temperatures for each agent |

---

## Experimental evaluation

Four experiments are available to benchmark the pipeline:

| Experiment | What it measures |
|------------|-----------------|
| `convergence` | Score stability across N independent runs |
| `ablation` | Each evaluator agent's contribution (disable one at a time) |
| `baseline` | Pipeline vs. single-LLM comparison via a blind LLM judge |
| `runtime` | Wall time, API call counts, and estimated cost |

Run any experiment with:

```bash
python experiments/run_experiment.py --config experiments/configs/convergence_example.yaml
python experiments/run_experiment.py --config experiments/configs/ablation_example.yaml
python experiments/run_experiment.py --config experiments/configs/baseline_example.yaml
python experiments/run_experiment.py --config experiments/configs/runtime_example.yaml
```

Summarize an existing result without re-running:

```bash
python experiments/run_experiment.py --summarize experiments/results/<id>/results.json
```

Results are saved to `experiments/results/` as JSON.

---

## Project structure

```
├── main.py                        # CLI entry point
├── config.yaml                    # Pipeline configuration
├── state.py                       # Shared LangGraph state (TypedDict)
├── agents/
│   ├── orchestrator.py            # Goal decomposition and coordination (Gemini)
│   ├── end_user.py                # End-user perspective evaluator (Ollama)
│   ├── policy.py                  # Policy / compliance evaluator (Ollama)
│   ├── software.py                # Software feasibility evaluator (Ollama)
│   ├── aggregator.py              # Confidence-weighted score aggregation
│   ├── coding.py                  # Solution design and refinement (Gemini)
│   ├── synthesis.py               # Final report generation (Gemini)
│   └── schemas.py                 # Pydantic output schemas for all agents
├── evaluation/
│   └── evaluator.py               # Convergence detection and routing
├── graph/
│   ├── builder.py                 # LangGraph StateGraph construction
│   └── runner.py                  # Pipeline execution and persistence
├── utils/
│   ├── config_loader.py
│   ├── logging_config.py
│   └── persistence.py
└── experiments/
    ├── run_experiment.py           # CLI entry point for experiments
    ├── configs/                    # YAML experiment configs + Pydantic schemas
    ├── runners/                    # convergence, ablation, baseline, runtime
    ├── judges/                     # Blind A/B LLM judge
    ├── baselines/                  # Single-LLM baseline generator
    └── analysis/                   # Result summarization utilities
```
