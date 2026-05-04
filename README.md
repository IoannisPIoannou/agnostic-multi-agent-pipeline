# Agnostic Multi-Agent Pipeline

A domain-agnostic multi-agent system built with [LangGraph](https://github.com/langchain-ai/langgraph) that iteratively evaluates and refines solutions to any goal through structured multi-perspective feedback, with an optional hard audit layer for feedback quality control.

## How it works

The pipeline accepts a free-text goal and runs an iterative **Think–Critique–Refine** loop until the solution converges or a quality threshold is reached.

```
START
  └─► orchestrator                    (Gemini 2.5 Flash) — decomposes goal, coordinates agents
        ├─► end_user  ──► end_user_audit  ─┐
        ├─► policy    ──► policy_audit    ─┤  (Ollama, parallel) — evaluate + audit feedback
        └─► software  ──► software_audit  ─┘
                └─► aggregator               — confidence-weighted score, conflict resolution
                      └─► coding             (Gemini 2.5 Flash) — refines solution artifact
                            └─► independent_evaluator  (Gemini 2.5 Flash) — scores vs goal
                                  └─► evaluator        — convergence / stopping check
                                        ├─[continue]─► orchestrator   (loop)
                                        └─[stop]─────► synthesis
                                                          └─► END  (Gemini) — final report
```

### Evaluator agents (parallel, Ollama local)

| Agent | Perspective | Model |
|---|---|---|
| `end_user` | Usability, clarity, cost, practicality | Llama 3.2 (3B) |
| `policy` | Safety, compliance, system-level impact | Llama 3.2 (3B) |
| `software` | Complexity, scalability, maintainability | Qwen2.5-Coder (7B) |

### Audit layer (optional, between evaluators and aggregator)

Each audit agent independently checks the **quality of the feedback** produced by its paired evaluator — not the solution itself. Feedback that fails the audit is sent back for revision (up to `max_audit_revisions` times per iteration). If the audit still fails after max revisions, the feedback is forwarded with a failure flag so the aggregator is always informed.

| Audit Agent | Checks | Model |
|---|---|---|
| `end_user_audit` | Role adherence, grounding, specificity, score–text consistency | Llama 3.2 (3B) |
| `policy_audit` | Safety/compliance focus, governance relevance | Llama 3.2 (3B) |
| `software_audit` | Technical specificity, feasibility depth | Qwen2.5-Coder (7B) |

Disable the audit layer without changing any code:
```yaml
# config.yaml
audit_layer:
  enabled: false
```

### Independent evaluator

A dedicated Gemini 2.5 Flash agent scores the solution against the original user goal on five fixed criteria (goal completeness, feasibility, clarity, innovation, overall quality). This score — not the parallel evaluator feedback — drives the convergence decision.

### Stopping conditions (any one is sufficient)
- Score ≥ `score_threshold` (default: 7.5)
- Score stable across consecutive iterations (convergence)
- Maximum iterations reached (default: 5)

---

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) running locally
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
# edit .env and set:
# GEMINI_API_KEY=your-key-here
```

Pull the required Ollama models:

```bash
ollama pull llama3.2:3b
ollama pull qwen2.5-coder:7b-instruct
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

Outputs are saved to `outputs/<run_id>/` as a JSON artifact (`final.json`) and a markdown report.

---

## Configuration

All pipeline settings live in `config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `pipeline.max_iterations` | 5 | Hard cap on refinement cycles |
| `pipeline.score_threshold` | 7.5 | Stop when independent evaluator score reaches this |
| `pipeline.convergence_delta` | 0.5 | Min score change between iterations to be considered improving |
| `pipeline.conflict_threshold` | 2.0 | Score gap between agents that triggers conflict resolution |
| `agent_weights` | driver=0.35, policy=0.30, software=0.35 | Relative evaluator weights (confidence-adjusted at runtime) |
| `audit_layer.enabled` | true | Enable/disable the hard audit layer |
| `audit_layer.audit_approval_threshold` | 8.5 | Minimum audit score (0–10) to approve feedback |
| `audit_layer.max_audit_revisions` | 2 | Max revision loops per evaluator branch per iteration |
| `models.*` | see config.yaml | Model names and temperatures for each agent |

---

## Experimental evaluation

Five experiment types are available to benchmark the pipeline:

| Experiment | What it measures |
|------------|-----------------|
| `convergence` | Score stability and consistency across N independent runs |
| `ablation` | Each evaluator agent's individual contribution (disable one at a time) |
| `baseline` | Pipeline vs. single-LLM comparison via a blind LLM judge |
| `runtime` | Wall time, iteration counts, and cost breakdown |
| `prompt_specificity` | How output quality varies with vague / moderate / highly specific prompts |

### Running experiments

Run all core experiments across all domains:
```bash
python experiments/run_all_experiments.py
```

Run specific experiment types:
```bash
python experiments/run_prompt_specificity.py    # prompt specificity sensitivity
python experiments/threshold_sweep.py           # convergence threshold sweep
python experiments/baseline_all_domains.py      # baseline comparison across all domains
python experiments/run_ablation_3x.py           # ablation with 3 runs per variant
```

Run the full Audit ON vs Audit OFF comparison batch:
```bash
python experiments/run_audit_on_experiments.py
```

Results are saved to `experiments/results/` as JSON.

---

## Project structure

```
├── main.py                            # CLI entry point
├── config.yaml                        # All pipeline configuration
├── state.py                           # Shared LangGraph state (TypedDict)
├── agents/
│   ├── orchestrator.py                # Goal decomposition and coordination (Gemini)
│   ├── end_user.py                    # End-user perspective evaluator (Llama 3.2 3B)
│   ├── policy.py                      # Policy / compliance evaluator (Llama 3.2 3B)
│   ├── software.py                    # Software feasibility evaluator (Qwen2.5-Coder 7B)
│   ├── end_user_audit.py              # Audit agent for end-user feedback quality
│   ├── policy_audit.py                # Audit agent for policy feedback quality
│   ├── software_audit.py              # Audit agent for software feedback quality
│   ├── independent_evaluator.py       # Solution scoring against goal (Gemini)
│   ├── aggregator.py                  # Confidence-weighted aggregation + conflict resolution
│   ├── coding.py                      # Solution design and refinement (Gemini)
│   ├── synthesis.py                   # Final report generation (Gemini)
│   └── schemas.py                     # Pydantic output schemas for all agents
├── evaluation/
│   └── evaluator.py                   # Convergence detection and stopping logic
├── graph/
│   ├── builder.py                     # LangGraph StateGraph construction
│   └── runner.py                      # Pipeline execution and output persistence
├── utils/
│   ├── config_loader.py
│   ├── logging_config.py
│   └── persistence.py                 # Intermediate + final output persistence
└── experiments/
    ├── run_all_experiments.py          # Run all experiments across all domains
    ├── run_audit_on_experiments.py     # Audit ON vs OFF comparison batch
    ├── run_prompt_specificity.py       # Prompt specificity sensitivity study
    ├── run_ablation_3x.py              # Ablation with 3 runs per variant
    ├── baseline_all_domains.py         # Baseline comparison across all domains
    ├── threshold_sweep.py              # Convergence threshold sweep
    ├── configs/                        # Pydantic config schemas + example YAMLs
    ├── runners/                        # convergence, ablation, baseline, runtime, prompt_specificity
    ├── judges/                         # Blind A/B LLM judge
    └── baselines/                      # Single-LLM baseline generator
```
