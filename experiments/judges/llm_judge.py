"""
experiments/judges/llm_judge.py — LLM-based solution quality judge.

Takes two solution artifacts (pipeline vs baseline) and asks a judge LLM to
score each on configurable criteria, blind to which is which (A/B labelling).

The judge sees both solutions anonymised as "Solution A" and "Solution B" to
reduce positional bias. The mapping (which is pipeline, which is baseline) is
recorded in the result but not shown to the judge.

When pipeline_text is provided (e.g. the synthesis final_report), both sides
are presented as prose blocks so the judge compares polished outputs rather than
raw structured JSON fields. Otherwise the structured artifact fields are used.

Output: JudgementResult with per-criterion scores and an overall preference.

Validity note:
    The judge and both solutions use the same underlying model family
    (gemini-2.5-flash). Same-model judges may have stylistic preferences that
    favour their own output style. For higher confidence, re-run with a
    different judge model. Each experiment run is a single comparison — treat
    results as indicative, not statistically robust.
"""

from __future__ import annotations

import os
import random
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from utils.logging_config import get_logger

logger = get_logger(__name__)


# ── Judge output schema ───────────────────────────────────────────────────────

class CriterionScore(BaseModel):
    criterion:   str
    score_a:     float = Field(ge=1, le=10, description="Score for Solution A")
    score_b:     float = Field(ge=1, le=10, description="Score for Solution B")
    rationale:   str   = Field(description="One sentence explaining the scores")


class JudgementResult(BaseModel):
    criterion_scores:  list[CriterionScore]
    preferred:         Literal["A", "B"] = Field(description="Must be exactly 'A' or 'B'")
    preference_reason: str   = Field(description="2-3 sentences explaining preference")
    overall_score_a:   float = Field(ge=1, le=10)
    overall_score_b:   float = Field(ge=1, le=10)


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM = """\
You are an impartial expert evaluator assessing two proposed system designs.
Your job is to score each solution on specific criteria and state which you prefer.

Be objective. Score independently on each criterion. Do not favour length or complexity.
Respond with a valid JSON object only — no markdown, no preamble.
"""

# Used when both sides are presented as structured artifact fields.
_USER_TEMPLATE = """\
## Problem Statement
{goal}

## Solution A
### Design Summary
{summary_a}

### Components
{components_a}

### Decision Logic
{decision_logic_a}

### Implementation Notes
{impl_notes_a}

## Solution B
### Design Summary
{summary_b}

### Components
{components_b}

### Decision Logic
{decision_logic_b}

### Implementation Notes
{impl_notes_b}

## Evaluation Criteria
Score each solution on the following criteria from 1 (very poor) to 10 (excellent):
{criteria_list}

For each criterion, provide a score for Solution A, a score for Solution B, and a
one-sentence rationale. Then provide an overall score for each and your overall preference.
"""

# Used when pipeline_text is provided — both sides shown as prose blocks so the
# comparison is between polished deliverables rather than raw structured fields.
_USER_TEMPLATE_TEXT = """\
## Problem Statement
{goal}

## Solution A
{text_a}

## Solution B
{text_b}

## Evaluation Criteria
Score each solution on the following criteria from 1 (very poor) to 10 (excellent):
{criteria_list}

For each criterion, provide a score for Solution A, a score for Solution B, and a
one-sentence rationale. Then provide an overall score for each and your overall preference.
"""


def _format_components(components: list[dict]) -> str:
    return "\n".join(
        f"  - {c.get('name', '?')}: {c.get('description', '')} "
        f"[{c.get('responsibility', '')}]"
        for c in components
    ) or "  (none)"


def _format_artifact_as_text(artifact: dict) -> str:
    """Convert a structured solution artifact dict to a readable prose block."""
    parts = []
    if summary := artifact.get("design_summary"):
        parts.append(f"**Design Summary**\n{summary}")
    if artifact.get("components"):
        parts.append(f"**Components**\n{_format_components(artifact['components'])}")
    if logic := artifact.get("decision_logic"):
        parts.append(f"**Decision Logic**\n{logic}")
    if notes := artifact.get("implementation_notes"):
        parts.append(f"**Implementation Notes**\n{notes}")
    return "\n\n".join(parts) or "N/A"


def judge_solutions(
    goal:               str,
    pipeline_artifact:  dict,
    baseline_artifact:  dict,
    criteria:           list[str],
    model:              str = "gemini-2.5-flash",
    pipeline_text:      str | None = None,
) -> dict:
    """
    Compare pipeline and baseline solutions using an LLM judge.

    Solutions are randomly assigned to A/B slots to reduce positional bias.

    If pipeline_text is provided (e.g. the synthesis final_report), both sides
    are presented as prose blocks via _USER_TEMPLATE_TEXT. Otherwise the
    structured artifact fields are used via _USER_TEMPLATE.

    Returns a dict with judge scores, preference, and the A/B assignment.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")

    # Random assignment to reduce positional bias
    if random.random() < 0.5:
        slot_a, slot_b = pipeline_artifact, baseline_artifact
        a_is_pipeline  = True
    else:
        slot_a, slot_b = baseline_artifact, pipeline_artifact
        a_is_pipeline  = False

    criteria_list = "\n".join(f"  - {c}" for c in criteria)

    if pipeline_text is not None:
        # Prose comparison: pipeline uses synthesis final_report,
        # baseline uses its structured artifact formatted as text.
        baseline_text = _format_artifact_as_text(baseline_artifact)
        text_a = pipeline_text  if a_is_pipeline else baseline_text
        text_b = baseline_text  if a_is_pipeline else pipeline_text
        user_content = _USER_TEMPLATE_TEXT.format(
            goal=goal,
            text_a=text_a,
            text_b=text_b,
            criteria_list=criteria_list,
        )
    else:
        user_content = _USER_TEMPLATE.format(
            goal=goal,
            summary_a=slot_a.get("design_summary", "N/A"),
            components_a=_format_components(slot_a.get("components", [])),
            decision_logic_a=slot_a.get("decision_logic", "N/A"),
            impl_notes_a=slot_a.get("implementation_notes", "N/A"),
            summary_b=slot_b.get("design_summary", "N/A"),
            components_b=_format_components(slot_b.get("components", [])),
            decision_logic_b=slot_b.get("decision_logic", "N/A"),
            impl_notes_b=slot_b.get("implementation_notes", "N/A"),
            criteria_list=criteria_list,
        )

    logger.info(
        f"[judge] Comparing solutions using {model} | "
        f"mode={'text' if pipeline_text else 'structured'} | criteria={criteria}",
        extra={"node": "judge"},
    )

    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0.1,
        google_api_key=api_key,
    )
    structured_llm = llm.with_structured_output(JudgementResult)
    messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=user_content)]
    judgement: JudgementResult = structured_llm.invoke(messages)

    # Normalise to bare "A" or "B" before comparing — guards against the LLM
    # returning "Solution A", "option a", etc. despite the Literal constraint.
    preferred_slot     = judgement.preferred.strip().upper()[0]
    pipeline_slot      = "A" if a_is_pipeline else "B"
    pipeline_preferred = preferred_slot == pipeline_slot

    result = {
        "judge_model":           model,
        "criteria":              criteria,
        "a_is_pipeline":         a_is_pipeline,
        "pipeline_slot":         pipeline_slot,
        "preferred_slot":        preferred_slot,
        "pipeline_preferred":    pipeline_preferred,
        "preference_reason":     judgement.preference_reason,
        "used_final_report":     pipeline_text is not None,
        "overall_score_pipeline": (
            judgement.overall_score_a if a_is_pipeline else judgement.overall_score_b
        ),
        "overall_score_baseline": (
            judgement.overall_score_b if a_is_pipeline else judgement.overall_score_a
        ),
        "criterion_scores": [
            {
                "criterion":      cs.criterion,
                "score_pipeline": cs.score_a if a_is_pipeline else cs.score_b,
                "score_baseline": cs.score_b if a_is_pipeline else cs.score_a,
                "rationale":      cs.rationale,
            }
            for cs in judgement.criterion_scores
        ],
    }

    logger.info(
        f"[judge] Pipeline score={result['overall_score_pipeline']:.1f} | "
        f"Baseline score={result['overall_score_baseline']:.1f} | "
        f"Preferred={'pipeline' if pipeline_preferred else 'baseline'}",
        extra={"node": "judge"},
    )
    return result
