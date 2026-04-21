"""
experiments/baselines/simple_llm.py — Single-LLM baseline solution generator.

Generates a solution artifact using a single Gemini call with no iterative
refinement, no multi-agent evaluation, and no structured feedback loop.
This is the "naive" baseline the pipeline is benchmarked against.

The output format mirrors SolutionArtifact from agents/schemas.py so the
judge can compare both solutions on equal structural footing.
"""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from utils.logging_config import get_logger

logger = get_logger(__name__)

_SYSTEM = """\
You are a solution architect.
Given a problem statement, design a clear, practical, implementable solution.

Your output must be a structured JSON object with these exact fields:
  - design_summary: high-level description of the solution (2-3 sentences)
  - components: list of {name, description, responsibility} objects
  - decision_logic: the core algorithm or decision-making approach
  - implementation_notes: key technical notes for implementors
  - evaluation_criteria: list of {name, description, metric} objects
  - addressed_concerns: leave as empty list []

Be specific, practical, and complete.
Respond with only a valid JSON object — no markdown, no explanation.
"""

_USER_TEMPLATE = """\
## Problem Statement
{goal}

Design a complete solution addressing this goal.
"""


class BaselineSolutionArtifact(BaseModel):
    """Mirrors CodingOutput / SolutionArtifact for structural comparability."""
    design_summary:      str
    components:          list[dict]
    decision_logic:      str
    implementation_notes:str
    evaluation_criteria: list[dict]
    addressed_concerns:  list[str] = Field(default_factory=list)


def generate_baseline(goal: str, model: str = "gemini-2.5-flash") -> dict:
    """
    Generate a single-pass solution using one LLM call.

    Returns a dict matching the solution_artifact structure used by the pipeline,
    with a 'baseline': True marker added for identification.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")

    logger.info(
        f"[baseline] Generating baseline solution with {model}",
        extra={"node": "baseline"},
    )

    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0.3,
        google_api_key=api_key,
    )
    structured_llm = llm.with_structured_output(BaselineSolutionArtifact)
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=_USER_TEMPLATE.format(goal=goal)),
    ]

    result: BaselineSolutionArtifact = structured_llm.invoke(messages)
    artifact = result.model_dump()
    artifact["baseline"] = True
    artifact["model"]    = model

    logger.info(
        f"[baseline] Generated: {len(artifact['components'])} components",
        extra={"node": "baseline"},
    )
    return artifact
