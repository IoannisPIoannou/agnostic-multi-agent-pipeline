"""
agents/schemas.py — Pydantic models for all structured agent outputs.

Every agent returns one of these models. Using Pydantic ensures:
  - Type-safe structured outputs via LangChain's with_structured_output()
  - Automatic validation of LLM responses
  - Clean serialisation to dict for state storage

Design note: Ollama agents use simpler schemas (fewer nested types) to
maximise JSON parse reliability from local models.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Orchestrator ──────────────────────────────────────────────────────────────

class SubProblem(BaseModel):
    id: str = Field(description="Short unique identifier, e.g. 'SP-1'")
    title: str = Field(description="Brief name for this subproblem")
    description: str = Field(description="Clear description of what needs to be solved")
    priority: int = Field(ge=1, le=5, description="1 = highest priority")
    relevant_agents: list[str] = Field(
        description="Which agents should focus on this: driver, policy, software"
    )


class OrchestratorOutput(BaseModel):
    """Complete output from the orchestrator agent."""
    subproblems: list[SubProblem] = Field(description="Decomposed subproblems")
    constraints: list[str] = Field(description="Hard constraints the solution must respect")
    success_criteria: list[str] = Field(description="Measurable criteria for a good solution")
    coordination_notes: str = Field(description="High-level coordination strategy")
    driver_instruction: str = Field(description="Specific evaluation mandate for the driver agent")
    policy_instruction: str = Field(description="Specific evaluation mandate for the policy agent")
    software_instruction: str = Field(description="Specific evaluation mandate for the software agent")
    revision_strategy: str | None = Field(
        default=None,
        description="On iteration > 1: what specifically to improve this cycle"
    )
    priority_focus: list[str] = Field(
        default_factory=list,
        description="Ordered list of improvement areas for the coding agent"
    )


# ── Evaluator agents (kept minimal for Ollama reliability) ────────────────────

class DriverFeedback(BaseModel):
    """Driver / end-user perspective evaluation."""
    usability_score: float = Field(ge=0, le=10, description="How easy is the solution to use?")
    clarity_score: float = Field(ge=0, le=10, description="How clear is the information presented?")
    cost_score: float = Field(ge=0, le=10, description="How affordable / cost-effective?")
    practicality_score: float = Field(ge=0, le=10, description="How well does it fit real-world usage?")
    confidence: float = Field(ge=0, le=1, description="Agent's confidence in its own assessment (0=uncertain, 1=very confident)")
    overall_summary: str = Field(description="2-3 sentence assessment from the driver perspective")
    key_concerns: list[str] = Field(description="Top concerns from the user's point of view")
    recommendations: list[str] = Field(description="Concrete improvements to address concerns")


class PolicyFeedback(BaseModel):
    """Policy / stakeholder / systemic evaluation."""
    safety_score: float = Field(ge=0, le=10, description="How safe is the proposed system?")
    compliance_score: float = Field(ge=0, le=10, description="How well does it meet regulatory / policy requirements?")
    system_impact_score: float = Field(ge=0, le=10, description="How positive is the systemic / societal impact?")
    confidence: float = Field(ge=0, le=1, description="Agent's confidence in its own assessment")
    overall_summary: str = Field(description="2-3 sentence assessment from the policy perspective")
    key_concerns: list[str] = Field(description="Top policy or safety concerns")
    recommendations: list[str] = Field(description="Concrete improvements to address concerns")


class SoftwareFeedback(BaseModel):
    """Software feasibility evaluation."""
    complexity_score: float = Field(ge=0, le=10, description="Implementation complexity (10 = very simple)")
    scalability_score: float = Field(ge=0, le=10, description="How well does it scale?")
    maintainability_score: float = Field(ge=0, le=10, description="How maintainable is the design?")
    confidence: float = Field(ge=0, le=1, description="Agent's confidence in its own assessment")
    overall_summary: str = Field(description="2-3 sentence assessment from the software perspective")
    key_concerns: list[str] = Field(description="Top implementation or architecture concerns")
    recommendations: list[str] = Field(description="Concrete improvements to address concerns")


# ── Coding agent ──────────────────────────────────────────────────────────────

class Component(BaseModel):
    name: str = Field(description="Component name")
    description: str = Field(description="What this component does")
    responsibility: str = Field(description="Its specific role in the overall system")


class EvaluationCriterion(BaseModel):
    name: str = Field(description="Criterion name")
    description: str = Field(description="What it measures")
    metric: str = Field(description="How to measure it (e.g. 'avg detour < 5 km')")


class CodingOutput(BaseModel):
    """Solution design produced or updated by the coding agent."""
    design_summary: str = Field(description="High-level description of the proposed solution")
    components: list[Component] = Field(description="Key components of the system")
    decision_logic: str = Field(description="Core decision-making algorithm or logic")
    evaluation_criteria: list[EvaluationCriterion] = Field(description="Measurable criteria to validate the solution")
    implementation_notes: str = Field(description="Key technical or practical notes for implementors")
    addressed_concerns: list[str] = Field(description="Which agent concerns were explicitly addressed in this revision")
