"""
graph/builder.py — LangGraph StateGraph construction.

Builds the compiled graph once and caches it.  The graph topology is:

    START
      └─► orchestrator
            ├─► end_user  ──┐
            ├─► policy    ──┤ (parallel fan-out)
            └─► software  ──┘
                  └─► aggregator
                        └─► coding
                              └─► evaluator
                                    ├─[continue]─► orchestrator   (loop)
                                    └─[stop]─────► synthesis
                                                      └─► END

Fan-out rationale:
  When a node has multiple outgoing edges, LangGraph executes all downstream
  nodes concurrently (as separate async tasks in the same event loop or as
  threaded workers depending on the runtime).  Each parallel node writes to a
  unique state key (driver_feedback / policy_feedback / software_feedback), so
  (driver_feedback is the state key used by the end_user node for historical reasons)
  there is no merge conflict.  The fan-in to aggregator happens automatically
  once all three upstream nodes have completed.

Loop rationale:
  The conditional edge from evaluator routes back to orchestrator when
  should_stop == False, creating a stateful cycle.  The iteration counter
  (incremented inside orchestrator_node) acts as the cycle guard.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.aggregator import aggregator_node
from agents.coding import coding_node
from agents.end_user import end_user_node
from agents.orchestrator import orchestrator_node
from agents.policy import policy_node
from agents.software import software_node
from agents.synthesis import synthesis_node
from evaluation.evaluator import evaluator_node, routing_function
from state import PipelineState

_compiled_graph = None  # module-level cache


def reset_graph() -> None:
    """Invalidate the compiled graph cache.

    Call this in tests or dev workflows that need a fresh graph build,
    e.g. after patching node functions or reconfiguring state.
    """
    global _compiled_graph
    _compiled_graph = None


def build_graph():
    """
    Construct and compile the pipeline StateGraph.

    Returns a compiled LangGraph runnable.  The graph is built once and
    the result is cached; subsequent calls return the cached graph.
    """
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    builder = StateGraph(PipelineState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("end_user",     end_user_node)
    builder.add_node("policy",       policy_node)
    builder.add_node("software",     software_node)
    builder.add_node("aggregator",   aggregator_node)
    builder.add_node("coding",       coding_node)
    builder.add_node("evaluator",    evaluator_node)
    builder.add_node("synthesis",    synthesis_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.add_edge(START, "orchestrator")

    # ── Fan-out: orchestrator → three parallel evaluator agents ───────────────
    builder.add_edge("orchestrator", "end_user")
    builder.add_edge("orchestrator", "policy")
    builder.add_edge("orchestrator", "software")

    # ── Fan-in: all three agents → aggregator (waits for all three) ───────────
    builder.add_edge("end_user",  "aggregator")
    builder.add_edge("policy",    "aggregator")
    builder.add_edge("software",  "aggregator")

    # ── Sequential: aggregator → coding → evaluator ───────────────────────────
    builder.add_edge("aggregator", "coding")
    builder.add_edge("coding",     "evaluator")

    # ── Conditional loop or termination ───────────────────────────────────────
    builder.add_conditional_edges(
        "evaluator",
        routing_function,
        {
            "continue": "orchestrator",  # loop back for another iteration
            "stop":     "synthesis",     # generate final report then end
        },
    )

    # ── Synthesis → END ───────────────────────────────────────────────────────
    builder.add_edge("synthesis", END)

    # No checkpointer: designed for single-run synchronous execution only.
    # Add a checkpointer here if resumable or async execution is needed.
    _compiled_graph = builder.compile()
    return _compiled_graph
