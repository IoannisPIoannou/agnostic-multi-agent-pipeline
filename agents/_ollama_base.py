"""
agents/_ollama_base.py — Shared helpers for Ollama-backed evaluator agents.

Provides:
  _call_ollama()   — invoke with structured output; retry once; safe fallback
  _safe_fallback() — neutral, low-confidence response when parsing fails completely
  extract_json_block() — brace-depth-aware JSON extraction via raw_decode

Design principle: the pipeline must NEVER crash due to a local model parse
failure.  A safe fallback (all scores = 5.0, confidence = 0.0) is returned
instead, which:
  - Keeps the aggregator from biasing the result (neutral scores)
  - Triggers weight fallback in the aggregator (confidence = 0.0)
  - Prevents false convergence (confident_enough = False)
  - Gets surfaced in log_entries with event="parse_error" / "fallback_activated"
"""

from __future__ import annotations

import json
import logging
import os
import typing
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ValidationError

from utils.logging_config import make_log_entry

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Parse-level errors: distinguishable from connection/runtime failures
_PARSE_ERRORS = (json.JSONDecodeError, ValueError, ValidationError)


def extract_json_block(text: str) -> str:
    """Extract the first valid JSON object from arbitrary text.

    Uses json.JSONDecoder.raw_decode to respect brace depth, so nested objects
    and multiple JSON fragments in the same string are handled correctly.
    """
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text, i)
                return json.dumps(obj)
            except json.JSONDecodeError:
                continue
    raise ValueError("No JSON object found in model response")


def _safe_fallback(schema: type[T], agent_name: str) -> T:
    """
    Return a neutral, zero-confidence instance of the schema.

    All *_score fields → 5.0 (neutral, won't skew weighted average)
    confidence → 0.0  (propagates failure signal through aggregator + convergence)

    Raises TypeError for unhandled field types so the guarded caller can log
    and re-raise rather than silently producing invalid data.
    """
    data: dict[str, Any] = {}
    for field_name, field_info in schema.model_fields.items():
        ann = field_info.annotation
        origin = typing.get_origin(ann)
        args = typing.get_args(ann)

        if ann is float:
            data[field_name] = 0.0 if field_name == "confidence" else 5.0
        elif ann is int:
            data[field_name] = 0
        elif ann is str:
            data[field_name] = f"[Parse failure — {agent_name} low-confidence fallback]"
        elif origin is list:
            data[field_name] = []
        elif origin is dict:
            data[field_name] = {}
        elif origin is typing.Union and type(None) in args:
            # Optional[X] — None is the safest neutral value
            data[field_name] = None
        else:
            raise TypeError(
                f"_safe_fallback: unhandled field type '{ann}' for field "
                f"'{field_name}' in schema '{schema.__name__}'. "
                "Add explicit handling to _safe_fallback."
            )
    return schema.model_validate(data)


def _call_ollama(
    llm: ChatOllama,
    messages: list,
    schema: type[T],
    agent_name: str,
    state_run_id: str,
    state_iteration: int,
) -> tuple[T, list[dict]]:
    """
    Attempt structured output from Ollama with one retry and a safe fallback.

    Returns:
        (result, log_entries_to_append)

    Attempt 0: llm.with_structured_output(schema)
    Attempt 1: plain invoke + manual JSON extraction + Pydantic parse
    Fallback:  _safe_fallback() — guarded; logs ERROR and re-raises if it fails
    """
    log_entries: list[dict] = []

    for attempt in range(2):
        try:
            if attempt == 0:
                result = llm.with_structured_output(schema).invoke(messages)
            else:
                # Retry: append a plain HumanMessage demanding strict JSON
                retry_msg = HumanMessage(
                    content=(
                        "IMPORTANT: Respond with ONLY a valid JSON object "
                        "matching the required schema.  No prose, no markdown."
                    )
                )
                raw = llm.invoke(messages + [retry_msg])
                json_str = extract_json_block(raw.content)
                result = schema.model_validate_json(json_str)

            return result, log_entries

        except _PARSE_ERRORS as exc:
            warning_msg = (
                f"[{agent_name}] JSON parse failed (attempt {attempt + 1}): {exc}"
            )
            logger.warning(warning_msg)
            log_entries.append(make_log_entry(
                event="parse_error",
                node=agent_name,
                run_id=state_run_id,
                iteration=state_iteration,
                message=warning_msg,
                attempt=attempt + 1,
                error=str(exc),
            ))

        except Exception as exc:  # noqa: BLE001 — connection/runtime errors
            warning_msg = (
                f"[{agent_name}] Connection/runtime error (attempt {attempt + 1}): {exc}"
            )
            logger.warning(warning_msg)
            log_entries.append(make_log_entry(
                event="connection_error",
                node=agent_name,
                run_id=state_run_id,
                iteration=state_iteration,
                message=warning_msg,
                attempt=attempt + 1,
                error=str(exc),
            ))

        if attempt == 1:
            # Both attempts failed — record activation, then return safe fallback
            logger.warning(
                f"[{agent_name}] Using safe low-confidence fallback after 2 failed attempts."
            )
            log_entries.append(make_log_entry(
                event="fallback_activated",
                node=agent_name,
                run_id=state_run_id,
                iteration=state_iteration,
                message=f"[{agent_name}] Returning safe fallback after 2 failed attempts.",
                attempt=2,
            ))
            try:
                return _safe_fallback(schema, agent_name), log_entries
            except Exception as fb_exc:
                logger.error(f"[{agent_name}] _safe_fallback itself failed: {fb_exc}")
                raise

    # Unreachable, but satisfies type checker
    return _safe_fallback(schema, agent_name), log_entries


def build_ollama_llm(agent_key: str) -> ChatOllama:
    """Construct a ChatOllama instance from config."""
    from utils.config_loader import get_config
    cfg = get_config()
    model_cfg = cfg["models"][agent_key]
    kwargs: dict = dict(
        model=model_cfg["model"],
        temperature=model_cfg["temperature"],
        base_url=cfg["pipeline"]["ollama_base_url"],
        format="json",
    )
    seed_str = os.environ.get("OLLAMA_SEED")
    if seed_str is not None:
        kwargs["seed"] = int(seed_str)
    return ChatOllama(**kwargs)
