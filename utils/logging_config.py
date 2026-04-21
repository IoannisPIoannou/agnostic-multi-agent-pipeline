"""
utils/logging_config.py — Structured JSON logging setup.

Provides:
  - setup_logging(): configures root logger with JSON formatter → file + console
  - get_logger(name): returns a module-level logger
  - make_log_entry(): builds a structured dict for inclusion in state["log_entries"]

JSON fields on every log record:
    timestamp, level, logger, node, run_id, iteration, message, [extra fields]
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class _JsonFormatter(logging.Formatter):
    """Formats each LogRecord as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        base = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge any extra fields attached via logger.info(..., extra={...})
        for key, value in record.__dict__.items():
            if key not in (
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "id", "levelname", "levelno",
                "lineno", "module", "msecs", "message", "msg", "name",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName",
            ):
                base[key] = value
        return json.dumps(base, default=str)


def setup_logging(cfg: dict) -> None:
    """
    Configure the root logger.  Call once at application startup.

    Args:
        cfg: The full config dict (reads cfg["logging"]["level"] and cfg["logging"]["file"]).
    """
    log_cfg = cfg.get("logging", {})
    level_name = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = log_cfg.get("file", "outputs/pipeline.log")

    # Ensure the output directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    formatter = _JsonFormatter()

    # Console handler — human-readable fallback using standard formatter
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    )

    # File handler — structured JSON
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)


class _PipelineLogger(logging.Logger):
    """Logger subclass that silently drops 'message' from extra= dicts.

    make_log_entry() returns dicts containing a 'message' key. Passing
    those dicts directly as extra= to logger calls conflicts with the
    reserved LogRecord.message attribute and raises KeyError in Python 3.12+.
    This subclass strips the key before delegating to makeRecord.
    """

    def makeRecord(self, name, level, fn, lno, msg, args, exc_info,
                   func=None, extra=None, sinfo=None):
        if extra and "message" in extra:
            extra = {k: v for k, v in extra.items() if k != "message"}
        return super().makeRecord(name, level, fn, lno, msg, args, exc_info,
                                  func, extra, sinfo)


# Register before any named logger is created so all getLogger() calls
# in agent modules produce _PipelineLogger instances.
logging.setLoggerClass(_PipelineLogger)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.  Must call setup_logging() first."""
    return logging.getLogger(name)


def make_log_entry(
    event: str,
    node: str,
    run_id: str,
    iteration: int,
    message: str,
    **extra: Any,
) -> dict:
    """
    Build a structured log dict suitable for appending to state["log_entries"].

    These entries are also written to the final.json diagnostics section.
    """
    return {
        "event": event,
        "node": node,
        "run_id": run_id,
        "iteration": iteration,
        "message": message,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        **extra,
    }
