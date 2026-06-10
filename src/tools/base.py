"""
Abstract Base Tool for the Agentic Data Analysis System.

Every tool in src/tools/ must subclass BaseTool.

Contract:
  - Deterministic: same inputs → same outputs (no randomness unless seeded).
  - Structured output: every execute() must return a dict with a "summary" key.
  - Typed: full PEP 484 type hints required.
  - Safe failure: raise ToolExecutionError on expected failures; never swallow
    exceptions silently.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from src.core.memory import ToolResult


class ToolExecutionError(Exception):
    """Raised when a tool encounters an unrecoverable, expected error."""
    pass


class BaseTool(ABC):
    """Abstract base class for all deterministic analysis tools."""

    #: Unique snake_case name referenced by the LLM in its JSON plans.
    name: str
    #: Human-readable description injected into LLM system prompts.
    description: str

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Execute the tool with the given parameters.

        Returns:
            JSON-serialisable dict.  Must always include:
              - "summary": one-sentence human-readable outcome description.
        Raises:
            ToolExecutionError: On expected, recoverable failures.
        """
        ...

    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """
        Return the parameter schema for LLM prompt injection.

        Returns:
            {param_name: {"type": str, "description": str, "required": bool}}
        """
        ...

    def run(self, **kwargs: Any) -> ToolResult:
        """
        Public wrapper — timing, error handling, ToolResult packaging.

        Always call this method rather than execute() directly.
        Wraps ToolExecutionError and any unexpected exception into a
        structured ToolResult so the agent loop never crashes.
        """
        start = time.monotonic()
        try:
            output = self.execute(**kwargs)
            elapsed = (time.monotonic() - start) * 1000
            return ToolResult(
                tool_name=self.name,
                status="success",
                output=output,
                execution_time_ms=elapsed,
            )
        except ToolExecutionError as exc:
            elapsed = (time.monotonic() - start) * 1000
            return ToolResult(
                tool_name=self.name,
                status="error",
                output={},
                error_message=str(exc),
                execution_time_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            return ToolResult(
                tool_name=self.name,
                status="error",
                output={},
                error_message=f"Unexpected error in {self.name}: {exc!r}",
                execution_time_ms=elapsed,
            )

    def to_prompt_description(self) -> str:
        """Format tool info for LLM system prompt injection."""
        schema = self.get_schema()
        params = "\n".join(
            f"  - {k} ({v.get('type', 'any')}) "
            f"{'[required]' if v.get('required') else '[optional]'}: "
            f"{v.get('description', '')}"
            for k, v in schema.items()
        )
        return (
            f"tool_name: {self.name}\n"
            f"description: {self.description}\n"
            f"parameters:\n{params}"
        )
