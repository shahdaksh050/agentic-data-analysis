"""
Abstract Base Tool for the Agentic Data Analysis System.

Every tool in src/tools/ must subclass BaseTool.
Tools must:
  - Be deterministic (same inputs → same outputs).
  - Return a structured dict (JSON-serializable).
  - Raise ToolExecutionError on failure (never crash silently).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from src.core.memory import ToolResult


class ToolExecutionError(Exception):
    """Raised when a tool encounters an unrecoverable error."""
    pass


class BaseTool(ABC):
    """
    Abstract base class for all analysis tools.

    Each subclass represents one deterministic, type-safe capability
    of the agent's execution layer.
    """

    #: Unique name used by the LLM to call this tool (e.g. "clean_data").
    name: str
    #: Human-readable description injected into LLM system prompts.
    description: str

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Execute the tool with the provided parameters.

        Args:
            **kwargs: Tool-specific parameters as returned by the LLM JSON plan.

        Returns:
            A JSON-serializable dict of results. Must always include a "summary"
            key with a one-sentence human-readable description of the outcome.

        Raises:
            ToolExecutionError: If the tool cannot complete successfully.
        """
        ...

    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """
        Return the tool's parameter schema for LLM prompt injection.

        Returns:
            A dict describing parameters: {name: {type, description, required}}.
        """
        ...

    def run(self, **kwargs: Any) -> ToolResult:
        """
        Wraps execute() with timing, error handling, and ToolResult packaging.

        Always call this method rather than execute() directly.
        """
        start_ms = time.monotonic() * 1000
        try:
            output = self.execute(**kwargs)
            elapsed = time.monotonic() * 1000 - start_ms
            return ToolResult(
                tool_name=self.name,
                status="success",
                output=output,
                execution_time_ms=elapsed,
            )
        except ToolExecutionError as exc:
            elapsed = time.monotonic() * 1000 - start_ms
            return ToolResult(
                tool_name=self.name,
                status="error",
                output={},
                error_message=str(exc),
                execution_time_ms=elapsed,
            )
        except Exception as exc:  # noqa: BLE001
            # Catch-all: surface unexpected errors as structured results
            elapsed = time.monotonic() * 1000 - start_ms
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
            f"  - {k}: ({v.get('type', 'any')}) {v.get('description', '')} "
            f"{'[required]' if v.get('required') else '[optional]'}"
            for k, v in schema.items()
        )
        return f"Tool: {self.name}\nDescription: {self.description}\nParameters:\n{params}"
