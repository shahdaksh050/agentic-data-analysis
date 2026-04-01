"""
Memory System for the Agentic Data Analysis System.

Maintains persistent state across reasoning-execution cycles,
storing dataset metadata, tool results, and analysis history.
Implements the 'external environment' concept from RLM paradigm.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@dataclass
class DatasetMetadata:
    """Structured metadata extracted from an uploaded dataset."""

    file_path: str
    row_count: int
    column_count: int
    columns: dict[str, str]          # {col_name: dtype}
    missing_values: dict[str, int]   # {col_name: missing_count}
    numerical_cols: list[str]
    categorical_cols: list[str]
    target_column: str | None = None
    task_type: str | None = None     # "classification", "regression", "clustering"
    summary_stats: dict[str, Any] = field(default_factory=dict)

    def to_prompt_string(self) -> str:
        """Render a compact, LLM-friendly metadata summary."""
        lines = [
            f"Dataset: {Path(self.file_path).name}",
            f"Shape: {self.row_count} rows × {self.column_count} columns",
            f"Numerical columns ({len(self.numerical_cols)}): {', '.join(self.numerical_cols[:10])}",
            f"Categorical columns ({len(self.categorical_cols)}): {', '.join(self.categorical_cols[:10])}",
            f"Missing values: {sum(self.missing_values.values())} total cells",
            f"Target: {self.target_column or 'Not identified'}",
            f"Task type: {self.task_type or 'Not identified'}",
        ]
        return "\n".join(lines)


@dataclass
class ToolResult:
    """The structured output of a single tool execution."""

    tool_name: str
    status: str          # "success" | "error" | "skipped"
    output: dict[str, Any]
    error_message: str | None = None
    execution_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "output": self.output,
            "error": self.error_message,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp,
        }


@dataclass
class AnalysisStep:
    """A single step in the analysis plan, as planned by the LLM."""

    step_number: int
    tool_name: str
    parameters: dict[str, Any]
    rationale: str
    result: ToolResult | None = None

    def is_complete(self) -> bool:
        return self.result is not None and self.result.status == "success"


class MemorySystem:
    """
    Central state store for the agent's working memory.

    Implements the 'external environment' (REPL state) from the RLM paradigm:
    - Dataset metadata is stored here, NOT in the LLM context window.
    - Only compact summaries are passed to the LLM at each reasoning call.
    - All tool results and intermediate analysis steps are accumulated here.
    """

    def __init__(self, persist_path: str | None = None) -> None:
        """
        Initialize the memory system.

        Args:
            persist_path: Optional path to a JSON file for persistent storage.
        """
        self.dataset_metadata: DatasetMetadata | None = None
        self.analysis_plan: list[AnalysisStep] = []
        self.tool_results: list[ToolResult] = []
        self.iteration_count: int = 0
        self.session_id: str = str(int(time.time()))
        self.persist_path: str | None = persist_path
        self._additional_context: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Dataset Metadata
    # ------------------------------------------------------------------ #

    def store_dataset_metadata(self, metadata: DatasetMetadata) -> None:
        """Store extracted dataset metadata as the foundational context."""
        self.dataset_metadata = metadata
        console.print(
            Panel(
                f"[bold green]✓ Dataset metadata stored[/]\n{metadata.to_prompt_string()}",
                title="[bold cyan]Memory System",
                border_style="cyan",
            )
        )

    def get_metadata_prompt(self) -> str:
        """Return compact metadata string suitable for LLM context."""
        if not self.dataset_metadata:
            raise ValueError("No dataset metadata available. Ingest a dataset first.")
        return self.dataset_metadata.to_prompt_string()

    # ------------------------------------------------------------------ #
    # Analysis Plan
    # ------------------------------------------------------------------ #

    def store_analysis_plan(self, steps: list[AnalysisStep]) -> None:
        """Store the LLM-generated analysis plan."""
        self.analysis_plan = steps
        console.print(f"[cyan]📋 Analysis plan stored: {len(steps)} steps[/]")

    def get_pending_steps(self) -> list[AnalysisStep]:
        """Return steps that have not yet been successfully executed."""
        return [s for s in self.analysis_plan if not s.is_complete()]

    def mark_step_complete(self, step_number: int, result: ToolResult) -> None:
        """Record the result of a completed analysis step."""
        for step in self.analysis_plan:
            if step.step_number == step_number:
                step.result = result
                break

    # ------------------------------------------------------------------ #
    # Tool Results
    # ------------------------------------------------------------------ #

    def append_tool_result(self, result: ToolResult) -> None:
        """Append a tool result to the accumulated history."""
        self.tool_results.append(result)
        status_color = "green" if result.status == "success" else "red"
        console.print(
            f"  [{status_color}]{'✓' if result.status == 'success' else '✗'} "
            f"{result.tool_name}[/] → {result.status} "
            f"[dim]({result.execution_time_ms:.0f}ms)[/]"
        )

    def get_results_summary(self) -> str:
        """Compact summary of all tool results for LLM context injection."""
        if not self.tool_results:
            return "No tool results yet."
        lines = []
        for r in self.tool_results:
            if r.status == "success":
                # Only inject key output fields, not full raw data
                key_outputs = {k: v for k, v in r.output.items() if k != "raw_data"}
                lines.append(f"[{r.tool_name}] → {json.dumps(key_outputs, default=str)[:300]}")
            else:
                lines.append(f"[{r.tool_name}] → ERROR: {r.error_message}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Generic Context Store (for RLM sub-call environments)
    # ------------------------------------------------------------------ #

    def set_context(self, key: str, value: Any) -> None:
        """Store arbitrary key-value context for RLM sub-calls."""
        self._additional_context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Retrieve a stored context value."""
        return self._additional_context.get(key, default)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self) -> None:
        """Serialize memory state to disk (optional persistence)."""
        if not self.persist_path:
            return
        state = {
            "session_id": self.session_id,
            "iteration_count": self.iteration_count,
            "dataset_metadata": (
                self.dataset_metadata.__dict__ if self.dataset_metadata else None
            ),
            "tool_results": [r.to_dict() for r in self.tool_results],
            "additional_context": self._additional_context,
        }
        Path(self.persist_path).write_text(json.dumps(state, indent=2, default=str))

    # ------------------------------------------------------------------ #
    # Display Helpers
    # ------------------------------------------------------------------ #

    def print_status(self) -> None:
        """Render a Rich status table to the terminal."""
        table = Table(title="Memory State", show_header=True, header_style="bold magenta")
        table.add_column("Key", style="cyan")
        table.add_column("Value")

        table.add_row("Session ID", self.session_id)
        table.add_row("Iterations", str(self.iteration_count))
        table.add_row("Dataset Loaded", "✓" if self.dataset_metadata else "✗")
        table.add_row("Analysis Steps", str(len(self.analysis_plan)))
        table.add_row("Tool Results", str(len(self.tool_results)))
        pending = len(self.get_pending_steps())
        table.add_row("Pending Steps", str(pending))

        console.print(table)
