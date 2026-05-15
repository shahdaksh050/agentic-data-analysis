"""
Memory System for the Agentic Data Analysis System.

Maintains persistent state across all reasoning-execution cycles.
Implements the 'external environment' (REPL state) from the RLM paradigm:
  - Full dataset metadata lives here, NOT in the LLM context window.
  - Only compact, token-efficient summaries are passed to the LLM.
  - All tool results and analysis steps are accumulated and queryable.

Workflow position: used by every other layer — ingested first, read last.
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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DatasetMetadata:
    """Structured metadata extracted from an uploaded dataset."""

    file_path: str
    row_count: int
    column_count: int
    columns: dict[str, str]           # {col_name: dtype_string}
    missing_values: dict[str, int]    # {col_name: n_missing}
    numerical_cols: list[str]
    categorical_cols: list[str]
    target_column: str | None = None
    task_type: str | None = None      # "classification" | "regression" | "clustering"
    summary_stats: dict[str, Any] = field(default_factory=dict)
    class_balance: dict[str, int] = field(default_factory=dict)  # for classification targets
    high_cardinality_cols: list[str] = field(default_factory=list)
    column_nunique: dict[str, int] = field(default_factory=dict)   # {col: nunique count}

    # Confidence thresholds for auto-detection decisions
    _AUTODETECT_HIGH: float = 0.75   # proceed autonomously
    _AUTODETECT_LOW: float = 0.40    # prompt the user (CLI) or use best guess (UI)

    # ------------------------------------------------------------------
    def detect_target_with_confidence(self) -> tuple[str | None, float]:
        """
        Heuristic target detection returning (column_name, confidence).

        Scoring pipeline:
          1. Naming convention match — exact keywords, domain names, partial hints
          2. Statistical adjustments — cardinality, dtype, high-cardinality flag
          3. Positional fallback — last column when no keyword match found

        Returns:
            (column, confidence) where confidence ∈ [0.0, 1.0].
            (None, 0.0) when the dataset has no columns.
        """
        _PRIMARY = frozenset({"target", "label", "class", "outcome", "y"})
        _DOMAIN = frozenset({
            "churn", "survived", "fraud", "default", "purchased", "converted",
            "cancelled", "canceled", "clicked", "subscribed", "is_fraud",
            "is_churn", "is_default",
        })
        _NUMERIC_TARGETS = frozenset({
            "profit", "sales", "revenue", "price", "amount", "total", "cost",
            "score", "rating", "demand", "margin", "result", "status",
        })
        _PARTIAL = ("target", "label", "class", "outcome", "predict", "response")

        col_names = list(self.columns.keys())
        if not col_names:
            return None, 0.0

        lower_to_orig = {c.lower(): c for c in col_names}
        candidates: list[tuple[str, float]] = []

        for lower, original in lower_to_orig.items():
            base = 0.0
            if lower in _PRIMARY:
                base = 0.95
            elif lower in _DOMAIN:
                base = 0.90
            elif lower in _NUMERIC_TARGETS:
                base = 0.85
            elif any(h in lower for h in _PARTIAL):
                base = 0.75

            if base == 0.0:
                continue

            # Statistical adjustments
            nunique = self.column_nunique.get(original, -1)
            if nunique == 2:                          # binary → strong target signal
                base = min(1.0, base + 0.05)
            elif nunique > 100:                       # very high cardinality → likely ID
                base = max(0.0, base - 0.15)
            if original in self.high_cardinality_cols:  # >50 unique categoricals
                base = max(0.0, base - 0.20)
            if original in self.categorical_cols:       # categorical dtype → slight bonus
                base = min(1.0, base + 0.03)

            candidates.append((original, round(base, 4)))

        if not candidates:
            # Positional fallback — last column is the common ML convention
            last_col = col_names[-1]
            pos_score = 0.55
            nunique = self.column_nunique.get(last_col, -1)
            if nunique == 2:
                pos_score = min(0.70, pos_score + 0.15)
            elif nunique > 100:
                pos_score = max(0.25, pos_score - 0.15)
            if last_col in self.high_cardinality_cols:
                pos_score = max(0.20, pos_score - 0.20)
            candidates.append((last_col, round(pos_score, 4)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]

    def auto_detect_target(self) -> str | None:
        """Compatibility shim — delegates to detect_target_with_confidence()."""
        col, _ = self.detect_target_with_confidence()
        return col

    def infer_task_type(self) -> str | None:
        """
        Infer the ML task type from the target column.

        Rules
        -----
        - No target → try auto_detect_target(); if still none → EDA mode
        - Binary / low-cardinality integer (≤20 unique) → classification
        - High-cardinality numeric → regression
        """
        if not self.target_column:
            raise ValueError("No target column detected.")
        dtype = self.columns.get(self.target_column, "")
        if "int" in dtype or "bool" in dtype:
            return "classification"
        if "float" in dtype:
            return "regression"
        return "classification"  # default for object/string targets

    def to_prompt_string(self) -> str:
        """Render a compact, LLM-friendly metadata summary (≈200 tokens)."""
        missing_total = sum(self.missing_values.values())
        target_note = self.target_column or "None (perform full EDA — describe distributions, correlations, outliers, and key patterns)"
        lines = [
            f"Dataset: {Path(self.file_path).name}",
            f"Shape: {self.row_count:,} rows × {self.column_count} columns",
            f"Numerical ({len(self.numerical_cols)}): {', '.join(self.numerical_cols[:12])}",
            f"Categorical ({len(self.categorical_cols)}): {', '.join(self.categorical_cols[:8])}",
            f"Missing values: {missing_total:,} total cells across "
            f"{sum(1 for v in self.missing_values.values() if v > 0)} columns",
            f"Target: {target_note}",
            f"Task type: {self.task_type or 'Not identified'}",
        ]
        if self.class_balance:
            bal = ", ".join(f"{k}={v}" for k, v in list(self.class_balance.items())[:6])
            lines.append(f"Class balance: {bal}")
        if self.high_cardinality_cols:
            lines.append(f"High-cardinality columns: {', '.join(self.high_cardinality_cols)}")
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
    """A single step in the LLM-generated analysis plan."""

    step_number: int
    tool_name: str
    parameters: dict[str, Any]
    rationale: str
    result: ToolResult | None = None
    retry_count: int = 0

    def is_complete(self) -> bool:
        return self.result is not None and self.result.status == "success"

    def is_failed(self) -> bool:
        return self.result is not None and self.result.status == "error"


# ---------------------------------------------------------------------------
# Memory System
# ---------------------------------------------------------------------------

class MemorySystem:
    """
    Central state store — the RLM 'external environment'.

    Design principles (from Zhang et al., 2024):
      1. Context offloading: full data stays here; LLM gets summaries only.
      2. Persistent REPL state: survives across all reasoning iterations.
      3. Generic context store: supports arbitrary RLM sub-call variables.
    """

    def __init__(self, persist_path: str | None = None) -> None:
        self.dataset_metadata: DatasetMetadata | None = None
        self.analysis_plan: list[AnalysisStep] = []
        self.tool_results: list[ToolResult] = []
        self.iteration_count: int = 0
        self.session_id: str = str(int(time.time()))
        self.persist_path: str | None = persist_path
        self._ctx: dict[str, Any] = {}     # generic RLM sub-call context store

    # ------------------------------------------------------------------
    # Stage 1 — Dataset metadata
    # ------------------------------------------------------------------

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
        """Return compact metadata string for LLM context injection."""
        if not self.dataset_metadata:
            raise ValueError("No dataset metadata. Call store_dataset_metadata() first.")
        return self.dataset_metadata.to_prompt_string()

    # ------------------------------------------------------------------
    # Stage 2 — Analysis plan (LLM-generated)
    # ------------------------------------------------------------------

    def store_analysis_plan(self, steps: list[AnalysisStep]) -> None:
        self.analysis_plan = steps
        console.print(f"[cyan]📋 Plan stored: {len(steps)} step(s)[/]")

    def get_pending_steps(self) -> list[AnalysisStep]:
        return [s for s in self.analysis_plan if not s.is_complete()]

    def get_failed_steps(self) -> list[AnalysisStep]:
        return [s for s in self.analysis_plan if s.is_failed()]

    def mark_step_complete(self, step_number: int, result: ToolResult) -> None:
        for step in self.analysis_plan:
            if step.step_number == step_number:
                step.result = result
                break

    def increment_retry(self, step_number: int) -> None:
        for step in self.analysis_plan:
            if step.step_number == step_number:
                step.retry_count += 1
                break

    # ------------------------------------------------------------------
    # Stage 3 — Tool results (execution layer outputs)
    # ------------------------------------------------------------------

    def append_tool_result(self, result: ToolResult) -> None:
        self.tool_results.append(result)
        color = "green" if result.status == "success" else "red"
        icon = "✓" if result.status == "success" else "✗"
        console.print(
            f"  [{color}]{icon} {result.tool_name}[/] → {result.status} "
            f"[dim]({result.execution_time_ms:.0f} ms)[/]"
        )

    def get_results_summary(self, max_chars_per_result: int = 350) -> str:
        """
        Compact summary of all tool outputs for LLM context injection.

        Only non-raw-data fields are included to keep tokens low.
        """
        if not self.tool_results:
            return "No tool results yet."
        lines: list[str] = []
        for r in self.tool_results:
            if r.status == "success":
                slim = {k: v for k, v in r.output.items() if k not in {"raw_data", "dataframe"}}
                serialised = json.dumps(slim, default=str)[:max_chars_per_result]
                lines.append(f"[{r.tool_name}] SUCCESS → {serialised}")
            else:
                lines.append(f"[{r.tool_name}] ERROR → {r.error_message}")
        return "\n".join(lines)

    def get_last_result_for(self, tool_name: str) -> ToolResult | None:
        """Return the most recent result for a given tool name."""
        for r in reversed(self.tool_results):
            if r.tool_name == tool_name:
                return r
        return None

    # ------------------------------------------------------------------
    # Generic context store — RLM sub-call environments (Stage 6)
    # ------------------------------------------------------------------

    def set_context(self, key: str, value: Any) -> None:
        self._ctx[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        return self._ctx.get(key, default)

    def list_context_keys(self) -> list[str]:
        return list(self._ctx.keys())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        if not self.persist_path:
            return
        state = {
            "session_id": self.session_id,
            "iteration_count": self.iteration_count,
            "dataset_metadata": self.dataset_metadata.__dict__ if self.dataset_metadata else None,
            "tool_results": [r.to_dict() for r in self.tool_results],
            "additional_context": {k: v for k, v in self._ctx.items() if isinstance(v, (str, int, float, bool, list, dict))},
        }
        Path(self.persist_path).write_text(json.dumps(state, indent=2, default=str))

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def print_status(self) -> None:
        table = Table(title="Memory State", show_header=True, header_style="bold magenta")
        table.add_column("Key", style="cyan")
        table.add_column("Value")
        table.add_row("Session ID", self.session_id)
        table.add_row("Iterations", str(self.iteration_count))
        table.add_row("Dataset loaded", "✓" if self.dataset_metadata else "✗")
        table.add_row("Plan steps", str(len(self.analysis_plan)))
        table.add_row("Tool results", str(len(self.tool_results)))
        table.add_row("Pending steps", str(len(self.get_pending_steps())))
        table.add_row("Failed steps", str(len(self.get_failed_steps())))
        table.add_row("Context keys", str(len(self._ctx)))
        console.print(table)
