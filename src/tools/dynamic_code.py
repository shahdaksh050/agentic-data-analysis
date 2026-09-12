"""
Dynamic Code Execution Tool — Execution Layer.

Wraps src.core.sandbox.run_sandboxed as a BaseTool so the existing
profile-driven planner can select it like any other tool. Additive:
use only for questions no other tool in the registry answers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.sandbox import run_sandboxed
from src.tools.base import BaseTool, ToolExecutionError


class DynamicCodeExecutionTool(BaseTool):
    """Execute custom Python code against the dataset in an isolated subprocess."""

    requires_llm = True

    name = "execute_dynamic_code"
    description = (
        "Execute custom Python code against the dataset when no other tool "
        "answers the question. A DataFrame `df` and column-kind `SCHEMA` "
        "are pre-loaded; assign the answer to RESULT. Runs in an isolated, "
        "restricted subprocess — no file/network access, no imports outside "
        "pandas/numpy/scipy/sklearn/duckdb/polars/math/statistics/json/"
        "datetime/re/collections/itertools. Use only for questions the "
        "other analysis tools cannot address."
    )

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        code: str,
        **_: Any,
    ) -> dict[str, Any]:
        if not code or not code.strip():
            raise ToolExecutionError("No code was provided to execute.")
        if not Path(file_path).exists():
            raise ToolExecutionError(f"Dataset file not found: {file_path}")

        sandbox_result = run_sandboxed(code=code, dataset_ref=file_path)

        if sandbox_result.status == "ok":
            return {
                "summary": f"Executed successfully in {sandbox_result.duration_ms:.0f} ms.",
                "status": "ok",
                "result": sandbox_result.result,
                "stdout": sandbox_result.stdout,
                "error_type": None,
                "traceback": None,
                "hint": None,
                "duration_ms": sandbox_result.duration_ms,
            }

        return {
            "summary": (
                f"Execution failed ({sandbox_result.error_type}): "
                f"{sandbox_result.hint or 'see traceback for details.'}"
            ),
            "status": "error",
            "result": None,
            "stdout": sandbox_result.stdout,
            "error_type": sandbox_result.error_type,
            "traceback": sandbox_result.traceback,
            "hint": sandbox_result.hint,
            "duration_ms": sandbox_result.duration_ms,
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Path to the (cleaned) dataset.",
                "required": True,
            },
            "code": {
                "type": "string",
                "description": (
                    "Python code to execute against the dataset. Must assign "
                    "the final answer to a variable named RESULT at the top "
                    "level. A pandas DataFrame `df` and a `SCHEMA` dict "
                    "(column name -> semantic kind) are pre-loaded — do not "
                    "read or parse any file. Only pandas, numpy, scipy, "
                    "sklearn, duckdb, polars, math, statistics, json, "
                    "datetime, re, collections, itertools are available."
                ),
                "required": True,
            },
        }
