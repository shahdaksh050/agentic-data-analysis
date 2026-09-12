"""Tests for src/tools/dynamic_code.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.tools.base import ToolExecutionError
from src.tools.dynamic_code import DynamicCodeExecutionTool
from tests.fixtures import single_column


class TestDynamicCodeExecutionTool:
    def test_execute_success(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        tool = DynamicCodeExecutionTool()
        output = tool.execute(
            file_path=str(dataset), code="RESULT = float(df['value'].mean())\n"
        )
        assert "summary" in output
        assert output["status"] == "ok"
        assert isinstance(output["result"], float)

    def test_execute_reports_sandbox_failure_without_raising(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        tool = DynamicCodeExecutionTool()
        output = tool.execute(file_path=str(dataset), code="x = 1\n")  # no RESULT
        assert output["status"] == "error"
        assert output["error_type"] == "static_check"
        assert "summary" in output

    def test_run_wraps_sandbox_failure_as_tool_success(self, tmp_path: Path) -> None:
        # The tool did its job (ran the sandbox and reported faithfully);
        # the *generated code* failing is not a tool-infrastructure error.
        dataset = single_column(tmp_path)
        tool = DynamicCodeExecutionTool()
        tool_result = tool.run(file_path=str(dataset), code="x = 1\n")
        assert tool_result.status == "success"
        assert tool_result.output["status"] == "error"

    def test_empty_code_raises(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        tool = DynamicCodeExecutionTool()
        with pytest.raises(ToolExecutionError):
            tool.execute(file_path=str(dataset), code="")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        tool = DynamicCodeExecutionTool()
        with pytest.raises(ToolExecutionError):
            tool.execute(file_path=str(tmp_path / "nope.csv"), code="RESULT = 1\n")

    def test_get_schema(self) -> None:
        tool = DynamicCodeExecutionTool()
        schema = tool.get_schema()
        assert schema["file_path"]["required"] is True
        assert schema["code"]["required"] is True


class TestRegistration:
    def test_tool_is_registered(self) -> None:
        from src.core.controller import ToolRegistry

        registry = ToolRegistry()
        assert registry.has("execute_dynamic_code")
